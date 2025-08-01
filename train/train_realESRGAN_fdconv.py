# ========== 改进版 Real-ESRGAN 训练脚本（适配DL-SMLM数据集） ==========
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import tifffile
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 导入Real-ESRGAN模型
from basicsr.archs.rrdbnet_arch import RRDBNet

# ========== 配置参数 ==========
DATA_ROOT = './DL_SMLM'          # 数据集路径
BATCH_SIZE = 4                   # 批大小
NUM_EPOCHS = 100                 # 训练轮数
LR = 5e-4                        # 学习率
NUM_WORKERS = 0                  # 数据加载线程数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_PATH = './realESRGAN_dlsmlm_best.pth'
VAL_SPLIT = 0.2                  # 验证集比例
DEBUG_SAMPLE = True              # 调试时显示样本图像

# ========== 改进的数据集类 ==========
class SMLMDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 读取WF图像（低分辨率）
        wf_img = tifffile.imread(sample['wf_path'])
        if wf_img.ndim == 3:
            wf_img = wf_img.sum(axis=0)  # 多帧求和（如有）
        wf_img = wf_img.astype('float32') / 65535.0  # 归一化到[0,1]
        
        # 2. 改进的SR图像处理（关键修改）
        sr_img = tifffile.imread(sample['sr_path'])
        sr_img = sr_img.astype('float32')
        
        # 方法1：对数变换 + 分位数归一化（推荐）
        sr_img = np.log1p(sr_img)  # 压缩高动态范围
        sr_img = sr_img / np.percentile(sr_img, 99.9)  # 用99.9%分位数归一化
        sr_img = np.clip(sr_img, 0, 1)  # 限制到[0,1]
        
        # 方法2（可选）：高斯平滑密度图
        # sr_img = gaussian_filter(sr_img, sigma=2)
        # sr_img = sr_img / sr_img.max()
        
        # 调试输出第一个样本
        if DEBUG_SAMPLE and idx == 0:
            print(f"[Debug] SR stats: min={sr_img.min():.4f}, max={sr_img.max():.4f}, mean={sr_img.mean():.4f}")
            plt.figure(figsize=(12,4))
            plt.subplot(121); plt.imshow(wf_img, cmap='gray'); plt.title('WF (LR)')
            plt.subplot(122); plt.imshow(sr_img, cmap='hot'); plt.title('Processed SR (HR)')
            plt.colorbar(); plt.show()
        
        # 转换为Tensor
        wf_img = torch.from_numpy(wf_img).unsqueeze(0).float()  # 添加通道维度
        sr_img = torch.from_numpy(sr_img).unsqueeze(0).float()
        return wf_img, sr_img

# ========== 数据加载 ==========
def get_samples(root_dir):
    samples = []
    for subdir in os.listdir(root_dir):
        subpath = os.path.join(root_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        for cell in os.listdir(subpath):
            cell_path = os.path.join(subpath, cell)
            if not os.path.isdir(cell_path):
                continue
            # 匹配WF和SR文件（不区分大小写）
            wf_files = [f for f in os.listdir(cell_path) if ('WF' in f.upper()) and f.endswith('.tif')]
            sr_files = [f for f in os.listdir(cell_path) if ('SR' in f.upper()) and f.endswith('.tif')]
            # 创建配对样本
            for wf_file in wf_files:
                for sr_file in sr_files:
                    samples.append({
                        'wf_path': os.path.join(cell_path, wf_file),
                        'sr_path': os.path.join(cell_path, sr_file)
                    })
    return samples

# 数据集划分
all_samples = get_samples(DATA_ROOT)
random.shuffle(all_samples)
split_idx = int(len(all_samples) * (1 - VAL_SPLIT))
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]

train_dataset = SMLMDataset(train_samples)
val_dataset = SMLMDataset(val_samples)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ========== 改进的模型定义 ==========
class CustomRRDBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rrdb = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=23, num_grow_ch=32)
        self.sigmoid = nn.Sigmoid()  # 输出限制到[0,1]
    
    def forward(self, x):
        x = self.rrdb(x)
        return self.sigmoid(x)

# ========== 训练流程 ==========
def train():
    model = CustomRRDBNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # 改进的损失函数：加权L1损失
    def weighted_l1_loss(pred, target):
        weight = torch.clamp(target * 10, min=0.1, max=1.0)  # 热点区域权重更高
        return torch.mean(weight * torch.abs(pred - target))
    
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for wf_imgs, sr_imgs in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
            wf_imgs = wf_imgs.to(DEVICE)
            sr_imgs = sr_imgs.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(wf_imgs)
            # 自动上采样到SR目标尺寸
            preds = torch.nn.functional.interpolate(preds, size=sr_imgs.shape[-2:], mode='bilinear', align_corners=False)
            loss = weighted_l1_loss(preds, sr_imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for wf_imgs, sr_imgs in val_loader:
                wf_imgs = wf_imgs.to(DEVICE)
                sr_imgs = sr_imgs.to(DEVICE)
                preds = model(wf_imgs)
                preds = torch.nn.functional.interpolate(preds, size=sr_imgs.shape[-2:], mode='bilinear', align_corners=False)
                val_loss += weighted_l1_loss(preds, sr_imgs).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'[Saved] Best model at epoch {epoch+1}')

if __name__ == '__main__':
    train()