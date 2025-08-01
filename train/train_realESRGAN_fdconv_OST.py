# ========== OST数据集自动遍历训练+验证集脚本（Real-ESRGAN主干+随机降质） ========== 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import tifffile
import random
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F

# 导入Real-ESRGAN模型
from basicsr.archs.rrdbnet_arch import RRDBNet

# ========== 可配置参数 ==========
DATA_ROOT = './OST'  # 数据集根目录
BATCH_SIZE = 4
NUM_EPOCHS = 100
LR = 5e-4
NUM_WORKERS = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_PATH = './realESRGAN_fdconv_OST_best.pth'
VAL_SPLIT = 0.2  # 验证集比例

# ========== 随机降质参数 ==========
SCALE_FACTORS = [2, 4, 8]  # 下采样倍数
BLUR_KERNEL_SIZES = [3, 5, 7]  # 模糊核大小
NOISE_LEVELS = [0.01, 0.02, 0.05]  # 噪声强度

# ========== 降质函数 ==========
def apply_downsampling(img, scale_factor):
    """下采样"""
    h, w = img.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # 再上采样回原尺寸
    img_upsampled = cv2.resize(img_resized, (w, h), interpolation=cv2.INTER_CUBIC)
    return img_upsampled

def apply_blur(img, kernel_size):
    """模糊"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_noise(img, noise_level):
    """添加高斯噪声"""
    noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
    img_noisy = img + noise
    return np.clip(img_noisy, 0, 1)

def random_degradation(img):
    """随机降质：6种组合方式（排除三种同时使用）"""
    # 转换为numpy格式
    if isinstance(img, torch.Tensor):
        img_np = img.squeeze().cpu().numpy()
    else:
        img_np = img.copy()
    
    # 确保图像在0-1范围内
    img_np = img_np.astype(np.float32)
    if img_np.max() > 1:
        img_np = img_np / img_np.max()
    
    # 6种降质组合方式
    degradation_types = [
        'downsample',  # 仅下采样
        'blur',        # 仅模糊
        'noise',       # 仅噪声
        'downsample_blur',    # 下采样+模糊
        'downsample_noise',   # 下采样+噪声
        'blur_noise'          # 模糊+噪声
    ]
    
    # 随机选择一种降质方式
    deg_type = random.choice(degradation_types)
    
    # 随机选择参数
    scale_factor = random.choice(SCALE_FACTORS)
    kernel_size = random.choice(BLUR_KERNEL_SIZES)
    noise_level = random.choice(NOISE_LEVELS)
    
    # 应用降质
    if deg_type == 'downsample':
        img_deg = apply_downsampling(img_np, scale_factor)
    elif deg_type == 'blur':
        img_deg = apply_blur(img_np, kernel_size)
    elif deg_type == 'noise':
        img_deg = apply_noise(img_np, noise_level)
    elif deg_type == 'downsample_blur':
        img_deg = apply_downsampling(img_np, scale_factor)
        img_deg = apply_blur(img_deg, kernel_size)
    elif deg_type == 'downsample_noise':
        img_deg = apply_downsampling(img_np, scale_factor)
        img_deg = apply_noise(img_deg, noise_level)
    elif deg_type == 'blur_noise':
        img_deg = apply_blur(img_np, kernel_size)
        img_deg = apply_noise(img_deg, noise_level)
    
    return img_deg, deg_type

# ========== 数据集类：自动遍历OST结构，读取图像并实时生成降质版本 ==========
class OSTDataset(Dataset):
    def __init__(self, image_paths, target_size=2048):
        self.image_paths = image_paths
        self.target_size = target_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 读取高分辨率图像（Ground Truth）
        ext = os.path.splitext(img_path)[-1].lower()
        if ext in ['.tif', '.tiff']:
            hr_img = tifffile.imread(img_path)
            if hr_img.ndim == 3:
                hr_img = hr_img.mean(axis=0)  # 转为灰度
            hr_img = hr_img.astype('float32')
            # 归一化
            if hr_img.max() > 1:
                hr_img = hr_img / hr_img.max()
        else:
            hr_img = Image.open(img_path).convert('L')
            hr_img = np.array(hr_img).astype('float32') / 255.0
        
        # 调整到目标尺寸
        if hr_img.shape != (self.target_size, self.target_size):
            hr_img = cv2.resize(hr_img, (self.target_size, self.target_size), interpolation=cv2.INTER_CUBIC)
        
        # 随机生成降质版本作为输入
        lr_img, deg_type = random_degradation(hr_img)
        
        # 转换为tensor
        lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr_img).unsqueeze(0).float()
        
        return lr_tensor, hr_tensor

# ========== 样本收集 ==========
def get_all_image_paths(root_dir):
    """遍历OST文件夹，收集所有图像路径"""
    image_paths = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    for subdir in os.listdir(root_dir):
        subpath = os.path.join(root_dir, subdir)
        if not os.path.isdir(subpath):
            continue
            
        print(f"Processing category: {subdir}")
        for filename in os.listdir(subpath):
            filepath = os.path.join(subpath, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[-1].lower()
                if ext in supported_extensions:
                    image_paths.append(filepath)
    
    print(f"Found {len(image_paths)} images in total")
    return image_paths

# 收集所有图像路径
all_image_paths = get_all_image_paths(DATA_ROOT)
random.shuffle(all_image_paths)

# 划分训练集和验证集
split_idx = int(len(all_image_paths) * (1 - VAL_SPLIT))
train_paths = all_image_paths[:split_idx]
val_paths = all_image_paths[split_idx:]

print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")

# 创建数据集和数据加载器
train_dataset = OSTDataset(train_paths)
val_dataset = OSTDataset(val_paths)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ========== 训练主流程 ==========
def train():
    # Real-ESRGAN主干，单通道输入输出
    model = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=23, num_grow_ch=32)
    model = model.to(DEVICE)
    
    # 损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_loss = float('inf')
    
    print(f"Starting training on device: {DEVICE}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")
    
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        
        for lr_img, hr_img in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} - Train'):
            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 前向传播
            pred = model(lr_img)
            
            # 确保预测结果与目标尺寸一致
            if pred.shape[-2:] != hr_img.shape[-2:]:
                pred = F.interpolate(pred, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
            
            # 计算损失
            loss = criterion(pred, hr_img)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * lr_img.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 训练损失爆炸检测
        if train_loss > 10:
            print(f'[Stop] Train loss爆炸（{train_loss:.4f}），自动停止训练！')
            break
        
        # 验证阶段
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img = lr_img.to(DEVICE)
                hr_img = hr_img.to(DEVICE)
                
                pred = model(lr_img)
                
                # 确保预测结果与目标尺寸一致
                if pred.shape[-2:] != hr_img.shape[-2:]:
                    pred = F.interpolate(pred, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
                
                loss = criterion(pred, hr_img)
                val_loss += loss.item() * lr_img.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'[Save] Best model at epoch {epoch+1} | Val Loss={val_loss:.4f}')

if __name__ == '__main__':
    # 检查数据集是否存在
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Dataset directory '{DATA_ROOT}' not found!")
        print("Please make sure the OST folder exists in the current directory.")
        exit(1)
    
    # 检查是否有图像文件
    if len(all_image_paths) == 0:
        print(f"Error: No images found in '{DATA_ROOT}'!")
        print("Please make sure the OST folder contains subdirectories with image files.")
        exit(1)
    
    train()
