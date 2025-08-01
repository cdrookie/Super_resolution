# ========== DL-SMLM数据集自动遍历训练+验证集脚本 ========== 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import tifffile
import random

# 动态导入U2NET和FDConv
import importlib.util
u2net_path = os.path.join(os.path.dirname(__file__), 'U_2_Net_master', 'model', 'u2net.py')
spec_u2net = importlib.util.spec_from_file_location('u2net', u2net_path)
u2net_module = importlib.util.module_from_spec(spec_u2net)
spec_u2net.loader.exec_module(u2net_module)
U2NET = u2net_module.U2NET
fdconv_path = os.path.join(os.path.dirname(__file__), 'FDConv_main', 'FDConv_detection', 'mmdet_custom', 'FDConv.py')
spec_fdconv = importlib.util.spec_from_file_location('FDConv', fdconv_path)
fdconv_module = importlib.util.module_from_spec(spec_fdconv)
spec_fdconv.loader.exec_module(fdconv_module)
FDConv = fdconv_module.FDConv

# ========== 可配置参数 ==========
DATA_ROOT = './DL_SMLM'  # 数据集根目录
BATCH_SIZE = 4
NUM_EPOCHS = 100
LR = 5e-4  # 调大学习率
NUM_WORKERS = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_PATH = './u2net_fdconv_best_1.pth'
VAL_SPLIT = 0.2  # 验证集比例

# ========== 替换U2NET部分卷积为FDConv（示例：只替换stage4的第一个卷积） ==========
def replace_conv_with_fdconv(model):
    for name, module in model.stage4.named_children():
        if isinstance(module, nn.Conv2d):
            in_ch = module.in_channels
            out_ch = module.out_channels
            kernel_size = module.kernel_size
            padding = module.padding
            bias = module.bias is not None
            fdconv = FDConv(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias)
            setattr(model.stage4, name, fdconv)
    return model

# ========== 数据集类：自动遍历DL_SMLM结构，读取WF和SR图像 ==========
class SMLMFolderDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        wf_img = tifffile.imread(sample['wf_path'])
        if wf_img.ndim == 3:
            wf_img = wf_img.sum(axis=0)
        wf_img = wf_img.astype('float32') / (2**16 - 1)
        sr_img = tifffile.imread(sample['sr_path'])
        sr_img = sr_img.astype('float32') / (2**32 - 1)
        wf_img = torch.from_numpy(wf_img).unsqueeze(0).float()
        sr_img = torch.from_numpy(sr_img).unsqueeze(0).float()
        return wf_img, sr_img

# ========== 样本划分 ==========
def get_all_samples(root_dir):
    samples = []
    for subdir in os.listdir(root_dir):
        subpath = os.path.join(root_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        for cell in os.listdir(subpath):
            cell_path = os.path.join(subpath, cell)
            if not os.path.isdir(cell_path):
                continue
            wf_files = [f for f in os.listdir(cell_path) if ('WF' in f or 'wf' in f) and f.endswith('.tif')]
            sr_files = [f for f in os.listdir(cell_path) if ('SR' in f or 'sr' in f) and f.endswith('.tif')]
            for wf_file in wf_files:
                for sr_file in sr_files:
                    samples.append({
                        'wf_path': os.path.join(cell_path, wf_file),
                        'sr_path': os.path.join(cell_path, sr_file)
                    })
    return samples

all_samples = get_all_samples(DATA_ROOT)
random.shuffle(all_samples)
split_idx = int(len(all_samples) * (1 - VAL_SPLIT))
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]
train_dataset = SMLMFolderDataset(train_samples)
val_dataset = SMLMFolderDataset(val_samples)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ========== 训练主流程 ==========
def train():
    model = U2NET(in_ch=1, out_ch=1)
    model = replace_conv_with_fdconv(model)
    model = model.to(DEVICE)
    criterion = nn.L1Loss()  # 损失函数改为L1Loss
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float('inf')
    import torch.nn.functional as F
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for wf_img, sr_img in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} - Train'):
            wf_img = wf_img.to(DEVICE)
            sr_img = sr_img.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(wf_img)[0]
            pred = torch.sigmoid(outputs)
            pred_up = F.interpolate(pred, size=(2048, 2048), mode='bilinear', align_corners=False)
            loss = criterion(pred_up, sr_img)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * wf_img.size(0)
        train_loss /= len(train_loader.dataset)
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for wf_img, sr_img in val_loader:
                wf_img = wf_img.to(DEVICE)
                sr_img = sr_img.to(DEVICE)
                outputs = model(wf_img)[0]
                pred = torch.sigmoid(outputs)
                pred_up = F.interpolate(pred, size=(2048, 2048), mode='bilinear', align_corners=False)
                loss = criterion(pred_up, sr_img)
                val_loss += loss.item() * wf_img.size(0)
        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'[Save] Best model at epoch {epoch+1} | Val Loss={val_loss:.4f}')

if __name__ == '__main__':
    train()
