# ========== 超快训练脚本：使用SRCNN轻量模型 ========== 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import time

# ========== 轻量SRCNN模型定义 ==========
class SRCNN(nn.Module):
    """
    超轻量的SRCNN模型，参数量极少，训练速度极快
    原始SRCNN只有3层卷积，非常适合快速实验
    """
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        # 特征提取层
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        # 非线性映射层
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        # 重构层
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class UltraLightSR(nn.Module):
    """
    极轻量超分模型，参数量更少
    """
    def __init__(self, num_channels=1, scale_factor=4):
        super(UltraLightSR, self).__init__()
        self.scale_factor = scale_factor
        
        # 轻量特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 输出层
        self.output_layer = nn.Conv2d(32, num_channels, 3, padding=1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        return output

# ========== 超快训练参数 ==========
DATA_ROOT = './OST'
BATCH_SIZE = 8  # 轻量模型可以用更大批次
NUM_EPOCHS = 10  # 更少的训练轮数
LR = 2e-3  # 更高的学习率
NUM_WORKERS = 0
MODEL_SAVE_PATH = './srcnn_ultrafast_best.pth'
VAL_SPLIT = 0.05  # 更少的验证数据
USE_MIXED_PRECISION = True

# ========== 训练优化参数 ==========
SCALE_FACTORS = [2, 4]  # 只训练主要scale
TARGET_SIZE = 128  # 更小的图像尺寸
SAMPLE_SUBSET = 0.1  # 只用10%数据快速训练
MAX_IMAGES_PER_CATEGORY = 100  # 每类最多100张图

# GPU优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True

# ========== 超快数据处理 ==========
def apply_resolution_degradation(img, scale_factor):
    """超快降质处理"""
    h, w = img.shape[:2]
    new_h, new_w = max(1, h // scale_factor), max(1, w // scale_factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def upsample_to_target_size(img, target_size):
    """超快上采样"""
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    return cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)

# ========== 超快数据集 ==========
class UltraFastDataset(Dataset):
    def __init__(self, image_paths, target_size=TARGET_SIZE, scale_factors=SCALE_FACTORS):
        self.target_size = target_size
        self.scale_factors = scale_factors
        
        # 大幅减少数据量
        if SAMPLE_SUBSET < 1.0:
            num_samples = max(100, int(len(image_paths) * SAMPLE_SUBSET))
            self.image_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
        else:
            self.image_paths = image_paths
            
        print(f"Dataset size: {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # 超快图像处理
            if img_path.lower().endswith(('.tif', '.tiff')):
                # 简化TIFF处理
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("Failed to load TIFF")
                hr_img = img.astype('float32') / 255.0
            else:
                hr_img = Image.open(img_path).convert('L')
                hr_img = np.array(hr_img).astype('float32') / 255.0
            
            # 快速resize
            hr_img = cv2.resize(hr_img, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
            
            # 随机降质
            scale_factor = random.choice(self.scale_factors)
            lr_img = apply_resolution_degradation(hr_img, scale_factor)
            lr_img = upsample_to_target_size(lr_img, self.target_size)
            
            # 转tensor
            lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).float()
            hr_tensor = torch.from_numpy(hr_img).unsqueeze(0).float()
            
            return lr_tensor, hr_tensor
        
        except Exception as e:
            # fallback数据
            lr_tensor = torch.rand(1, self.target_size, self.target_size) * 0.5
            hr_tensor = torch.rand(1, self.target_size, self.target_size)
            return lr_tensor, hr_tensor

# ========== 超快数据加载 ==========
def get_ultra_fast_image_paths(root_dir):
    """超快数据收集，大幅限制数据量"""
    image_paths = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']  # 排除TIFF加快速度
    
    if not os.path.exists(root_dir):
        print(f"Error: Dataset directory '{root_dir}' not found!")
        return []
    
    total_collected = 0
    for subdir in os.listdir(root_dir):
        if total_collected >= 1000:  # 最多1000张图片
            break
            
        subpath = os.path.join(root_dir, subdir)
        if not os.path.isdir(subpath):
            continue
            
        category_count = 0
        for filename in os.listdir(subpath):
            if category_count >= MAX_IMAGES_PER_CATEGORY:
                break
                
            filepath = os.path.join(subpath, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[-1].lower()
                if ext in supported_extensions:
                    image_paths.append(filepath)
                    category_count += 1
                    total_collected += 1
        
        print(f"Category {subdir}: {category_count} images")
    
    print(f"Total collected: {len(image_paths)} images")
    return image_paths

# ========== 超快训练主函数 ==========
def ultra_fast_train():
    print("Starting ultra-fast training...")
    start_time = time.time()
    
    # 数据加载
    print("Loading data...")
    all_image_paths = get_ultra_fast_image_paths(DATA_ROOT)
    
    if len(all_image_paths) == 0:
        print("No images found!")
        return
    
    # 快速数据划分
    random.shuffle(all_image_paths)
    split_idx = int(len(all_image_paths) * (1 - VAL_SPLIT))
    train_paths = all_image_paths[:split_idx]
    val_paths = all_image_paths[split_idx:] if split_idx < len(all_image_paths) else all_image_paths[-10:]
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # 创建数据集
    train_dataset = UltraFastDataset(train_paths)
    val_dataset = UltraFastDataset(val_paths)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 设置设备和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 选择超轻量模型
    model = UltraLightSR(num_channels=1)  # 或者使用 SRCNN()
    model = model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.3f}M)")
    
    # 优化器和损失
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # 混合精度
    scaler = GradScaler() if USE_MIXED_PRECISION and torch.cuda.is_available() else None
    
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for lr_img, hr_img in pbar:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    pred = model(lr_img)
                    loss = criterion(pred, hr_img)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(lr_img)
                loss = criterion(pred, hr_img)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # 验证（每2个epoch一次）
        if (epoch + 1) % 2 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for lr_img, hr_img in val_loader:
                    lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                    pred = model(lr_img)
                    val_loss += criterion(pred, hr_img).item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f'[Save] Best model | Val Loss: {val_loss:.4f}')
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}: Train={train_loss:.4f}, Time={epoch_time:.1f}s')
        
        # GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f'\nUltra-fast training completed in {total_time/60:.1f} minutes!')
    print(f'Average: {total_time/NUM_EPOCHS:.1f}s per epoch')

if __name__ == '__main__':
    print("=" * 50)
    print("Ultra-Fast Super Resolution Training")
    print("=" * 50)
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Data subset: {SAMPLE_SUBSET*100:.1f}%")
    print("=" * 50)
    
    ultra_fast_train()
