# ========== 轻量化Real-ESRGAN训练脚本（快速训练版） ========== 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import tifffile
import random
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import time

# 导入Real-ESRGAN模型
from basicsr.archs.rrdbnet_arch import RRDBNet

# ========== 快速训练优化参数 ==========
DATA_ROOT = './OST'
BATCH_SIZE = 4  # 增加批次大小提高GPU利用率
NUM_EPOCHS = 15  # 减少训练轮数
LR = 1e-3  # 提高学习率加快收敛
NUM_WORKERS = 0  # 避免多进程问题
MODEL_SAVE_PATH = './realESRGAN_lightweight_best.pth'
VAL_SPLIT = 0.1  # 减少验证集比例
USE_MIXED_PRECISION = True  # 使用混合精度训练

# ========== 轻量化模型参数 ==========
# 原始：num_feat=64, num_block=23, num_grow_ch=32
# 轻量化：减少特征数和块数
LITE_MODEL_CONFIG = {
    'num_in_ch': 1,
    'num_out_ch': 1, 
    'num_feat': 32,      # 从64减少到32
    'num_block': 8,      # 从23减少到8
    'num_grow_ch': 16    # 从32减少到16
}

# ========== GPU内存优化 ==========
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ========== 训练优化参数 ==========
SCALE_FACTORS = [2, 4]  # 只训练2x和4x，减少3x
TARGET_SIZE = 256  # 进一步减小图像尺寸
SAMPLE_SUBSET = 0.3  # 只使用30%的数据集进行快速训练

# ========== 分辨率降质函数（优化版） ==========
def apply_resolution_degradation(img, scale_factor):
    """快速分辨率降质"""
    h, w = img.shape[:2]
    new_h, new_w = h // scale_factor, w // scale_factor
    # 使用更快的插值方法
    lr_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return lr_img

def upsample_to_target_size(img, target_size):
    """快速上采样"""
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    upsampled_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    return upsampled_img

# ========== 快速数据集类 ==========
class FastOSTDataset(Dataset):
    def __init__(self, image_paths, target_size=TARGET_SIZE, scale_factors=SCALE_FACTORS):
        self.image_paths = image_paths
        self.target_size = target_size
        self.scale_factors = scale_factors
        
        # 预处理：只保留指定比例的数据
        if SAMPLE_SUBSET < 1.0:
            num_samples = int(len(image_paths) * SAMPLE_SUBSET)
            self.image_paths = random.sample(image_paths, num_samples)
            print(f"Using subset: {len(self.image_paths)} samples ({SAMPLE_SUBSET*100:.1f}% of total)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # 快速图像读取和处理
            ext = os.path.splitext(img_path)[-1].lower()
            if ext in ['.tif', '.tiff']:
                hr_img = tifffile.imread(img_path)
                if hr_img.ndim == 3:
                    hr_img = hr_img.mean(axis=0)
                hr_img = hr_img.astype('float32')
                if hr_img.max() > 1:
                    hr_img = hr_img / hr_img.max()
            else:
                hr_img = Image.open(img_path).convert('L')
                hr_img = np.array(hr_img).astype('float32') / 255.0
            
            # 快速resize
            if hr_img.shape != (self.target_size, self.target_size):
                hr_img = cv2.resize(hr_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            
            # 随机选择降质倍数
            scale_factor = random.choice(self.scale_factors)
            
            # 生成低分辨率版本
            lr_img = apply_resolution_degradation(hr_img, scale_factor)
            lr_img_upsampled = upsample_to_target_size(lr_img, self.target_size)
            
            # 转换为tensor
            lr_tensor = torch.from_numpy(lr_img_upsampled).unsqueeze(0).float()
            hr_tensor = torch.from_numpy(hr_img).unsqueeze(0).float()
            
            return lr_tensor, hr_tensor, scale_factor
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回随机数据作为fallback
            lr_tensor = torch.rand(1, self.target_size, self.target_size)
            hr_tensor = torch.rand(1, self.target_size, self.target_size)
            return lr_tensor, hr_tensor, 2

# ========== 快速数据收集 ==========
def get_all_image_paths(root_dir, max_per_category=500):
    """快速收集图像路径，限制每个类别的最大数量"""
    image_paths = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    if not os.path.exists(root_dir):
        print(f"Error: Dataset directory '{root_dir}' not found!")
        return []
    
    for subdir in os.listdir(root_dir):
        subpath = os.path.join(root_dir, subdir)
        if not os.path.isdir(subpath):
            continue
            
        print(f"Processing category: {subdir}")
        category_paths = []
        
        for filename in os.listdir(subpath):
            if len(category_paths) >= max_per_category:
                break
            filepath = os.path.join(subpath, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[-1].lower()
                if ext in supported_extensions:
                    category_paths.append(filepath)
        
        image_paths.extend(category_paths)
        print(f"  Found {len(category_paths)} images in {subdir}")
    
    print(f"Found {len(image_paths)} images in total (limited per category)")
    return image_paths

# 快速数据加载
print("Loading image paths...")
start_time = time.time()
all_image_paths = get_all_image_paths(DATA_ROOT, max_per_category=300)  # 限制每类最多300张
load_time = time.time() - start_time
print(f"Data loading completed in {load_time:.2f}s")

if len(all_image_paths) == 0:
    print(f"Error: No images found in '{DATA_ROOT}'!")
    exit(1)

# 快速划分数据集
random.shuffle(all_image_paths)
split_idx = int(len(all_image_paths) * (1 - VAL_SPLIT))
train_paths = all_image_paths[:split_idx]
val_paths = all_image_paths[split_idx:]

print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")
print(f"Scale factors: {SCALE_FACTORS}")
print(f"Target resolution: {TARGET_SIZE}x{TARGET_SIZE}")

# 创建快速数据加载器
train_dataset = FastOSTDataset(train_paths)
val_dataset = FastOSTDataset(val_paths)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

# ========== 快速训练函数 ==========
def fast_train():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # 创建轻量化模型
    model = RRDBNet(**LITE_MODEL_CONFIG)
    model = model.to(device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # 学习率调度器（更激进）
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=NUM_EPOCHS, 
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    # 混合精度训练
    scaler = GradScaler() if USE_MIXED_PRECISION and torch.cuda.is_available() else None
    
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    print(f"Starting fast training...")
    print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Mixed precision: {USE_MIXED_PRECISION and torch.cuda.is_available()}")
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        for batch_idx, (lr_img, hr_img, scale_factors) in enumerate(train_pbar):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            if scaler is not None:
                with autocast():
                    pred = model(lr_img)
                    if pred.shape[-2:] != hr_img.shape[-2:]:
                        pred = F.interpolate(pred, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
                    loss = criterion(pred, hr_img)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(lr_img)
                if pred.shape[-2:] != hr_img.shape[-2:]:
                    pred = F.interpolate(pred, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(pred, hr_img)
                loss.backward()
                optimizer.step()
            
            scheduler.step()  # OneCycleLR需要每个batch更新
            
            batch_loss = loss.item()
            train_loss += batch_loss
            
            train_pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 更频繁的内存清理
            if batch_idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        
        # 快速验证（每3个epoch验证一次）
        if (epoch + 1) % 3 == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for lr_img, hr_img, scale_factors in val_loader:
                    lr_img = lr_img.to(device, non_blocking=True)
                    hr_img = hr_img.to(device, non_blocking=True)
                    
                    if scaler is not None:
                        with autocast():
                            pred = model(lr_img)
                            if pred.shape[-2:] != hr_img.shape[-2:]:
                                pred = F.interpolate(pred, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
                            loss = criterion(pred, hr_img)
                    else:
                        pred = model(lr_img)
                        if pred.shape[-2:] != hr_img.shape[-2:]:
                            pred = F.interpolate(pred, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
                        loss = criterion(pred, hr_img)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f'[Save] Best model at epoch {epoch+1} | Val Loss={val_loss:.4f}')
        else:
            val_loss = 0
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Time={epoch_time:.1f}s')
    
    total_time = time.time() - training_start_time
    print(f'\nTraining completed in {total_time/60:.1f} minutes!')
    print(f'Average time per epoch: {total_time/NUM_EPOCHS:.1f}s')
    print(f'Best validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    print("=" * 60)
    print("Fast Real-ESRGAN Training Script (Lightweight)")
    print("=" * 60)
    print(f"Dataset: {DATA_ROOT}")
    print(f"Model config: {LITE_MODEL_CONFIG}")
    print(f"Scale factors: {SCALE_FACTORS}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Sample subset: {SAMPLE_SUBSET*100:.1f}%")
    print("=" * 60)
    
    fast_train()
