# ========== 现有脚本的快速训练优化版本 ========== 
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

# ========== 快速训练配置（基于你现有的脚本优化） ==========
DATA_ROOT = './OST'
BATCH_SIZE = 2  # 适中的批次大小
NUM_EPOCHS = 8   # 大幅减少epoch数
LR = 2e-3        # 提高学习率加快收敛
NUM_WORKERS = 0  # 保持单进程避免共享内存问题
MODEL_SAVE_PATH = './realESRGAN_fast_training.pth'
VAL_SPLIT = 0.05  # 减少验证集到5%

# ========== 训练加速策略 ==========
USE_MIXED_PRECISION = True  # 混合精度训练
GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积模拟更大batch size
VALIDATION_FREQUENCY = 2  # 每2个epoch验证一次
CHECKPOINT_FREQUENCY = 4  # 每4个epoch保存一次

# ========== 模型轻量化配置 ==========
# 方案A：轻量RRDBNet（推荐）
LITE_RRDB_CONFIG = {
    'num_in_ch': 1,
    'num_out_ch': 1,
    'num_feat': 32,      # 原64->32 (减少50%特征数)
    'num_block': 12,     # 原23->12 (减少约50%层数)
    'num_grow_ch': 16    # 原32->16 (减少50%增长通道)
}

# 方案B：超轻量RRDBNet（最快）
ULTRA_LITE_CONFIG = {
    'num_in_ch': 1,
    'num_out_ch': 1,
    'num_feat': 24,      # 进一步减少
    'num_block': 8,      # 进一步减少
    'num_grow_ch': 12    # 进一步减少
}

# ========== 数据优化策略 ==========
SCALE_FACTORS = [2, 4]  # 只训练2x和4x，移除3x
TARGET_SIZE = 256       # 从512进一步减小到256
FAST_RESIZE = True      # 使用最快的resize方法
SAMPLE_RATIO = 0.2      # 只使用20%的数据进行快速训练

# GPU优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ========== 快速数据处理函数 ==========
def fast_resize(img, size, method='nearest'):
    """快速resize，使用最快的插值方法"""
    if method == 'nearest':
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    elif method == 'linear':
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

def apply_fast_degradation(img, scale_factor):
    """快速降质处理"""
    h, w = img.shape[:2]
    new_h, new_w = max(1, h // scale_factor), max(1, w // scale_factor)
    # 使用最快的nearest neighbor
    lr_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return lr_img

# ========== 快速数据集类 ==========
class FastTrainingDataset(Dataset):
    def __init__(self, image_paths, target_size=TARGET_SIZE, scale_factors=SCALE_FACTORS, sample_ratio=SAMPLE_RATIO):
        self.target_size = target_size
        self.scale_factors = scale_factors
        
        # 采样减少数据量
        if sample_ratio < 1.0:
            num_samples = max(200, int(len(image_paths) * sample_ratio))
            self.image_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
            print(f"Using {len(self.image_paths)} samples ({sample_ratio*100:.1f}% of total)")
        else:
            self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # 快速图像加载
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
            
            # 快速resize到目标尺寸
            if hr_img.shape != (self.target_size, self.target_size):
                hr_img = fast_resize(hr_img, self.target_size, 'nearest' if FAST_RESIZE else 'linear')
            
            # 快速降质
            scale_factor = random.choice(self.scale_factors)
            lr_img = apply_fast_degradation(hr_img, scale_factor)
            lr_img = fast_resize(lr_img, self.target_size, 'nearest' if FAST_RESIZE else 'linear')
            
            # 转tensor
            lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).float()
            hr_tensor = torch.from_numpy(hr_img).unsqueeze(0).float()
            
            return lr_tensor, hr_tensor, scale_factor
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Fallback数据
            lr_tensor = torch.rand(1, self.target_size, self.target_size) * 0.8
            hr_tensor = torch.rand(1, self.target_size, self.target_size)
            return lr_tensor, hr_tensor, 2

# ========== 快速数据收集（限制数量） ==========
def get_fast_image_paths(root_dir, max_per_category=200):
    """快速数据收集，限制每类图片数量"""
    image_paths = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    if not os.path.exists(root_dir):
        print(f"Error: Dataset directory '{root_dir}' not found!")
        return []
    
    total_time = time.time()
    for subdir in os.listdir(root_dir):
        subpath = os.path.join(root_dir, subdir)
        if not os.path.isdir(subpath):
            continue
            
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
        print(f"{subdir}: {len(category_paths)} images")
    
    load_time = time.time() - total_time
    print(f"Data loading completed in {load_time:.2f}s, found {len(image_paths)} images")
    return image_paths

# ========== 快速训练主函数 ==========
def fast_train():
    print("Starting optimized fast training...")
    training_start_time = time.time()
    
    # 快速数据加载
    all_image_paths = get_fast_image_paths(DATA_ROOT, max_per_category=300)
    
    if len(all_image_paths) == 0:
        print("No images found!")
        return
    
    # 快速数据划分
    random.shuffle(all_image_paths)
    split_idx = int(len(all_image_paths) * (1 - VAL_SPLIT))
    train_paths = all_image_paths[:split_idx]
    val_paths = all_image_paths[split_idx:]
    
    print(f"Training: {len(train_paths)}, Validation: {len(val_paths)}")
    
    # 创建数据集和数据加载器
    train_dataset = FastTrainingDataset(train_paths)
    val_dataset = FastTrainingDataset(val_paths, sample_ratio=1.0)  # 验证集不采样
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        
        # 显示GPU内存
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {total_memory:.1f}GB")
    
    # 创建轻量化模型（可选择配置）
    model_config = ULTRA_LITE_CONFIG  # 使用超轻量配置，如需要可改为LITE_RRDB_CONFIG
    model = RRDBNet(**model_config)
    model = model.to(device)
    
    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Model config: {model_config}")
    
    # 优化器和调度器
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # 使用OneCycleLR进行快速训练
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LR, 
        epochs=NUM_EPOCHS, 
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 混合精度训练
    scaler = GradScaler() if USE_MIXED_PRECISION and torch.cuda.is_available() else None
    if scaler:
        print("Using mixed precision training")
    
    best_val_loss = float('inf')
    accumulated_loss = 0
    
    print(f"Training configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"  Scale factors: {SCALE_FACTORS}")
    
    # 训练循环
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        for batch_idx, (lr_img, hr_img, scale_factors) in enumerate(train_pbar):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)
            
            # 混合精度前向传播
            if scaler:
                with autocast():
                    pred = model(lr_img)
                    if pred.shape[-2:] != hr_img.shape[-2:]:
                        pred = F.interpolate(pred, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
                    loss = criterion(pred, hr_img) / GRADIENT_ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                
                # 梯度累积
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                pred = model(lr_img)
                if pred.shape[-2:] != hr_img.shape[-2:]:
                    pred = F.interpolate(pred, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(pred, hr_img) / GRADIENT_ACCUMULATION_STEPS
                
                loss.backward()
                accumulated_loss += loss.item()
                
                # 梯度累积
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # 更新进度条
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                train_pbar.set_postfix({
                    'Loss': f'{accumulated_loss:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                accumulated_loss = 0
            
            # 定期清理GPU内存
            if batch_idx % 30 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        
        # 验证阶段（按频率执行）
        val_loss = 0
        if (epoch + 1) % VALIDATION_FREQUENCY == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc='Validation', leave=False)
                for lr_img, hr_img, scale_factors in val_pbar:
                    lr_img = lr_img.to(device, non_blocking=True)
                    hr_img = hr_img.to(device, non_blocking=True)
                    
                    if scaler:
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
                    val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
            
            val_loss /= len(val_loader)
            
            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f'[Save] Best model at epoch {epoch+1} | Val Loss={val_loss:.4f}')
        
        # 定期保存检查点
        if (epoch + 1) % CHECKPOINT_FREQUENCY == 0:
            checkpoint_path = MODEL_SAVE_PATH.replace('.pth', f'_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Time={epoch_time:.1f}s')
        
        # GPU内存状态
        if torch.cuda.is_available() and (epoch + 1) % 2 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f'GPU Memory: {allocated:.2f}GB allocated')
    
    total_time = time.time() - training_start_time
    avg_epoch_time = total_time / NUM_EPOCHS
    
    print(f'\nFast training completed!')
    print(f'Total time: {total_time/60:.1f} minutes')
    print(f'Average per epoch: {avg_epoch_time:.1f}s')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Speed improvement: ~{120/avg_epoch_time:.1f}x faster than original')

if __name__ == '__main__':
    print("=" * 60)
    print("Optimized Fast Real-ESRGAN Training")
    print("=" * 60)
    print(f"Dataset: {DATA_ROOT}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Scale factors: {SCALE_FACTORS}")
    print(f"Sample ratio: {SAMPLE_RATIO*100:.1f}%")
    print("=" * 60)
    
    fast_train()
