# ========== OST数据集自动遍历训练+验证集脚本（Real-ESRGAN主干+分辨率降质+多GPU） ========== 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import tifffile
import random
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import argparse

# 导入Real-ESRGAN模型
from basicsr.archs.rrdbnet_arch import RRDBNet

# ========== 可配置参数 ==========
DATA_ROOT = './OST'  # 数据集根目录
BATCH_SIZE = 1  # 进一步减小批次大小以适应内存限制
NUM_EPOCHS = 100
LR = 5e-4
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 0))  # 默认使用单进程，避免共享内存问题
MODEL_SAVE_PATH = './realESRGAN_fdconv_OST_1_best.pth'
VAL_SPLIT = 0.2  # 验证集比例

# ========== 多GPU训练参数 ==========
USE_MULTI_GPU = True  # 是否使用多GPU训练
USE_DDP = False  # True=DistributedDataParallel, False=DataParallel
WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1

# ========== GPU内存优化设置 ==========
import os
# 设置PyTorch使用可扩展内存段，减少内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 启用CUDA内存缓存优化
torch.backends.cudnn.benchmark = True
# 启用混合精度训练
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 设置CUDA可见设备（可选，默认使用所有可用GPU）
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # 指定使用哪些GPU

# ========== 分辨率降质参数（可手动调整） ==========
SCALE_FACTORS = [2, 3, 4]  # 分辨率降低倍数：2倍、3倍、4倍
TARGET_SIZE = 512  # 进一步减小目标高分辨率尺寸以适应GPU内存限制

# ========== 分辨率降质函数 ==========
def apply_resolution_degradation(img, scale_factor):
    """
    分辨率降质：将图像分辨率降低指定倍数
    Args:
        img: 输入高分辨率图像 (numpy array)
        scale_factor: 降质倍数 (2, 3, 4)
    Returns:
        lr_img: 低分辨率图像 (numpy array)
    """
    h, w = img.shape[:2]
    
    # 计算低分辨率尺寸
    new_h, new_w = h // scale_factor, w // scale_factor
    
    # 下采样到低分辨率
    lr_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return lr_img

def upsample_to_target_size(img, target_size):
    """
    将低分辨率图像上采样到目标尺寸
    Args:
        img: 输入低分辨率图像
        target_size: 目标尺寸
    Returns:
        upsampled_img: 上采样后的图像
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    upsampled_img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    return upsampled_img

# ========== 数据集类：自动遍历OST结构，读取图像并生成不同分辨率版本 ==========
class OSTResolutionDataset(Dataset):
    def __init__(self, image_paths, target_size=TARGET_SIZE, scale_factors=SCALE_FACTORS):
        self.image_paths = image_paths
        self.target_size = target_size
        self.scale_factors = scale_factors
    
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
        
        # 调整到目标尺寸（高分辨率）
        if hr_img.shape != (self.target_size, self.target_size):
            hr_img = cv2.resize(hr_img, (self.target_size, self.target_size), interpolation=cv2.INTER_CUBIC)
        
        # 随机选择一个降质倍数
        scale_factor = random.choice(self.scale_factors)
        
        # 生成低分辨率版本
        lr_img = apply_resolution_degradation(hr_img, scale_factor)
        
        # 将低分辨率图像上采样回目标尺寸（用于训练时尺寸匹配）
        lr_img_upsampled = upsample_to_target_size(lr_img, self.target_size)
        
        # 转换为tensor
        lr_tensor = torch.from_numpy(lr_img_upsampled).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr_img).unsqueeze(0).float()
        
        return lr_tensor, hr_tensor, scale_factor

# ========== 样本收集 ==========
def get_all_image_paths(root_dir):
    """遍历OST文件夹，收集所有图像路径"""
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
        category_count = 0
        for filename in os.listdir(subpath):
            filepath = os.path.join(subpath, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[-1].lower()
                if ext in supported_extensions:
                    image_paths.append(filepath)
                    category_count += 1
        print(f"  Found {category_count} images in {subdir}")
    
    print(f"Found {len(image_paths)} images in total")
    return image_paths

# 收集所有图像路径
all_image_paths = get_all_image_paths(DATA_ROOT)

if len(all_image_paths) == 0:
    print(f"Error: No images found in '{DATA_ROOT}'!")
    print("Please make sure the OST folder contains subdirectories with image files.")
    exit(1)

# 打乱并划分训练集和验证集
random.shuffle(all_image_paths)
split_idx = int(len(all_image_paths) * (1 - VAL_SPLIT))
train_paths = all_image_paths[:split_idx]
val_paths = all_image_paths[split_idx:]

print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")
print(f"Scale factors: {SCALE_FACTORS}")
print(f"Target resolution: {TARGET_SIZE}x{TARGET_SIZE}")

# 创建数据集和数据加载器
train_dataset = OSTResolutionDataset(train_paths)
val_dataset = OSTResolutionDataset(val_paths)

# 根据NUM_WORKERS值创建不同配置的DataLoader
if NUM_WORKERS > 0:
    print(f"Using multiprocessing with {NUM_WORKERS} workers")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
else:
    print("Using single-process data loading (NUM_WORKERS=0)")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )

# ========== 训练主流程 ==========
def setup_device_and_model():
    """设置设备和模型（支持多GPU）"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s)")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device('cpu')
        num_gpus = 0
        print("CUDA not available, using CPU")
    
    # 创建模型
    model = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=23, num_grow_ch=32)
    
    # 多GPU设置
    if USE_MULTI_GPU and num_gpus > 1:
        model = model.to(device)
        if USE_DDP:
            print(f"Using DistributedDataParallel with {num_gpus} GPUs")
            # DDP需要在多进程环境中使用
            model = DDP(model)
        else:
            print(f"Using DataParallel with {num_gpus} GPUs")
            model = DataParallel(model)
    else:
        model = model.to(device)
        print(f"Using single GPU/CPU training")
    
    return model, device, num_gpus

def train():
    # 设置设备和模型
    model, device, num_gpus = setup_device_and_model()
    
    # GPU内存清理和优化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 显示当前GPU内存状态
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        cached_memory = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Total: {total_memory:.2f}GB, Allocated: {allocated_memory:.2f}GB, Cached: {cached_memory:.2f}GB")
    
    # 损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float('inf')
    
    print(f"Starting training on device: {device}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")
    print(f"Training with resolution scale factors: {SCALE_FACTORS}")
    if USE_MULTI_GPU and num_gpus > 1:
        print(f"Effective batch size: {BATCH_SIZE * num_gpus}")
    
    # 用于统计不同scale factor的使用情况
    scale_factor_stats = {sf: 0 for sf in SCALE_FACTORS}
    
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        epoch_scale_stats = {sf: 0 for sf in SCALE_FACTORS}
        
        # 使用tqdm显示进度
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} - Train')
        
        for batch_idx, (lr_img, hr_img, scale_factors) in enumerate(train_pbar):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            
            # 统计scale factor使用情况
            for sf in scale_factors:
                epoch_scale_stats[sf.item()] += 1
                scale_factor_stats[sf.item()] += 1
            
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
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss * lr_img.size(0)
            
            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 内存清理（每50个batch清理一次，更频繁的清理）
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 如果内存使用过高，强制清理
            if torch.cuda.is_available() and batch_idx % 20 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                if allocated > 1.5:  # 如果使用超过1.5GB，清理缓存
                    torch.cuda.empty_cache()
        
        train_loss /= len(train_loader.dataset)
        
        # 训练损失爆炸检测
        if train_loss > 10:
            print(f'[Stop] Train loss爆炸（{train_loss:.4f}），自动停止训练！')
            break
        
        # 验证阶段
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} - Val')
            for lr_img, hr_img, scale_factors in val_pbar:
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)
                
                pred = model(lr_img)
                
                # 确保预测结果与目标尺寸一致
                if pred.shape[-2:] != hr_img.shape[-2:]:
                    pred = F.interpolate(pred, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
                
                loss = criterion(pred, hr_img)
                val_loss += loss.item() * lr_img.size(0)
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader.dataset)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={optimizer.param_groups[0]["lr"]:.2e}')
        
        # 每10个epoch打印scale factor统计
        if (epoch + 1) % 10 == 0:
            print(f'Scale factor usage in epoch {epoch+1}: {epoch_scale_stats}')
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 如果使用多GPU，需要保存module的state_dict
            if isinstance(model, (DataParallel, DDP)):
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
            else:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'[Save] Best model at epoch {epoch+1} | Val Loss={val_loss:.4f}')
        
        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            checkpoint_path = MODEL_SAVE_PATH.replace('.pth', f'_epoch_{epoch+1}.pth')
            if isinstance(model, (DataParallel, DDP)):
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
            print(f'[Checkpoint] Saved at epoch {epoch+1}')
    
    # 训练结束后打印总体统计
    print(f'\nTraining completed!')
    print(f'Total scale factor usage: {scale_factor_stats}')
    print(f'Best validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    print("=" * 60)
    print("Real-ESRGAN OST Resolution Training Script (Multi-GPU)")
    print("=" * 60)
    print(f"Dataset: {DATA_ROOT}")
    print(f"Scale factors: {SCALE_FACTORS}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Batch size per GPU: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Multi-GPU enabled: {USE_MULTI_GPU}")
    print(f"Available GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print("=" * 60)
    
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
