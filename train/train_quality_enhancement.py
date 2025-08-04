# ========== 图像清晰度增强训练脚本 (尺寸不变, 质量提升) ==========
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import numpy as np
from PIL import Image, ImageFilter
import torch.nn.functional as F
import json
import time
import cv2

# 导入Real-ESRGAN模型
from basicsr.archs.rrdbnet_arch import RRDBNet

# ========== 训练配置参数 ==========
HR_DATA_ROOT = './OST'  # 高质量图像目录
LR_DATA_ROOT = './OST_LQ'  # 生成的低质量图像目录
IMAGE_SIZE = 512  # 固定图像尺寸（输入输出相同）
BATCH_SIZE = 2  # 批次大小
NUM_EPOCHS = 25  # 训练轮数
LR = 2e-4  # 学习率
NUM_WORKERS = 0  # 数据加载进程数
MODEL_SAVE_PATH = './realESRGAN_quality_enhancement.pth'
VAL_SPLIT = 0.1  # 验证集比例

# ========== 图像质量增强模型配置 ==========
# 专门用于质量增强的配置 - RGB彩色图像
MODEL_CONFIGS = {
    'ultra_lite': {
        'num_in_ch': 3, 'num_out_ch': 3,  # RGB彩色图像
        'num_feat': 32, 'num_block': 10, 'num_grow_ch': 16
    },
    'lite': {
        'num_in_ch': 3, 'num_out_ch': 3,  # RGB彩色图像
        'num_feat': 48, 'num_block': 16, 'num_grow_ch': 24
    },
    'standard': {
        'num_in_ch': 3, 'num_out_ch': 3,  # RGB彩色图像
        'num_feat': 64, 'num_block': 20, 'num_grow_ch': 32
    }
}

CURRENT_MODEL = 'lite'  # 选择模型复杂度

# ========== GPU优化设置 ==========
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True

# ========== 图像质量降解函数 ==========
def add_degradation(img, degradation_type='mixed'):
    """
    对高质量图像添加各种降解效果，模拟真实世界的低质量图像
    
    Args:
        img: PIL图像
        degradation_type: 降解类型
    
    Returns:
        degraded_img: 降解后的PIL图像
    """
    if degradation_type == 'blur':
        # 模糊降解
        blur_radius = random.uniform(0.5, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    
    elif degradation_type == 'noise':
        # 噪声降解
        img_array = np.array(img).astype('float32')
        noise_std = random.uniform(5, 25)
        noise = np.random.normal(0, noise_std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255)
        img = Image.fromarray(img_array.astype('uint8'))
    
    elif degradation_type == 'jpeg':
        # JPEG压缩降解
        import io
        quality = random.randint(30, 70)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer)
    
    elif degradation_type == 'mixed':
        # 混合降解（最真实）
        # 1. 先添加轻微模糊
        if random.random() < 0.7:
            blur_radius = random.uniform(0.3, 1.2)
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # 2. 添加噪声
        if random.random() < 0.6:
            img_array = np.array(img).astype('float32')
            noise_std = random.uniform(3, 15)
            noise = np.random.normal(0, noise_std, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
            img = Image.fromarray(img_array.astype('uint8'))
        
        # 3. JPEG压缩
        if random.random() < 0.8:
            import io
            quality = random.randint(40, 80)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
    
    return img

# ========== 图像质量增强数据集类 ==========
class QualityEnhancementDataset(Dataset):
    """图像质量增强数据集 - 相同尺寸的低质量到高质量"""
    
    def __init__(self, hr_root, image_files, image_size=512):
        self.hr_root = hr_root
        self.image_files = image_files
        self.image_size = image_size
        
        print(f"图像质量增强数据集初始化: {len(image_files)} 张图像")
        print(f"任务: {image_size}×{image_size} RGB 低质量 → {image_size}×{image_size} RGB 高质量")
        print(f"目标: 提升清晰度、去噪、去模糊（尺寸不变）")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # 加载原始高质量图像
            hr_img = self.load_image(img_path)
            
            # 创建低质量版本（在线降解）
            hr_pil = self.tensor_to_pil(hr_img)
            lq_pil = add_degradation(hr_pil, 'mixed')
            lq_img = self.pil_to_tensor(lq_pil)
            
            return lq_img, hr_img
        
        except Exception as e:
            print(f"❌ 加载图像失败: {img_path}, 错误: {e}")
            # 返回随机数据作为fallback
            lq_tensor = torch.rand(3, self.image_size, self.image_size) * 0.8
            hr_tensor = torch.rand(3, self.image_size, self.image_size)
            return lq_tensor, hr_tensor
    
    def load_image(self, img_path):
        """加载图像并转换为tensor"""
        img = Image.open(img_path).convert('RGB')
        
        # 调整到目标尺寸
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # 转换为tensor
        img_array = np.array(img).astype('float32') / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1))).float()
        
        return img_tensor
    
    def tensor_to_pil(self, tensor):
        """tensor转PIL图像"""
        img_array = tensor.numpy()
        img_array = np.transpose(img_array, (1, 2, 0))
        img_array = (img_array * 255).astype('uint8')
        return Image.fromarray(img_array)
    
    def pil_to_tensor(self, pil_img):
        """PIL图像转tensor"""
        img_array = np.array(pil_img).astype('float32') / 255.0
        return torch.from_numpy(np.transpose(img_array, (2, 0, 1))).float()

# ========== 收集图像文件函数 ==========
def collect_image_files(hr_root):
    """收集所有图像文件路径"""
    image_files = []
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    print(f"🔍 搜索图像文件...")
    print(f"  图像目录: {hr_root}")
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(hr_root):
        for file in files:
            if os.path.splitext(file.lower())[1] in supported_formats:
                img_path = os.path.join(root, file)
                image_files.append(img_path)
    
    print(f"✅ 总计找到 {len(image_files)} 张图像")
    return image_files

# ========== 训练函数 ==========
def train_quality_enhancement():
    """图像质量增强模型训练"""
    print("=" * 70)
    print("🚀 Real-ESRGAN 图像质量增强训练 (RGB彩色图像)")
    print("=" * 70)
    
    # 检查目录存在性
    if not os.path.exists(HR_DATA_ROOT):
        print(f"❌ 高质量图像目录不存在: {HR_DATA_ROOT}")
        return
    
    # 收集所有图像文件
    image_files = collect_image_files(HR_DATA_ROOT)
    if len(image_files) == 0:
        print("❌ 未找到任何图像文件!")
        return
    
    # 划分训练集和验证集
    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - VAL_SPLIT))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"📊 数据集划分:")
    print(f"  训练集: {len(train_files)} 张图像")
    print(f"  验证集: {len(val_files)} 张图像")
    
    # 创建数据集和数据加载器
    train_dataset = QualityEnhancementDataset(HR_DATA_ROOT, train_files, IMAGE_SIZE)
    val_dataset = QualityEnhancementDataset(HR_DATA_ROOT, val_files, IMAGE_SIZE)
    
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
    
    # 设置设备和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 训练设备: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # 创建模型
    model_config = MODEL_CONFIGS[CURRENT_MODEL]
    model = RRDBNet(**model_config)
    model = model.to(device)
    
    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型配置: {CURRENT_MODEL}")
    print(f"📊 模型参数: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"⚙️  模型详情: {model_config}")
    
    # 损失函数和优化器
    criterion = nn.L1Loss()
    # 添加感知损失会更好，但这里先用L1
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=NUM_EPOCHS, 
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    # 训练配置信息
    print(f"\n🚀 图像质量增强训练配置:")
    print(f"  输入尺寸: {IMAGE_SIZE}×{IMAGE_SIZE} RGB")
    print(f"  输出尺寸: {IMAGE_SIZE}×{IMAGE_SIZE} RGB")
    print(f"  放大倍数: 1x (尺寸不变)")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  学习率: {LR}")
    print(f"  任务类型: 图像质量增强（去模糊、去噪、细节恢复）")
    
    # 开始训练
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    training_log = {
        'config': model_config,
        'task_type': 'quality_enhancement',
        'input_size': [IMAGE_SIZE, IMAGE_SIZE],
        'output_size': [IMAGE_SIZE, IMAGE_SIZE],
        'epochs': [],
        'best_val_loss': float('inf'),
        'total_time': 0
    }
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} - 训练')
        
        for batch_idx, (lq_img, hq_img) in enumerate(train_pbar):
            lq_img = lq_img.to(device, non_blocking=True)
            hq_img = hq_img.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播
            pred = model(lq_img)
            
            # 计算损失
            loss = criterion(pred, hq_img)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss
            
            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'In': f'{lq_img.shape[-2:]}',
                'Out': f'{pred.shape[-2:]}'
            })
            
            # 定期清理GPU内存
            if batch_idx % 30 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        
        # 验证阶段（每2个epoch一次）
        val_loss = None
        if (epoch + 1) % 2 == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} - 验证', leave=False)
                for lq_img, hq_img in val_pbar:
                    lq_img = lq_img.to(device, non_blocking=True)
                    hq_img = hq_img.to(device, non_blocking=True)
                    
                    pred = model(lq_img)
                    loss = criterion(pred, hq_img)
                    val_loss += loss.item()
                    
                    val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
            
            val_loss /= len(val_loader)
            
            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f'[保存] 最佳质量增强模型 Epoch {epoch+1} | 验证损失={val_loss:.4f}')
        
        epoch_time = time.time() - epoch_start_time
        
        # 记录训练日志
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss if val_loss is not None else "未计算",
            'time': epoch_time,
            'lr': optimizer.param_groups[0]['lr']
        }
        training_log['epochs'].append(epoch_log)
        
        # 显示训练结果
        if val_loss is not None:
            print(f'Epoch {epoch+1}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}, 时间={epoch_time:.1f}s')
        else:
            print(f'Epoch {epoch+1}: 训练损失={train_loss:.4f}, 验证损失=跳过验证, 时间={epoch_time:.1f}s')
        
        # GPU内存状态
        if torch.cuda.is_available() and (epoch + 1) % 3 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f'GPU内存: {allocated:.2f}GB')
    
    # 训练完成统计
    total_time = time.time() - training_start_time
    training_log['total_time'] = total_time
    training_log['best_val_loss'] = best_val_loss if best_val_loss != float('inf') else None
    
    # 保存训练日志
    log_file = MODEL_SAVE_PATH.replace('.pth', '_training_log.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 图像质量增强训练完成!")
    print(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
    print(f"⚡ 平均每轮: {total_time/NUM_EPOCHS:.1f} 秒")
    print(f"🏆 最佳验证损失: {best_val_loss:.4f}" if best_val_loss != float('inf') else "🏆 未进行验证")
    print(f"💾 模型已保存: {MODEL_SAVE_PATH}")
    print(f"📊 训练日志: {log_file}")
    print(f"🔍 模型能力: {IMAGE_SIZE}×{IMAGE_SIZE} RGB 质量增强")
    print(f"🎯 功能: 去模糊、去噪、细节恢复（尺寸不变）")

# ========== 主函数 ==========
if __name__ == '__main__':
    print("开始图像质量增强训练...")
    
    # 检查高质量图像目录
    if not os.path.exists(HR_DATA_ROOT):
        print(f"❌ 高质量图像目录不存在: {HR_DATA_ROOT}")
        print("请确保OST目录包含高质量的训练图像")
        exit(1)
    
    # 开始质量增强训练
    train_quality_enhancement()
