# ========== 使用预生成低分辨率图像的训练脚本 ==========
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
import json
import time

# 导入Real-ESRGAN模型
from basicsr.archs.rrdbnet_arch import RRDBNet

# ========== 训练配置参数 ==========
HR_DATA_ROOT = './OST'  # 高分辨率图像目录
LR_DATA_ROOT = './OST_LR'  # 预生成的低分辨率图像目录
SCALE_FACTOR = 4  # 使用的降质倍数
BATCH_SIZE = 2  # 批次大小
NUM_EPOCHS = 15  # 训练轮数
LR = 1e-3  # 学习率
NUM_WORKERS = 0  # 数据加载进程数
MODEL_SAVE_PATH = './realESRGAN_from_pregenerated.pth'
VAL_SPLIT = 0.1  # 验证集比例

# ========== 模型配置 ==========
# 可选择不同复杂度的模型
MODEL_CONFIGS = {
    'ultra_lite': {
        'num_in_ch': 1, 'num_out_ch': 1,
        'num_feat': 24, 'num_block': 8, 'num_grow_ch': 12
    },
    'lite': {
        'num_in_ch': 1, 'num_out_ch': 1,
        'num_feat': 32, 'num_block': 12, 'num_grow_ch': 16
    },
    'standard': {
        'num_in_ch': 1, 'num_out_ch': 1,
        'num_feat': 48, 'num_block': 16, 'num_grow_ch': 24
    }
}

CURRENT_MODEL = 'ultra_lite'  # 选择模型复杂度

# ========== GPU优化设置 ==========
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True

# ========== 预生成图像数据集类 ==========
class PreGeneratedDataset(Dataset):
    """使用预生成的低分辨率图像的数据集"""
    
    def __init__(self, hr_root, lr_root, scale_factor, image_pairs):
        self.hr_root = hr_root
        self.lr_root = lr_root
        self.scale_factor = scale_factor
        self.image_pairs = image_pairs
        
        print(f"数据集初始化: {len(image_pairs)} 个图像对")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        hr_path, lr_upsampled_path = self.image_pairs[idx]
        
        try:
            # 加载高分辨率图像（Ground Truth）
            hr_img = self.load_image(hr_path)
            
            # 加载预生成的低分辨率上采样图像（模型输入）
            lr_img = self.load_image(lr_upsampled_path)
            
            # 转换为tensor
            lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).float()
            hr_tensor = torch.from_numpy(hr_img).unsqueeze(0).float()
            
            return lr_tensor, hr_tensor
        
        except Exception as e:
            print(f"❌ 加载图像对失败: {hr_path}, {lr_upsampled_path}, 错误: {e}")
            # 返回随机数据作为fallback
            size = 512  # 假设的图像尺寸
            lr_tensor = torch.rand(1, size, size) * 0.8
            hr_tensor = torch.rand(1, size, size)
            return lr_tensor, hr_tensor
    
    def load_image(self, img_path):
        """加载单张图像并确保尺寸一致"""
        img = Image.open(img_path).convert('L')
        img = np.array(img).astype('float32') / 255.0
        
        # 确保所有图像都是相同尺寸 - 修复tensor尺寸不匹配问题
        target_size = 512  # 固定目标尺寸
        if img.shape != (target_size, target_size):
            # 使用PIL进行resize以避免cv2依赖问题
            img_pil = Image.fromarray((img * 255).astype('uint8'))
            img_pil_resized = img_pil.resize((target_size, target_size), Image.LANCZOS)
            img = np.array(img_pil_resized).astype('float32') / 255.0
        
        return img

# ========== 数据配对函数 ==========
def find_image_pairs(hr_root, lr_root, scale_factor):
    """
    找到高分辨率图像和对应的低分辨率图像对
    Returns:
        image_pairs: [(hr_path, lr_upsampled_path), ...] 的列表
    """
    image_pairs = []
    lr_scale_dir = os.path.join(lr_root, f'scale_{scale_factor}x')
    
    if not os.path.exists(lr_scale_dir):
        print(f"❌ 低分辨率目录不存在: {lr_scale_dir}")
        return []
    
    print(f"🔍 搜索图像对...")
    print(f"  高分辨率目录: {hr_root}")
    print(f"  低分辨率目录: {lr_scale_dir}")
    
    # 遍历高分辨率图像目录
    for category in os.listdir(hr_root):
        hr_category_path = os.path.join(hr_root, category)
        lr_category_path = os.path.join(lr_scale_dir, category)
        
        if not os.path.isdir(hr_category_path) or not os.path.exists(lr_category_path):
            continue
        
        category_pairs = 0
        for hr_filename in os.listdir(hr_category_path):
            hr_path = os.path.join(hr_category_path, hr_filename)
            
            if not os.path.isfile(hr_path):
                continue
            
            # 构建对应的低分辨率上采样图像路径
            base_name = os.path.splitext(hr_filename)[0]
            lr_upsampled_filename = f"{base_name}_upsampled_{scale_factor}x.png"
            lr_upsampled_path = os.path.join(lr_category_path, lr_upsampled_filename)
            
            if os.path.exists(lr_upsampled_path):
                image_pairs.append((hr_path, lr_upsampled_path))
                category_pairs += 1
        
        print(f"  类别 {category}: {category_pairs} 个图像对")
    
    print(f"✅ 总计找到 {len(image_pairs)} 个有效图像对")
    return image_pairs

# ========== 训练函数 ==========
def train_with_pregenerated_data():
    """使用预生成数据进行训练"""
    print("=" * 60)
    print("使用预生成低分辨率图像的Real-ESRGAN训练")
    print("=" * 60)
    
    # 检查目录存在性
    if not os.path.exists(HR_DATA_ROOT):
        print(f"❌ 高分辨率数据目录不存在: {HR_DATA_ROOT}")
        return
    
    if not os.path.exists(LR_DATA_ROOT):
        print(f"❌ 低分辨率数据目录不存在: {LR_DATA_ROOT}")
        print(f"请先运行 generate_lr_images.py 生成低分辨率图像")
        return
    
    # 找到图像对
    image_pairs = find_image_pairs(HR_DATA_ROOT, LR_DATA_ROOT, SCALE_FACTOR)
    if len(image_pairs) == 0:
        print("❌ 未找到任何有效的图像对!")
        return
    
    # 划分训练集和验证集
    random.shuffle(image_pairs)
    split_idx = int(len(image_pairs) * (1 - VAL_SPLIT))
    train_pairs = image_pairs[:split_idx]
    val_pairs = image_pairs[split_idx:]
    
    print(f"📊 数据集划分:")
    print(f"  训练集: {len(train_pairs)} 个图像对")
    print(f"  验证集: {len(val_pairs)} 个图像对")
    
    # 创建数据集和数据加载器
    train_dataset = PreGeneratedDataset(HR_DATA_ROOT, LR_DATA_ROOT, SCALE_FACTOR, train_pairs)
    val_dataset = PreGeneratedDataset(HR_DATA_ROOT, LR_DATA_ROOT, SCALE_FACTOR, val_pairs)
    
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
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=NUM_EPOCHS, 
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    # 训练配置信息
    print(f"\n🚀 训练配置:")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  学习率: {LR}")
    print(f"  降质倍数: {SCALE_FACTOR}x")
    
    # 开始训练
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    training_log = {
        'config': model_config,
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
        
        for batch_idx, (lr_img, hr_img) in enumerate(train_pbar):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)
            
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
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss
            
            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 定期清理GPU内存
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        
        # 验证阶段（每2个epoch一次）
        val_loss = 0
        if (epoch + 1) % 2 == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} - 验证', leave=False)
                for lr_img, hr_img in val_pbar:
                    lr_img = lr_img.to(device, non_blocking=True)
                    hr_img = hr_img.to(device, non_blocking=True)
                    
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
                print(f'[保存] 最佳模型 Epoch {epoch+1} | 验证损失={val_loss:.4f}')
        
        epoch_time = time.time() - epoch_start_time
        
        # 记录训练日志
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time': epoch_time,
            'lr': optimizer.param_groups[0]['lr']
        }
        training_log['epochs'].append(epoch_log)
        
        print(f'Epoch {epoch+1}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}, 时间={epoch_time:.1f}s')
        
        # GPU内存状态
        if torch.cuda.is_available() and (epoch + 1) % 3 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f'GPU内存: {allocated:.2f}GB')
    
    # 训练完成统计
    total_time = time.time() - training_start_time
    training_log['total_time'] = total_time
    training_log['best_val_loss'] = best_val_loss
    
    # 保存训练日志
    log_file = MODEL_SAVE_PATH.replace('.pth', '_training_log.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 训练完成!")
    print(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
    print(f"⚡ 平均每轮: {total_time/NUM_EPOCHS:.1f} 秒")
    print(f"🏆 最佳验证损失: {best_val_loss:.4f}")
    print(f"💾 模型已保存: {MODEL_SAVE_PATH}")
    print(f"📊 训练日志: {log_file}")

# ========== 主函数 ==========
if __name__ == '__main__':
    print("检查预生成数据...")
    
    # 检查是否存在预生成的数据
    lr_scale_dir = os.path.join(LR_DATA_ROOT, f'scale_{SCALE_FACTOR}x')
    if not os.path.exists(lr_scale_dir):
        print(f"❌ 未找到预生成的低分辨率数据: {lr_scale_dir}")
        print("请先运行以下命令生成低分辨率图像:")
        print(f"python generate_lr_images.py --scales {SCALE_FACTOR}")
        exit(1)
    
    # 开始训练
    train_with_pregenerated_data()
