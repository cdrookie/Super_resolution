# ========== 真正的超分辨率训练脚本 (4x分辨率提升, RGB彩色) ==========
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
BATCH_SIZE = 1  # 批次大小 (减小以适应更大的图像)
NUM_EPOCHS = 20  # 训练轮数
LR = 5e-4  # 学习率 (降低以获得更稳定的训练)
NUM_WORKERS = 0  # 数据加载进程数
MODEL_SAVE_PATH = './realESRGAN_4x_super_resolution.pth'
VAL_SPLIT = 0.1  # 验证集比例

# ========== 超分辨率模型配置 ==========
# 真正的超分辨率配置 - RGB彩色图像
MODEL_CONFIGS = {
    'ultra_lite': {
        'num_in_ch': 3, 'num_out_ch': 3,  # RGB彩色图像
        'num_feat': 32, 'num_block': 12, 'num_grow_ch': 16
    },
    'lite': {
        'num_in_ch': 3, 'num_out_ch': 3,  # RGB彩色图像
        'num_feat': 48, 'num_block': 16, 'num_grow_ch': 24
    },
    'standard': {
        'num_in_ch': 3, 'num_out_ch': 3,  # RGB彩色图像
        'num_feat': 64, 'num_block': 23, 'num_grow_ch': 32
    }
}

CURRENT_MODEL = 'lite'  # 选择模型复杂度

# ========== GPU优化设置 ==========
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True

# ========== 超分辨率数据集类 ==========
class SuperResolutionDataset(Dataset):
    """真正的超分辨率数据集 - 低分辨率输入到高分辨率输出"""
    
    def __init__(self, hr_root, lr_root, scale_factor, image_pairs):
        self.hr_root = hr_root
        self.lr_root = lr_root
        self.scale_factor = scale_factor
        self.image_pairs = image_pairs
        
        print(f"超分辨率数据集初始化: {len(image_pairs)} 个图像对")
        print(f"任务: {256}×{256} RGB → {1024}×{1024} RGB (4x超分辨率)")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        hr_path, lr_upsampled_path = self.image_pairs[idx]
        
        try:
            # 加载高分辨率图像（Ground Truth）- 1024x1024
            hr_img = self.load_hr_image(hr_path)
            
            # 加载低分辨率图像（模型输入）- 256x256
            lr_img = self.load_lr_image(lr_upsampled_path)
            
            # 转换为tensor (已经是CHW格式)
            lr_tensor = torch.from_numpy(lr_img).float()
            hr_tensor = torch.from_numpy(hr_img).float()
            
            return lr_tensor, hr_tensor
        
        except Exception as e:
            print(f"❌ 加载图像对失败: {hr_path}, {lr_upsampled_path}, 错误: {e}")
            # 返回随机数据作为fallback
            lr_tensor = torch.rand(3, 256, 256) * 0.8   # RGB低分辨率
            hr_tensor = torch.rand(3, 1024, 1024)       # RGB高分辨率
            return lr_tensor, hr_tensor
    
    def load_hr_image(self, img_path):
        """加载高分辨率图像 - 目标输出"""
        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype('float32') / 255.0
        
        # 高分辨率目标尺寸: 1024x1024
        target_size = 1024
        if img.shape[:2] != (target_size, target_size):
            img_pil = Image.fromarray((img * 255).astype('uint8'))
            img_pil_resized = img_pil.resize((target_size, target_size), Image.LANCZOS)
            img = np.array(img_pil_resized).astype('float32') / 255.0
        
        # 转换为CHW格式 (channels, height, width)
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img
    
    def load_lr_image(self, img_path):
        """加载低分辨率图像 - 模型输入"""
        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype('float32') / 255.0
        
        # 低分辨率输入尺寸: 256x256 (1024/4)
        target_size = 256
        if img.shape[:2] != (target_size, target_size):
            img_pil = Image.fromarray((img * 255).astype('uint8'))
            img_pil_resized = img_pil.resize((target_size, target_size), Image.LANCZOS)
            img = np.array(img_pil_resized).astype('float32') / 255.0
        
        # 转换为CHW格式 (channels, height, width)
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
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
    
    print(f"🔍 搜索超分辨率图像对...")
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
def train_super_resolution():
    """超分辨率模型训练"""
    print("=" * 70)
    print("🚀 Real-ESRGAN 4x超分辨率训练 (RGB彩色图像)")
    print("=" * 70)
    
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
    train_dataset = SuperResolutionDataset(HR_DATA_ROOT, LR_DATA_ROOT, SCALE_FACTOR, train_pairs)
    val_dataset = SuperResolutionDataset(HR_DATA_ROOT, LR_DATA_ROOT, SCALE_FACTOR, val_pairs)
    
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
    print(f"\n🚀 超分辨率训练配置:")
    print(f"  输入尺寸: 256×256 RGB")
    print(f"  输出尺寸: 1024×1024 RGB")
    print(f"  放大倍数: {SCALE_FACTOR}x")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  学习率: {LR}")
    print(f"  任务类型: 真正的超分辨率重建")
    
    # 开始训练
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    training_log = {
        'config': model_config,
        'task_type': '4x_super_resolution',
        'input_size': [256, 256],
        'output_size': [1024, 1024],
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
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'In': f'{lr_img.shape[-2:]}',
                'Out': f'{pred.shape[-2:]}'
            })
            
            # 定期清理GPU内存
            if batch_idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        
        # 验证阶段（每2个epoch一次，节省时间）
        val_loss = None
        if (epoch + 1) % 2 == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            val_loss = 0
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
                print(f'[保存] 最佳超分辨率模型 Epoch {epoch+1} | 验证损失={val_loss:.4f}')
        
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
    
    print(f"\n🎉 超分辨率训练完成!")
    print(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
    print(f"⚡ 平均每轮: {total_time/NUM_EPOCHS:.1f} 秒")
    print(f"🏆 最佳验证损失: {best_val_loss:.4f}" if best_val_loss != float('inf') else "🏆 未进行验证")
    print(f"💾 模型已保存: {MODEL_SAVE_PATH}")
    print(f"📊 训练日志: {log_file}")
    print(f"🔍 模型能力: 256×256 RGB → 1024×1024 RGB (4x超分辨率)")

# ========== 主函数 ==========
if __name__ == '__main__':
    print("检查超分辨率预生成数据...")
    
    # 检查是否存在预生成的数据
    lr_scale_dir = os.path.join(LR_DATA_ROOT, f'scale_{SCALE_FACTOR}x')
    if not os.path.exists(lr_scale_dir):
        print(f"❌ 未找到预生成的低分辨率数据: {lr_scale_dir}")
        print("请先运行以下命令生成低分辨率图像:")
        print(f"python generate_lr_images.py --scales {SCALE_FACTOR}")
        exit(1)
    
    # 开始超分辨率训练
    train_super_resolution()
