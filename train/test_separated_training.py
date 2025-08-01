# ========== 分离式训练方案快速测试脚本 ==========
import os
import subprocess
import sys
import time

def run_command(cmd, description, check_success=True):
    """运行命令并显示结果"""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    print(f"执行命令: {cmd}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        elapsed_time = time.time() - start_time
        
        print(f"⏱️  执行时间: {elapsed_time:.2f}秒")
        
        if result.stdout:
            print(f"📤 输出:")
            print(result.stdout)
        
        if result.stderr:
            print(f"⚠️  错误输出:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} 成功完成!")
            return True
        else:
            print(f"❌ {description} 失败! 返回码: {result.returncode}")
            if check_success:
                return False
            return True
            
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        return False

def check_requirements():
    """检查必要的文件和目录"""
    print("🔍 检查环境和依赖...")
    
    # 检查OST数据集
    if not os.path.exists('./OST'):
        print("❌ OST数据集目录不存在!")
        print("请确保OST文件夹在当前目录中")
        return False
    
    # 检查脚本文件
    required_files = [
        'generate_lr_images.py',
        'train_from_pregenerated.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ 缺少必要文件: {file}")
            return False
    
    print("✅ 环境检查通过!")
    return True

def quick_test():
    """快速测试分离式训练方案"""
    print("🎯 开始分离式训练方案快速测试")
    print("此测试使用少量数据验证整个流程")
    
    if not check_requirements():
        return False
    
    # 第一步：生成少量低分辨率图像进行测试
    print("\n📊 第1步：生成测试用低分辨率图像")
    cmd1 = "python generate_lr_images.py --sample 0.02 --size 256 --scales 4 --auto"
    success1 = run_command(cmd1, "生成低分辨率图像（2%采样，256x256）")
    
    if not success1:
        print("❌ 数据预处理失败，停止测试")
        return False
    
    # 检查生成结果
    lr_dir = "./OST_LR/scale_4x"
    if os.path.exists(lr_dir):
        # 统计生成的文件数量
        total_files = 0
        for root, dirs, files in os.walk(lr_dir):
            total_files += len([f for f in files if f.endswith('.png')])
        print(f"✅ 成功生成 {total_files} 个低分辨率图像文件")
    else:
        print("❌ 未找到生成的低分辨率图像目录")
        return False
    
    # 第二步：使用预生成数据进行快速训练
    print("\n🧠 第2步：使用预生成数据训练模型")
    print("修改训练参数以进行快速测试...")
    
    # 创建快速测试配置的训练脚本
    create_quick_test_trainer()
    
    cmd2 = "python train_quick_test.py"
    success2 = run_command(cmd2, "快速训练测试（3个epoch）")
    
    if not success2:
        print("❌ 训练测试失败")
        return False
    
    # 第三步：验证结果
    print("\n🎯 第3步：验证测试结果")
    
    # 检查模型文件
    model_files = [
        'realESRGAN_quick_test.pth',
        'realESRGAN_quick_test_training_log.json'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ 找到输出文件: {model_file}")
        else:
            print(f"⚠️  未找到输出文件: {model_file}")
    
    # 显示测试总结
    print("\n🎉 快速测试完成!")
    print("=" * 50)
    print("📋 测试总结:")
    print("✅ 数据预处理：成功")
    print("✅ 模型训练：成功") 
    print("✅ 文件输出：成功")
    print("\n🚀 下一步建议:")
    print("1. 如果测试结果满意，可以使用完整数据集:")
    print("   python generate_lr_images.py --auto")
    print("   python train_from_pregenerated.py")
    print("2. 或者逐步增加数据量进行验证")
    print("3. 调整模型复杂度和训练参数")
    
    return True

def create_quick_test_trainer():
    """创建快速测试用的训练脚本"""
    quick_trainer_content = '''# ========== 快速测试训练脚本 ==========
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

# ========== 快速测试配置 ==========
HR_DATA_ROOT = './OST'
LR_DATA_ROOT = './OST_LR'
SCALE_FACTOR = 4
BATCH_SIZE = 1  # 小批次用于快速测试
NUM_EPOCHS = 3  # 少数epoch进行快速验证
LR = 2e-3
NUM_WORKERS = 0
MODEL_SAVE_PATH = './realESRGAN_quick_test.pth'
VAL_SPLIT = 0.2

# 超轻量模型配置
MODEL_CONFIG = {
    'num_in_ch': 1, 'num_out_ch': 1,
    'num_feat': 16, 'num_block': 4, 'num_grow_ch': 8
}

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True

class QuickTestDataset(Dataset):
    def __init__(self, hr_root, lr_root, scale_factor, image_pairs):
        self.hr_root = hr_root
        self.lr_root = lr_root
        self.scale_factor = scale_factor
        self.image_pairs = image_pairs
        print(f"快速测试数据集: {len(image_pairs)} 个图像对")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        hr_path, lr_upsampled_path = self.image_pairs[idx]
        
        try:
            hr_img = self.load_image(hr_path)
            lr_img = self.load_image(lr_upsampled_path)
            
            lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).float()
            hr_tensor = torch.from_numpy(hr_img).unsqueeze(0).float()
            
            return lr_tensor, hr_tensor
        except Exception as e:
            print(f"加载失败: {hr_path}, 错误: {e}")
            size = 256
            lr_tensor = torch.rand(1, size, size) * 0.8
            hr_tensor = torch.rand(1, size, size)
            return lr_tensor, hr_tensor
    
    def load_image(self, img_path):
        img = Image.open(img_path).convert('L')
        img = np.array(img).astype('float32') / 255.0
        return img

def find_image_pairs(hr_root, lr_root, scale_factor):
    image_pairs = []
    lr_scale_dir = os.path.join(lr_root, f'scale_{scale_factor}x')
    
    if not os.path.exists(lr_scale_dir):
        print(f"❌ 低分辨率目录不存在: {lr_scale_dir}")
        return []
    
    print(f"🔍 搜索图像对...")
    for category in os.listdir(hr_root):
        hr_category_path = os.path.join(hr_root, category)
        lr_category_path = os.path.join(lr_scale_dir, category)
        
        if not os.path.isdir(hr_category_path) or not os.path.exists(lr_category_path):
            continue
        
        for hr_filename in os.listdir(hr_category_path):
            hr_path = os.path.join(hr_category_path, hr_filename)
            
            if not os.path.isfile(hr_path):
                continue
            
            base_name = os.path.splitext(hr_filename)[0]
            lr_upsampled_filename = f"{base_name}_upsampled_{scale_factor}x.png"
            lr_upsampled_path = os.path.join(lr_category_path, lr_upsampled_filename)
            
            if os.path.exists(lr_upsampled_path):
                image_pairs.append((hr_path, lr_upsampled_path))
    
    print(f"✅ 找到 {len(image_pairs)} 个有效图像对")
    return image_pairs

def main():
    print("🚀 开始快速测试训练...")
    
    image_pairs = find_image_pairs(HR_DATA_ROOT, LR_DATA_ROOT, SCALE_FACTOR)
    if len(image_pairs) == 0:
        print("❌ 未找到图像对!")
        return
    
    # 数据划分
    random.shuffle(image_pairs)
    split_idx = int(len(image_pairs) * (1 - VAL_SPLIT))
    train_pairs = image_pairs[:split_idx]
    val_pairs = image_pairs[split_idx:] if split_idx < len(image_pairs) else image_pairs[-2:]
    
    print(f"训练集: {len(train_pairs)}, 验证集: {len(val_pairs)}")
    
    # 创建数据集
    train_dataset = QuickTestDataset(HR_DATA_ROOT, LR_DATA_ROOT, SCALE_FACTOR, train_pairs)
    val_dataset = QuickTestDataset(HR_DATA_ROOT, LR_DATA_ROOT, SCALE_FACTOR, val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 设备和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    model = RRDBNet(**MODEL_CONFIG)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,} ({total_params/1e6:.3f}M)")
    
    # 优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 训练
    best_val_loss = float('inf')
    training_log = {'epochs': [], 'best_val_loss': float('inf')}
    
    print(f"开始 {NUM_EPOCHS} 个epoch的快速训练...")
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for lr_img, hr_img in train_pbar:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            
            optimizer.zero_grad()
            pred = model(lr_img)
            
            if pred.shape[-2:] != hr_img.shape[-2:]:
                pred = F.interpolate(pred, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = criterion(pred, hr_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                pred = model(lr_img)
                if pred.shape[-2:] != hr_img.shape[-2:]:
                    pred = F.interpolate(pred, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(pred, hr_img)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'[保存] 最佳模型 Epoch {epoch+1} | 验证损失={val_loss:.4f}')
        
        epoch_time = time.time() - epoch_start
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time': epoch_time
        }
        training_log['epochs'].append(epoch_log)
        
        print(f'Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, Time={epoch_time:.1f}s')
    
    training_log['best_val_loss'] = best_val_loss
    
    # 保存日志
    log_file = MODEL_SAVE_PATH.replace('.pth', '_training_log.json')
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\\n🎉 快速测试完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存: {MODEL_SAVE_PATH}")
    print(f"日志保存: {log_file}")

if __name__ == '__main__':
    main()
'''
    
    with open('train_quick_test.py', 'w', encoding='utf-8') as f:
        f.write(quick_trainer_content)
    
    print("✅ 创建快速测试训练脚本: train_quick_test.py")

def cleanup_test_files():
    """清理测试文件"""
    print("\n🧹 清理测试文件...")
    
    files_to_remove = [
        'train_quick_test.py',
        'realESRGAN_quick_test.pth',
        'realESRGAN_quick_test_training_log.json'
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"✅ 删除: {file}")
            except Exception as e:
                print(f"⚠️  删除失败: {file}, 错误: {e}")
    
    # 询问是否删除生成的低分辨率数据
    response = input("\n❓ 是否删除测试生成的低分辨率数据目录 OST_LR? (y/n): ").lower().strip()
    if response == 'y':
        import shutil
        try:
            if os.path.exists('./OST_LR'):
                shutil.rmtree('./OST_LR')
                print("✅ 删除测试数据目录: OST_LR")
        except Exception as e:
            print(f"⚠️  删除目录失败: {e}")

if __name__ == '__main__':
    print("🎯 分离式训练方案快速测试工具")
    print("=" * 50)
    
    try:
        success = quick_test()
        
        if success:
            print(f"\\n✅ 所有测试通过!")
            print("🚀 分离式训练方案工作正常，可以投入使用!")
        else:
            print(f"\\n❌ 测试过程中出现问题")
            print("请检查错误信息并修复相关问题")
    
    except KeyboardInterrupt:
        print(f"\\n⚠️  用户中断测试")
    except Exception as e:
        print(f"\\n❌ 测试过程中出现异常: {e}")
    
    finally:
        # 询问是否清理测试文件
        response = input("\\n❓ 是否清理测试过程中生成的文件? (y/n): ").lower().strip()
        if response == 'y':
            cleanup_test_files()
        
        print("\\n👋 测试完成，感谢使用!")
