# ========== 超分辨率数据预处理脚本 (RGB彩色, 4x分辨率提升) ==========
import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import time

def create_lr_hr_pairs(hr_image_path, lr_output_path, scale_factor=4):
    """
    从高分辨率图像创建低分辨率-高分辨率图像对
    
    Args:
        hr_image_path: 高分辨率图像路径
        lr_output_path: 低分辨率图像输出路径
        scale_factor: 降质倍数
    
    Returns:
        success: 是否成功处理
    """
    try:
        # 加载高分辨率图像 (RGB彩色)
        hr_img = Image.open(hr_image_path).convert('RGB')
        
        # 高分辨率目标尺寸 (用于训练的ground truth)
        hr_target_size = 1024  # 1024x1024
        
        # 低分辨率输入尺寸 (模型输入)
        lr_input_size = hr_target_size // scale_factor  # 256x256
        
        # 1. 调整高分辨率图像到目标尺寸
        hr_resized = hr_img.resize((hr_target_size, hr_target_size), Image.LANCZOS)
        
        # 2. 创建低分辨率版本 (通过下采样)
        lr_downsampled = hr_resized.resize((lr_input_size, lr_input_size), Image.LANCZOS)
        
        # 3. 将低分辨率图像上采样回高分辨率尺寸 (这是模型的输入)
        lr_upsampled = lr_downsampled.resize((hr_target_size, hr_target_size), Image.LANCZOS)
        
        # 4. 但是实际模型训练时，我们使用原始的低分辨率作为输入
        # 所以我们保存的是256x256的低分辨率图像
        lr_upsampled_for_training = lr_downsampled  # 256x256作为模型输入
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(lr_output_path), exist_ok=True)
        
        # 保存低分辨率图像 (256x256, 作为模型输入)
        lr_upsampled_for_training.save(lr_output_path)
        
        return True, {
            'hr_size': hr_resized.size,
            'lr_size': lr_upsampled_for_training.size,
            'scale_factor': scale_factor
        }
        
    except Exception as e:
        print(f"❌ 处理失败 {hr_image_path}: {e}")
        return False, None

def process_dataset(hr_root, lr_root, scales, max_images_per_category=None):
    """
    处理整个数据集，创建超分辨率训练数据
    
    Args:
        hr_root: 高分辨率图像根目录
        lr_root: 低分辨率图像输出根目录
        scales: 降质倍数列表
        max_images_per_category: 每个类别的最大图像数量
    """
    print("=" * 70)
    print("🚀 超分辨率数据预处理 (RGB彩色图像)")
    print("=" * 70)
    
    if not os.path.exists(hr_root):
        print(f"❌ 高分辨率数据目录不存在: {hr_root}")
        return
    
    # 支持的图像格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    processing_stats = {
        'start_time': time.time(),
        'scales': scales,
        'categories': {},
        'total_processed': 0,
        'total_failed': 0
    }
    
    for scale in scales:
        print(f"\n📊 处理 {scale}x 降质...")
        
        lr_scale_root = os.path.join(lr_root, f'scale_{scale}x')
        os.makedirs(lr_scale_root, exist_ok=True)
        
        scale_stats = {
            'processed': 0,
            'failed': 0,
            'categories': {}
        }
        
        # 遍历所有类别目录
        categories = [d for d in os.listdir(hr_root) if os.path.isdir(os.path.join(hr_root, d))]
        
        for category in categories:
            hr_category_path = os.path.join(hr_root, category)
            lr_category_path = os.path.join(lr_scale_root, category)
            
            # 创建输出目录
            os.makedirs(lr_category_path, exist_ok=True)
            
            # 获取该类别下的所有图像文件
            image_files = []
            for file in os.listdir(hr_category_path):
                if os.path.splitext(file.lower())[1] in supported_formats:
                    image_files.append(file)
            
            # 限制每个类别的图像数量
            if max_images_per_category and len(image_files) > max_images_per_category:
                image_files = image_files[:max_images_per_category]
            
            category_processed = 0
            category_failed = 0
            
            print(f"  📁 类别 {category}: {len(image_files)} 张图像")
            
            # 使用进度条处理图像
            pbar = tqdm(image_files, desc=f"    处理 {category}", leave=False)
            
            for filename in pbar:
                hr_image_path = os.path.join(hr_category_path, filename)
                
                # 生成输出文件名
                base_name = os.path.splitext(filename)[0]
                lr_filename = f"{base_name}_upsampled_{scale}x.png"
                lr_output_path = os.path.join(lr_category_path, lr_filename)
                
                # 跳过已存在的文件
                if os.path.exists(lr_output_path):
                    category_processed += 1
                    continue
                
                # 处理图像
                success, info = create_lr_hr_pairs(hr_image_path, lr_output_path, scale)
                
                if success:
                    category_processed += 1
                else:
                    category_failed += 1
                
                # 更新进度条
                pbar.set_postfix({
                    '成功': category_processed,
                    '失败': category_failed
                })
            
            scale_stats['categories'][category] = {
                'processed': category_processed,
                'failed': category_failed,
                'total': len(image_files)
            }
            
            scale_stats['processed'] += category_processed
            scale_stats['failed'] += category_failed
            
            print(f"    ✅ 完成: {category_processed}/{len(image_files)} 张图像")
        
        processing_stats['categories'][f'scale_{scale}x'] = scale_stats
        processing_stats['total_processed'] += scale_stats['processed']
        processing_stats['total_failed'] += scale_stats['failed']
        
        print(f"  📈 {scale}x 降质完成: {scale_stats['processed']} 成功, {scale_stats['failed']} 失败")
    
    # 计算总耗时
    total_time = time.time() - processing_stats['start_time']
    processing_stats['total_time'] = total_time
    
    # 保存处理统计
    stats_file = os.path.join(lr_root, 'processing_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(processing_stats, f, indent=2, ensure_ascii=False)
    
    # 显示最终统计
    print(f"\n🎉 超分辨率数据预处理完成!")
    print(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
    print(f"✅ 成功处理: {processing_stats['total_processed']} 张图像")
    print(f"❌ 处理失败: {processing_stats['total_failed']} 张图像")
    print(f"📊 统计文件: {stats_file}")
    print(f"🔍 数据特点: RGB彩色图像, 真正的超分辨率 (256→1024)")
    
    # 显示每个scale的详细统计
    for scale in scales:
        scale_key = f'scale_{scale}x'
        if scale_key in processing_stats['categories']:
            stats = processing_stats['categories'][scale_key]
            print(f"\n📋 {scale}x降质统计:")
            for category, cat_stats in stats['categories'].items():
                success_rate = cat_stats['processed'] / cat_stats['total'] * 100 if cat_stats['total'] > 0 else 0
                print(f"  {category}: {cat_stats['processed']}/{cat_stats['total']} ({success_rate:.1f}%)")

def verify_generated_data(lr_root, scales):
    """验证生成的数据"""
    print(f"\n🔍 验证生成的数据...")
    
    for scale in scales:
        lr_scale_root = os.path.join(lr_root, f'scale_{scale}x')
        if not os.path.exists(lr_scale_root):
            print(f"❌ {scale}x数据目录不存在: {lr_scale_root}")
            continue
        
        total_images = 0
        categories = []
        
        for category in os.listdir(lr_scale_root):
            category_path = os.path.join(lr_scale_root, category)
            if os.path.isdir(category_path):
                category_count = len([f for f in os.listdir(category_path) 
                                    if f.endswith('.png')])
                categories.append((category, category_count))
                total_images += category_count
        
        print(f"✅ {scale}x降质: {total_images} 张图像")
        for category, count in categories:
            print(f"  📁 {category}: {count} 张")

def main():
    parser = argparse.ArgumentParser(description='超分辨率数据预处理工具 - RGB彩色图像')
    parser.add_argument('--hr_root', type=str, default='./OST',
                        help='高分辨率图像根目录')
    parser.add_argument('--lr_root', type=str, default='./OST_LR',
                        help='低分辨率图像输出根目录')
    parser.add_argument('--scales', type=int, nargs='+', default=[4],
                        help='降质倍数列表，默认: [4]')
    parser.add_argument('--max_per_category', type=int, default=None,
                        help='每个类别的最大图像数量')
    parser.add_argument('--verify', action='store_true',
                        help='仅验证已生成的数据')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_generated_data(args.lr_root, args.scales)
    else:
        process_dataset(args.hr_root, args.lr_root, args.scales, args.max_per_category)
        verify_generated_data(args.lr_root, args.scales)

if __name__ == '__main__':
    print("=" * 70)
    print("🎨 超分辨率数据预处理工具")
    print("🔍 功能: 创建RGB彩色图像的超分辨率训练数据")
    print("📐 任务: 高分辨率(1024×1024) → 低分辨率(256×256)")
    print("=" * 70)
    
    import sys
    if len(sys.argv) == 1:
        print("📖 使用示例:")
        print("\n1. 生成4x降质数据:")
        print("   python generate_lr_images_sr.py --scales 4")
        print("\n2. 生成多个尺度的数据:")
        print("   python generate_lr_images_sr.py --scales 2 4 8")
        print("\n3. 限制每个类别的图像数量:")
        print("   python generate_lr_images_sr.py --scales 4 --max_per_category 1000")
        print("\n4. 验证已生成的数据:")
        print("   python generate_lr_images_sr.py --verify --scales 4")
        print("\n📝 参数说明:")
        print("   --hr_root: 高分辨率图像目录 (默认: ./OST)")
        print("   --lr_root: 低分辨率图像输出目录 (默认: ./OST_LR)")
        print("   --scales: 降质倍数 (默认: [4])")
        print("   --max_per_category: 每类最大图像数")
        print("   --verify: 仅验证数据")
        print(f"\n🎯 特点: RGB彩色, 真正超分辨率, 256×256→1024×1024")
        sys.exit(0)
    
    main()
