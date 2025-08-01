# ========== OST数据集低分辨率图像生成脚本 ==========
import os
import numpy as np
import cv2
from PIL import Image
import tifffile
import random
from tqdm import tqdm
import argparse
import json
import time

# ========== 配置参数 ==========
DATA_ROOT = './OST'  # 原始数据集根目录
OUTPUT_ROOT = './OST_LR'  # 低分辨率输出目录
SCALE_FACTORS = [4]  # 降质倍数：[2, 3, 4]
TARGET_SIZE = 512  # 目标图像尺寸
INTERPOLATION_METHOD = cv2.INTER_CUBIC  # 插值方法
PROCESS_ALL = True  # 是否处理所有图像，False则按比例采样
SAMPLE_RATIO = 1.0  # 采样比例（当PROCESS_ALL=False时使用）

# ========== 分辨率降质函数 ==========
def apply_resolution_degradation(img, scale_factor, method=cv2.INTER_CUBIC):
    """
    分辨率降质：将图像分辨率降低指定倍数
    Args:
        img: 输入高分辨率图像 (numpy array)
        scale_factor: 降质倍数 (2, 3, 4)
        method: 插值方法
    Returns:
        lr_img: 低分辨率图像 (numpy array)
    """
    h, w = img.shape[:2]
    
    # 计算低分辨率尺寸
    new_h, new_w = max(1, h // scale_factor), max(1, w // scale_factor)
    
    # 下采样到低分辨率
    lr_img = cv2.resize(img, (new_w, new_h), interpolation=method)
    
    return lr_img

def upsample_to_target_size(img, target_size, method=cv2.INTER_CUBIC):
    """
    将低分辨率图像上采样到目标尺寸
    Args:
        img: 输入低分辨率图像
        target_size: 目标尺寸 (int 或 tuple)
        method: 插值方法
    Returns:
        upsampled_img: 上采样后的图像
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    upsampled_img = cv2.resize(img, target_size, interpolation=method)
    return upsampled_img

# ========== 图像读取和预处理 ==========
def load_and_preprocess_image(img_path, target_size):
    """
    加载并预处理图像
    Args:
        img_path: 图像路径
        target_size: 目标尺寸
    Returns:
        processed_img: 预处理后的图像 (float32, [0,1])
    """
    try:
        ext = os.path.splitext(img_path)[-1].lower()
        
        if ext in ['.tif', '.tiff']:
            # 处理TIFF文件
            img = tifffile.imread(img_path)
            if img.ndim == 3:
                img = img.mean(axis=0)  # 转为灰度
            img = img.astype('float32')
            # 归一化到[0,1]
            if img.max() > 1:
                img = img / img.max()
        else:
            # 处理常规图像文件
            img = Image.open(img_path).convert('L')
            img = np.array(img).astype('float32') / 255.0
        
        # 调整到目标尺寸
        if img.shape != (target_size, target_size):
            img = cv2.resize(img, (target_size, target_size), interpolation=INTERPOLATION_METHOD)
        
        return img
    
    except Exception as e:
        print(f"❌ 处理图像失败: {img_path}, 错误: {e}")
        return None

# ========== 数据收集 ==========
def get_all_image_paths(root_dir, sample_ratio=1.0):
    """
    遍历OST文件夹，收集所有图像路径
    Args:
        root_dir: 数据集根目录
        sample_ratio: 采样比例
    Returns:
        image_paths: 图像路径列表
        category_info: 类别信息字典
    """
    image_paths = []
    category_info = {}
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    if not os.path.exists(root_dir):
        print(f"❌ 数据集目录不存在: '{root_dir}'")
        return [], {}
    
    print(f"📂 扫描数据集目录: {root_dir}")
    
    for subdir in os.listdir(root_dir):
        subpath = os.path.join(root_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        
        print(f"处理类别: {subdir}")
        category_paths = []
        
        for filename in os.listdir(subpath):
            filepath = os.path.join(subpath, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[-1].lower()
                if ext in supported_extensions:
                    category_paths.append(filepath)
        
        # 按比例采样
        if sample_ratio < 1.0 and len(category_paths) > 0:
            num_samples = max(1, int(len(category_paths) * sample_ratio))
            category_paths = random.sample(category_paths, num_samples)
        
        image_paths.extend(category_paths)
        category_info[subdir] = len(category_paths)
        print(f"  找到 {len(category_paths)} 张图像")
    
    print(f"📊 总计找到 {len(image_paths)} 张图像")
    return image_paths, category_info

# ========== 批量处理函数 ==========
def generate_lr_images(input_paths, output_root, scale_factors, target_size):
    """
    批量生成低分辨率图像
    Args:
        input_paths: 输入图像路径列表
        output_root: 输出根目录
        scale_factors: 降质倍数列表
        target_size: 目标图像尺寸
    """
    # 创建输出目录结构
    os.makedirs(output_root, exist_ok=True)
    
    # 为每个scale factor创建子目录
    scale_dirs = {}
    for sf in scale_factors:
        scale_dir = os.path.join(output_root, f'scale_{sf}x')
        os.makedirs(scale_dir, exist_ok=True)
        scale_dirs[sf] = scale_dir
        print(f"📁 创建输出目录: {scale_dir}")
    
    # 统计信息
    stats = {
        'total_processed': 0,
        'total_failed': 0,
        'scale_factor_counts': {sf: 0 for sf in scale_factors},
        'processing_times': []
    }
    
    print(f"🚀 开始处理 {len(input_paths)} 张图像...")
    start_time = time.time()
    
    # 使用tqdm显示进度
    for img_path in tqdm(input_paths, desc="生成低分辨率图像"):
        try:
            # 加载和预处理原始图像
            hr_img = load_and_preprocess_image(img_path, target_size)
            if hr_img is None:
                stats['total_failed'] += 1
                continue
            
            # 获取相对路径信息以保持目录结构
            rel_path = os.path.relpath(img_path, DATA_ROOT)
            filename = os.path.basename(img_path)
            category = os.path.dirname(rel_path)
            
            # 为每个scale factor生成低分辨率版本
            for scale_factor in scale_factors:
                process_start = time.time()
                
                # 创建类别子目录
                category_output_dir = os.path.join(scale_dirs[scale_factor], category)
                os.makedirs(category_output_dir, exist_ok=True)
                
                # 生成低分辨率图像
                lr_img = apply_resolution_degradation(hr_img, scale_factor, INTERPOLATION_METHOD)
                
                # 上采样回目标尺寸（用于训练时的尺寸匹配）
                lr_upsampled = upsample_to_target_size(lr_img, target_size, INTERPOLATION_METHOD)
                
                # 保存图像时确保尺寸完全一致
                # 保存原始低分辨率版本
                lr_filename = f"{os.path.splitext(filename)[0]}_lr_{scale_factor}x.png"
                lr_output_path = os.path.join(category_output_dir, lr_filename)
                lr_img_uint8 = (np.clip(lr_img, 0, 1) * 255).astype(np.uint8)
                # 确保保存的图像尺寸正确
                if lr_img_uint8.shape != (target_size//scale_factor, target_size//scale_factor):
                    lr_img_uint8 = cv2.resize(lr_img_uint8, (target_size//scale_factor, target_size//scale_factor), interpolation=INTERPOLATION_METHOD)
                cv2.imwrite(lr_output_path, lr_img_uint8)
                
                # 保存上采样版本（用于训练）- 确保尺寸完全一致
                upsampled_filename = f"{os.path.splitext(filename)[0]}_upsampled_{scale_factor}x.png"
                upsampled_output_path = os.path.join(category_output_dir, upsampled_filename)
                upsampled_img_uint8 = (np.clip(lr_upsampled, 0, 1) * 255).astype(np.uint8)
                # 强制确保上采样图像尺寸为目标尺寸
                if upsampled_img_uint8.shape != (target_size, target_size):
                    upsampled_img_uint8 = cv2.resize(upsampled_img_uint8, (target_size, target_size), interpolation=INTERPOLATION_METHOD)
                cv2.imwrite(upsampled_output_path, upsampled_img_uint8)
                
                # 统计处理时间
                process_time = time.time() - process_start
                stats['processing_times'].append(process_time)
                stats['scale_factor_counts'][scale_factor] += 1
            
            stats['total_processed'] += 1
            
        except Exception as e:
            print(f"❌ 处理失败: {img_path}, 错误: {e}")
            stats['total_failed'] += 1
    
    total_time = time.time() - start_time
    
    # 保存处理统计信息
    stats['total_time'] = total_time
    stats['avg_time_per_image'] = total_time / len(input_paths) if input_paths else 0
    stats['avg_processing_time'] = np.mean(stats['processing_times']) if stats['processing_times'] else 0
    
    stats_file = os.path.join(output_root, 'processing_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 打印结果统计
    print(f"\n🎉 处理完成!")
    print(f"✅ 成功处理: {stats['total_processed']} 张图像")
    print(f"❌ 处理失败: {stats['total_failed']} 张图像")
    print(f"⏱️  总耗时: {total_time/60:.2f} 分钟")
    print(f"⚡ 平均速度: {stats['avg_time_per_image']:.3f} 秒/图像")
    
    print(f"\n📊 各scale factor统计:")
    for sf, count in stats['scale_factor_counts'].items():
        print(f"  Scale {sf}x: {count} 张图像")
    
    print(f"\n📁 输出目录: {output_root}")
    print(f"📄 统计文件: {stats_file}")

# ========== 输出目录结构预览 ==========
def preview_output_structure(output_root, scale_factors, category_info):
    """预览输出目录结构"""
    print(f"\n📋 输出目录结构预览:")
    print(f"{output_root}/")
    for sf in scale_factors:
        print(f"├── scale_{sf}x/")
        for category, count in category_info.items():
            print(f"│   ├── {category}/")
            print(f"│   │   ├── image1_lr_{sf}x.png")
            print(f"│   │   ├── image1_upsampled_{sf}x.png")
            print(f"│   │   └── ... ({count*2} 个文件)")
    print(f"└── processing_stats.json")

# ========== 主函数 ==========
def main():
    print("=" * 60)
    print("OST数据集低分辨率图像生成脚本")
    print("=" * 60)
    
    # 显示配置信息
    print(f"📂 输入目录: {DATA_ROOT}")
    print(f"📁 输出目录: {OUTPUT_ROOT}")
    print(f"🔢 降质倍数: {SCALE_FACTORS}")
    print(f"📐 目标尺寸: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"🎨 插值方法: {INTERPOLATION_METHOD}")
    print(f"📊 处理方式: {'全部处理' if PROCESS_ALL else f'按比例采样({SAMPLE_RATIO*100:.1f}%)'}")
    print("=" * 60)
    
    # 收集图像路径
    image_paths, category_info = get_all_image_paths(
        DATA_ROOT, 
        sample_ratio=1.0 if PROCESS_ALL else SAMPLE_RATIO
    )
    
    if len(image_paths) == 0:
        print("❌ 未找到任何图像文件!")
        return
    
    # 预览输出结构
    preview_output_structure(OUTPUT_ROOT, SCALE_FACTORS, category_info)
    
    # 确认处理
    estimated_files = len(image_paths) * len(SCALE_FACTORS) * 2  # 每个scale factor生成2个文件
    print(f"\n📋 处理计划:")
    print(f"  - 输入图像: {len(image_paths)} 张")
    print(f"  - Scale factors: {len(SCALE_FACTORS)} 个")
    print(f"  - 预计生成: {estimated_files} 个文件")
    
    response = input(f"\n❓ 确认开始处理? (y/n): ").lower().strip()
    if response != 'y':
        print("❌ 已取消处理")
        return
    
    # 开始处理
    generate_lr_images(image_paths, OUTPUT_ROOT, SCALE_FACTORS, TARGET_SIZE)

# ========== 命令行参数支持 ==========
def parse_args():
    parser = argparse.ArgumentParser(description='生成OST数据集的低分辨率版本')
    parser.add_argument('--input', '-i', type=str, default='./OST',
                       help='输入数据集目录 (默认: ./OST)')
    parser.add_argument('--output', '-o', type=str, default='./OST_LR',
                       help='输出目录 (默认: ./OST_LR)')
    parser.add_argument('--scales', '-s', type=int, nargs='+', default=[4],
                       help='降质倍数列表 (默认: [4])')
    parser.add_argument('--size', type=int, default=512,
                       help='目标图像尺寸 (默认: 512)')
    parser.add_argument('--sample', type=float, default=1.0,
                       help='采样比例 (默认: 1.0, 处理全部)')
    parser.add_argument('--method', type=str, default='cubic',
                       choices=['nearest', 'linear', 'cubic'],
                       help='插值方法 (默认: cubic)')
    parser.add_argument('--auto', action='store_true',
                       help='自动处理，跳过确认')
    
    return parser.parse_args()

if __name__ == '__main__':
    # 支持命令行参数
    args = parse_args()
    
    # 更新全局配置
    DATA_ROOT = args.input
    OUTPUT_ROOT = args.output
    SCALE_FACTORS = args.scales
    TARGET_SIZE = args.size
    SAMPLE_RATIO = args.sample
    PROCESS_ALL = args.sample >= 1.0
    
    # 设置插值方法
    method_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC
    }
    INTERPOLATION_METHOD = method_map[args.method]
    
    if args.auto:
        # 自动模式：直接处理，不需要确认
        image_paths, category_info = get_all_image_paths(DATA_ROOT, SAMPLE_RATIO)
        if len(image_paths) > 0:
            generate_lr_images(image_paths, OUTPUT_ROOT, SCALE_FACTORS, TARGET_SIZE)
        else:
            print("❌ 未找到任何图像文件!")
    else:
        # 交互模式
        main()
