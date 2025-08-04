# ========== Real-ESRGAN 灰度图像增强测试脚本 ==========
import os
import torch
import numpy as np
from PIL import Image
import time
import argparse

# 导入Real-ESRGAN模型
from basicsr.archs.rrdbnet_arch import RRDBNet

# ========== 模型配置 ==========
MODEL_CONFIGS = {
    'ultra_lite': {
        'num_in_ch': 1, 'num_out_ch': 1,  # 灰度图像
        'num_feat': 24, 'num_block': 8, 'num_grow_ch': 12
    },
    'lite': {
        'num_in_ch': 1, 'num_out_ch': 1,  # 灰度图像
        'num_feat': 32, 'num_block': 12, 'num_grow_ch': 16
    },
    'standard': {
        'num_in_ch': 1, 'num_out_ch': 1,  # 灰度图像
        'num_feat': 48, 'num_block': 16, 'num_grow_ch': 24
    }
}

class GrayscaleEnhancer:
    """灰度图像增强器"""
    
    def __init__(self, model_path, model_config='ultra_lite', device=None):
        self.model_path = model_path
        self.model_config = model_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_size = 512  # 固定处理尺寸
        
        # 加载模型
        self.model = self.load_model()
        
        print(f"🚀 灰度图像增强器初始化完成")
        print(f"📱 设备: {self.device}")
        print(f"🧠 模型配置: {model_config}")
        print(f"📄 模型路径: {model_path}")
        print(f"🔍 功能: 灰度图像质量增强")
        print(f"📐 处理尺寸: {self.target_size}×{self.target_size}")
    
    def load_model(self):
        """加载训练好的模型"""
        config = MODEL_CONFIGS[self.model_config]
        model = RRDBNet(**config)
        
        if os.path.exists(self.model_path):
            print(f"✅ 加载模型权重: {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            print(f"⚠️  模型文件不存在: {self.model_path}")
            print("将使用随机初始化的模型进行测试")
        
        model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 模型参数: {total_params:,} ({total_params/1e6:.2f}M)")
        
        return model
    
    def preprocess_image(self, image_path):
        """预处理输入图像"""
        # 加载图像并转换为灰度
        img = Image.open(image_path).convert('L')
        original_size = img.size
        
        # 调整到目标尺寸
        if img.size != (self.target_size, self.target_size):
            img = img.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        # 转换为numpy数组并归一化
        img_array = np.array(img).astype('float32') / 255.0
        
        # 添加batch和channel维度
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return img_tensor, original_size
    
    def postprocess_image(self, tensor, target_size=None):
        """后处理输出tensor"""
        # 移除batch和channel维度
        tensor = tensor.squeeze(0).squeeze(0)
        
        # 裁剪到[0,1]范围
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为numpy
        img_array = tensor.cpu().numpy()
        
        # 转换为PIL图像
        img_array = (img_array * 255).astype('uint8')
        img_pil = Image.fromarray(img_array, mode='L')
        
        # 如果需要调整回原始尺寸
        if target_size and img_pil.size != target_size:
            img_pil = img_pil.resize(target_size, Image.LANCZOS)
        
        return img_pil
    
    def enhance_single(self, input_path, output_path, preserve_size=True):
        """对单张图像进行增强"""
        start_time = time.time()
        
        # 预处理
        img_tensor, original_size = self.preprocess_image(input_path)
        
        print(f"📥 输入图像: {input_path}")
        print(f"📐 原始尺寸: {original_size}")
        print(f"📐 处理尺寸: {img_tensor.shape}")
        
        # 模型推理
        with torch.no_grad():
            enhanced_tensor = self.model(img_tensor)
        
        print(f"📐 输出尺寸: {enhanced_tensor.shape}")
        
        # 后处理
        target_size = original_size if preserve_size else None
        enhanced_image = self.postprocess_image(enhanced_tensor, target_size)
        
        # 保存结果
        enhanced_image.save(output_path)
        
        process_time = time.time() - start_time
        
        print(f"💾 输出图像: {output_path}")
        print(f"📐 最终尺寸: {enhanced_image.size}")
        print(f"⏱️  处理时间: {process_time:.2f}秒")
        print(f"🎯 增强类型: 灰度图像质量增强")
        
        return enhanced_image
    
    def enhance_batch(self, input_dir, output_dir, max_images=None, preserve_size=True):
        """批量增强处理"""
        if not os.path.exists(input_dir):
            print(f"❌ 输入目录不存在: {input_dir}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 支持的图像格式
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 获取所有图像文件
        image_files = []
        for file in os.listdir(input_dir):
            if os.path.splitext(file.lower())[1] in supported_formats:
                image_files.append(file)
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"🔍 找到 {len(image_files)} 张图像进行增强")
        print(f"📁 输入目录: {input_dir}")
        print(f"📁 输出目录: {output_dir}")
        
        total_start_time = time.time()
        
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_dir, filename)
            
            # 生成输出文件名
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_enhanced.png"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"\n[{i}/{len(image_files)}] 处理: {filename}")
            
            try:
                self.enhance_single(input_path, output_path, preserve_size)
                print(f"✅ 完成")
            except Exception as e:
                print(f"❌ 失败: {e}")
        
        total_time = time.time() - total_start_time
        print(f"\n🎉 批量增强完成!")
        print(f"⏱️  总耗时: {total_time:.1f}秒")
        print(f"⚡ 平均每张: {total_time/len(image_files):.1f}秒")
    
    def compare_results(self, original_path, enhanced_path):
        """比较原始和增强结果"""
        # 加载图像
        original_img = Image.open(original_path).convert('L')
        enhanced_img = Image.open(enhanced_path).convert('L')
        
        print(f"\n📊 增强结果比较:")
        print(f"🔹 原始图像: {original_img.size}")
        print(f"🔹 增强图像: {enhanced_img.size}")
        
        # 创建对比图像
        target_size = max(original_img.size[0], enhanced_img.size[0])
        original_resized = original_img.resize((target_size, target_size), Image.LANCZOS)
        enhanced_resized = enhanced_img.resize((target_size, target_size), Image.LANCZOS)
        
        # 创建水平拼接的对比图
        comparison = Image.new('L', (target_size * 2, target_size))
        comparison.paste(original_resized, (0, 0))
        comparison.paste(enhanced_resized, (target_size, 0))
        
        comparison_path = enhanced_path.replace('.png', '_comparison.png')
        comparison.save(comparison_path)
        print(f"💾 对比图像: {comparison_path}")
        print(f"📋 布局: 原始图像 | 增强图像")
        
        return comparison

def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN 灰度图像增强测试')
    parser.add_argument('--model', type=str, default='./realESRGAN_from_pregenerated.pth',
                        help='模型权重文件路径')
    parser.add_argument('--config', type=str, default='ultra_lite', 
                        choices=['ultra_lite', 'lite', 'standard'],
                        help='模型配置')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像或目录路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出图像或目录路径')
    parser.add_argument('--mode', type=str, default='single', 
                        choices=['single', 'batch'],
                        help='处理模式')
    parser.add_argument('--max_images', type=int, default=None,
                        help='批量处理时的最大图像数量')
    parser.add_argument('--preserve_size', action='store_true', default=True,
                        help='保持原始图像尺寸')
    parser.add_argument('--compare', action='store_true',
                        help='生成对比图像')
    
    args = parser.parse_args()
    
    # 创建增强器
    enhancer = GrayscaleEnhancer(args.model, args.config)
    
    if args.mode == 'single':
        # 单张图像处理
        print(f"\n🎯 单张灰度图像增强")
        enhanced_image = enhancer.enhance_single(args.input, args.output, args.preserve_size)
        
        # 如果需要生成对比图像
        if args.compare:
            enhancer.compare_results(args.input, args.output)
    
    elif args.mode == 'batch':
        # 批量处理
        print(f"\n🎯 批量灰度图像增强")
        enhancer.enhance_batch(args.input, args.output, args.max_images, args.preserve_size)

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Real-ESRGAN 灰度图像增强测试工具")
    print("=" * 70)
    
    # 如果没有命令行参数，显示使用示例
    import sys
    if len(sys.argv) == 1:
        print("📖 使用示例:")
        print("\n1. 单张图像增强:")
        print("   python test_grayscale_enhancement.py --input image.jpg --output enhanced.png --mode single")
        print("\n2. 单张图像增强（带对比）:")
        print("   python test_grayscale_enhancement.py --input image.jpg --output enhanced.png --mode single --compare")
        print("\n3. 批量处理:")
        print("   python test_grayscale_enhancement.py --input ./inputs --output ./outputs --mode batch")
        print("\n4. 使用不同模型配置:")
        print("   python test_grayscale_enhancement.py --input image.jpg --output enhanced.png --config lite")
        print("\n📝 参数说明:")
        print("   --model: 模型权重文件路径 (默认: ./realESRGAN_from_pregenerated.pth)")
        print("   --config: 模型配置 (ultra_lite/lite/standard)")
        print("   --input: 输入图像或目录")
        print("   --output: 输出图像或目录")
        print("   --mode: 处理模式 (single/batch)")
        print("   --preserve_size: 保持原始尺寸")
        print("   --compare: 生成对比图像")
        print("\n🔍 功能: 灰度图像质量增强 (512×512处理)")
        sys.exit(0)
    
    main()
