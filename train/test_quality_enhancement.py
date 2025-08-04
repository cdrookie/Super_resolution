# ========== 图像质量增强测试脚本 ==========
import os
import torch
import numpy as np
from PIL import Image, ImageFilter
import time
import argparse

# 导入Real-ESRGAN模型
from basicsr.archs.rrdbnet_arch import RRDBNet

# ========== 模型配置 ==========
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

class QualityEnhancer:
    """图像质量增强器"""
    
    def __init__(self, model_path, model_config='lite', device=None):
        self.model_path = model_path
        self.model_config = model_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self.load_model()
        
        print(f"🚀 图像质量增强器初始化完成")
        print(f"📱 设备: {self.device}")
        print(f"🧠 模型配置: {model_config}")
        print(f"📄 模型路径: {model_path}")
        print(f"🔍 功能: 图像质量增强（去模糊、去噪、细节恢复）")
        print(f"📐 尺寸: 保持原始尺寸不变")
    
    def load_model(self):
        """加载训练好的质量增强模型"""
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
    
    def preprocess_image(self, image_path, target_size=None):
        """预处理输入图像"""
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        # 如果指定目标尺寸，调整图像
        if target_size and img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.LANCZOS)
        
        # 转换为numpy数组并归一化
        img_array = np.array(img).astype('float32') / 255.0
        
        # 转换为CHW格式
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        
        # 添加batch维度并转换为tensor
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).to(self.device)
        
        return img_tensor, original_size
    
    def postprocess_image(self, tensor, target_size=None):
        """后处理输出tensor"""
        # 移除batch维度
        tensor = tensor.squeeze(0)
        
        # 裁剪到[0,1]范围
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为numpy
        img_array = tensor.cpu().numpy()
        
        # 转换为HWC格式
        img_array = np.transpose(img_array, (1, 2, 0))  # CHW -> HWC
        
        # 转换为PIL图像
        img_array = (img_array * 255).astype('uint8')
        img_pil = Image.fromarray(img_array)
        
        # 如果需要调整回原始尺寸
        if target_size and img_pil.size != target_size:
            img_pil = img_pil.resize(target_size, Image.LANCZOS)
        
        return img_pil
    
    def enhance_single(self, input_path, output_path, preserve_size=True):
        """对单张图像进行质量增强"""
        start_time = time.time()
        
        # 预处理
        img_tensor, original_size = self.preprocess_image(input_path, 512 if not preserve_size else None)
        
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
        print(f"🎯 增强类型: 质量提升（去模糊、去噪、细节恢复）")
        
        return enhanced_image
    
    def enhance_batch(self, input_dir, output_dir, max_images=None, preserve_size=True):
        """批量质量增强处理"""
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
        
        print(f"🔍 找到 {len(image_files)} 张图像进行质量增强")
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
        print(f"\n🎉 批量质量增强完成!")
        print(f"⏱️  总耗时: {total_time:.1f}秒")
        print(f"⚡ 平均每张: {total_time/len(image_files):.1f}秒")
    
    def create_degraded_test(self, input_path, output_dir):
        """创建降解图像用于对比测试"""
        img = Image.open(input_path).convert('RGB')
        
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # 1. 原始图像
        original_path = os.path.join(output_dir, f"{base_name}_original.png")
        img.save(original_path)
        
        # 2. 模糊版本
        blurred = img.filter(ImageFilter.GaussianBlur(1.5))
        blur_path = os.path.join(output_dir, f"{base_name}_blurred.png")
        blurred.save(blur_path)
        
        # 3. 噪声版本
        img_array = np.array(img).astype('float32')
        noise = np.random.normal(0, 15, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype('uint8')
        noisy = Image.fromarray(noisy_array)
        noise_path = os.path.join(output_dir, f"{base_name}_noisy.png")
        noisy.save(noise_path)
        
        # 4. JPEG压缩版本
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=50)
        buffer.seek(0)
        compressed = Image.open(buffer)
        jpeg_path = os.path.join(output_dir, f"{base_name}_compressed.png")
        compressed.save(jpeg_path)
        
        print(f"📊 创建测试图像:")
        print(f"  原始: {original_path}")
        print(f"  模糊: {blur_path}")
        print(f"  噪声: {noise_path}")
        print(f"  压缩: {jpeg_path}")
        
        return [original_path, blur_path, noise_path, jpeg_path]
    
    def compare_results(self, original_path, degraded_path, enhanced_path):
        """比较原始、降解和增强结果"""
        # 加载图像
        original_img = Image.open(original_path).convert('RGB')
        degraded_img = Image.open(degraded_path).convert('RGB')
        enhanced_img = Image.open(enhanced_path).convert('RGB')
        
        print(f"\n📊 质量增强结果比较:")
        print(f"🔹 原始图像: {original_img.size}")
        print(f"🔹 降解图像: {degraded_img.size}")
        print(f"🔹 增强图像: {enhanced_img.size}")
        
        # 创建对比图像
        target_size = original_img.size
        original_resized = original_img.resize(target_size, Image.LANCZOS)
        degraded_resized = degraded_img.resize(target_size, Image.LANCZOS)
        enhanced_resized = enhanced_img.resize(target_size, Image.LANCZOS)
        
        # 创建水平拼接的对比图
        comparison = Image.new('RGB', (target_size[0] * 3, target_size[1]))
        comparison.paste(degraded_resized, (0, 0))
        comparison.paste(enhanced_resized, (target_size[0], 0))
        comparison.paste(original_resized, (target_size[0] * 2, 0))
        
        comparison_path = enhanced_path.replace('.png', '_comparison.png')
        comparison.save(comparison_path)
        print(f"💾 对比图像: {comparison_path}")
        print(f"📋 布局: 降解图像 | 增强图像 | 原始图像")
        
        return comparison

def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN 图像质量增强测试')
    parser.add_argument('--model', type=str, default='./realESRGAN_quality_enhancement.pth',
                        help='模型权重文件路径')
    parser.add_argument('--config', type=str, default='lite', 
                        choices=['ultra_lite', 'lite', 'standard'],
                        help='模型配置')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像或目录路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出图像或目录路径')
    parser.add_argument('--mode', type=str, default='single', 
                        choices=['single', 'batch', 'test_degraded'],
                        help='处理模式')
    parser.add_argument('--max_images', type=int, default=None,
                        help='批量处理时的最大图像数量')
    parser.add_argument('--preserve_size', action='store_true', default=True,
                        help='保持原始图像尺寸')
    parser.add_argument('--compare', type=str, default=None,
                        help='用于比较的原始图像路径(仅单张模式)')
    
    args = parser.parse_args()
    
    # 创建质量增强器
    enhancer = QualityEnhancer(args.model, args.config)
    
    if args.mode == 'single':
        # 单张图像处理
        print(f"\n🎯 单张图像质量增强")
        enhanced_image = enhancer.enhance_single(args.input, args.output, args.preserve_size)
        
        # 如果提供了原始图像进行比较
        if args.compare:
            enhancer.compare_results(args.compare, args.input, args.output)
    
    elif args.mode == 'batch':
        # 批量处理
        print(f"\n🎯 批量图像质量增强")
        enhancer.enhance_batch(args.input, args.output, args.max_images, args.preserve_size)
    
    elif args.mode == 'test_degraded':
        # 创建降解测试图像
        print(f"\n🎯 创建降解测试图像")
        test_images = enhancer.create_degraded_test(args.input, args.output)
        
        # 对每个降解图像进行增强
        for degraded_path in test_images[1:]:  # 跳过原始图像
            name = os.path.basename(degraded_path).replace('.png', '')
            enhanced_path = os.path.join(args.output, f"{name}_enhanced.png")
            enhancer.enhance_single(degraded_path, enhanced_path, args.preserve_size)
            enhancer.compare_results(test_images[0], degraded_path, enhanced_path)

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Real-ESRGAN 图像质量增强测试工具")
    print("=" * 70)
    
    # 如果没有命令行参数，显示使用示例
    import sys
    if len(sys.argv) == 1:
        print("📖 使用示例:")
        print("\n1. 单张图像质量增强:")
        print("   python test_quality_enhancement.py --input blurry.jpg --output clear.png --mode single")
        print("\n2. 批量处理:")
        print("   python test_quality_enhancement.py --input ./inputs --output ./outputs --mode batch")
        print("\n3. 创建降解测试图像:")
        print("   python test_quality_enhancement.py --input high_quality.jpg --output ./test --mode test_degraded")
        print("\n4. 与原始图像对比:")
        print("   python test_quality_enhancement.py --input blurry.jpg --output clear.png --compare original.jpg --mode single")
        print("\n📝 参数说明:")
        print("   --model: 模型权重文件路径")
        print("   --config: 模型配置 (ultra_lite/lite/standard)")
        print("   --input: 输入图像或目录")
        print("   --output: 输出图像或目录")
        print("   --mode: 处理模式 (single/batch/test_degraded)")
        print("   --preserve_size: 保持原始尺寸")
        print("   --compare: 用于比较的原始图像")
        print("\n🔍 功能: 图像质量增强（去模糊、去噪、细节恢复，尺寸不变）")
        sys.exit(0)
    
    main()
