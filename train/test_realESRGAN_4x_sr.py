# ========== Real-ESRGAN 4x超分辨率测试脚本 ==========
import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import time
import argparse

# 导入Real-ESRGAN模型
from basicsr.archs.rrdbnet_arch import RRDBNet

# ========== 模型配置 ==========
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

class SuperResolutionTester:
    """4x超分辨率测试器"""
    
    def __init__(self, model_path, model_config='lite', device=None):
        self.model_path = model_path
        self.model_config = model_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self.load_model()
        
        print(f"🚀 超分辨率测试器初始化完成")
        print(f"📱 设备: {self.device}")
        print(f"🧠 模型配置: {model_config}")
        print(f"📄 模型路径: {model_path}")
        print(f"🔍 功能: 256×256 RGB → 1024×1024 RGB (4x超分辨率)")
    
    def load_model(self):
        """加载训练好的超分辨率模型"""
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
    
    def preprocess_image(self, image_path, target_size=256):
        """预处理输入图像"""
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        # 调整到目标尺寸 (256x256)
        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
        
        # 转换为numpy数组并归一化
        img_array = np.array(img_resized).astype('float32') / 255.0
        
        # 转换为CHW格式
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        
        # 添加batch维度并转换为tensor
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).to(self.device)
        
        return img_tensor, original_size
    
    def postprocess_image(self, tensor):
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
        
        return img_pil
    
    def super_resolve_single(self, input_path, output_path):
        """对单张图像进行4x超分辨率"""
        start_time = time.time()
        
        # 预处理
        lr_tensor, original_size = self.preprocess_image(input_path)
        
        print(f"📥 输入图像: {input_path}")
        print(f"📐 原始尺寸: {original_size}")
        print(f"📐 模型输入: {lr_tensor.shape}")
        
        # 模型推理
        with torch.no_grad():
            sr_tensor = self.model(lr_tensor)
        
        print(f"📐 模型输出: {sr_tensor.shape}")
        
        # 后处理
        sr_image = self.postprocess_image(sr_tensor)
        
        # 保存结果
        sr_image.save(output_path)
        
        process_time = time.time() - start_time
        
        print(f"💾 输出图像: {output_path}")
        print(f"📐 输出尺寸: {sr_image.size}")
        print(f"⏱️  处理时间: {process_time:.2f}秒")
        print(f"🎯 放大倍数: 4x (256→1024)")
        
        return sr_image
    
    def super_resolve_batch(self, input_dir, output_dir, max_images=None):
        """批量超分辨率处理"""
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
        
        print(f"🔍 找到 {len(image_files)} 张图像进行处理")
        print(f"📁 输入目录: {input_dir}")
        print(f"📁 输出目录: {output_dir}")
        
        total_start_time = time.time()
        
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_dir, filename)
            
            # 生成输出文件名
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_4x_sr.png"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"\n[{i}/{len(image_files)}] 处理: {filename}")
            
            try:
                self.super_resolve_single(input_path, output_path)
                print(f"✅ 完成")
            except Exception as e:
                print(f"❌ 失败: {e}")
        
        total_time = time.time() - total_start_time
        print(f"\n🎉 批量处理完成!")
        print(f"⏱️  总耗时: {total_time:.1f}秒")
        print(f"⚡ 平均每张: {total_time/len(image_files):.1f}秒")
    
    def compare_results(self, lr_path, hr_path, sr_path):
        """比较低分辨率、高分辨率和超分辨率结果"""
        # 加载图像
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB') if os.path.exists(hr_path) else None
        sr_img = Image.open(sr_path).convert('RGB')
        
        print(f"\n📊 结果比较:")
        print(f"🔹 低分辨率 (输入): {lr_img.size}")
        print(f"🔹 超分辨率 (输出): {sr_img.size}")
        if hr_img:
            print(f"🔹 高分辨率 (真实): {hr_img.size}")
        
        # 创建对比图像
        if hr_img:
            # 调整所有图像到相同尺寸进行比较
            target_size = (1024, 1024)
            lr_resized = lr_img.resize(target_size, Image.NEAREST)  # 最近邻放大
            hr_resized = hr_img.resize(target_size, Image.LANCZOS)
            sr_resized = sr_img.resize(target_size, Image.LANCZOS)
            
            # 创建水平拼接的对比图
            comparison = Image.new('RGB', (target_size[0] * 3, target_size[1]))
            comparison.paste(lr_resized, (0, 0))
            comparison.paste(sr_resized, (target_size[0], 0))
            comparison.paste(hr_resized, (target_size[0] * 2, 0))
            
            comparison_path = sr_path.replace('.png', '_comparison.png')
            comparison.save(comparison_path)
            print(f"💾 对比图像: {comparison_path}")
            print(f"📋 布局: 低分辨率 | 超分辨率 | 高分辨率")
        
        return sr_img

def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN 4x超分辨率测试')
    parser.add_argument('--model', type=str, default='./realESRGAN_4x_super_resolution.pth',
                        help='模型权重文件路径')
    parser.add_argument('--config', type=str, default='lite', 
                        choices=['ultra_lite', 'lite', 'standard'],
                        help='模型配置')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像或目录路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出图像或目录路径')
    parser.add_argument('--mode', type=str, default='single', 
                        choices=['single', 'batch'],
                        help='处理模式: single(单张) 或 batch(批量)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='批量处理时的最大图像数量')
    parser.add_argument('--compare', type=str, default=None,
                        help='用于比较的高分辨率图像路径(仅单张模式)')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = SuperResolutionTester(args.model, args.config)
    
    if args.mode == 'single':
        # 单张图像处理
        print(f"\n🎯 单张图像超分辨率处理")
        sr_image = tester.super_resolve_single(args.input, args.output)
        
        # 如果提供了高分辨率图像进行比较
        if args.compare:
            tester.compare_results(args.input, args.compare, args.output)
    
    elif args.mode == 'batch':
        # 批量处理
        print(f"\n🎯 批量超分辨率处理")
        tester.super_resolve_batch(args.input, args.output, args.max_images)

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Real-ESRGAN 4x超分辨率测试工具")
    print("=" * 70)
    
    # 如果没有命令行参数，显示使用示例
    import sys
    if len(sys.argv) == 1:
        print("📖 使用示例:")
        print("\n1. 单张图像超分辨率:")
        print("   python test_realESRGAN_4x_sr.py --input test.jpg --output test_4x.png --mode single")
        print("\n2. 批量处理:")
        print("   python test_realESRGAN_4x_sr.py --input ./inputs --output ./outputs --mode batch")
        print("\n3. 与高分辨率图像对比:")
        print("   python test_realESRGAN_4x_sr.py --input lr.jpg --output sr.png --compare hr.jpg --mode single")
        print("\n4. 使用不同模型配置:")
        print("   python test_realESRGAN_4x_sr.py --input test.jpg --output test_4x.png --config standard")
        print("\n📝 参数说明:")
        print("   --model: 模型权重文件路径")
        print("   --config: 模型配置 (ultra_lite/lite/standard)")
        print("   --input: 输入图像或目录")
        print("   --output: 输出图像或目录")
        print("   --mode: 处理模式 (single/batch)")
        print("   --max_images: 批量处理的最大图像数")
        print("   --compare: 用于比较的高分辨率图像")
        print("\n🔍 功能: 将256×256图像放大到1024×1024 (4x超分辨率)")
        sys.exit(0)
    
    main()
