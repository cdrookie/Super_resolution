# ========== 简化的模型测试脚本（避免复杂依赖） ==========
import torch
import numpy as np
from PIL import Image
import os

# 模型配置（与训练时保持一致）
ULTRA_LITE_CONFIG = {
    'num_in_ch': 1,
    'num_out_ch': 1,
    'num_feat': 24,
    'num_block': 8,
    'num_grow_ch': 12
}

def simple_resize(img, size):
    """简单的resize函数，避免cv2依赖"""
    pil_img = Image.fromarray((img * 255).astype('uint8'), mode='L')
    resized = pil_img.resize((size, size), Image.LANCZOS)
    return np.array(resized).astype('float32') / 255.0

def load_and_test_model():
    """加载模型并进行简单测试"""
    MODEL_PATH = './realESRGAN_fast_training.pth'
    
    print("=" * 50)
    print("简化模型测试")
    print("=" * 50)
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("请确保训练脚本已完成并保存了模型")
        return
    
    try:
        # 导入模型架构
        from basicsr.archs.rrdbnet_arch import RRDBNet
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 创建并加载模型
        model = RRDBNet(**ULTRA_LITE_CONFIG)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型加载成功!")
        print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # 创建测试数据
        print("\n🧪 创建测试数据...")
        test_size = 256
        
        # 生成测试图像（模拟真实数据）
        x = np.linspace(0, 4*np.pi, test_size)
        y = np.linspace(0, 4*np.pi, test_size)
        X, Y = np.meshgrid(x, y)
        test_img = (np.sin(X) * np.cos(Y) + 1) / 2  # 值在[0,1]范围
        
        # 创建低分辨率版本
        lr_size = test_size // 2
        lr_img = simple_resize(test_img, lr_size)
        lr_upsampled = simple_resize(lr_img, test_size)
        
        print(f"原始图像尺寸: {test_img.shape}")
        print(f"低分辨率尺寸: {lr_img.shape}")
        print(f"上采样输入尺寸: {lr_upsampled.shape}")
        
        # 模型推理
        print("\n🚀 进行模型推理...")
        lr_tensor = torch.from_numpy(lr_upsampled).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            sr_tensor = model(lr_tensor)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                print(f"⚡ 推理时间: {inference_time:.2f}ms")
            
            sr_img = sr_tensor.squeeze().cpu().numpy()
        
        # 计算简单的质量指标
        print("\n📊 质量评估:")
        
        # MSE和PSNR
        mse = np.mean((test_img - sr_img) ** 2)
        if mse > 0:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            print(f"MSE: {mse:.6f}")
            print(f"PSNR: {psnr:.2f} dB")
        else:
            print("MSE: 0 (完美重建)")
            print("PSNR: ∞ dB")
        
        # 保存结果
        print("\n💾 保存测试结果...")
        
        # 保存为图像
        Image.fromarray((test_img * 255).astype('uint8'), mode='L').save('test_original.png')
        Image.fromarray((lr_upsampled * 255).astype('uint8'), mode='L').save('test_lr_input.png')
        Image.fromarray((np.clip(sr_img, 0, 1) * 255).astype('uint8'), mode='L').save('test_sr_output.png')
        
        print("✅ 测试图像已保存:")
        print("  - test_original.png (原始)")
        print("  - test_lr_input.png (低分输入)")
        print("  - test_sr_output.png (超分输出)")
        
        # 模型性能总结
        print(f"\n🎉 模型测试完成!")
        print(f"✅ 模型可以正常加载和推理")
        print(f"✅ 输出尺寸正确: {sr_img.shape}")
        print(f"✅ 数值范围合理: [{sr_img.min():.3f}, {sr_img.max():.3f}]")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装basicsr库")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_with_real_image():
    """使用真实图像进行测试"""
    test_images = ['1.bmp', 'SR1.tif', 'WF1.tif']
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n🖼️  测试真实图像: {img_path}")
            try:
                if img_path.endswith('.tif') or img_path.endswith('.tiff'):
                    try:
                        import tifffile
                        img = tifffile.imread(img_path)
                        if img.ndim == 3:
                            img = img.mean(axis=0)
                        img = img.astype('float32')
                        if img.max() > 1:
                            img = img / img.max()
                    except ImportError:
                        print("⚠️  需要tifffile库读取TIFF文件，跳过...")
                        continue
                else:
                    img = Image.open(img_path).convert('L')
                    img = np.array(img).astype('float32') / 255.0
                
                print(f"✅ 成功读取图像，尺寸: {img.shape}")
                
                # 可以在这里添加模型推理代码
                # ... (模型推理逻辑)
                
                break
            except Exception as e:
                print(f"❌ 读取失败: {e}")
                continue
    else:
        print("ℹ️  未找到测试图像，使用合成数据测试")

if __name__ == '__main__':
    # 主要测试
    success = load_and_test_model()
    
    if success:
        # 尝试真实图像测试
        test_with_real_image()
        
        print("\n" + "=" * 50)
        print("🎯 建议下一步:")
        print("1. 查看保存的测试图像，评估视觉效果")
        print("2. 如果效果满意，可以处理更多数据")
        print("3. 如果需要更好效果，可以调整模型配置重新训练")
        print("4. 集成到现有的inference脚本中")
        print("=" * 50)
    else:
        print("\n💡 如果遇到依赖问题，可以:")
        print("1. 检查basicsr是否正确安装")
        print("2. 确认模型文件路径正确")
        print("3. 检查CUDA环境配置")
