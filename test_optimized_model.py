# ========== 快速测试优化后的模型 ==========
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from basicsr.archs.rrdbnet_arch import RRDBNet
import os

# 模型配置（与训练时保持一致）
ULTRA_LITE_CONFIG = {
    'num_in_ch': 1,
    'num_out_ch': 1,
    'num_feat': 24,
    'num_block': 8,
    'num_grow_ch': 12
}

MODEL_PATH = './realESRGAN_fast_training.pth'
TEST_IMAGE = './1.bmp'  # 或者其他测试图像

def load_model():
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet(**ULTRA_LITE_CONFIG)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ 模型加载成功: {MODEL_PATH}")
    else:
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return None, device
    
    model = model.to(device)
    model.eval()
    return model, device

def test_super_resolution(model, device, image_path, scale_factor=2):
    """测试超分辨率效果"""
    if not os.path.exists(image_path):
        print(f"❌ 测试图像不存在: {image_path}")
        return None, None, None
    
    # 读取测试图像
    if image_path.lower().endswith(('.tif', '.tiff')):
        import tifffile
        img = tifffile.imread(image_path)
        if img.ndim == 3:
            img = img.mean(axis=0)
        img = img.astype('float32')
        if img.max() > 1:
            img = img / img.max()
    else:
        img = Image.open(image_path).convert('L')
        img = np.array(img).astype('float32') / 255.0
    
    print(f"原始图像尺寸: {img.shape}")
    
    # 创建低分辨率版本（模拟输入）
    h, w = img.shape
    lr_h, lr_w = h // scale_factor, w // scale_factor
    lr_img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    
    # 将低分辨率图像上采样到原始尺寸（作为模型输入）
    lr_upsampled = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 转换为tensor
    lr_tensor = torch.from_numpy(lr_upsampled).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # 模型推理
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        sr_img = sr_tensor.squeeze().cpu().numpy()
    
    # 确保值在[0,1]范围内
    sr_img = np.clip(sr_img, 0, 1)
    
    print(f"低分辨率图像尺寸: {lr_img.shape}")
    print(f"超分结果尺寸: {sr_img.shape}")
    
    return img, lr_upsampled, sr_img

def visualize_results(original, lr_input, sr_output, save_path='test_results.png'):
    """可视化对比结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original (Ground Truth)')
    axes[0].axis('off')
    
    # 低分辨率输入
    axes[1].imshow(lr_input, cmap='gray')
    axes[1].set_title('Low Resolution Input')
    axes[1].axis('off')
    
    # 超分结果
    axes[2].imshow(sr_output, cmap='gray')
    axes[2].set_title('Super Resolution Output')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 结果已保存: {save_path}")

def calculate_metrics(original, sr_output):
    """计算评估指标"""
    # PSNR
    mse = np.mean((original - sr_output) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # SSIM (简化版)
    from scipy.ndimage import gaussian_filter
    mu1 = gaussian_filter(original, sigma=1.5)
    mu2 = gaussian_filter(sr_output, sigma=1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(original * original, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(sr_output * sr_output, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(original * sr_output, sigma=1.5) - mu1_mu2
    
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    
    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim = np.mean(ssim)
    
    return psnr, ssim

def main():
    print("=" * 60)
    print("测试优化训练后的Real-ESRGAN模型")
    print("=" * 60)
    
    # 加载模型
    model, device = load_model()
    if model is None:
        return
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 测试超分辨率
    original, lr_input, sr_output = test_super_resolution(model, device, TEST_IMAGE)
    
    if original is not None:
        # 计算评估指标
        psnr, ssim = calculate_metrics(original, sr_output)
        print(f"\n📊 评估指标:")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim:.4f}")
        
        # 可视化结果
        visualize_results(original, lr_input, sr_output)
        
        # 保存单独的结果图像
        cv2.imwrite('sr_result.png', (sr_output * 255).astype(np.uint8))
        print("✅ 超分结果已保存: sr_result.png")
    
    print("\n🎉 测试完成！")

if __name__ == '__main__':
    main()
