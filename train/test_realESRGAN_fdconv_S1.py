# ========== Real-ESRGAN超分模型推理脚本（DL-SMLM原始流程+后处理增强） ========== 
import os
import torch
import tifffile
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch.nn.functional as F
from PIL import Image
import numpy as np

# ========== 配置参数 ========== 
MODEL_PATH = './realESRGAN_fdconv_best.pth'  # 训练好的模型权重
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 加载模型 ========== 
def load_model():
    model = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=23, num_grow_ch=32)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model

# ========== 图像读取与归一化 ========== 
def load_image(input_path):
    ext = os.path.splitext(input_path)[-1].lower()
    if ext in ['.tif', '.tiff']:
        img = tifffile.imread(input_path)
        if img.ndim == 3:
            img = img.sum(axis=0)
        img = img.astype('float32') / 65535.0  # WF/SR均归一化到0~1
    else:
        img = Image.open(input_path).convert('L')
        img = np.array(img).astype('float32') / 255.0
    print(f'input img stats: min={img.min()}, max={img.max()}, mean={img.mean()}')
    return img

# ========== 推理与后处理 ========== 
def super_resolve(input_path, output_path):
    img = load_image(input_path)
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # [1,1,H,W]
    model = load_model()
    with torch.no_grad():
        pred = model(img_tensor)
        pred_up = F.interpolate(pred, size=(2048, 2048), mode='bilinear', align_corners=False)
        sr_img = pred_up.squeeze().cpu().numpy()
        print(f'sr_img stats: min={sr_img.min()}, max={sr_img.max()}, mean={sr_img.mean()}')
    # 反归一化（如SR为点云计数图，假定未用对数变换则直接线性逆归一化）
    sr_img_inv = sr_img * 65535.0  # 若训练用其他分位数或对数变换请自定义
    sr_img_inv = np.clip(sr_img_inv, 0, None)
    # ===== 点云图可视化增强：对数变换+分位数归一化+伪彩色 =====
    # 1. 对数变换，提升稀疏热点可见性
    sr_img_log = np.log1p(sr_img)
    # 2. 分位数归一化（如99.9%分位数），避免极端值影响
    p999 = np.percentile(sr_img_log, 99.9)
    sr_img_norm = sr_img_log / (p999 if p999 > 0 else 1)
    sr_img_norm = np.clip(sr_img_norm, 0, 1)
    # 3. 线性拉伸到0~255
    sr_img_vis = (sr_img_norm * 255.0).astype('uint8')
    # 4. 伪彩色可视化（matplotlib colormap）
    try:
        import matplotlib.pyplot as plt
        sr_img_color = plt.get_cmap('hot')(sr_img_norm)[:, :, :3]  # 取RGB
        sr_img_color = (sr_img_color * 255).astype('uint8')
    except ImportError:
        sr_img_color = None
    # 自动根据输出扩展名保存
    ext = os.path.splitext(output_path)[-1].lower()
    if ext in ['.tif', '.tiff']:
        tifffile.imwrite(output_path, sr_img_vis.astype('uint8'))  # 灰度可视化tif
        if sr_img_color is not None:
            color_path = output_path.replace('.tif', '_color.tif').replace('.tiff', '_color.tiff')
            tifffile.imwrite(color_path, sr_img_color)
        inv_path = output_path.replace('.tif', '_inv.tif').replace('.tiff', '_inv.tiff')
        tifffile.imwrite(inv_path, sr_img_inv.astype('float32'))  # 反归一化tif
        print(f'Saved tif (vis) to {output_path}, tif (inv) to {inv_path}')
        if sr_img_color is not None:
            print(f'Saved tif (color) to {color_path}')
    elif ext in ['.jpg', '.jpeg', '.png']:
        Image.fromarray(sr_img_vis).save(output_path)
        if sr_img_color is not None:
            color_path = output_path.replace(ext, '_color.png')
            Image.fromarray(sr_img_color).save(color_path)
        inv_path = output_path.replace(ext, '_inv.tif')
        tifffile.imwrite(inv_path, sr_img_inv.astype('float32'))
        print(f'Saved {ext} (vis) to {output_path}, tif (inv) to {inv_path}')
        if sr_img_color is not None:
            print(f'Saved color png to {color_path}')
    else:
        Image.fromarray(sr_img_vis).save(output_path + '.png')
        if sr_img_color is not None:
            color_path = output_path + '_color.png'
            Image.fromarray(sr_img_color).save(color_path)
        vis_path = output_path + '.png'
        inv_path = output_path + '_inv.tif'
        tifffile.imwrite(inv_path, sr_img_inv.astype('float32'))
        print(f'Saved png (vis) to {vis_path}, tif (inv) to {inv_path}')
        if sr_img_color is not None:
            print(f'Saved color png to {color_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Real-ESRGAN超分推理脚本（DL-SMLM原始流程+后处理增强）')
    parser.add_argument('--input', type=str, required=True, help='输入低分辨率tif/jpg/png文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出高分辨率tif/jpg/png文件路径')
    args = parser.parse_args()
    super_resolve(args.input, args.output)
