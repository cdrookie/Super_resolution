#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========== Real-ESRGAN超分模型推理脚本 ========== 
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

# ========== 推理函数 ========== 

def load_image(input_path):
    ext = os.path.splitext(input_path)[-1].lower()
    if ext in ['.tif', '.tiff']:
        img = tifffile.imread(input_path)
        if img.ndim == 3:
            img = img.sum(axis=0)
        # 自动检测数据类型，归一化到0~1（与训练一致，65535）
        img = img.astype('float32') / 65535.0
    else:
        img = Image.open(input_path).convert('L')
        img = np.array(img).astype('float32')
        img = img / 255.0  # 归一化到0~1
    print(f'input img stats: min={img.min()}, max={img.max()}, mean={img.mean()}')
    return img

def super_resolve(input_path, output_path):
    # 读取低分辨率图片
    img = load_image(input_path)
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # [1,1,H,W]
    # 加载模型
    model = load_model()
    with torch.no_grad():
        pred = model(img_tensor)
        pred_up = F.interpolate(pred, size=(2048, 2048), mode='bilinear', align_corners=False)
        sr_img = pred_up.squeeze().cpu().numpy()
        print(f'sr_img stats: min={sr_img.min()}, max={sr_img.max()}, mean={sr_img.mean()}')
    # 反归一化：逆分位数归一化+逆对数变换（假设训练用99.9%分位数归一化）
    percentile_99_9 = 1.0  # 若训练时用1.0归一化，可改为实际分位数
    sr_img_inv = np.expm1(sr_img * percentile_99_9)
    sr_img_inv = np.clip(sr_img_inv, 0, None)
    # 可视化增强：拉伸到0~255
    sr_img_vis = (sr_img / sr_img.max() * 255.0).clip(0,255).astype('uint8') if sr_img.max() > 0 else (sr_img * 255.0).astype('uint8')
    # 自动根据输出扩展名保存
    ext = os.path.splitext(output_path)[-1].lower()
    if ext in ['.tif', '.tiff']:
        tifffile.imwrite(output_path, sr_img_vis.astype('uint8'))  # 可视化tif
        tifffile.imwrite(output_path.replace('.tif', '_inv.tif').replace('.tiff', '_inv.tiff'), sr_img_inv.astype('float32'))  # 反归一化tif
        print(f'Saved tif (vis) to {output_path}, tif (inv) to {output_path.replace('.tif', '_inv.tif').replace('.tiff', '_inv.tiff')}')
    elif ext in ['.jpg', '.jpeg', '.png']:
        Image.fromarray(sr_img_vis).save(output_path)
        tifffile.imwrite(output_path.replace(ext, '_inv.tif'), sr_img_inv.astype('float32'))
        print(f'Saved {ext} (vis) to {output_path}, tif (inv) to {output_path.replace(ext, '_inv.tif')}')
    else:
        # 默认保存为png
        Image.fromarray(sr_img_vis).save(output_path + '.png')
        tifffile.imwrite(output_path + '_inv.tif', sr_img_inv.astype('float32'))
        print(f'Saved png (vis) to {output_path + '.png'}, tif (inv) to {output_path + '_inv.tif'}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Real-ESRGAN超分推理脚本')
    parser.add_argument('--input', type=str, required=True, help='输入低分辨率tif文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出高分辨率tif文件路径')
    args = parser.parse_args()
    super_resolve(args.input, args.output)
