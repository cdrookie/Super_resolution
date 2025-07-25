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
        img = img.astype('float32') / (2**16 - 1)
    else:
        img = Image.open(input_path).convert('L')
        img = np.array(img).astype('float32') / 255.0
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
    # 保存高分辨率图片
    ext = os.path.splitext(output_path)[-1].lower()
    if ext in ['.tif', '.tiff']:
        tifffile.imwrite(output_path, (sr_img * (2**32 - 1)).astype('uint32'))
    else:
        out_img = (sr_img * 255.0).clip(0,255).astype('uint8')
        Image.fromarray(out_img).save(output_path)
    print(f'Saved super-resolved image to {output_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Real-ESRGAN超分推理脚本')
    parser.add_argument('--input', type=str, required=True, help='输入低分辨率tif文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出高分辨率tif文件路径')
    args = parser.parse_args()
    super_resolve(args.input, args.output)
