# Super_resolution
# u2net代码：https://github.com/xuebinqin/U-2-Net/tree/master
# fdconv代码：https://github.com/Linwei-Chen/FDConv/tree/main
# 还需加载：mmcv-1.7.2
# 数据集采用：DL-SMLM
# 还需准备：mac版本winscp类似软件，mac版本putty，配置云服务器的python环境
# 扩散超分模型：https://github.com/cszn/KAIR/blob/master/README.md
# SR3 (Image Super-Resolution via Iterative Refinement)
# 论文原作者官方实现（TensorFlow）：https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
# PyTorch 高质量复现（推荐）：https://github.com/lyndonzheng/SRDiff https://github.com/baofff/DM-SRGAN
# IDDPM/IDDPM-SR (Improved Denoising Diffusion Probabilistic Models)
# 官方 PyTorch 实现（含超分脚本）：https://github.com/openai/improved-diffusion
# SwinIR-Diffusion（基于SwinIR骨干的扩散超分）https://github.com/cszn/KAIR/tree/main/main_diffusion https://github.com/wyhuai/DDPM-SwinIR
# 其他通用扩散超分框架：https://github.com/CompVis/latent-diffusion（可用于超分任务） https://github.com/ermongroup/ddim

# 扩散超分训练难度更大，但是比较适合医学领域，建议加入尝试

# requirements.txt使用方法（环境依赖移植）
python -m venv venv
source venv/bin/activate  # Linux
.\\venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
# 这样就可以移植配置环境了

# test脚本使用命令 python test_realESRGAN_fdconv.py --input 输入图片路径 --output 输出图片路径