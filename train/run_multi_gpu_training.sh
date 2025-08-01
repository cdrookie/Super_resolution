#!/bin/bash
# ========== 多GPU训练启动脚本 ==========

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定使用的GPU ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 检查GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $NUM_GPUS 个GPU"

# 显示GPU信息
echo "GPU信息："
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

# 方法1: 使用DataParallel (简单，推荐用于单机多卡)
echo "========== 方法1: DataParallel训练 =========="
echo "启动多GPU训练 (DataParallel)..."
python train_realESRGAN_fdconv_OST_1.py

# 方法2: 使用DistributedDataParallel (更高效，推荐用于多机多卡)
# echo "========== 方法2: DistributedDataParallel训练 =========="
# echo "启动分布式训练 (DistributedDataParallel)..."
# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train_realESRGAN_fdconv_OST_1.py

# 方法3: 使用torchrun (PyTorch 1.10+推荐)
# echo "========== 方法3: torchrun训练 =========="
# echo "启动torchrun分布式训练..."
# torchrun --nproc_per_node=$NUM_GPUS train_realESRGAN_fdconv_OST_1.py
