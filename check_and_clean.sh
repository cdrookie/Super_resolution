#!/bin/bash
# ========== 磁盘空间和内存检查脚本 ==========

echo "========== 系统资源检查 =========="

# 检查磁盘空间
echo "磁盘空间使用情况："
df -h

echo ""
echo "当前目录空间使用："
du -sh ./*

echo ""
echo "========== 内存使用情况 =========="
free -h

echo ""
echo "========== 共享内存使用情况 =========="
df -h /dev/shm

echo ""
echo "========== 清理临时文件 =========="

# 清理PyTorch临时文件
echo "清理PyTorch临时文件..."
rm -rf /tmp/torch_*
rm -rf /dev/shm/torch_*

# 清理Python缓存
echo "清理Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# 清理conda缓存
echo "清理Conda缓存..."
conda clean --all -y 2>/dev/null

# 清理pip缓存
echo "清理pip缓存..."
pip cache purge 2>/dev/null

echo ""
echo "========== 设置共享内存大小 =========="
# 增加共享内存大小（需要root权限）
# mount -o remount,size=8G /dev/shm

echo "建议手动运行以下命令增加共享内存（需要root权限）："
echo "sudo mount -o remount,size=8G /dev/shm"

echo ""
echo "========== 清理完成后的磁盘空间 =========="
df -h

echo ""
echo "========== GPU内存使用情况 =========="
nvidia-smi

echo ""
echo "========== 建议的训练参数 =========="
echo "如果仍然遇到内存问题，建议使用以下参数："
echo "NUM_WORKERS = 0"
echo "BATCH_SIZE = 1"
echo "TARGET_SIZE = 512"
echo "pin_memory = False"
