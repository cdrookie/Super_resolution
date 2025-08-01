#!/bin/bash
# ========== 共享内存和系统优化脚本 ==========

echo "========== 当前系统状态 =========="
echo "共享内存使用情况："
df -h /dev/shm

echo ""
echo "系统内存使用情况："
free -h

echo ""
echo "文件描述符限制："
ulimit -n

echo ""
echo "========== 优化系统设置 =========="

# 1. 增加共享内存大小
echo "1. 增加共享内存大小到16GB..."
if [ "$EUID" -eq 0 ]; then
    mount -o remount,size=16G /dev/shm
    echo "共享内存已增加到16GB"
else
    echo "需要root权限来增加共享内存，请手动运行："
    echo "sudo mount -o remount,size=16G /dev/shm"
fi

# 2. 增加文件描述符限制
echo ""
echo "2. 增加文件描述符限制..."
ulimit -n 65536
echo "文件描述符限制已设置为65536"

# 3. 设置PyTorch环境变量
echo ""
echo "3. 设置PyTorch环境变量..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# 4. 清理临时文件
echo ""
echo "4. 清理临时文件..."
rm -rf /tmp/torch_* 2>/dev/null
rm -rf /dev/shm/torch_* 2>/dev/null
rm -rf /tmp/.pytorch_* 2>/dev/null

# 5. 创建临时环境变量脚本
echo ""
echo "5. 创建环境设置脚本..."
cat > set_training_env.sh << 'EOF'
#!/bin/bash
# 训练环境设置脚本

# 设置共享内存（需要root权限）
# sudo mount -o remount,size=16G /dev/shm

# 设置文件描述符限制
ulimit -n 65536

# 设置PyTorch环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# 设置CUDA环境
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

echo "训练环境已设置完成！"
echo "共享内存: $(df -h /dev/shm | tail -1 | awk '{print $2}')"
echo "文件描述符限制: $(ulimit -n)"
EOF

chmod +x set_training_env.sh

echo ""
echo "========== 优化后的系统状态 =========="
echo "共享内存使用情况："
df -h /dev/shm

echo ""
echo "文件描述符限制："
ulimit -n

echo ""
echo "========== 使用说明 =========="
echo "1. 如果有root权限，请运行："
echo "   sudo mount -o remount,size=16G /dev/shm"
echo ""
echo "2. 在训练前运行："
echo "   source set_training_env.sh"
echo ""
echo "3. 然后运行训练脚本："
echo "   python train_realESRGAN_fdconv_OST_1.py"
echo ""
echo "4. 推荐的NUM_WORKERS设置："
echo "   - 小数据集: NUM_WORKERS = 2-4"
echo "   - 大数据集: NUM_WORKERS = 4-8"
echo "   - 如果仍有问题: NUM_WORKERS = 0"
