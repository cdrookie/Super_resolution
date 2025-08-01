#!/usr/bin/env python3
"""
GPU内存检查和训练启动脚本
检查GPU内存使用情况，如果有其他进程占用大量内存则提示用户
"""

import torch
import subprocess
import sys
import os

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if not torch.cuda.is_available():
        print("CUDA不可用，将使用CPU训练")
        return True
    
    print("=" * 60)
    print("GPU内存检查")
    print("=" * 60)
    
    # 获取GPU信息
    device_count = torch.cuda.device_count()
    print(f"可用GPU数量: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3
        print(f"GPU {i}: {props.name}")
        print(f"  总内存: {total_memory:.2f} GB")
        
        # 检查当前内存使用
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free_memory = total_memory - allocated - reserved
        
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已保留: {reserved:.2f} GB")
        print(f"  可用: {free_memory:.2f} GB")
        
        if free_memory < 2.0:
            print(f"  ⚠️  警告: GPU {i} 可用内存不足2GB!")
            return False
        elif free_memory < 5.0:
            print(f"  ⚠️  注意: GPU {i} 可用内存较少，建议减小batch size")
        else:
            print(f"  ✅ GPU {i} 内存充足")
    
    return True

def check_other_processes():
    """检查是否有其他进程占用GPU"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("当前GPU进程信息:")
            print("=" * 60)
            lines = result.stdout.split('\n')
            in_processes_section = False
            
            for line in lines:
                if 'Processes:' in line:
                    in_processes_section = True
                    continue
                if in_processes_section and line.strip():
                    if 'No running processes found' in line:
                        print("✅ 没有其他进程占用GPU")
                        break
                    elif any(c.isdigit() for c in line):
                        print(line)
    except FileNotFoundError:
        print("nvidia-smi命令不可用，无法检查GPU进程")

def get_recommended_settings():
    """根据GPU内存推荐训练设置"""
    if not torch.cuda.is_available():
        return {}
    
    # 检查可用内存
    torch.cuda.empty_cache()
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / 1024**3
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free_memory = total_memory - allocated - reserved
    
    settings = {}
    
    if free_memory >= 8:
        settings['BATCH_SIZE'] = '2'
        settings['TARGET_SIZE'] = '1024'
        settings['NUM_WORKERS'] = '2'
    elif free_memory >= 4:
        settings['BATCH_SIZE'] = '1'
        settings['TARGET_SIZE'] = '1024'
        settings['NUM_WORKERS'] = '1'
    elif free_memory >= 2:
        settings['BATCH_SIZE'] = '1'
        settings['TARGET_SIZE'] = '512'
        settings['NUM_WORKERS'] = '0'
    else:
        settings['BATCH_SIZE'] = '1'
        settings['TARGET_SIZE'] = '256'
        settings['NUM_WORKERS'] = '0'
    
    return settings

def main():
    print("Real-ESRGAN训练前GPU检查工具")
    print("=" * 60)
    
    # 检查GPU内存
    memory_ok = check_gpu_memory()
    
    # 检查其他进程
    check_other_processes()
    
    # 获取推荐设置
    settings = get_recommended_settings()
    
    print("\n" + "=" * 60)
    print("推荐的训练设置:")
    print("=" * 60)
    for key, value in settings.items():
        print(f"{key}={value}")
    
    if not memory_ok:
        print("\n⚠️  GPU内存不足，建议:")
        print("1. 关闭其他占用GPU的程序")
        print("2. 使用更小的batch size和图像尺寸")
        print("3. 设置NUM_WORKERS=0避免多进程问题")
        
        choice = input("\n是否继续训练? (y/n): ").lower().strip()
        if choice != 'y':
            print("训练已取消")
            return
    
    print("\n" + "=" * 60)
    print("启动训练...")
    print("=" * 60)
    
    # 设置环境变量
    for key, value in settings.items():
        os.environ[key] = value
        print(f"设置 {key}={value}")
    
    # 设置CUDA内存配置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # 启动训练脚本
    try:
        subprocess.run([sys.executable, 'train_realESRGAN_fdconv_OST_1.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"训练脚本执行失败: {e}")
    except KeyboardInterrupt:
        print("\n训练被用户中断")

if __name__ == '__main__':
    main()
