#!/usr/bin/env python3
"""
GPU内存清理脚本
强制清理GPU内存缓存，释放被PyTorch占用的内存
"""

import torch
import gc

def clear_gpu_memory():
    """清理GPU内存"""
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return
    
    print("清理GPU内存...")
    
    # 清理Python垃圾回收
    gc.collect()
    
    # 清理PyTorch缓存
    torch.cuda.empty_cache()
    
    # 同步GPU操作
    torch.cuda.synchronize()
    
    # 显示清理后的内存状态
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        torch.cuda.set_device(i)
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free_memory = total_memory - allocated - reserved
        
        print(f"GPU {i}: {props.name}")
        print(f"  总内存: {total_memory:.2f} GB")
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已保留: {reserved:.2f} GB")
        print(f"  可用: {free_memory:.2f} GB")

if __name__ == '__main__':
    clear_gpu_memory()
    print("GPU内存清理完成!")
