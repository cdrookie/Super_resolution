# Real-ESRGAN 4x超分辨率训练和测试指南

## 📖 概述

这是一套完整的Real-ESRGAN超分辨率解决方案，专门用于：
- **RGB彩色图像**的4x超分辨率
- **256×256 → 1024×1024** 真正的分辨率提升
- 支持训练和测试的完整流程

## 🗂️ 文件结构

```
├── generate_lr_images_sr.py          # 超分辨率数据预处理脚本
├── train_from_pregenerated_2.py      # 超分辨率训练脚本
├── test_realESRGAN_4x_sr.py          # 超分辨率测试脚本
└── README_SuperResolution.md         # 本文档
```

## 🚀 使用流程

### 步骤1: 数据预处理

首先生成用于训练的低分辨率图像对：

```bash
# 生成4x降质数据 (RGB彩色)
python generate_lr_images_sr.py --scales 4

# 或者限制每个类别的图像数量
python generate_lr_images_sr.py --scales 4 --max_per_category 1000

# 验证生成的数据
python generate_lr_images_sr.py --verify --scales 4
```

**数据预处理特点：**
- 输入：高分辨率RGB图像 (任意尺寸)
- 输出：256×256 RGB低分辨率图像 (模型输入)
- 目标：1024×1024 RGB高分辨率图像 (训练目标)

### 步骤2: 模型训练

使用预处理的数据训练超分辨率模型：

```bash
# 开始训练
python train_from_pregenerated_2.py
```

**训练配置：**
- 模型：RRDBNet (3→3通道，RGB)
- 输入尺寸：256×256 RGB
- 输出尺寸：1024×1024 RGB
- 放大倍数：4x
- 批次大小：1 (可调整)
- 训练轮数：20

**模型选择：**
- `ultra_lite`: 轻量级 (32特征, 12块)
- `lite`: 标准 (48特征, 16块) **默认**
- `standard`: 重型 (64特征, 23块)

### 步骤3: 模型测试

训练完成后，使用测试脚本进行超分辨率推理：

```bash
# 单张图像测试
python test_realESRGAN_4x_sr.py --input test.jpg --output test_4x.png --mode single

# 批量处理
python test_realESRGAN_4x_sr.py --input ./inputs --output ./outputs --mode batch

# 与高分辨率图像对比
python test_realESRGAN_4x_sr.py --input lr.jpg --output sr.png --compare hr.jpg --mode single

# 使用不同模型配置
python test_realESRGAN_4x_sr.py --input test.jpg --output test_4x.png --config standard
```

## 📊 关键特性

### 真正的超分辨率
- ✅ **分辨率提升**: 256×256 → 1024×1024 (4x)
- ✅ **RGB彩色支持**: 完整的3通道彩色图像处理
- ✅ **细节重建**: 通过深度学习重建高频细节

### 与之前版本的区别
| 特性 | 之前版本 | 新版本 (\_2) |
|------|----------|--------------|
| 输入通道 | 1 (灰度) | 3 (RGB) |
| 输出通道 | 1 (灰度) | 3 (RGB) |
| 输入尺寸 | 512×512 | 256×256 |
| 输出尺寸 | 512×512 | 1024×1024 |
| 任务类型 | 图像增强 | 超分辨率 |
| 放大倍数 | 1x | 4x |

## 🛠️ 配置选项

### 训练参数 (train_from_pregenerated_2.py)

```python
# 主要配置
SCALE_FACTOR = 4          # 放大倍数
BATCH_SIZE = 1            # 批次大小
NUM_EPOCHS = 20           # 训练轮数
LR = 5e-4                 # 学习率
CURRENT_MODEL = 'lite'    # 模型复杂度
```

### 测试参数 (test_realESRGAN_4x_sr.py)

```bash
--model: 模型权重文件路径
--config: 模型配置 (ultra_lite/lite/standard)
--input: 输入图像或目录
--output: 输出图像或目录
--mode: 处理模式 (single/batch)
--max_images: 批量处理的最大图像数
--compare: 用于比较的高分辨率图像
```

## 📁 目录结构

```
Super_resolution/
├── OST/                              # 高分辨率训练数据
│   ├── category1/
│   ├── category2/
│   └── ...
├── OST_LR/                           # 生成的低分辨率数据
│   └── scale_4x/
│       ├── category1/
│       ├── category2/
│       └── ...
├── inputs/                           # 测试输入图像
├── outputs/                          # 测试输出结果
└── realESRGAN_4x_super_resolution.pth # 训练的模型权重
```

## 🔧 依赖环境

确保安装以下依赖：

```bash
pip install torch torchvision
pip install pillow numpy tqdm
pip install basicsr  # Real-ESRGAN模型库
```

## 💡 使用技巧

### 训练优化
1. **GPU内存管理**: 批次大小设为1以节省显存
2. **数据预处理**: 使用generate_lr_images_sr.py预生成数据
3. **模型选择**: 根据硬件能力选择模型复杂度

### 测试优化
1. **批量处理**: 使用batch模式提高效率
2. **结果对比**: 使用--compare参数评估效果
3. **格式支持**: 支持jpg, png, bmp, tiff等格式

## 🎯 预期效果

- **输入**: 256×256像素的低分辨率RGB图像
- **输出**: 1024×1024像素的高分辨率RGB图像
- **提升倍数**: 4x分辨率提升
- **质量**: 通过深度学习重建的高质量细节

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小批次大小 (BATCH_SIZE = 1)
   - 使用更小的模型配置 (ultra_lite)

2. **训练数据不存在**
   - 确保运行generate_lr_images_sr.py生成数据
   - 检查OST目录是否包含训练图像

3. **模型导入错误**
   - 安装basicsr: `pip install basicsr`
   - 检查Real-ESRGAN环境配置

## 📈 性能基准

典型性能表现：
- **训练速度**: ~30-60秒/轮 (取决于数据量和硬件)
- **推理速度**: ~1-3秒/张 (GPU)
- **内存占用**: ~2-4GB GPU显存
- **模型大小**: 15-60MB (取决于配置)

---

*本指南涵盖了完整的4x超分辨率训练和测试流程，专为RGB彩色图像的真正分辨率提升而设计。*
