# 全量 RGB 训练测试报告

## 1. 任务概述
- **目标**: 在完整 MARIDA 数据集 (RGB 3通道) 上训练 YOLOv11n 模型。
- **数据集**: `mask_bbox_rescue_split` (Train: 347, Val: 114, Test: 53)。
- **输入格式**: `.npy` (预处理后的 NumPy 数组, 归一化并缩放至 0-255)。
- **模型**: YOLOv11n (预训练权重 `yolo11n.pt`)。

## 2. 训练配置
- **Epochs**: 150 (实际运行 59 Epochs, Early Stopping 触发)。
- **Batch Size**: 16。
- **Image Size**: 640。
- **Optimizer**: Auto。
- **Patience**: 20。

## 3. 训练结果 (Best Epoch: 39)
- **mAP50**: 0.407
- **mAP50-95**: 0.245
- **Precision**: 0.413
- **Recall**: 0.410

### 类别性能详情
| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **All** | 114 | 235 | 0.413 | 0.410 | 0.407 | 0.245 |
| Marine_Debris | 77 | 192 | 0.434 | 0.656 | 0.486 | 0.306 |
| Dense_Sargassum | 3 | 3 | 0.501 | 0.667 | 0.672 | 0.421 |
| Sparse_Sargassum | 3 | 3 | 0.602 | 0.333 | 0.342 | 0.273 |
| Ship | 19 | 19 | 0.674 | 0.654 | 0.682 | 0.379 |
| Foam | 11 | 16 | 0.251 | 0.107 | 0.160 | 0.0799 |
| Wake | 2 | 2 | 0.0177 | 0.0442 | 0.101 | 0.0101 |

## 4. 问题与解决方案
- **问题**: `RuntimeError: No valid images found`。
  - **原因**: 数据集目录中混杂了原始 `.tif` 文件和预处理后的 `.npy` 文件。自定义 Dataset 类在扫描文件时包含了 `.tif` 文件，导致加载失败。
  - **解决**: 
    1. 修改 `ultralytics_plugins/mask_rescue_dataset.py`，在 `get_img_files` 中显式过滤掉 `.tif` 文件，仅保留 `.npy`。
    2. 使用 Monkey Patch 技术 (`train_full_rgb.py`) 强制 Ultralytics 使用修正后的 `MaskRescueNPYDataset` 类。
    3. 修复了 `MaskRescueNPYDataset` 初始化时传递 `use_segments` 参数导致的 `TypeError`。

## 5. 结论
- 全量训练流程已跑通。
- 模型在主要类别 (Marine_Debris, Ship) 上表现尚可，但在少样本类别 (Wake, Foam) 上表现较差。
- 建议后续尝试多光谱 (6通道) 训练以提升性能。
