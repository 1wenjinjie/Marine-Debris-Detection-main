# 7ch Baseline 系统分析（验证版）

本文档基于当前项目代码与已完成验证结果整理，用于论文写作与技术复盘。

## 1. 结论先行

验证结论：文件结构导航核心准确，7ch Baseline 分析总体正确，但有少量路径与条件描述需要修正。

- Baseline 主线成立：`YOLOv11n + 7ch 输入 + CIoU + knowledge heatmap`。
- 核心实现均可在代码中定位。
- 关键性能指标可追溯且一致。

## 2. 已验证为正确的关键文件

| 文件 | 状态 | 实际验证点 |
|---|---|---|
| `configs/models/yolov11n_7ch.yaml` | ✅ 存在 | `ch: 7`，定义 7 通道输入模型 |
| `configs/datasets/marida_rescue_7ch.yaml` | ✅ 存在 | `channels: 7` 正确；`path` 需指向实际样本目录后才可直接训练 |
| `ultralytics_plugins/mask_rescue_dataset_fixed.py` | ✅ 存在 | 包含 `MaskRescueNPYDataset` 核心实现 |
| `ultralytics_plugins/mask_rescue_trainer.py` | ✅ 存在 | `MaskRescueTrainer(DetectionTrainer)` 自定义训练器 |
| `scripts/preprocessing/preprocess_generate_knowledge_channel.py` | ✅ 存在 | 第 7 通道热图生成脚本 |
| `scripts/training/train_mask_rescue.py` | ✅ 存在 | 训练入口，调用 `MaskRescueTrainer` |
| `scripts/visualize_knowledge_overlay.py` | ✅ 存在 | 第 7 通道可视化与叠加检查 |

## 3. 需要修正的细节

### 3.1 数据集配置文件名

分析中曾出现：`marida_mask_rescue_7ch.yaml`

实际应为：`marida_rescue_7ch.yaml`

训练脚本默认参数也使用实际文件名：

```python
# scripts/training/train_mask_rescue.py
parser.add_argument("--data", default="marida_rescue_7ch.yaml", ...)
```

说明：该差异是命名层面的，不影响系统功能与实验结论。

### 3.2 数据集 `path` 指向

当前配置文件中：`path: D:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split`

核验结果：

- 该目录存在，但不包含可直接训练所需的 `train.txt` 与训练样本。
- 实际可用数据目录为：`data/processed/mask_bbox_rescue_split`（包含 `train.txt` 且有训练样本）。

说明：若直接按当前 YAML `path` 训练，可能出现数据加载异常；应以实际数据目录为准。

## 4. 核心技术实现验证

### 4.1 置信度加权采样

- 位置：`mask_rescue_dataset_fixed.py`（约第 522 行）
- 方法：`_align_sample_weights()`
- 权重思路：基于样本统计对采样分布进行加权，缓解数据不均衡与难例不足问题。

### 4.2 ClassAwareCopyPaste

- 位置：`mask_rescue_dataset_fixed.py`（类定义约第 47 行）
- 作用：按类别约束进行 Copy-Paste，避免无约束粘贴带来的噪声。
- 启用条件：仅在 `copy_paste > 0` 且提供了类别配置（或环境变量）时启用。

### 4.3 11-to-6 波段选择

- 位置：`mask_rescue_dataset_fixed.py`（约第 433 行）
- 代码逻辑：`arr = arr[..., [3, 2, 1, 7, 9, 10]]`
- 映射关系：`Red(B4), Green(B3), Blue(B2), NIR(B8), SWIR1(B11), SWIR2(B12)`

### 4.4 第 7 通道（知识热图）读取

- 可视化侧验证：`scripts/visualize_knowledge_overlay.py`
- 核心读取方式：`heatmap = arr[:, :, -1]`（最后一个通道）

## 5. Baseline 训练配置核对（关键增强策略）

在 `train_mask_rescue.py` 中，已确认以下多光谱友好配置：

- `hsv_h = 0.0`：禁用色调增强（保护光谱特征）
- `hsv_s = 0.0`：禁用饱和度增强
- `hsv_v = 0.0`：禁用亮度增强
- `degrees = 180.0`：启用强几何旋转增强
- `flipud = 0.5`：垂直翻转
- `fliplr = 0.5`：水平翻转
- `mosaic = 0.0`：禁用 Mosaic（避免破坏 float32 多通道数据分布）
- `mixup = 0.0`：禁用 Mixup
- `copy_paste = 0.0`：默认标准 CP 关闭；`ClassAwareCopyPaste` 不会在默认参数下自动启用

## 6. 修正后的基线文件结构导航（论文可引用）

| 模块 | 关键文件 | 作用说明 | 验证状态 |
|---|---|---|---|
| 配置 | `configs/models/yolov11n_7ch.yaml` | 定义 7 通道模型结构（`ch: 7`） | ✅ |
| 配置 | `configs/datasets/marida_rescue_7ch.yaml` | 类别定义正确（`channels: 7`）；`path` 需指向实际样本目录 | ✅ |
| 数据加载 | `ultralytics_plugins/mask_rescue_dataset_fixed.py` | NPY 读取、采样权重、通道选择、增强 | ✅ |
| 训练器 | `ultralytics_plugins/mask_rescue_trainer.py` | 训练入口对接 `MaskRescueNPYDataset` | ✅ |
| 预处理 | `scripts/preprocessing/preprocess_generate_knowledge_channel.py` | 生成第 7 通道热图与统计 | ✅ |
| 训练入口 | `scripts/training/train_mask_rescue.py` | Baseline 启动脚本与训练超参数 | ✅ |
| 可视化 | `scripts/visualize_knowledge_overlay.py` | 热图与 RGB 叠加检查 | ✅ |

## 7. Baseline 结果与解释（摘要）

当前最优主线：7ch Baseline（CIoU）

- F1: 0.6774
- Precision: 0.6281
- Recall: 0.7351
- mAP50: 0.6673

对比结论：

- 7ch 主线优于 10ch 最优对照（SIoU）
- SIoU 在 10ch 中有效，但在 7ch 中未进一步提升主线表现
- 说明“通道设计 + 先验机制 + 损失函数”存在耦合关系，不能跨配置直接迁移

## 8. 最终结论

你的 7ch Baseline 系统分析结论总体成立：

- 关键文件齐全
- 核心技术已实现
- 代码定位可追溯
- 训练策略与多光谱场景一致

本次核验主要修正了三点：可视化脚本路径、数据集 `path` 可用性描述、`ClassAwareCopyPaste` 启用条件。修正后文档可直接用于论文与答辩材料。

---

## 补充：7ch Baseline 系统完整文件清单（原文保留）

### 一、核心代码文件（跑通系统的必需文件）

#### 1.1 模型配置

| 文件路径 | 作用 | 关键内容 |
|---|---|---|
| `yolov11n_7ch.yaml` | 定义 7 通道模型架构 | `ch: 7`, `nc: 7` (7类) |
| `yolov11n_7ch_init.pt` | 预训练权重（7ch 初始化） | 从 3ch 预训练权重扩展而来 |

#### 1.2 数据集配置

| 文件路径 | 作用 | 关键内容 |
|---|---|---|
| `marida_rescue_7ch.yaml` | 主数据集配置 | `channels: 7`, 数据路径 |
| `marida_mask_rescue_7ch.yaml` | 备用配置（旧版） | 功能同上 |
| `configs/datasets/marida_rescue_split_*.yaml` | 其他通道变体 | 3ch/4ch/5ch/6ch 配置 |

#### 1.3 数据加载核心

| 文件路径 | 作用 | 关键技术 |
|---|---|---|
| `mask_rescue_dataset_fixed.py` | 7ch NPY 数据集加载器 | ✅ `MaskRescueNPYDataset` 类<br>✅ 置信度加权采样 `_align_sample_weights()`<br>✅ ClassAwareCopyPaste<br>✅ 11-to-6 波段选择 `[3,2,1,7,9,10]`<br>✅ P01-P99 归一化 |
| `mask_rescue_trainer.py` | 自定义训练器 | 绑定 `MaskRescueNPYDataset` 到训练循环 |
| `ultralytics_plugins/mask_rescue_validator.py` | 自定义验证器 | 支持 7ch 推理验证 |

备份文件（可忽略）：

- `mask_rescue_dataset.py.backup`
- `mask_rescue_dataset.py.broken`
- `mask_rescue_dataset_backup.py`

#### 1.4 训练入口脚本

| 文件路径 | 作用 | 默认参数 |
|---|---|---|
| `train_mask_rescue.py` | 主训练脚本 | `--data marida_rescue_7ch.yaml`<br>`--model yolov11n_7ch.yaml`<br>`--epochs 50`, `--batch 8` |
| `scripts/training/train_7ch_main.py` | 备用训练脚本 | 功能相似 |
| `scripts/training/train_7ch_main_fixed.py` | 修复版训练脚本 | 功能相似 |

#### 1.5 数据预处理

| 文件路径 | 作用 | 核心算法 |
|---|---|---|
| `scripts/preprocessing/preprocess_generate_knowledge_channel.py` | 生成第 7 通道热图 | ✅ `cv2.distanceTransform(mask, DIST_L2, 5)`<br>✅ 归一化到 [0,1]<br>✅ 生成 `knowledge_generation_summary.json` |
| `scripts/preprocessing/build_mask_rescue_dataset.py` | 构建 7ch 数据集 | 6ch + Heatmap -> 7ch NPY |
| `scripts/create_rescue_manifests.py` | 生成 train/val/test.txt | 数据集分割清单 |

#### 1.6 工具脚本

| 文件路径 | 作用 |
|---|---|
| `visualize_knowledge_overlay.py` | 可视化第 7 通道热图叠加效果 |
| `scripts/evaluation/validate_mask_rescue_model.py` | 验证训练好的 7ch 模型 |
| `scripts/utils/infer_mask_rescue.py` | 7ch 模型推理脚本 |
| `scripts/utils/cleanup_mask_rescue_nan.py` | 清理数据中的 NaN 值 |

### 二、性能证明文件（Baseline SOTA 结果）

#### 2.1 关键结果文件

| 文件路径 | 作用 | 核心数据 |
|---|---|---|
| `runs_marida/CRITICAL_7ch_vs_10ch_analysis.json` | 7ch vs 10ch 对比分析 | ✅ F1=0.6774 (7ch SOTA)<br>✅ Precision=0.6281<br>✅ Recall=0.7351<br>✅ mAP50=0.6673 |
| `train_summary.json` | Baseline 训练摘要 | 50 epochs, batch=8, imgsz=512 |
| `experiments/runs/baseline_sota_aligned_1223_2219/results.csv` | 完整训练曲线 | 每个 epoch 的指标记录 |
| `best.pt` | 最佳权重 | F1=0.6774 的模型权重 |

#### 2.2 通道统计与元数据

| 文件路径 | 作用 | 内容 |
|---|---|---|
| `experiments/runs/marida_runs/channel_stats_7ch.json` | 7 通道归一化统计 | P01/P25/P50/P75/P99 分位数<br>Mean, Std 统计 |
| `data/processed/mask_bbox_rescue_split/knowledge_generation_summary.json` | 热图生成统计 | 1383 样本的 `confidence_weight` |

#### 2.3 可视化结果

实验目录 `experiments/runs/baseline_sota_aligned_1223_2219/` 包含：

- `confusion_matrix.png` / `confusion_matrix_normalized.png`
- `BoxF1_curve.png`, `BoxPR_curve.png`
- `results.png` (训练曲线)
- `val_batch*_pred.jpg` (验证集预测可视化)

#### 2.4 实验对比结果

| 文件路径 | 内容 |
|---|---|
| `analysis_results.json` | Exp-F (7ch + SIoU) 失败分析 |
| `EXPERIMENT_SUMMARY_FINAL.json` | 所有实验汇总（包含 7ch baseline） |

### 三、数据文件（运行时必需）

#### 3.1 数据集目录规划（用户确认版）

```text
D:/OceanEdge_Group/Marine-Debris-Detection-main/
│
├── data/
│   ├── raw/
│   │   └── MARIDA/                          # 原始 MARIDA 数据集
│   │       ├── patches/                     # 原始影像和掩码
│   │       │   ├── S2_1-12-19_48MYU/
│   │       │   │   ├── S2_1-12-19_48MYU_0.tif       # 11 波段原始影像
│   │       │   │   ├── S2_1-12-19_48MYU_0_cl.tif    # GT 分割掩码
│   │       │   │   ├── S2_1-12-19_48MYU_0_conf.tif  # 置信度图
│   │       │   │   └── ...
│   │       │   └── (其他 scene 目录)
│   │       ├── shapefiles/                  # 原始标注文件
│   │       └── splits/                      # 官方数据集分割
│   │
│   └── processed/
│       └── mask_bbox_rescue_split/          # ✅ 7ch 处理后的数据（真实位置）
│           ├── train/
│           │   ├── images/                  # 训练集图像
│           │   │   ├── S2_1-12-19_48MYU_0.npy   # 7ch NPY (256×256×7, float32)
│           │   │   ├── S2_1-12-19_48MYU_0.tif   # 备份 TIF 格式
│           │   │   └── ...
│           │   ├── labels/                  # YOLO 格式标注
│           │   │   ├── S2_1-12-19_48MYU_0.txt
│           │   │   └── ...
│           │   ├── images_aug/              # 增强后的图像（可选）
│           │   └── labels_aug/              # 增强后的标注（可选）
│           │
│           ├── val/                         # 验证集（同样结构）
│           │   ├── images/
│           │   └── labels/
│           │
│           ├── test/                        # 测试集（同样结构）
│           │   ├── images/
│           │   └── labels/
│           │
│           ├── train.txt                    # 训练集清单（相对路径）
│           ├── val.txt                      # 验证集清单
│           ├── test.txt                     # 测试集清单
│           ├── classes.txt                  # 类别名称
│           ├── knowledge_generation_summary.json  # 热图生成元数据
│           ├── class_balance_weights.json   # 类别平衡权重
│           └── channel_report.json          # 通道统计报告
│
└── mask_bbox_rescue_split/                  # ❌ 空目录（配置错误指向这里）
    └── mask_bbox_rescue_split/
        └── knowledge_generation_summary.json  # 只有这一个文件
```

#### 3.2 路径使用结论（训练必须遵循）

- 训练与验证应使用：`data/processed/mask_bbox_rescue_split`。
- 顶层 `mask_bbox_rescue_split/` 为错误指向路径（历史遗留），不可作为训练数据根目录。
- `marida_rescue_7ch.yaml` 中的 `path` 字段需与上述真实路径保持一致。

### 四、核心技术实现位置

| 技术点 | 实现位置 | 代码行号/方法 |
|---|---|---|
| 输入级知识蒸馏（第 7 通道） | `preprocess_generate_knowledge_channel.py` | `mask_to_heatmap()` 函数<br>`cv2.distanceTransform()` |
| 置信度加权采样 | `mask_rescue_dataset_fixed.py` | 第 522 行 `_align_sample_weights()`<br>权重公式: `0.4 + 0.1*(mean+coverage)/2` |
| ClassAwareCopyPaste | `mask_rescue_dataset_fixed.py` | 第 47 行 `ClassAwareCopyPaste` 类<br>第 368 行动态替换逻辑 |
| 11-to-6 波段选择 | `mask_rescue_dataset_fixed.py` | 第 433 行 `arr[..., [3,2,1,7,9,10]]` |
| P01-P99 归一化 | `mask_rescue_dataset_fixed.py` | 第 156-168 行 `_load_channel_stats()`<br>使用 `channel_stats_7ch.json` |
| 光谱保护增强策略 | `train_mask_rescue.py` | 第 59-66 行<br>`hsv_h/s/v=0.0, degrees=180, mosaic=0` |
| 热图可视化 | `visualize_knowledge_overlay.py` | 第 100 行 `heatmap = arr[:,:, -1]` |

### 五、完整运行流程

#### 5.1 数据准备（已完成，无需重跑）

```bash
# 1. 提取 6 通道多光谱数据
python scripts/preprocessing/preprocess_extract_bands.py --bands 6

# 2. 生成第 7 通道热图
python scripts/preprocessing/preprocess_generate_knowledge_channel.py \
  --source-dataset MARIDA_YOLO_6band_split \
  --target-dataset mask_bbox_rescue_split \
  --mask-root MARIDA/patches

# 3. 生成数据集清单
python scripts/create_rescue_manifests.py
```

#### 5.2 训练 7ch Baseline（复现 F1=0.6774）

```bash
python scripts/training/train_mask_rescue.py \
  --model configs/models/yolov11n_7ch.yaml \
  --data configs/datasets/marida_rescue_7ch.yaml \
  --epochs 50 \
  --batch 8 \
  --imgsz 512 \
  --device 0 \
  --name baseline_7ch_reproduce
```

#### 5.3 验证模型

```bash
python scripts/evaluation/validate_mask_rescue_model.py \
  --weights experiments/runs/baseline_sota_aligned_1223_2219/weights/best.pt \
  --data configs/datasets/marida_rescue_7ch.yaml
```

### 六、性能指标总结

7ch Baseline SOTA (来自 `CRITICAL_7ch_vs_10ch_analysis.json`):

- F1 Score: 0.6774 ✅
- Precision: 0.6281
- Recall: 0.7351 (高召回是核心优势)
- mAP50: 0.6673
- 训练轮数: 50 epochs (最佳在 epoch 38)
- 模型大小: YOLOv11n (2.59M params)

对比实验:

- 10ch + CIoU: F1=0.462 (比 7ch 低 46.6%)
- 10ch + SIoU: F1=0.628 (比 7ch 低 7.3%)
- 7ch + SIoU: F1=0.648 (比 7ch 低 4.3%, 失败)

