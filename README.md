# Marine Debris Detection Dataset

A comprehensive toolkit for processing and analyzing the MARIDA (Marine Debris Archive) dataset for deep learning-based marine debris detection using multispectral satellite imagery.

## v2 六通道基线复现入口（独立实验）

以下入口对应 `marida_v2` 三类检测独立实验；历史 v1 七通道论文主线及其 oracle 限制见后文。当前仅完成公开发布准备的一部分，尚未验证全新环境下的训练复现；仓库是否 Public、许可和归档 DOI 均待确认。发布步骤与阻断项见 [公开发布清单](docs/PUBLICATION.md)。

### 环境

在独立 Python 环境中，先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择与本机 CUDA/驱动匹配的 `torch`、`torchvision`，再安装项目依赖和 Ultralytics：

```bash
python -m pip install -r requirements.txt
python -m pip install ultralytics
```

`requirements.txt` 不包含完整训练依赖，也未锁定版本；以上安装命令不是已验证的环境锁文件。本项目使用 Ultralytics 内部接口，最新版可能不兼容。发布前必须从**实际生成论文结果的环境**记录并锁定 Python、torch、torchvision、ultralytics 及其他包版本，同时记录 CUDA、驱动和硬件；不要用未运行过实验的环境生成“复现环境”。

### 数据与路径

从 MARIDA 原始发布记录获取数据，并在最终发布元数据中核实原记录 DOI 和许可；MADOS 外部验证也应引用其原始发布记录。原始数据不随代码重新分发。v2 运行需要对应的预处理产物、场景划分和通道统计；下方历史预处理入口不等于 v2 数据构建流程。

```text
data/processed/marida_v2/
  train.txt / val.txt / test.txt
  train/images/*.npy     # val/、test/ 同结构，float32 六通道
  train/labels/*.txt     # YOLO 归一化 xywh
  channel_stats_6ch.json
```

通道顺序为 B4、B3、B2、B8、B11、B12；类别为 Marine_Debris (0)、Algae_Organic (1)、Ship (2)。修改 `configs/datasets/marida_v2_6ch.yaml` 的 `path`，使其指向本机的上述数据目录。检查三个 manifest 中每一行：旧机器的绝对路径必须重定位，也可改成相对 manifest 所在目录的路径（如 `train/images/tile.npy`）。保持原始场景划分、标签和通道统计一致，确认所有条目均存在，避免加载器跳过缺失样本。

### 训练、测试与逐实例诊断

从仓库根目录执行，训练默认使用 GPU 0；以下命令作为复现入口，未在本轮执行：

```bash
python scripts/v2_baseline_train.py --seed 0
python scripts/v2_evaluate.py --weights runs_v2/baseline_6ch_s0/weights/best.pt --data configs/datasets/marida_v2_6ch.yaml --channels 6 --split test --imgsz 512 --conf 0.001 --iou 0.6
python scripts/v2_sweep_thresholds.py --weights runs_v2/baseline_6ch_s0/weights/best.pt --tag baseline_s0 --channels 6 --split test --imgsz 512
```

训练后自动测试并写入 `runs_v2/baseline_6ch_s0/eval_test.json`。多种子训练可使用 `--seed 0 1 2`。评估必须经过 `scripts/v2_evaluate.py` 的多通道数据加载器，不能直接调用标准 `model.val()`。独立评估输出 JSON 可增加 `--output` 参数。逐实例阈值诊断不等同于标准 AP 评估；`--imgsz` 必须与训练一致，1024 模型请使用 `--imgsz 1024`。测试集诊断只能用于报告错误，不能据此选择最终阈值或模型。

## 🌊 Project Overview

This project provides tools for preprocessing, analyzing, and preparing the MARIDA dataset for YOLO-based object detection models. It includes support for multispectral band optimization, data augmentation, and comprehensive validation utilities.

## 📋 Features

### Data Processing
- **Band Extraction**: Extract and process Sentinel-2 multispectral bands (RGB, NIR, SWIR)
- **Format Conversion**: Convert GeoTIFF masks to YOLO annotation format
- **Dataset Splitting**: Automated train/val/test split with stratification
- **Data Augmentation**: Advanced augmentation pipeline with albumentations
- **Spectral Indices + Texture**: `preprocess_add_indices_texture.py` adds NDVI/NDWI/FDI and GLCM maps for enriched inputs
- **Image Resizing**: Standardize images to uniform dimensions

### Analysis Tools
- **Multispectral Optimization**: Test 8 different band configurations (3-8 channels)
- **Band Verification**: Validate correct channel loading and data quality
- **Annotation Validation**: Verify YOLO annotations against original masks
- **Dataset Visualization**: Generate sample visualizations for quality control

### Supported Band Configurations
1. **RGB** (3 channels): B04, B03, B02
2. **RGB+NIR** (4 channels): B04, B03, B02, B08
3. **RGB+SWIR** (5 channels): B04, B03, B02, B11, B12
4. **RGB+NIR+SWIR** (6 channels): B04, B03, B02, B08, B11, B12 ⭐ Recommended
5. **RGB+NIR+NDVI** (5 channels): RGB+NIR + NDVI index
6. **RGB+NIR+NDWI** (5 channels): RGB+NIR + NDWI index
7. **RGB+NIR+3 Indices** (7 channels): RGB+NIR + NDVI, NDWI, FDI
8. **All Bands+Indices** (8 channels): All bands + NDVI, NDWI

## 🚀 Quick Start

### Prerequisites (historical preprocessing tools)

```bash
python -m pip install -r requirements.txt
```

Required packages:
- rasterio
- numpy
- opencv-python
- matplotlib
- albumentations
- tqdm
- scikit-image

### Usage (historical preprocessing tools)

These commands point to the existing legacy tools. Check their local paths and input configuration before running; this pipeline does not establish the v2 scene split.

#### 1. Run Complete Preprocessing Pipeline

```bash
python scripts/run_preprocessing_pipeline.py
```

This will:
- Extract multispectral bands
- Convert masks to YOLO format
- Split dataset (train/val/test)
- Apply data augmentation
- Validate outputs

#### 2. Test Multispectral Configurations

```bash
python tests/unit/test_multispectral_optimization.py
```

Generates comparison visualizations for different band configurations.

#### 3. Verify Band Outputs

```bash
python scripts/evaluation/verify_band_outputs.py
```

Validates that all band configurations are loaded correctly.

#### 4. Visualize Dataset

```bash
python scripts/visualization/visualize_dataset.py
```

Generate sample visualizations from the dataset.

### 5. MaskRescue Inference + Threshold Sweeps

These are historical 7-channel tools. GT-derived knowledge-channel input makes the corresponding results oracle results; see the warning below before interpreting or citing them.

- **Batch inference for NPY/TIF patches**

	```bash
	python scripts/utils/infer_mask_rescue.py \
		runs_marida/yolov11n_7ch_sparse_weight8_wake_down_cos2/weights/best.pt \
		data/processed/mask_bbox_rescue_split/val.txt \
		--data configs/datasets/marida_rescue_7ch.yaml \
		--imgsz 512 --device 0 --conf 0.35 --iou 0.5 \
		--project runs_marida --name infer_job \
		--save-img --save-txt --save-conf
	```

	- Accepts manifests, directories, or individual `.npy/.tif` tensors
	- Outputs annotated previews under `runs_marida/infer_job/images/` and YOLO TXT predictions under `.../labels/`
	- Writes aggregate stats to `.../inference_summary.json` for quick QA
	- Use `--max-images N` for quick smoke tests, or `--channel-stats` to override clipping JSON

- **Confidence / IoU threshold sweeps**

	```bash
	python scripts/utils/sweep_inference_thresholds.py \
		runs_marida/yolov11n_7ch_sparse_weight8_wake_down_cos2/weights/best.pt \
		--data configs/datasets/marida_rescue_7ch.yaml \
		--conf-list 0.25,0.3,0.35,0.4 \
		--iou-list 0.45,0.5 \
		--project runs_marida --name threshold_scan
	```

	- Runs `model.val` for every conf/IoU pair using the MaskRescue dataset builder
	- Produces `threshold_scan/threshold_sweep.{json,csv}` summarizing mean metrics and per-class maps
	- Add `--keep-artifacts` to retain Ultralytics plots/PR curves for each sweep point

## 📁 Project Structure

```
Marine-Debris-Detection-main/
├── scripts/preprocessing/           # Band extraction, labels, splits, augmentation
├── scripts/run_preprocessing_pipeline.py  # Historical preprocessing pipeline
├── scripts/evaluation/              # Band and annotation verification
├── scripts/visualization/           # Dataset visualization
├── scripts/v2_baseline_train.py     # Independent v2 six-channel baseline
├── scripts/v2_evaluate.py           # Multi-channel evaluation
├── ultralytics_plugins/             # Multi-channel dataset and trainers
├── configs/                         # Dataset and model configuration
├── tests/unit/                      # Unit tests and historical checks
└── docs/analysis/                   # Dataset analysis and historical reports

```

## 📊 Dataset Information

### MARIDA Dataset
- **Source**: Marine Debris Archive for Deep Learning
- **Sensor**: Sentinel-2 Multispectral Imagery (L2A, surface reflectance)
- **Bands**: 11 bands (B1-B12 excluding B10), 60m/20m/10m original GSD, all resampled by MARIDA authors to a single 10m grid in each patch
- **Patch size**: 256×256 (native, as published by MARIDA) — NOT resampled by us
- **Training `imgsz`**: `imgsz=512` / `imgsz=1024` is letterbox up-sampling inside the YOLO data pipeline, not a higher-resolution sensor input
- **Classes**: 15 raw segmentation classes; we map to either 7-class YOLO (v1, paper main line) or 3-class detection (v2, parked)

### Output Format
- **Images**: GeoTIFF (.tif) or PNG (.png)
- **Annotations**: YOLO format (.txt)
- **Structure**: Standard YOLO directory layout (images/ and labels/)

## 🎯 Recommended Configuration

Based on extensive testing and validation:

**RGB+NIR+SWIR (6 channels)** - Best balance of:
- Information completeness
- Computational efficiency
- Detection accuracy for marine debris

This configuration provides:
- RGB for visible features
- NIR for water/debris separation
- SWIR for material differentiation

## 📈 Validation Results

All 8 band configurations tested with **100% success rate**:
- ✅ Correct channel counts
- ✅ Proper band mapping
- ✅ Valid data ranges
- ✅ No missing or corrupted data

See `multispectral_verification/` for detailed reports.

### v1 7ch Oracle Baseline (Historical Reference)

> **⚠️ Oracle / Privileged-Information Result.** The 7th channel of this
> model is a GT-derived distance-transform heatmap (see
> `scripts/preprocessing/preprocess_generate_knowledge_channel.py`). At
> validation/test time the model still receives that GT-derived channel,
> so the numbers below are an **upper bound under privileged input**, not
> a deployable detection result. With the heatmap zeroed at eval, the
> same checkpoint collapses to mAP50 ≈ 0.06. Quote this as an oracle
> teacher / upper bound only — never as a clean Sentinel-2 baseline.
> Full audit: [docs/paper/7CH_BASELINE_CURRENT_AUDIT_2026-05-18.md](docs/paper/7CH_BASELINE_CURRENT_AUDIT_2026-05-18.md).

| Item | Details |
| --- | --- |
| **Checkpoint** | `experiments/runs/runs_marida/yolov11n_7ch_512_finetune_wakeboost/weights/best.pt` |
| **Checkpoint SHA256** | `74e841801a4d0e7b884751c7dbeb1824013899a0ae754b9dbc47d7667f282468` |
| **Training setup** | imgsz 512, 30 epochs, lr0 5e-4 → cosine decay, mosaic disabled, translate 0.05, scale 0.2, erasing 0.2, `MASKRESCUE_CLASS_BALANCE_POWER=1.3` |
| **Validation pipeline** | `scripts/v1_reproduce_7ch_baseline.py` (pins YAML, channel_stats, loader, CWD, checkpoint SHA) |
| **Publication F1** | Precision 0.826 · Recall 0.787 · F1 0.806 @ IoU 0.5 (best threshold 0.53; major-class only) |
| **`model.val()` on val split** | mAP50 0.524 · mAP50-95 0.265 (reproduced 2026-05-18 within 1% of audit) |

Reproduce:

```bash
python scripts/v1_reproduce_7ch_baseline.py --split val
```

## 📝 Documentation

- [Class Strategy Analysis](docs/analysis/CLASS_STRATEGY_ANALYSIS.md)
- [Multispectral Analysis](docs/analysis/MULTISPECTRAL_ANALYSIS.md)
- [Dataset Analysis](docs/analysis/README_DATASET_ANALYSIS.md)
- [Test Report](docs/analysis/TEST_REPORT.md)
- [公开发布清单](docs/PUBLICATION.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This repository currently has no `LICENSE` file. The rights holders must confirm a license before public release; no MIT permission is implied. Ultralytics is distributed under AGPL-3.0, with other licensing options described in its [official licensing guidance](https://www.ultralytics.com/license). Review the [upstream license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), the project's use of it, and third-party data/model terms before choosing the project license.

## 🙏 Acknowledgments

- MARIDA dataset creators
- Sentinel-2 satellite imagery (ESA Copernicus)
- Albumentations library
- YOLO object detection framework

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: Large data files (MARIDA dataset, generated images) are excluded from the repository via `.gitignore`. Please download the MARIDA dataset separately from the official source.
