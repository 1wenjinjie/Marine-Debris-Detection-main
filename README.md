# Marine Debris Detection Dataset

A comprehensive toolkit for processing and analyzing the MARIDA (Marine Debris Archive) dataset for deep learning-based marine debris detection using multispectral satellite imagery.

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

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

Required packages:
- rasterio
- numpy
- opencv-python
- matplotlib
- albumentations
- tqdm
- scikit-image

### Usage

#### 1. Run Complete Preprocessing Pipeline

```bash
python run_preprocessing_pipeline.py
```

This will:
- Extract multispectral bands
- Convert masks to YOLO format
- Split dataset (train/val/test)
- Apply data augmentation
- Validate outputs

#### 2. Test Multispectral Configurations

```bash
python test_multispectral_optimization.py
```

Generates comparison visualizations for different band configurations.

#### 3. Verify Band Outputs

```bash
python verify_band_outputs.py
```

Validates that all band configurations are loaded correctly.

#### 4. Visualize Dataset

```bash
python visualize_dataset.py
```

Generate sample visualizations from the dataset.

### 5. MaskRescue Inference + Threshold Sweeps

- **Batch inference for NPY/TIF patches**

	```bash
	python tools/infer_mask_rescue.py \
		runs_marida/yolov11n_7ch_sparse_weight8_wake_down_cos2/weights/best.pt \
		mask_bbox_rescue_split/val.txt \
		--data marida_rescue_7ch.yaml \
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
	python tools/sweep_inference_thresholds.py \
		runs_marida/yolov11n_7ch_sparse_weight8_wake_down_cos2/weights/best.pt \
		--data marida_rescue_7ch.yaml \
		--conf-list 0.25,0.3,0.35,0.4 \
		--iou-list 0.45,0.5 \
		--project runs_marida --name threshold_scan
	```

	- Runs `model.val` for every conf/IoU pair using the MaskRescue dataset builder
	- Produces `threshold_scan/threshold_sweep.{json,csv}` summarizing mean metrics and per-class maps
	- Add `--keep-artifacts` to retain Ultralytics plots/PR curves for each sweep point

## 📁 Project Structure

```
Marine_dataset/
├── preprocess_extract_bands.py      # Extract Sentinel-2 bands
├── preprocess_mask_to_yolo.py       # Convert masks to YOLO format
├── preprocess_split_dataset.py      # Split dataset
├── preprocess_augmentation.py       # Data augmentation
├── preprocess_resize.py             # Image resizing
├── run_preprocessing_pipeline.py    # Complete pipeline
├── test_multispectral_optimization.py  # Band optimization tests
├── verify_band_outputs.py           # Band verification
├── validate_annotations.py          # Annotation validation
├── visualize_dataset.py             # Dataset visualization
├── CLASS_STRATEGY_ANALYSIS.md       # Class strategy analysis
├── MULTISPECTRAL_ANALYSIS.md        # Multispectral analysis report
├── README_DATASET_ANALYSIS.md       # Dataset analysis documentation
└── TEST_REPORT.md                   # Testing report

```

## 📊 Dataset Information

### MARIDA Dataset
- **Source**: Marine Debris Archive for Deep Learning
- **Sensor**: Sentinel-2 Multispectral Imagery
- **Bands**: 11 bands (Blue, Green, Red, NIR, SWIR1, SWIR2, etc.)
- **Resolution**: Various (resampled to 256x256)
- **Classes**: Multiple marine debris types

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

### Current SOTA (Major Classes)

| Item | Details |
| --- | --- |
| **Checkpoint** | `runs_marida/yolov11n_7ch_512_finetune_wakeboost/weights/best.pt` |
| **Training setup** | imgsz 512, 30 epochs, lr0 5e-4 → cosine decay, mosaic disabled, translate 0.05, scale 0.2, erasing 0.2, `MASKRESCUE_CLASS_BALANCE_POWER=1.3` |
| **Validation pipeline** | `tools/run_val_with_outputs.py` + `tools/plot_publication_metrics.py` (MaskRescue dataset builder, val split 235 GT boxes) |
| **Best threshold** | 0.53 |
| **Metrics @ IoU 0.5** | Precision 0.826 · Recall 0.787 · F1 0.806 |

This run (`yolov11n_7ch_512_finetune_wakeboost`) is **Current SOTA (Major Classes)** and should be referenced for publication-ready results.

## 📝 Documentation

- [Class Strategy Analysis](CLASS_STRATEGY_ANALYSIS.md)
- [Multispectral Analysis](MULTISPECTRAL_ANALYSIS.md)
- [Dataset Analysis](README_DATASET_ANALYSIS.md)
- [Test Report](TEST_REPORT.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MARIDA dataset creators
- Sentinel-2 satellite imagery (ESA Copernicus)
- Albumentations library
- YOLO object detection framework

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: Large data files (MARIDA dataset, generated images) are excluded from the repository via `.gitignore`. Please download the MARIDA dataset separately from the official source.
