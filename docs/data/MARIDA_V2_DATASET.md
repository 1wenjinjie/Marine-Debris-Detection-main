# MARIDA v2 Dataset

## Overview

MARIDA v2 is rebuilt from raw MARIDA (Marine Debris Archive) patches for marine debris detection using Sentinel-2 multispectral imagery with YOLO. It fixes critical quality issues found in v1 and serves as the canonical dataset for all subsequent experiments.

**Build script**: `scripts/preprocessing/build_marida_v2.py`
**Output directory**: `data/processed/marida_v2/`
**Configs**: `configs/datasets/marida_v2_6ch.yaml`, `configs/datasets/marida_v2_7ch.yaml`

## Task Definition (Plan A: 3 Classes)

| YOLO Class | Name | MARIDA Source Classes |
|---|---|---|
| 0 | Marine_Debris | 1 (Marine Debris) |
| 1 | Algae_Organic | 2 (Dense Sargassum), 3 (Sparse Sargassum), 4 (Natural Organic Material) |
| 2 | Ship | 5 (Ship) |

MARIDA classes 6-15 (Clouds, Marine Water, Sediment-Laden Water, Foam, Turbid Water, Shallow Water, Waves, Cloud Shadows, Wakes, Mixed Water) are all treated as background. Foam and Wakes were excluded because of insufficient test data and poor bbox quality at 10m resolution.

## Band Selection (6 Channels)

| Channel | Sentinel-2 Band | Wavelength | Spatial Res |
|---|---|---|---|
| 0 | B4 (Red) | 665 nm | 10 m |
| 1 | B3 (Green) | 560 nm | 10 m |
| 2 | B2 (Blue) | 490 nm | 10 m |
| 3 | B8 (NIR) | 842 nm | 10 m |
| 4 | B11 (SWIR1) | 1610 nm | 20 m |
| 5 | B12 (SWIR2) | 2190 nm | 20 m |

Rasterio 1-indexed band indices: `[4, 3, 2, 8, 10, 11]` from the 11-band MARIDA .tif files.

## 7th Knowledge Channel

For Privileged Knowledge Distillation (PKD) experiments, a 7th channel is generated:

- **Source**: Ground truth class mask (`_cl.tif`)
- **Method**: `cv2.distanceTransform(filtered_binary, cv2.DIST_L2, 5)`, normalized to [0, 1]
- **Consistency guarantee**: Only pixels from connected components >= 4 pixels (same threshold as bbox extraction) contribute to the heatmap. Patches with empty labels have an all-zero 7th channel.
- **Storage**: Separate `images_7ch/` directory with corresponding `*_7ch.txt` manifest files

## Dataset Statistics

| Metric | Value |
|---|---|
| Total patches | 1,381 |
| With objects | 367 (26.6%) |
| Pure background | 1,014 (73.4%) |
| Image size | 256 x 256 x 6 (float32) |
| Source | All 1,381 raw MARIDA patches from 63 scenes, 17 MGRS tiles |

### Split Distribution

| Split | Patches | Ratio | Scenes |
|---|---|---|---|
| Train | 973 | 70.5% | 44 |
| Val | 199 | 14.4% | 10 |
| Test | 209 | 15.1% | 9 |

### Class Instance Distribution

| Class | Total | Train | Val | Test |
|---|---|---|---|---|
| Marine_Debris | 212 | 122 | 77 | 13 |
| Algae_Organic | 273 | 173 | 43 | 57 |
| Ship | 202 | 141 | 26 | 35 |
| **TOTAL** | **687** | **436** | **146** | **105** |

### Known Limitations

- **Marine_Debris test count = 13**: Per-class AP for debris in the test set will have high variance. Report with confidence intervals in the paper. Consider supplementary evaluation at different confidence thresholds.
- **Small object sizes**: Marine debris at Sentinel-2 10m resolution is typically 1-8 pixels. Many components are just 4-6 pixels after filtering. YOLO may struggle with sub-pixel-scale targets.
- **Geographic overlap at tile level**: Scene-level splitting allows different temporal acquisitions of the same MGRS tile to appear in different splits. This is a deliberate tradeoff — strict tile-level splitting leaves Marine_Debris absent from val/test entirely (only 17 tiles, debris concentrated in ~5 tiles).

## Splitting Strategy

### Rationale

v1 had 100% test-train scene overlap (all 26 test scenes appeared in train), causing data leakage. v2 uses **scene-level splitting** where patches from the same satellite acquisition (date + tile) always stay in the same split.

### Algorithm

1. Group all 1,381 patches by scene (63 unique scenes = date + MGRS tile)
2. Initial assignment from MARIDA official split files via majority vote per scene
3. **Stratified rebalancing**: Scenes containing Marine_Debris are distributed across train/val/test using interleaved assignment to ensure all splits contain debris
4. Non-debris scenes fill remaining quota to achieve ~70/15/15 ratio
5. Verification: zero scene overlap between any two splits

### Trade-off: Scene vs Tile Splitting

| Approach | Pros | Cons |
|---|---|---|
| Tile-level | Zero geographic overlap | Only 17 tiles; Marine_Debris absent from val/test |
| Scene-level (chosen) | All classes in all splits; balanced ratios | Same geography may appear at different dates across splits |

Scene-level splitting was chosen because tile-level splitting makes the dataset unusable for Marine_Debris evaluation. The remaining geographic correlation is mitigated by the fact that debris is transient — same location at different dates has different debris patterns.

## Bbox Extraction

- **Method**: Connected component analysis (`scipy.ndimage.label`) on per-class binary masks from `_cl.tif`
- **Minimum component size**: 4 pixels (filters single-pixel noise from atmospheric/sensor artifacts)
- **Bbox padding**: 2 pixels around each component (prevents cutting object edges)
- **Format**: YOLO normalized `class cx cy w h` in `.txt` files
- **Background patches**: Patches with no target-class components >= 4px get an empty `.txt` label file

## Directory Structure

```
data/processed/marida_v2/
  train.txt              # 6ch manifest (train/images/S2_xxx.npy)
  val.txt
  test.txt
  train_7ch.txt          # 7ch manifest (train/images_7ch/S2_xxx.npy)
  val_7ch.txt
  test_7ch.txt
  classes.txt            # Marine_Debris, Algae_Organic, Ship
  build_audit.json       # Full audit log with scene assignments, class counts
  train/
    images/              # 256x256x6 float32 .npy
    images_7ch/          # 256x256x7 float32 .npy (6ch + knowledge heatmap)
    labels/              # YOLO format .txt
  val/
    images/
    images_7ch/
    labels/
  test/
    images/
    images_7ch/
    labels/
```

## Fixes vs v1

| Issue | v1 | v2 |
|---|---|---|
| Scene leakage | 100% test scenes in train | 0% scene overlap |
| Pure background | 0 patches | 1,014 patches (73.4%) |
| Dead classes | Natural_Organic = 0 instances | Merged into Algae_Organic |
| Test coverage | Missing 3 of 7 classes in test | All 3 classes in test |
| Augmented ghosts | 869 broken references | No augmentation (clean baseline) |
| Class count | 7 (extreme imbalance) | 3 (balanced by design) |
| Heatmap consistency | N/A (generated separately) | Heatmap filtered identically to labels |
| Reproducibility | Multi-step manual pipeline | Single script, all rules hardcoded |

## Usage

### Rebuild from scratch

```bash
python scripts/preprocessing/build_marida_v2.py
```

Requires: `rasterio`, `opencv-python`, `scipy`, `numpy`, `tqdm`

### Training configs

- 6ch student baseline: `configs/datasets/marida_v2_6ch.yaml`
- 7ch teacher (PKD): `configs/datasets/marida_v2_7ch.yaml`

### Audit

Full build audit including per-scene split assignments and per-split class counts:
`data/processed/marida_v2/build_audit.json`
