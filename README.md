# Leakage Audit and Reproducible Evaluation for Sentinel-2 Marine Debris Detection

Official code, evaluation protocols, audit manifests, and figure scripts for quantifying evaluation leakage (label-derived inputs, same-scene sample exposure, and MARIDA–MADOS cross-dataset overlap) and benchmarking multispectral marine debris detection in Sentinel-2 imagery.

## Key Results Summary

All YOLO11n detection models use lossless $D_4$ geometric augmentation (dihedral group of 90-degree rotations and reflections), 6-band Sentinel-2 inputs (`B4, B3, B2, B8, B11, B12`; C6) or 7-band inputs with inverted NDWI (`(1 - NDWI) / 2`; C7), and scene-grouped evaluation with 10,000 paired scene-cluster bootstrap replicates:

- **MARIDA–MADOS cross-dataset overlap (Section 4.1, Table 1)**: 1,013 of 1,381 MARIDA patches (73.4%) originate from the same Sentinel-2 acquisitions as MADOS (`IoU = 1.00` on overlapping Marine Debris masks). Under official splits, 282/728 (38.7%) of MADOS test crops share acquisitions with MARIDA official test scenes; under the 10-scene resplit, 203/728 (27.9%) overlap with MARIDA train/val scenes.
- **9-fold 2 x 2 factorial leakage audit (Section 4.2, Table 2, Fig. 2)**: Label-derived distance-map input inflates pooled OOF AP50 by `+0.187 [0.089, 0.283]`, and ~50% same-scene non-overlapping exposure inflates AP50 by `+0.099 [0.023, 0.159]` with no detectable interaction (`+0.013 [-0.102, 0.117]`), increasing AP50 from `0.176` to `0.462` (2.6x).
- **Leakage-free 9-fold scene CV and baselines (Sections 4.3–4.4, Table 3, Fig. 3)**: Validation-calibrated pooled OOF AP50 on MARIDA (63 scenes, 1,381 patches, 212 MIN4 targets):
  - `C6 (6-band YOLO11n)`: `0.200 [0.109, 0.342]`
  - `C7 (+inv-NDWI YOLO11n)`: `0.146 [0.072, 0.268]`
  - `Pixel-level Random Forest (11 bands + 8 indices)`: `0.064 [0.028, 0.123]` calibrated (`0.084 [0.038, 0.146]` uncalibrated)
  - `FDI threshold baseline (q = 0.90)`: `0.004 [0.002, 0.008]`
- **Zero-shot external tests (Sections 4.3 & 4.5.4)**: Deduplicated MADOS test set (525 crops, 79 targets): C6 `0.118`, C7 `0.166`; PLP2021 (Sen2Cor L2A, 21 overpasses): static MARIDA thresholds transfer poorly (`0/21` HDPE, `4.4%` wood), rising to `23.8%` HDPE and `77.8%` wood at `conf >= 0.001`.

## Repository Structure

```text
.
├── ultralytics_plugins/          # Custom multi-channel .npy dataset (lossless D4), trainer, and validator
├── configs/
│   ├── models/                   # YOLO11n 6-channel and 7-channel architecture YAMLs
│   └── datasets/mpb/             # Dataset YAML definitions (relative paths under data/)
├── scripts/
│   ├── mpb/                      # End-to-end protocol prep, D4 CV/factorial training, external tests, overlap audit
│   ├── figures/                  # Scripts to reproduce Main Figures 1-6, Supplementary Figs. S1-S3, Graphical Abstract
│   ├── training/                 # Shared locked training hyperparameters
│   ├── utils/                    # Multi-channel COCO-mean first-conv weight initialization
│   ├── v2_cv_factorial.py        # 9-fold 2 x 2 factorial dataset builder and analyzer
│   ├── cv_factorial_scheduler.py # Multi-worker locking and task scheduler
│   ├── v2_fdi_baseline.py        # FDI threshold baseline on the 9-fold CV split
│   ├── v2_fdi_sensitivity.py     # FDI quantile grid (0.50-0.9999) and water-mask sensitivity
│   ├── v2_rf_baseline.py         # Pixel-level Random Forest baseline (11 bands + 8 indices)
│   ├── v2_cv_fold_ap.py          # Per-fold AP and provenance checks
│   ├── v2_relaxed_match.py       # Relaxed localization matching (IoU & centroid distance)
│   ├── v2_error_analysis.py      # Unmatched prediction attribution and stratified recall
│   └── v2_evaluate.py            # Multi-channel evaluation entry point (patches Ultralytics loader)
├── artifacts/
│   ├── protocol/                 # Pre-specified factorial analysis plan + SHA-256, 9-fold & 10-scene split manifests
│   ├── overlap_audit/            # MARIDA-MADOS duplicate map, split crosstab, mask agreement, protocol survey
│   └── results/                  # Summary tables and JSON ledgers for Tables 1-3 and Figs. 2-4
├── tests/unit/                   # Unit tests for lossless D4 transforms and factorial scheduler
├── requirements.txt
└── README.md
```

## Environment Setup

Experiments were executed with **Python 3.11**, **PyTorch 2.6.0 (CUDA 12.4)**, and **Ultralytics 8.4.69**.

1. Install PyTorch matching your CUDA toolkit (see [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch==2.6.0 torchvision==0.21.0
```

2. Install Ultralytics and remaining dependencies:

```bash
pip install ultralytics==8.4.69
pip install -r requirements.txt
```

3. Verify installation with unit tests:

```bash
python -m unittest discover -s tests/unit -v
```

> **Important evaluation note**: Standard `YOLO.val()` expects 3-channel uint8 RGB images and silently misreads 6/7-channel float32 `.npy` tensors. Always evaluate via `scripts/v2_evaluate.py` or the scripts in `scripts/mpb/`.

## Public Datasets

Raw satellite imagery and trained model weights are not redistributed in this repository. Download the public datasets from their official archives and place them under `data/raw/` (or `data/MADOS/`):

| Dataset | Official Archive | Expected Local Path |
|---|---|---|
| **MARIDA** (Kikaki et al., 2022) | Zenodo: https://doi.org/10.5281/zenodo.5151941 | `data/raw/MARIDA/` (`patches/`, `splits/`) |
| **MADOS** (Kikaki et al., 2024) | Zenodo: https://doi.org/10.5281/zenodo.10664073 | `data/MADOS/` (`Scene_*/`, `splits/`) |
| **PLP2021** (Papageorgiou et al., 2022) | Zenodo: https://doi.org/10.5281/zenodo.7085112 | Downloaded/materialized via `scripts/mpb/materialize_plp2021_native256.py` |

Download the pretrained COCO `yolo11n.pt` checkpoint from Ultralytics into the repository root before running training (6-channel and 7-channel first-convolution weights are initialized automatically via `scripts/utils/prepare_multichannel_yolo_weights.py`).

## Reproduction Workflow

Run all commands from the repository root. On Windows with Intel MKL, set `KMP_DUPLICATE_LIB_OK=TRUE` if required.

### 1. MARIDA–MADOS Cross-Dataset Overlap Audit (Section 4.1, Table 1)

```bash
python scripts/mpb/mados_overlap_t6.py --phase all
python scripts/mpb/mados_overlap_t1.py
python scripts/mpb/mados_overlap_t2.py
python scripts/mpb/mados_overlap_t3.py
```

Pre-computed duplicate maps, split cross-tabulations, and mask agreement tables are archived in `artifacts/overlap_audit/`.

### 2. Locked Protocol Preparation and 9-Fold Split Construction (Sections 3.1–3.5)

```bash
# Build 10-scene split variants (C6, C7, C7-shuffled, MIN1/2/4/8)
python scripts/mpb/prepare_protocol.py --stage audit
python scripts/mpb/prepare_protocol.py --stage build

# Build 9-fold scene-grouped CV datasets
python scripts/mpb/cv_pipeline.py --pilot

# Build 9-fold 2 x 2 factorial datasets (matches artifacts/protocol/analysis_plan.md)
python scripts/v2_cv_factorial.py --design
```

Verify the SHA-256 lock of the pre-specified factorial analysis plan:

```bash
sha256sum -c artifacts/protocol/analysis_plan.sha256
```

### 3. Main Model Training (Lossless $D_4$ Augmentation)

```bash
# 9-fold scene CV: C6 & C7 x 3 seeds (54 runs -> runs_cv_d4/)
python scripts/mpb/run_cv_d4.py --device 0 --jobs 6

# Stage 2A: 9-fold 2 x 2 factorial audit (108 runs -> runs_cv_factorial_d4/)
# Stage 2B: 10-scene split + MIN area sensitivity (27 runs -> runs_mpb_d4/)
python scripts/mpb/run_phase2_d4.py --stage all --device 0 --jobs 6
```

### 4. Traditional Baselines (FDI and Pixel-Level Random Forest)

```bash
# FDI threshold baseline + quantile/water-mask sensitivity
python scripts/v2_fdi_baseline.py --raw-dir data/raw/MARIDA/patches --ref-json runs_cv_d4/cv_stat_results.json
python scripts/v2_fdi_sensitivity.py --raw-dir data/raw/MARIDA/patches --ref-json runs_cv_d4/cv_stat_results.json

# Pixel-level Random Forest (11 bands + 8 spectral indices, 3 seeds)
python scripts/v2_rf_baseline.py --raw-dir data/raw/MARIDA/patches --ref-json runs_cv_d4/cv_stat_results.json
```

### 5. Post-Training Diagnostics and External Validation (Sections 4.3–4.5)

```bash
# Per-fold AP, relaxed localization, unmatched-prediction attribution, and strict val-only calibration
python scripts/mpb/run_phase3_cv_d4.py

# Materialize external datasets (first time only)
python scripts/mpb/materialize_mados_external.py
python scripts/mpb/materialize_plp2021_native256.py

# Zero-shot evaluation on MADOS (official & deduplicated) and PLP2021
python scripts/mpb/run_phase3_external_d4.py --device 0
```

### 6. Reproducing Figures (Figs. 1–6, Figs. S1–S3, Graphical Abstract)

```bash
python scripts/figures/run_all_figures.py --out-dir figures_out --png-dpi 600
```

Figures 2–4 can be generated directly from the tabular ledgers in `artifacts/results/`. Figures 1, 5, 6, and S1–S3 require raw MARIDA/MADOS GeoTIFFs (extracted automatically via `scripts/figures/extract_real_assets.py`).

## Pre-Computed Audit Artifacts

All lightweight CSV/JSON manifests and summary tables supporting the manuscript are provided in `artifacts/` for direct inspection without retraining:

- `artifacts/protocol/`: `analysis_plan.md` & `analysis_plan.sha256`, `cv9_fold_assignment.csv`, `split10_manifest.json`, `factorial_design.json`, `marine_debris_components.csv`.
- `artifacts/overlap_audit/`: `dup_map_full.csv`, `dup_crosstab.csv`, `label_agreement_summary.csv`, `t6_summary.json`, `published_protocols.csv`.
- `artifacts/results/`:
  - `factorial_2x2_d4/`: `summary.json`, `per_fold_per_seed.csv` (Table 2, Fig. 2)
  - `cv9_d4/`: `cv_stat_results.json`, `fold_ap.csv`, `val_calibration_summary.json`, `rf_val_calibrated.json`, `relaxed_ap.csv`, `error_analysis/`, `fig45/` (Table 3, Figs. 3–4)
  - `baselines/`: `fdi_summary.json`, `fdi_fold_results.csv`, `fdi_sensitivity_summary.json`, `rf_summary.json`, `rf_fold_results.csv`
  - `split10_and_external_d4/`: `split10_summary.json`, `mados_summary.json`, `mados_bootstrap_ci.json`, `plp2021_summary.json`, `plp2021_targets.csv`

## Third-Party Software and Licensing Notice

This codebase builds on [Ultralytics](https://github.com/ultralytics/ultralytics) (distributed under AGPL-3.0) and uses public datasets MARIDA, MADOS, and PLP2021 under their respective Zenodo licenses. Please review upstream license terms before redistribution or commercial use.
