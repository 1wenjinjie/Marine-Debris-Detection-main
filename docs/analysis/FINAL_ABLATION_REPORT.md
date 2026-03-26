# MARIDA Spectral Band Ablation Study Report (Final)

> Rerun completed 2025-12-01 using the aligned training scripts with channel guardrails.

## Objective
Determine the most effective multispectral stack (3–6 Sentinel-2 bands) for MARIDA marine-debris detection when training YOLO11n under a unified configuration.

## Methodology
- **Model**: YOLO11n (Ultralytics 8.2+) initialized from `yolo11n.pt`
- **Dataset root**: `mask_bbox_rescue_split` (256×256 MARIDA patches, see provenance below)
- **Splits**: `marida_rescue_split_[3|4|5|6]ch.yaml` (train 966 / val 276 / test 139 patches, identical geometry per split)
- **Hyper-parameters**: `epochs=100`, `batch=8`, `imgsz=640`, `workers=4`, `patience=20`, AMP on, cosine disabled
- **Hardware**: NVIDIA RTX 3090 Ti (24 GB), CUDA 12.4
- **Runs**: Executed via `train_{3ch,4ch_nir,5ch,6ch}.py` after applying YAML channel assertions + dataset loader guards.

## Results Summary

Best checkpoints were taken from each run’s `results.csv` (see `marida_runs/yolo11n_ablation_summary.json`).

| Experiment | Bands Used | Channels | mAP50 | mAP50-95 | Best Epoch | Run Directory |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (RGB)** | R, G, B | 3 | 0.373 | 0.217 | 32 | `marida_runs/yolo11n_3ch_rgb` |
| **RGB + NIR** | R, G, B, NIR | 4 | 0.432 | 0.256 | 17 | `marida_runs/yolo11n_4ch_nir` |
| **RGB + NIR + SWIR1** | R, G, B, NIR, SWIR1 | 5 | 0.405 | 0.204 | 8 | `marida_runs/yolo11n_5ch_swir1` |
| **Full Spectrum** | R, G, B, NIR, SWIR1, SWIR2 | 6 | **0.456** | **0.284** | 60 | `marida_runs/yolo11n_6ch_full` |

## Key Findings

1. **Six bands win overall** – The stacked RGB+NIR+SWIR1+SWIR2 input delivers the highest mAP50-95 (0.284), giving a +6.7 pt absolute lift over RGB (0.217) and +2.8 pt over the 4-channel variant.
2. **NIR still matters** – Moving from RGB to RGB+NIR contributes the single largest jump (+0.059 mAP50-95). Even if compute budgets rule out 6 channels, 4 channels remain a low-cost, high-return option.
3. **SWIR1 alone is marginal** – The 5-channel setup nudges recall but introduces enough noise that overall mAP50-95 slips by ~0.052 vs. 4-channel. SWIR2 is needed to recover the benefit.
4. **Training stability** – Guardrails prevented the earlier “3-channel masquerading as 6-channel” issue. Each log now shows matching `channels` metadata and consistent loss curves.

## Provenance & Guardrails

- **Data version**: `mask_bbox_rescue_split` snapshot (this repo). Split stats (from `validation_results.json`): train 966/273 boxes, val 276/118 boxes, test 139/42 boxes. All tensors stored as float32 reflectance scaled ×1500.
- **YAML integrity**: Every `train_*.py` asserts the `channels` field in the referenced YAML before training begins. Failure halts execution with a descriptive message.
- **Loader safeguards**: `MaskRescueNPYDataset` cross-checks the requested channel count against the YAML payload and raises if they diverge; initialization logs the resolved channel stack.
- **Metrics archive**: `marida_runs/yolo11n_ablation_summary.json` captures the best epoch + metrics for every rerun to avoid spreadsheet drift.

## Reproduction

```powershell
cd D:\OceanEdge_Group\Marine-Debris-Detection-main
python train_3ch.py --data marida_rescue_split_3ch.yaml --name yolo11n_3ch_rgb
python train_4ch_nir.py --data marida_rescue_split_4ch_nir.yaml --name yolo11n_4ch_nir
python train_5ch.py --data marida_rescue_split_5ch.yaml --name yolo11n_5ch_swir1
python train_6ch.py --data marida_rescue_split_6ch.yaml --name yolo11n_6ch_full
```

> All runs write under `marida_runs/` and can be summarized by rerunning the inline script that produces `yolo11n_ablation_summary.json`.

## Conclusion
The rerun confirms that **6-channel inputs (RGB + NIR + SWIR1 + SWIR2)** offer the best trade-off for MARIDA debris detection when paired with the latest preprocessing fixes. Four channels remain a strong fallback for lighter deployments, but adding both SWIR bands ultimately yields the most robust detector. Future work should explore larger backbones (e.g., YOLO11s/m) on the 6-channel stack and investigate SWIR-specific augmentations to further exploit the spectrum.
