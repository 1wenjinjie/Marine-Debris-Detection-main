# Marine Debris Detection on MARIDA v2 — Paper Draft (skeleton)

> **⚠️ 路径状态 (2026-05-20)：本稿基于 v2 3-class，已搁置。**
>
> 论文主线已切换到 **v1 7ch physidx on v2split** (honest setting，scene-clean，NDWI-inverse 物理先验，3-seed × 5-architecture × {imgsz=512, 1024} 完整 ablation 矩阵，BiFPN+imgsz=1024 = SOTA 已确认)。新主线证据汇总见 `docs/paper/7CH_BASELINE_CURRENT_AUDIT_2026-05-18.md` 和 `docs/paper/进度汇报_2026-05-19.md`。本稿暂保留作为：(a) v2 数据 confirmation 候选 (跑 1-2 个 BiFPN+1024 cell 复现主线结论)；(b) calibration / focal loss 讨论的素材来源。新论文稿建立后本文件移到 `archive/`。
>
> **呈现规范统一（v1 + v2 都遵守）：**
> - 主指标：**Marine_Debris AP50 / F1** (hard class，故事核心)
> - 次指标：mAP50 (all)，加注 "Marine_Debris dominates"
> - per-class 图表只画 / 列有 GT 的类；v1 test 0 GT 类 (Dense_Sarg / Sparse_Sarg / Wakes) 不报 AP
> - "imgsz=1024 helps" 必须描述为 model capacity / P3 head 设计区间命中，**不可写作"更高 sensor resolution"**

**Working title (v2 stub):** Confidence calibration is the bottleneck for rare
floating-object detection: a 3-seed study on MARIDA v2

**Last updated:** 2026-04-30 (frozen pending v1 paper draft)
**Status:** Skeleton with all results plugged in. Sections 1-3 are
short prose stubs to be expanded. Sections 4-6 are populated with the
final tables and figures (locations referenced below).

---

## 1. Introduction (stub)

Marine debris detection from Sentinel-2 multispectral imagery is a
data-scarce, scale-imbalanced task: floating debris instances are
~12-14 px wide and represent <2% of the labeled pixels. Existing
benchmarks (e.g. original MARIDA splits) were susceptible to scene-level
leakage. We work on **MARIDA v2**, a leak-free re-split with strict
scene separation, and study what design choices actually move
practical Marine_Debris recovery rather than just headline mAP.

Contributions (one sentence each, expand later):

- A 3-seed audit of YOLO-family structural modifications (CBAM,
  Bi-FPN, EIoU) on MARIDA v2: none provides a stable improvement over
  baseline once seed variability is accounted for.
- Identification of **confidence calibration**, not localization, as
  the operational bottleneck. The detector finds debris (IoU > 0.9)
  but assigns scores far below any usable operating threshold.
- A simple combination — focal loss (γ=2.0) + imgsz=1024 + post-hoc
  Platt scaling — that stably recovers `5.7 ± 0.6 / 13` debris instances
  at `conf=0.05` and `5.0 ± 0.0 / 13` at the more usable `conf=0.1`,
  versus baseline `1 / 13` and `0 / 13` respectively.
- Spectral-information ablation (3-channel RGB vs 6-channel
  RGB+NIR+SWIR) showing NIR/SWIR are necessary, not optional, for
  open-water debris recovery.

---

## 2. Related Work (stub)

- MARIDA dataset and prior segmentation/detection work
- YOLO architecture variants (v8 → v11)
- Attention modules in detection: CBAM, ECA, SE
- Feature pyramid variants: PANet, BiFPN
- Box regression losses: CIoU, GIoU, EIoU
- Confidence calibration: Platt scaling, temperature scaling, ECE
- Marine-remote-sensing spectral indices (FDI, NDWI)

---

## 3. Methodology (stub)

### 3.1 Dataset: MARIDA v2

Leak-free scene-level split. 1381 tiles total (256×256, 6 bands,
float32). 3 classes: Marine_Debris (cls=0), Algae_Organic (cls=1),
Ship (cls=2). Train/val/test counts and per-class instance counts in
**Figure 9** and `runs_v2/figures/fig9_summary.json`.

### 3.2 Channel selection and preprocessing

Six Sentinel-2 bands: B4, B3, B2 (RGB); B8 (NIR); B11, B12 (SWIR).
Per-channel `p01/p99` percentile normalization computed on the train
split (`channel_stats_6ch.json`). No HSV / mosaic / mixup
(spectral-unsafe). Geometric augmentation only: rotation 180°,
horizontal/vertical flip 50%.

### 3.3 Detector

Backbone: YOLOv11n (Ultralytics). Three detection heads at strides
8/16/32. We modify the input convolution to accept 6 channels;
otherwise the architecture is unchanged in our final pipeline. Custom
plugin (`ultralytics_plugins/`) handles 6-channel float32 `.npy` input
that the standard YOLO loader does not support.

### 3.4 Final pipeline — focal loss + 1024 + Platt scaling

See **Figure 1**. The final pipeline is the baseline backbone+neck+head,
trained with `imgsz=1024`, focal loss (γ=2.0, α=0.25) replacing BCE on
the classification head, followed by post-hoc Platt scaling
(`p_cal = σ(a · logit(p_raw) + b)`) fit on val-set TP/FP labels.

### 3.5 Structure ablation modules (rejected)

For comparison and per the original benchmark protocol we also
evaluated three architectural changes individually and combined.

- **CBAM**: channel + spatial attention applied after each backbone
  C3k2 block (P2 / P3 / P4 / P5).
- **BiFPN-lite**: weighted concat replacing standard concat at each
  fusion node in the head.
- **EIoU**: replaces CIoU in the box regression branch.
- **Full**: all three combined.

All four were trained at imgsz=512 with BCE classification loss,
matching the baseline. Implementation details and YAML files in
`configs/models/`.

### 3.6 Evaluation protocol

Two complementary measurements:

1. **Standard detector metrics** at `conf=0.001`, IoU=0.6 NMS — mAP@0.5,
   mAP@0.5:0.95, per-class P / R / AP50 (`scripts/v2_evaluate.py`).
2. **Marine_Debris threshold sweep** — for each of 13 debris GT
   instances on the test split, the maximum-IoU prediction's
   confidence; counted as TP if `max_iou > 0.5` AND
   `best_conf >= conf_thr`. Reported at
   `conf ∈ {0.001, 0.005, 0.01, 0.025, 0.05, 0.1}`
   (`scripts/v2_sweep_thresholds.py`).

We argue (Section 5) that the threshold-sweep metric reflects
operational utility better than mAP for rare-class detection.

---

## 4. Results

### 4.1 Structure ablation (Table 1, Figure 5a)

**Source:** `runs_v2/structure_ablation_table1.md`,
`runs_v2/figures/fig5_structure_ablation_bars.png`

> Insert Table 1 here.

Headline observations:

- All four structural variants either match or under-perform baseline
  on mAP@0.5 with substantial seed variance.
- CBAM appears to lift Marine_Debris AP50 from `0.016 ± 0.008` to
  `0.037 ± 0.025`; the std is comparable to the apparent gain. Per-seed:
  `s0 = 0.066`, `s1 = 0.024`, `s2 = 0.020`. We do not claim a
  significant improvement.
- EIoU is the only variant with stable mAP@0.5 gain
  (`0.216 ± 0.010`), driven by Ship (`+0.030`) and Algae (`+0.027`).
  Marine_Debris recovery does not improve.
- Full (CBAM+BiFPN+EIoU) collapses to mAP@0.5 = `0.161 ± 0.020`,
  worse than each component, suggesting combined-capacity
  over-saturation.

### 4.2 Engineering pipeline (Table 2, Figure 5b)

**Source:** `runs_v2/engineering_ablation_table2.md`,
`runs_v2/figures/fig5_engineering_pipeline_bars.png`

> Insert Table 2 here.

Key contrast: Marine_Debris TP at `conf=0.05` evolves as

```
baseline (BCE, 512)        → 1.0
+ imgsz=1024 (BCE)         → 0.0 ± 0.0    (regression!)
+ focal γ=2.0 (i1024)      → 5.7 ± 0.6
+ Platt calibration        → 6.0 ± 0.0
```

Note `imgsz=1024` alone is *worse* than baseline at the practical
threshold. This is reproduced across all 3 seeds and is the central
empirical anchor of our argument that focal loss is the necessary
ingredient.

At `conf=0.1`:

```
baseline                    → 0
+ imgsz=1024                → 0
+ focal γ=2.0 (i1024)       → 0.7 ± 0.6
+ Platt calibration         → 5.0 ± 0.0
```

Platt scaling makes `conf=0.1` a usable deployment threshold, not just
a closer-to-zero number.

### 4.3 Spectral-channel ablation (3ch vs 6ch)

**Source:** `runs_v2/ablation_3ch_focal_g2.0_i1024_s0/eval_test.json`,
`runs_v2/sweep_ablation_3ch_focal_g2.0_i1024_s0/sweep_results.json`

| Configuration | M.Debris AP50 | Algae AP50 | Ship AP50 | Debris TP @ conf=0.001 |
|---|---:|---:|---:|---:|
| 3ch (RGB only) | 0.021 | 0.024 | 0.363 | 2 / 13 |
| 6ch (RGB+NIR+SWIR) | 0.025 | 0.078 | 0.299 | 7 / 13 |

The decisive scene is `S2_7-3-20_18QYG` (5 GT instances): 3ch
recovers `0 / 5`, 6ch recovers `4 / 5`. Removing NIR and SWIR
collapses debris detection in the open-water case. We interpret this
as direct evidence that spectral information is necessary, not
incidental, for floating-debris recovery.

### 4.4 Scene-level recovery breakdown (Table 3)

**Source:** `runs_v2/scene_breakdown_table3.md`,
`runs_v2/scene_breakdown_summary.json`

> Insert Table 3 here.

Observations:

- `S2_7-3-20_18QYG` (5 GT) is consistently recoverable: 4/5 across all
  3 seeds at `conf=0.025-0.05`.
- `S2_23-9-20_16PCC` (2 GT) is fully recoverable at `conf=0.025-0.05`.
- `S2_29-8-17_51RVQ` (1 GT) is seed-dependent (`0.7 ± 0.6`).
- `S2_20-4-18_30VWH` (5 GT) is **never recovered** by either model.
  This is the residual-failure scene; future work should look at what
  is spectrally distinct about it.

### 4.5 Speed-quality trade-off (Figure 6)

**Source:** `runs_v2/figures/fig6_fps_map_tradeoff.png`

The structure-ablation variants pay 5-25% FPS for no mAP gain at the
same image size. The engineering pipeline (focal+1024) trades down
mAP@0.5 (because Ship class regresses) but is the only configuration
that produces non-trivial Marine_Debris TP@0.05 / 0.1.

### 4.6 Qualitative comparison (Figure 8)

**Source:** `runs_v2/figures/fig8_detection_overlay.png`

Side-by-side baseline vs proposed predictions on six representative
debris tiles. Red boxes are GT; blue dashed boxes are predictions
with alpha proportional to confidence. The proposed model produces
visibly higher-confidence boxes on debris regions while the baseline
either misses or assigns very low (near-zero) confidence.

### 4.7 Spatial distribution (Figure 9)

**Source:** `runs_v2/figures/fig9_overview.png`,
`runs_v2/figures/fig9_per_class.png`,
`runs_v2/figures/fig9_summary.json`

Tile-center distribution by class and split. MARIDA v2 sampling is
concentrated in three regions: SE Asia / Indonesia (~110-130°E,
0-20°N), Caribbean (~-90°W, 15°N), and the Mediterranean / North Sea.
Marine_Debris is the rarest class (80 tiles, 212 instances) and is
absent from the open-Atlantic and South-Pacific samples.

### 4.8 Training dynamics (Figures 3 / 4)

**Source:** `runs_v2/figures/fig3_training_curves_structure.png`,
`runs_v2/figures/fig3_training_curves_engineering.png`,
`runs_v2/figures/fig4_training_curves_combined.png`

Training-curve dynamics confirm that baseline and `imgsz=1024` reach
roughly the same val mAP@0.5 (~0.21-0.24 by epoch 50). The focal+1024
run plateaus lower (~0.13) in val mAP because it has been re-balanced
toward the rare class.

---

## 5. Discussion

> Full text in [`DISCUSSION_DRAFT.md`](./DISCUSSION_DRAFT.md). Keep
> the full draft separate so authoring revisions don't mix with the
> figure-caption pass.

Section overview:

- 5.1 Why structural attention/fusion modules failed to stabilize
- 5.2 Confidence calibration is the actual bottleneck
- 5.3 The mAP-vs-utility paradox
- 5.4 Multispectral information is necessary, not optional
- 5.5 Limitations (test set scale, scene-level splits, no domain
       transfer, fixed train resolution)
- 5.6 What we did *not* find (PKD, oversampling, P2 head — all
       abandoned, kept for record)
- 5.7 Summary

---

## 6. Conclusion (stub)

We do not propose a new attention module. We argue that for rare
small-object detection on multispectral satellite imagery, the
unglamorous combination of (i) increased input resolution,
(ii) focal-loss training, and (iii) post-hoc score calibration
produces stable, deployable Marine_Debris recovery, while several
popular architectural improvements (CBAM, BiFPN-lite, EIoU and their
combination) do not. We hope this clarifies the bar for future
detection methods on similar small-rare benchmarks.

---

## Appendix A. Reproducibility checklist

- Data: MARIDA v2 split files at
  `data/processed/marida_v2/{train,val,test}.txt`.
- Original Sentinel-2 GeoTIFFs:
  `data/raw/MARIDA/patches/<scene>/<tile>.tif` (preserved on disk;
  index at `data/processed/marida_v2/tile_geo_index.json`).
- Hyperparameters: `scripts/training/mask_rescue_hyperparams.py`.
- Final-model training: `scripts/v2_focal_loss_train.py` with
  `gamma=2.0, imgsz=1024`. 3-seed protocol: `--seed 0 1 2`.
- Calibration: `scripts/v2_calibrate.py` reads val-set TP/FP and fits
  Platt parameters. Output: `runs_v2/calib_*/calibration_results.json`.
- Evaluation: `scripts/v2_evaluate.py` (standard detector metrics).
- Threshold sweep: `scripts/v2_sweep_thresholds.py` (debris-specific).
- FPS benchmark: `scripts/v2_fps_benchmark.py --batch-runs structure_ablation`
  / `--batch-runs engineering`.
- Table aggregation: `scripts/v2_structure_ablation_summary.py`,
  `scripts/v2_engineering_ablation_summary.py`,
  `scripts/v2_make_table3_scene_breakdown.py`.
- Figure scripts: `scripts/v2_make_fig{1,3,5,6,8,9}_*.py`.

All seed=0/1/2 runs are deterministic given seed and hardware.

## Appendix B. List of figures

| Number | Figure | Path |
|---|---|---|
| 1 | Pipeline overview | `runs_v2/figures/fig1_pipeline.png` |
| 3a | Training curves (structure) | `runs_v2/figures/fig3_training_curves_structure.png` |
| 3b | Training curves (engineering) | `runs_v2/figures/fig3_training_curves_engineering.png` |
| 4 | mAP@0.5 vs epoch (combined) | `runs_v2/figures/fig4_training_curves_combined.png` |
| 5a | Structure ablation bars | `runs_v2/figures/fig5_structure_ablation_bars.png` |
| 5b | Engineering pipeline bars | `runs_v2/figures/fig5_engineering_pipeline_bars.png` |
| 6 | FPS-mAP trade-off | `runs_v2/figures/fig6_fps_map_tradeoff.png` |
| 8 | Detection overlay (qualitative) | `runs_v2/figures/fig8_detection_overlay.png` |
| 9 | Spatial distribution (overview) | `runs_v2/figures/fig9_overview.png` |
| 9 | Spatial distribution (per-class) | `runs_v2/figures/fig9_per_class.png` |

Not yet generated (lower priority / requires external data):

| Number | Figure | Reason |
|---|---|---|
| 2 | CBAM/BiFPN module diagrams | Final model has no attention modules; would only support the rejected ablation |
| 7 | Grad-CAM feature heatmaps | Lower priority; possible quick add later |
| 10 | Temporal variation by month | Requires per-scene timestamps + larger temporal sample |
| 11 | CMEMS current overlay | Requires Copernicus Marine data download; descriptive only — paper must not claim correlation significance (see DISCUSSION 5.5) |

## Appendix C. List of tables

| Number | Table | Path |
|---|---|---|
| 1 | Structure ablation (3-seed) | `runs_v2/structure_ablation_table1.md` |
| 2 | Engineering pipeline (3-seed) | `runs_v2/engineering_ablation_table2.md` |
| 3 | Scene-level recovery | `runs_v2/scene_breakdown_table3.md` |

Not yet generated:

| Number | Table | Status |
|---|---|---|
| 4 | Environmental factor descriptive stats | Requires CMEMS pull. **Paper must not write "correlation is significant"** -- only descriptive on a handful of case scenes |
