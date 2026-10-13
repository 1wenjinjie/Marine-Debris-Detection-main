# MARIDA v2.1 Dataset Specification (LOCKED 2026-05-XX)

This is the authoritative specification of the v2.1 dataset and
experimental protocol. All scripts under `scripts/v2_1/` must conform.
Deviations require updating this document first.

---

## 1. Data Source (unchanged)

- **Source**: MARIDA v1 (Kikaki et al., 2022), Zenodo
- **Path**: `data/raw/MARIDA/patches/` (preserved, never modified)
- **Raw format**: per-tile ACOLITE Rayleigh-corrected reflectance from
  Sentinel-2 L1C, 11 spectral bands, 256×256, float32, value range
  ~[0.005, 0.5].
- **Per-tile companions**: `_cl.tif` (15-class pixel mask, values 0..15
  where 0 = background/unlabeled), `_conf.tif` (4-level confidence,
  values 0/1/2/3 where 0 = unlabeled, 1 = High, 2 = Moderate, 3 = Low;
  values 0 and `cl == 0` always co-occur).
- **Scene inventory**: 63 scenes claimed by MARIDA paper; **59 usable**
  after filtering empty/corrupted scenes. Paper must state both numbers.

## 2. Atmospheric Correction Description (FIXED)

All prose, captions, and discussion **must** describe the product as:

> ACOLITE Rayleigh-corrected reflectance from Sentinel-2 L1C, as
> released in MARIDA (Kikaki et al., 2022).

**Forbidden phrases**: "Sentinel-2 L2A", "Sen2Cor-processed",
"surface reflectance".

## 3. Channel Selection — 9 channels (FIXED)

Six spectral bands + three computed indices. **Not** a full RF SS+SI
reproduction; we describe it as "selected spectral bands + indices".

| Index | Source | Description |
|---:|---|---|
| 0 | TIF band 4 | B4 Red (665 nm) |
| 1 | TIF band 3 | B3 Green (560 nm) |
| 2 | TIF band 2 | B2 Blue (490 nm) |
| 3 | TIF band 6 | **B6 Red Edge 2 (740 nm)** — required for FDI |
| 4 | TIF band 8 | B8 NIR (842 nm) |
| 5 | TIF band 10 | B11 SWIR-1 (1610 nm) |
| 6 | TIF band 11 | B12 SWIR-2 (2190 nm) |
| 7 | computed | **FDI** = R_B8 − R'_B8 ; R'_B8 = R_B6 + (R_B11 − R_B6) · (λ_B8 − λ_B6) / (λ_B11 − λ_B6) · 10 |
| 8 | computed | **NDVI** = (B8 − B4) / (B8 + B4 + ε) |
| 9 (optional) | computed | **NDWI** = (B3 − B8) / (B3 + B8 + ε) |

For initial Phase 1, **fix at 9 channels** (B4/B3/B2/B6/B8/B11/B12 +
FDI + NDVI). NDWI added later if ablation justifies.

**ε = 1e-6** to avoid divide-by-zero.

Note: the v2.0 selection (`[4, 3, 2, 8, 10, 11]` 1-indexed) corresponds
to B4/B3/B2/B8/B11/B12 (6ch, no B6, no indices). v2.1 explicitly adds
**B6 + 3 indices**.

## 4. valid_mask (FIXED — loss-domain, three-layer ignore)

NaN regions of the raw TIF are tracked as a per-tile boolean
`valid_mask` of shape (H, W), stored separately in `valid_masks/`.

**Implementation**: as a **loss-weight mask** (not an extra input
channel). Custom `WaterAwareSegLoss` applies **three-layer** ignore:

1. **GT ignore**: any GT instance whose bbox center falls in an
   invalid (NaN) region is dropped from loss computation entirely
   (does not produce box, cls, or mask gradient).
2. **Prediction ignore**: any model prediction whose bbox center
   falls in an invalid region is excluded from FP/FN counting and
   from loss matching (does not generate spurious objectness/cls
   gradient).
3. **Mask-pixel ignore**: within the surviving instance mask losses
   from (1)+(2), the per-pixel BCE/Dice loss is multiplied by
   `valid_weight ∈ {0, 1}` so NaN pixels contribute zero gradient.

Without (1)+(2), the model is implicitly punished/rewarded around NaN
edges in box and cls losses. Pixel-level masking alone is insufficient.

- `nc = 9` (NOT 10). Model input remains 9-channel.
- ultralytics' default seg loss does NOT support layers (1)+(2);
  custom `WaterAwareSegLoss` in `ultralytics_plugins_v2_1/` is required.
- valid_mask is **streamed alongside images** in the dataloader (as
  an extra dict key `batch["valid_mask"]`), NOT stacked as a 10th
  input channel. This keeps the model's input dimensionality the same
  as inference time and keeps the mask out of conv weights entirely.

## 5. Normalization Statistics (FIXED)

- **Pixel pool**: train split ∩ water-related pixels only. Water-related
  = MARIDA class IDs ∈ {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 15} (debris,
  algae, organic, ship, water variants, foam, mixed water, etc., but
  **excluding** clouds 6, cloud_shadows 13, wakes 14).
- **Statistics**: per-channel p01 and p99.
- **Output file**: `channel_stats_9ch_train_water.json`
  (filename encodes the constraint).
- **No test/val pixels** participate in computation (no leakage).
- For computed indices (FDI/NDVI/NDWI), p01/p99 are computed in the
  same way after index computation.

## 6. Annotation Confidence (FIXED — three label sets)

We generate **three parallel label sets** from the same raw `_cl.tif` +
`_conf.tif`:

| Set | Filter rule | Used in |
|---|---|---|
| `highonly` | `(cl > 0) & (conf == 1)` | Appendix ablation |
| `highmod`  | `(cl > 0) & (conf ∈ {1, 2})` | **★ Main experiment** |
| `all`      | `(cl > 0) & (conf ∈ {1, 2, 3})` | Appendix ablation |

The three label sets are stored side-by-side (`labels_highonly/`,
`labels_highmod/`, `labels_all/`). They share images, valid_masks, and
channel stats. **This three-way comparison is itself a paper
contribution** ("first systematic confidence-aware ablation on MARIDA").

## 7. MIN_COMPONENT_PIXELS (FIXED = 1, confirmed by Phase 0 sanity)

- **Final value**: `MIN_COMPONENT_PIXELS = 1` (no filtering)
- **Train-time augmentation**: dilate 3×3 to mitigate the
  IoU=1/9-ceiling problem for 1-px targets.
- **Eval**: report original GT (no dilation) — separate **tiny-bin**
  (≤3 px), **small** (4-15 px), **large** (>15 px) tracks.

### Phase 0 sanity check evidence (2026-05-XX, n=30 blind judgement)

| True conf level | n | real | fake | unclear | real rate |
|---|---:|---:|---:|---:|---:|
| **High** (conf=1) | 10 | **9** | 0 | 1 | **90%** |
| Moderate (conf=2) | 10 | 6 | 1 | 3 | 60% |
| Low (conf=3) | 10 | 8 | 0 | 2 | 80% |
| Total | 30 | 23 | 1 | 6 | 77% |

**Decision**: High-conf real rate 90% (≥ 70% threshold) → confirm
`MIN_COMPONENT_PIXELS = 1`.

**Side findings to document in paper**:
1. ~521 1-px Marine_Debris instances exist in MARIDA raw; ~400 of
   them carry real signal (77% × 521). The legacy v2.0 setting
   `MIN=4` discarded this entire pool.
2. Confidence-aware filtering is justified: Moderate has 1 confirmed
   false-positive label (10% of n=10), validating the `highonly`
   ablation track.
3. Low-confidence labels are not pure noise (80% real on n=10) —
   the `labels_all` ablation is meaningful, not just a negative
   reference.

Sanity check artifacts: `runs_v2_1/sanity/min1_panels/` (PNG, CSV,
keymap).

## 8. Bounding-Box Padding

- **No automatic padding** in v2.1. The bounding box of a connected
  component is exactly the axis-aligned tight box.
- For instance segmentation, the polygon (not the box) is the
  primary label. The box is derived for compatibility only.

## 9. Split Algorithm (FIXED — three-tier scene-group allocation, v2.1.3)

The naïve scene-group + 2D round-robin (v2.1.0) led to test top-1
share = 70%, because MARIDA's debris-box-per-scene distribution is
**power-law right-skewed** (4 mega scenes account for ~45% of all
debris). After two iterations (v2.1.1, v2.1.2) of mega-filtering +
constraint enforcement that left test at n=3 scenes / 29 boxes,
v2.1.3 adopts an explicit three-tier allocation:

**Tier 1 — mega scenes (debris_count > 50)**:
all forced into train. Empirically 5 scenes in MARIDA.

**Tier 2 — mid scenes (15 < debris_count ≤ 50)**:
sort by debris_count descending, then **alternate val / test** in
that order. This guarantees val and test each get a balanced sample
of medium and large mid-size scenes. Empirically 18 mid scenes →
9 val + 9 test.

**Tier 3 — small scenes (debris_count ≤ 15)**:
shuffled with `SPLIT_SEED = 42`, then standard 5:1:1 round-robin
into train / val / test. Empirically 25 small scenes → 18/3/4
roughly.

**Background-only scenes** (no debris GT): round-robin 5:1:1 over a
shuffled list (15 scenes).

### Empirical outcome (LOCKED 2026-05-16, v2.1.3 build)

| Split | Tiles | Total scenes | Debris scenes | Debris boxes | Top-1 share |
|---|---:|---:|---:|---:|---:|
| train | 740 | 35 | 24 | **608** | 27.3% |
| val | 369 | 14 | 12 | **247** | 16.2% |
| test | 272 | 14 | 12 | **229** | 15.7% |

Compared with v2.0 (test = 13 boxes / 4 scenes / top-1 ≈ 38%):
- **test box count × 18**
- **test scene diversity × 3**
- **test top-1 share halved**

**Reproducibility**: split seed is hardcoded `SPLIT_SEED = 42` in
`build_v2_1.py`; the full mapping is recorded in `split_manifest.json`.

## 10. Output Layout (FIXED)

```
data/processed/marida_v2_1/
├── train/
│   ├── images/                            # <tile>.npy  (H, W, 9) float32
│   ├── valid_masks/                       # <tile>.npy  (H, W) bool
│   ├── instance_masks_highonly/           # <tile>.npy  (H, W) int16  (0 = bg, k = instance k)
│   ├── instance_masks_highmod/
│   ├── instance_masks_all/
│   ├── labels_highonly/                   # <tile>.txt  YOLO polygon (cls x1 y1 x2 y2 ...)
│   ├── labels_highmod/                    # ★
│   ├── labels_all/
│   └── instance_metadata_highmod/         # <tile>.json {inst_id: {cls, n_pixels, source_cl_class}}
├── val/                                   # same structure
├── test/                                  # same structure
├── channel_stats_9ch_train_water.json
├── tile_geo_index.json                    # may be reused from v2.0 (geo unchanged)
├── split_manifest.json                    # {scene_id → split, strata, debris_count, mean_size}
└── build_report.json                      # version, hash, build stats, warnings
```

## 11. Task Definition (FIXED — binary instance segmentation)

- **Main experiment**: **binary** instance segmentation
  (Marine_Debris vs background).
- **Extended experiment** (appendix): **3-class** (Marine_Debris,
  Algae_Organic, Ship). Single seed.
- **Confounder classes** (Foam, Turbid Water, Waves, Mixed Water,
  Sediment-Laden Water, Cloud Shadows, Wakes, Shallow Water): treated
  as **hard-negative background**, NOT a class. Their pixels participate
  in normalization (water-related) and may be flagged in
  `instance_metadata_highmod` for error analysis ("FP-in-confounder-zone"
  reporting). They are NOT given instance IDs.

## 12. Class → v2.1 mapping

MARIDA original `_cl.tif` → v2.1 instance class:

| `_cl.tif` value | MARIDA name | v2.1 main | v2.1 ext |
|---:|---|---|---|
| 1 | Marine Debris | **0 (Marine_Debris)** | 0 |
| 2 | Dense Sargassum | dropped (binary) | 1 (Algae_Organic) |
| 3 | Sparse Sargassum | dropped | 1 |
| 4 | Natural Organic Material | dropped | 1 |
| 5 | Ship | dropped | 2 (Ship) |
| 8 | Sediment-Laden Water | confounder-bg | confounder-bg |
| 9 | Foam | confounder-bg | confounder-bg |
| 10 | Turbid Water | confounder-bg | confounder-bg |
| 11 | Shallow Water | confounder-bg | confounder-bg |
| 12 | Waves | confounder-bg | confounder-bg |
| 14 | Wakes | confounder-bg | confounder-bg |
| 15 | Mixed Water | confounder-bg | confounder-bg |
| 6 (clouds), 13 (cloud_shadows) | excluded from norm | excluded |
| 7 (Marine Water) | clean-bg | clean-bg |
| 0 (unlabeled) | unlabeled | unlabeled |

## 13. Evaluation (FIXED — three views)

| View | Primary metric | Source data | Notes |
|---|---|---|---|
| Detection (box) | mAP@0.5 | bbox from polygon | uses `labels_highmod` |
| Instance segmentation | mask IoU | polygon vs polygon | uses `labels_highmod` |
| Semantic segmentation | per-pixel IoU/F1 | raster vs raster | uses `instance_masks_highmod` |

**Size-bin diagnostic** (mandatory for the paper):
- tiny (≤3 px): report hit-rate + center-distance (IoU not meaningful)
- small (4-15 px): report all three views
- large (>15 px): report all three views

**Statistical analysis**:
- **Scene-level bootstrap** (B=1000), sampling 9 test scenes
  **with replacement**. Report mean ± 95% CI.
- Honest warning in Limitations: n_scene=9 limits effective bootstrap
  sample size; CI widths are lower bounds on true uncertainty.
- **F1-optimal threshold search** on val set (replaces Platt scaling).
- Report per-scene breakdown explicitly (Table 3 v2.1).

## 14. Training Matrix (FIXED)

| Run name | Task | Loss-set | imgsz | Seeds | Purpose |
|---|---|---:|---:|---|---|
| `baseline_v2_1_s{0,1,2}` | Binary MD seg | highmod | 512 | 3 | Main, low-res |
| `baseline_v2_1_i1024_s{0,1,2}` | Binary MD seg | highmod | 1024 | 3 | Main, hi-res |
| `ext_3class_s0` | 3-class seg | highmod | best of above | 1 | Appendix |
| `abl_minus_b6_s0` | Binary, no B6/FDI | highmod | best | 1 | Leave-one-out |
| `abl_minus_inflate_s0` | Binary, from scratch | highmod | best | 1 | Leave-one-out |
| `abl_minus_waterstats_s0` | Binary, full-pixel stats | highmod | best | 1 | Leave-one-out |
| `abl_minus_min1_s0` | Binary, MIN=4 | highmod | best | 1 | Leave-one-out |
| `abl_11band_s0` | Binary, 11 raw bands no indices | highmod | best | 1 | Trade-off ablation |
| `abl_highonly_s0` | Binary | highonly | best | 1 | Conf ablation |
| `abl_all_s0` | Binary | all | best | 1 | Conf ablation |

Total: **6 main runs + 7 ablation runs = 13 runs**.

## 15. Baseline Comparisons (FIXED — same split)

| Baseline | Input | Code source | Reported as |
|---|---|---|---|
| RF SS+SI (Kakogeorgiou 2022) | Full 11 band + indices + GLCM | their repo | re-run on v2.1 same split |
| U-Net (Kikaki 2022) | 11 raw bands | their repo | re-run on v2.1 same split |
| FDI > threshold | B6/B8/B11 | trivial impl. | re-run on v2.1 same split |
| YOLO-seg v2.1 (ours) | 9ch (selected + indices) | this repo | re-run on v2.1 same split |

**Honest input-trade-off disclosure** (mandatory in §3 or §5):

> Each baseline uses its original input configuration as published.
> Our method intentionally restricts input to a 9-channel subset
> (selected spectral bands + indices) to keep model size manageable;
> this is a trade-off in input information for inference efficiency,
> not a limitation of YOLO-seg. We additionally include
> `abl_11band` as a fairness ablation feeding raw 11-band into
> YOLO-seg.

Literature numbers (Bouchelaghem, Mäyrä, etc.) are reported in a
**separate** "Historical benchmarks (different protocols, not directly
comparable)" subsection and **never** mixed with v2.1 same-split
numbers.

## 16. Versioning & Reproducibility

- Every `data/processed/marida_v2_1/` build emits:
  - `build_report.json` with: build timestamp, git commit hash, MARIDA
    raw checksum, total tile/scene counts, per-split GT counts,
    elapsed time, all warnings.
  - SHA-256 of every produced `.npy` and `.json` (in `build_report.json`).
- Dataset YAML for training references `marida_v2_1@<hash>`. If the
  hash changes mid-experiment, training scripts refuse to run.
- All seeds, hyperparameters, environment versions logged into
  `runs_v2_1/<run>/manifest.json` at training start.

## 17. What is *not* changed from v2.0

- Raw MARIDA data in `data/raw/MARIDA/` (untouched)
- Original `tile_geo_index.json` (geo information unchanged; can be
  symlinked into `data/processed/marida_v2_1/`)
- The legacy v2.0 work (paper draft, figures, runs) remains frozen.

---

---

## 18. COCO Weight Inflation 3→9 ch (FIXED)

The first conv layer of YOLOv11 (`model.0.conv`, COCO-pretrained) has
shape `(out_channels=64, in_channels=3, k=3, k=3)`. We inflate this to
`(64, 9, 3, 3)` using the following deterministic strategy:

| New ch idx | Name | Initialization |
|---:|---|---|
| 0 | B4 Red (665 nm) | COCO R weight (slice `[:, 0]`) |
| 1 | B3 Green (560 nm) | COCO G weight (slice `[:, 1]`) |
| 2 | B2 Blue (490 nm) | COCO B weight (slice `[:, 2]`) |
| 3 | B6 Red Edge 2 (740 nm) | (COCO R + G) / 2  (no RGB analog; use the two nearest in wavelength) |
| 4 | B8 NIR (842 nm) | COCO R weight (R is the closest RGB band to NIR) |
| 5 | B11 SWIR-1 (1610 nm) | Xavier uniform (no RGB analog) |
| 6 | B12 SWIR-2 (2190 nm) | Xavier uniform |
| 7 | FDI | Xavier uniform (derived signed quantity) |
| 8 | NDVI | Xavier uniform (derived signed quantity) |

**Magnitude correction**: After the above assignment, multiply the
**entire** inflated tensor by `sqrt(3/9) = 0.5774` so the initial
activation magnitude after `model.0.conv` matches the COCO-pretrained
norm. Without this rescale, BN running statistics on the next layer
will diverge during the first ~100 steps.

**Differential learning rate (first 100 steps only)**: channels 5-8
(SWIR + indices) carry Xavier weights, while channels 0-4 carry
COCO-pretrained features. Apply `lr_multiplier = 3.0` on the
columns `[5, 6, 7, 8]` of `model.0.conv.weight` for the first 100
optimization steps so the un-pretrained channels catch up. After step
100, fall back to uniform lr.

Implementation: `ultralytics_plugins_v2_1/weight_inflate.py`.

## 19. Training-Time Polygon Dilation (FIXED)

Per Phase 1 sanity check, 83% of `labels_highmod` polygons span
≤ 2 pixels (in normalized coordinates). At imgsz=512 a 1-px GT
becomes a 2-px polygon and at imgsz=1024 a 4-px polygon — far below
the resolution at which YOLOv11n-seg mask prototypes can learn.

**Dataloader behavior**: at train time, expand every instance mask by
`morphological dilation 3×3` before computing the polygon from
connected components.

- Dilation is applied to **the rasterized instance mask**, not to the
  pre-computed `labels_*/.txt` polygons. The dataloader re-derives
  polygons from the dilated mask on the fly.
- Validation and test loops do **NOT** apply dilation. GT remains the
  original raster mask for honest evaluation.
- Train-time dilation is disclosed in §3 of the paper and reported
  as part of the augmentation pipeline, not as ground-truth
  modification.

## 20. Dataloader Augmentation (FIXED, explicit YAML)

The following must appear verbatim in every v2.1 model YAML to prevent
ultralytics from silently re-enabling defaults:

```yaml
mosaic:      0.0    # destroys per-tile spectral coherence
mixup:       0.0    # same
copy_paste:  0.0    # same
hsv_h:       0.0    # 9-channel data has no HSV concept
hsv_s:       0.0
hsv_v:       0.0
degrees:     180.0  # geometric only, full rotation OK
fliplr:      0.5
flipud:      0.5
scale:       0.5    # geometric scale jitter OK
shear:       0.0
perspective: 0.0
```

Any deviation is treated as a SPEC violation and refused by trainer.

## 21. Weighted Sampling (FIXED)

To compensate for the 73% / 27% empty-vs-debris tile ratio in train,
the dataloader uses `torch.utils.data.WeightedRandomSampler` with:

- `weight = 5.0` for tiles with at least one highmod debris box
- `weight = 1.0` for empty tiles

Result: ≈ 30% of batches contain at least one debris instance, vs the
natural rate of ≈ 22%. This is logged in `runs_v2_1/<run>/manifest.json`.

## 22. Per-Channel Stats Fallback (FIXED)

If any of the 9 channels yields `p99 - p01 < 1e-4` during stats
computation, log a warning and use `mean ± 3*std` (also computed on
the water pool) as a fallback. Refuse to proceed if both methods
collapse — that signals a corrupted training pool.

---

**Last updated**: 2026-05-16 (after Phase 1.3 mega-scene fix, §18-22 added)
**Next review point**: after Phase 2 implementation
