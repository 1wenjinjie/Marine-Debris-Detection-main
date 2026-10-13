# MARIDA v2.1 — Isolated Workspace

This directory tree is the **complete, independent** working area for the
v2.1 pipeline (binary instance segmentation, scene-group split,
confidence-aware labels, 9-channel input with FDI/NDVI/NDWI). It is
**completely disjoint** from the v2.0 work — no shared code, no shared
data, no shared run directories.

## Directory layout

```
data/processed/marida_v2_1/         # new dataset (not v2/)
  ├── train|val|test/
  │   ├── images/                   # 9ch .npy
  │   ├── valid_masks/              # bool .npy (NaN region)
  │   ├── instance_masks_highonly/  # int16 raster, one int per instance
  │   ├── instance_masks_highmod/   # ★ main experiment uses this
  │   ├── instance_masks_all/
  │   ├── labels_highonly/          # YOLO polygon .txt
  │   ├── labels_highmod/           # ★ main experiment
  │   ├── labels_all/
  │   └── instance_metadata_highmod/  # per-tile json {inst_id → (cls, n_px)}
  ├── channel_stats_9ch_train_water.json   # train-only, water-pixel-only stats
  ├── tile_geo_index.json           # carried over from v2.0 (compatible)
  ├── split_manifest.json           # scene → split + strata bin labels
  └── build_report.json             # version, hash, build stats

runs_v2_1/                          # new training outputs
  ├── sanity/                       # Phase 0 sanity experiments
  ├── baseline_v2_1_s{0,1,2}/       # binary MD instance seg, imgsz=512
  ├── baseline_v2_1_i1024_s{0,1,2}/ # binary MD instance seg, imgsz=1024
  ├── ablation_*/                   # leave-one-out
  └── baselines/                    # Kakogeorgiou RF, Kikaki U-Net, FDI thresh

scripts/v2_1/                       # all v2.1 scripts
  ├── preprocessing/
  ├── training/
  ├── evaluation/
  └── figures/

configs/datasets/v2_1/
configs/models/v2_1/

ultralytics_plugins_v2_1/           # v2.1-only plugins
                                    # (don't touch ultralytics_plugins/)

docs/paper/v2_1/
  ├── README.md                     # this file
  ├── v2_1_DATASET_SPEC.md          # locked decisions
  ├── PAPER_PROSE_DRAFT.md          # v2.1 paper writing
  └── EXECUTION_LOG.md              # Phase 0-6 progress
```

## What is *frozen* and where

| Frozen artifact | Location |
|---|---|
| Original MARIDA raw data | `data/raw/MARIDA/` — never touched |
| v2.0 work (legacy) | `data/processed/marida_v2/`, `runs_v2/`, `scripts/v2_*.py`, `docs/paper/PAPER_PROSE_DRAFT.md` etc. — frozen, not edited |
| v2.1 work | everything under `*_v2_1/` and this directory |

## Why isolated?

- v2.0 has 18+ trained models, frozen figures, frozen tables. They are
  documented and referenced in `docs/paper/PAPER_DRAFT.md`. Touching
  them invalidates the legacy paper draft.
- v2.1 is a clean re-implementation following the 5-step framework
  (see `v2_1_DATASET_SPEC.md`). Mixing the two would break
  reproducibility on both sides.
- If v2.1 produces strong results, v2.0 becomes Appendix material
  ("we initially attempted detection-only on v2.0; here is what we
  learned"). If v2.1 fails, v2.0 remains the submitted version.

## Phase status

See `EXECUTION_LOG.md` for live progress.
