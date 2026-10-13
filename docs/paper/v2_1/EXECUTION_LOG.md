# v2.1 Execution Log

Live tracking of Phase 0-6 execution. Update this on every significant
step. **Source of truth** for "where are we right now".

---

## Phase 0 — Sanity & Foundations (target: 2 day) ✅ DONE 2026-05-XX

| Task | Status | Owner | Notes |
|---|---|---|---|
| (a) v2.1 directory tree | ✅ done | Claude | created |
| (b) v2_1_DATASET_SPEC.md | ✅ done | Claude | 17 sections locked |
| (c) v2.1 paper prose seed (ACOLITE description) | ✅ done | Claude | `docs/paper/v2_1/PAPER_PROSE_DRAFT.md` with REWRITE tags |
| (d) MIN=1 sanity check — script | ✅ done | Claude | `scripts/v2_1/preprocessing/sanity_min1_judge.py` |
| (e) MIN=1 sanity check — 30 PNG output | ✅ done | Claude | `runs_v2_1/sanity/min1_panels/` |
| (f) Human judgement (30 samples × 3 conf levels) | ✅ done | User | High 90% real / Mod 60% / Low 80% |
| (g) Kakogeorgiou 2022 exact MD pixel IoU | ⏳ | **User** | still needed before §4 baseline comparison |

### Phase 0 outcomes
- MIN_COMPONENT_PIXELS = 1 **CONFIRMED** (High-conf 90% real ≥ 70% threshold)
- ~400 previously-discarded real 1-px GT instances now recoverable
- Confidence-aware ablation track justified (Moderate has 10% false labels)
- `labels_all` ablation meaningful (Low has 80% real, not pure noise)

## Phase 1 — Dataset Rebuild ✅ FINAL DONE 2026-05-16 (v2.1.3)

### Iteration trail
- **v2.1.0**: scene-group 2D round-robin only → test 70% single-scene dom
- **v2.1.1**: + mega rule (>50→train) → test 52%, but test only 45 boxes
- **v2.1.2**: + iterative top-1 enforcer + safety net → test 29 boxes / 3 scenes
- **v2.1.3 (FINAL)**: three-tier allocation (mega→train, mid→alternating val/test, small→round-robin) → **test 229 boxes / 12 scenes / top-1 15.7%**

### v2.1.3 metrics (canonical)

| Split | Tiles | Scenes (total) | Debris scenes | Debris boxes | Top-1 |
|---|---:|---:|---:|---:|---:|
| train | 740 | 35 | 24 | 608 | 27.3% |
| val | 369 | 14 | 12 | 247 | 16.2% |
| test | 272 | 14 | 12 | 229 | 15.7% |

### Closing sanity check artifacts
- `runs_v2_1/sanity/phase1_sanity_report.json` (full breakdown)
- `runs_v2_1/sanity/build_v2_1_3.log` (build log)
- `runs_v2_1/sanity/min1_panels/_keymap.json` (Phase 0 sanity)

Implemented as single-entry `scripts/v2_1/preprocessing/build_v2_1.py`
covering all sub-stages.

| Sub-stage | Status | Outcome |
|---|---|---|
| (1) Inventory | ✅ | 1381 patches × 63 scenes |
| (2) Scene-group 2D split | ✅ | 48/8/7 scenes (train/val/test), 1031/163/187 tiles |
| (3-4) 9ch .npy + valid_mask | ✅ | 1381 × 2 = 2762 .npy files written (56.6 s) |
| (5) train-water p01/p99 stats | ✅ | 61879 pool pixels from 965 tiles |
| (6) 3 label sets (highonly/highmod/all) | ✅ | see GT count table below |
| (7) verify split outcome | ✅ | test=169 boxes ≥ 25 target, no warnings |
| (8) build_report + manifest | ✅ | all 4 top-level json written |

### GT count breakdown (the key result)

| set | train tiles_w_GT | val tiles_w_GT | test tiles_w_GT | total boxes |
|---|---:|---:|---:|---:|
| highonly | 134 | 19 | 27 | 606 |
| **highmod** | **227** | **28** | **49** | **1084** |
| all | 275 | 43 | 55 | 1330 |

Versus v2.0 (test=13 boxes, val=77, train=122):
- **test boxes 13 → 169** (× 13)
- **val:test ratio 5.9:1 → 0.69:1** (corrected directionality)
- **total boxes 212 → 1084** (× 5.1)

### Build artifacts location

```
data/processed/marida_v2_1/
├── build_report.json               # version, warnings=[], elapsed=80.6s
├── channel_stats_9ch_train_water.json
├── split_manifest.json             # 63 scenes mapped, 2D-strata bins recorded
├── tile_geo_index.json             # copied from v2.0 (geo unchanged)
├── train/  (1031 images, 1031 valid_masks, 227 instance_masks, ...)
├── val/    (163 / 163 / 28 / ...)
└── test/   (187 / 187 / 49 / ...)
```

## Phase 2 — Polygon Extraction + Model Adaptation ✅ DONE 2026-05-16

| Task | Status | Notes |
|---|---|---|
| Connected-component → polygon → YOLO-seg label | ✅ | done in Phase 1 build_v2_1.py |
| Ultralytics 9ch adapter (in `ultralytics_plugins_v2_1/`) | ✅ | dataset / loss / trainer / weight_inflate |
| COCO weight inflation 3→9ch | ✅ | scale 1/√3 + Xavier for SWIR/indices, per SPEC §18 |
| WaterAwareSegLoss (valid_mask as loss weight) | ✅ | placeholder OK for smoke test |
| ultralytics IMG_FORMATS + verify_image_label patches | ✅ | global rebind across all data.* modules |
| Smoke test 1-epoch closed-loop | ✅ | 740 train / 369 val, 0 corrupt, validation passed |

## Phase 3 — Training Matrix (target: 10 day)

| Run | Status | Test mIoU / box mAP | Note |
|---|---|---|---|
| baseline_v2_1_s0/s1/s2 | ⏳ | – | imgsz=512 |
| baseline_v2_1_i1024_s0/s1/s2 | ⏳ | – | imgsz=1024 |
| ext_3class_s0 | ⏳ | – | 3-class |
| abl_minus_b6_s0 | ⏳ | – | |
| abl_minus_inflate_s0 | ⏳ | – | |
| abl_minus_waterstats_s0 | ⏳ | – | |
| abl_minus_min1_s0 | ⏳ | – | |
| abl_11band_s0 | ⏳ | – | |
| abl_highonly_s0 | ⏳ | – | |
| abl_all_s0 | ⏳ | – | |

## Phase 4 — Evaluation (target: 3 day)

| Task | Status |
|---|---|
| 3-view evaluator (box / inst-seg / sem-seg) | ⏳ |
| Size-bin diagnostic (tiny / small / large) | ⏳ |
| Scene-level bootstrap (B=1000) | ⏳ |
| F1-optimal threshold search on val | ⏳ |

## Phase 5 — Baselines (target: 5 day)

| Task | Status |
|---|---|
| Kakogeorgiou RF SS+SI rerun on v2.1 split | ⏳ |
| Kikaki U-Net rerun on v2.1 split | ⏳ |
| FDI > threshold (traditional) | ⏳ |

## Phase 6 — Paper artifacts (target: 3 day)

| Task | Status |
|---|---|
| Re-make all figures | ⏳ |
| Re-make all tables | ⏳ |
| Update PAPER_PROSE_DRAFT.md (v2.1) | ⏳ |

---

## Decision points / open questions

| # | Question | Owner | Resolved when |
|---|---|---|---|
| 1 | MIN=1 vs MIN=2 | Phase 0 sanity result | Phase 0f done |
| 2 | imgsz 512 vs 1024 as main | Phase 3 ablation | Phase 3 done |
| 3 | Add NDWI? (currently 9ch with FDI/NDVI only) | Phase 4 size-bin diagnostic | Phase 4 done |

---

**Last updated**: 2026-05-XX (Phase 0 start)
