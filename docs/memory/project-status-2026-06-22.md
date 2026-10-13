---
name: project-status-2026-06-22
description: "Current state of the v1 honest line: BiFPN+imgsz1024 honest SOTA, deployment F1=0.604, 老师 plan 9/11 figs done, remaining TODOs"
metadata: 
  node_type: memory
  type: project
  originSessionId: a3ca46b2-90f2-46df-b0b0-eb6ecf921caf
---

2026-06-22 snapshot of the v1 honest paper/汇报 line (主线 = v1 7ch physidx on v2split; see CLAUDE.md for data layout).

**Honest SOTA:** BiFPN @ imgsz=1024, seed-0. Per-tile deployment F1 = **0.604** (P 0.581 / R 0.630) @ τ=0.25, IoU≥0.5, Marine_Debris instance level. Verified by `scripts/v1_per_tile_test_inference.py` → `docs/paper/v1_test_per_tile.csv`. Test set = 73 tiles / 108 MD GT. Weights: `runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_bifpn_imgsz1024_clean_init_s0/weights/best.pt`.

**老师 plan figures:** 9/11 done. 图1/2 = designer SVG (`docs/paper/figures/fig1_pipeline.svg`, `fig2_modules.svg`); 图3-9 = python (ablation bars, PR/F1 curves, EigenCAM `v1_activation_heatmap.png`, cartopy `v1_spatial_distribution.png`, temporal `v1_temporal_distribution.png`). 图10 done as descriptive. 图11 (CMEMS) pending — needs `copernicusmarine` + Copernicus account; user undecided.

**Done 2026-06-23:** 汇报对账文档 `docs/paper/汇报_对账老师plan_2026-06-23.md` (老师 plan vs done/self-added/dropped); 硕士论文 8-章中文大纲 `docs/paper/硕士论文章节大纲_2026-06-23.md`. Thesis main line decided = **honest-evaluation framing** (NOT performance-race). Honest numbers confirmed: baseline 3-seed Best F1 0.409 → BiFPN×imgsz1024 0.479 (gain small, mostly within std); deployment F1 0.604 is a DIFFERENT metric (seed-0, instance-level, τ=0.25) — never conflate.

**Key open decision:** whether GT-leak-attribution becomes a formal thesis chapter (ch.4). I strongly recommend YES (strongest scientific point); user previously leaned 汇报-only. See [[feedback-paper-presentation]].

**Remaining small TODOs:** update `docs/paper/figures/README.md`; verify designer's 3 concerns (BiFPN node count vs YAML, FDI B6 outside detector channels, seed-0 vs 3-seed caption); oracle 4-cell exact numbers need metric-definition check.

**Biggest weakness:** statistical power. Headline F1/calibration/coast-table all seed-0 single-run; tables mix 3-seed mean and single-seed — must label or upgrade before submission.

**Why:** lets a new session resume without re-deriving the active line. **How to apply:** run scripts via the `marine` conda python (see [[project-env-4090-china-mirrors]]); honest line, NOT the leaky `mask_bbox_rescue_split` or shelved v2 3-class. Related: [[project-leak-attribution-2026-06]] [[project-v1-calibration-2026-06]] [[project-finding-d-segmentation]].
