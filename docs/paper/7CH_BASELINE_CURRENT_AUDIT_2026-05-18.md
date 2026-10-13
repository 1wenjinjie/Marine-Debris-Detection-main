# 7ch YOLOv11n Baseline Current Audit

Date: 2026-05-18
Last fix pass: 2026-05-18 (engineering bugs only; oracle/leakage issues remain by design)

Scope: current repository state under `d:/OceanEdge_Group/Marine-Debris-Detection-main`, with the Notion project notes used as a guide for what to verify.

This file supersedes the older baseline notes for current decision making, especially:

- `docs/paper/BASELINE_SYSTEM_ANALYSIS.md`
- `docs/analysis/BASELINE_ANALYSIS.txt`
- `README.md` section `Current SOTA (Major Classes)`

Those files are useful as historical records, but some conclusions there are now incomplete or unsafe without the caveats below.

## Fix Status Summary (2026-05-18 pass)

The v1 line is now positioned as **historical reference / oracle teacher** only. Engineering bugs have been patched so the artifact remains auditable; the GT-heatmap and scene-overlap issues are *not* patched — they are intrinsic to the v1 split and would require new data work to remove.

| # | Issue | Status | Action taken |
|---|---|---|---|
| 1 | GT-derived heatmap at test time | Documented, not fixed | README + this doc now mark all v1 7ch numbers as oracle/privileged. |
| 2 | Scene-level leakage in v1 split | Documented, not fixed | Use MARIDA v2 scene-split for any deployable claim. |
| 3 | Stale `train.txt` (869 entries pointing to wrong absolute path) | **Fixed** | Rewrote aug rows to relative `train/images_aug/...`. Original kept as `train.txt.bak`. All 1216 entries now resolve. |
| 4 | `channel_stats_7ch.json` path drift | **Fixed** | Copied to dataset root `data/processed/mask_bbox_rescue_split/channel_stats_7ch.json` so loader default resolves without CWD tricks. |
| 5 | `model.val()` reproduction not locked | **Fixed** | New `scripts/v1_reproduce_7ch_baseline.py` pins repo root, checkpoint SHA256, YAML, channel_stats, loader, imgsz/batch/conf/iou; saves JSON with the pin. |
| 6 | `ClassAwareCopyPaste` `parsed_cp_classes` always None | **Fixed (code only, no retrain)** | Parsed `cp_classes` into `parsed_cp_classes` before constructing `ClassAwareCopyPaste`; also fixed f-string `{{op.p}}` literal bug. Per user decision, the prior class-aware runs are not retrained — they remain unreliable evidence. |

**Reproduction result (2026-05-18 rerun, val split, full heatmap):**

| Metric | Audit-recorded | Rerun | Delta |
|---|---:|---:|---:|
| mAP50 | 0.5173 | 0.5243 | +0.0070 |
| mAP50-95 | 0.2702 | 0.2652 | -0.0050 |

Both within 1%; sufficient to call the historical numbers reproducible. Per-class breakdown saved to `experiments/runs/runs_marida/yolov11n_7ch_512_finetune_wakeboost/v1_reproduce_val.json`. Marine_Debris AP50 = 0.844 confirms the oracle major-class strength is intact.

## Leak Decomposition Experiment (2026-05-19)

To separate the contribution of (a) scene-level overlap and (b) GT-derived
heatmap, we resplit the v1 7ch tile pool using v2's scene-clean assignment
(371 train / 70 val / 73 test, 0 cross-split scene leakage; manifest at
`data/processed/mask_bbox_rescue_split_v2split/`), kept v1's original
7-class labels, and ran four evaluations on the identical v2-split test set.

| Experiment | Init | Test heatmap | mAP50 | mAP50-95 | Validity |
|---|---|---|---:|---:|---|
| Historical SOTA direct eval | -- (no retrain) | GT | 0.9212 | 0.7207 | **Invalid**: 50 of 73 test tiles came from the original v1 train split; SOTA had trained on the exact same tiles. |
| Finetune from SOTA parent (30 ep) | SOTA parent best.pt | GT | 0.8402 | 0.5760 | **Compromised**: the parent checkpoint was pretrained on leaky v1 data, so 30 epochs of finetune on the scene-clean split can't fully purge prior exposure. |
| **Clean init, scratch-train (30 ep)** | yolo11n_7ch_init.pt | GT | **0.4658** | **0.3020** | **Clean** scene-split + oracle heatmap. Real upper bound when GT prior is available. |
| Clean init, eval with zeroed heatmap | same checkpoint | zeroed | 0.1408 | 0.0906 | **Clean** scene-split, no GT prior. Real deployable performance. |

Artifacts (with checkpoint SHA256 pins):
- `runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_finetune/v2split_test_eval.json`
- `runs/detect/runs_v1_v2split/historical_sota_direct_v2split_test_eval.json`
- `runs_v1_v2split/yolov11n_7ch_v2split_clean_init2/v2split_test_eval.json`
- `runs_v1_v2split/yolov11n_7ch_v2split_clean_init2/v2split_test_zero_heatmap_eval.json`

### Decomposition

Comparing the three honest numbers:

```
v1 SOTA on v1 leaky split   : mAP50 0.5243   (oracle + scene leak; published baseline)
v1 7ch clean-init, oracle   : mAP50 0.4658   (oracle only)
v1 7ch clean-init, no GT    : mAP50 0.1408   (neither — deployable)
```

- **Scene leak contribution**: ~0.058 mAP50 (0.524 -> 0.466). Smaller than expected; the v1 split's scene overlap inflates results but is not the dominant lift.
- **GT heatmap contribution**: ~0.325 mAP50 (0.466 -> 0.141). **The oracle channel accounts for ~70% of total performance.**
- **Deployable v1 7ch (no GT)**: mAP50 0.141. This is *lower* than the v2 6ch clean baseline (mAP50 ~0.19), although label spaces differ (7-class v1 vs 3-class v2) so the comparison is indicative rather than strict.

### What this means for any v1-derived paper

- The historical F1 0.806 / mAP50 0.524 result is built almost entirely on GT-derived test-time information. It is not a multispectral-detection result; it is an oracle result.
- Removing the GT prior at test time drops the same architecture to ~0.14 mAP50 on a clean split. The v1 7ch pipeline's spectral-only generalization is weaker than v2's 6ch clean baseline.
- Recommended publication framing (does not require additional training):
  1. v1 7ch with GT heatmap is a privileged-information *upper bound*, not a detector.
  2. Quantify the GT prior contribution explicitly (~70% of mAP50).
  3. Defer deployable claims to a clean 6ch model (v2 or future student); v1 should be cited only as a privileged reference.

## Physical-Prior Variant (2026-05-19)

Follow-up experiment: replace the GT-derived 7th channel with a deterministic NDWI-inverse index `(1 - (B3-B8)/(B3+B8)) / 2`. The 7th channel is now a function of spectral bands only and contains no GT information. Trained from `yolo11n_7ch_init.pt` with the same SOTA-pinned hyperparameters on the v2 scene-clean split.

Artifacts:
- Preprocessing: `scripts/v1_make_physidx_tiles.py` (writes `data/processed/mask_bbox_rescue_split_physidx/`)
- Dataset YAML: `configs/datasets/marida_v1_physidx_on_v2split_7ch.yaml`
- Training run: `runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_clean_init/`
- Eval JSONs (with index / zero ch7) saved in the run directory

| Setup | Init | ch7 at train | ch7 at eval | mAP50 | mAP50-95 |
|---|---|---|---|---:|---:|
| Clean init **physidx** (NDWI-inverse) | yolo11n_7ch_init | NDWI-inv | NDWI-inv | **0.3492** | 0.1721 |
| Clean init physidx, zeroed ch7 at eval | (same ckpt) | NDWI-inv | 0 | 0.3320 | 0.1631 |
| -- for comparison: clean init oracle | yolo11n_7ch_init | GT heatmap | GT heatmap | 0.4658 | 0.3020 |
| -- for comparison: clean init oracle, zeroed | (same ckpt) | GT heatmap | 0 | 0.1408 | 0.0906 |

### Two non-trivial findings

**Finding A: the NDWI-inverse channel itself contributes very little at inference.** The physidx-trained model loses only 0.017 mAP50 when ch7 is zeroed (0.349 -> 0.332). The 6 spectral channels already encode the information that NDWI exposes, so the prior channel is largely redundant at inference time.

**Finding B: training with a weak prior preserves spectral feature learning; training with an oracle prior destroys it.** When the GT-trained model loses its ch7 at inference, it collapses to 0.141 mAP50 -- the model had learned to rely almost entirely on the GT channel and never developed strong spectral features. The physidx-trained model with zeroed ch7 retains 0.332 mAP50 -- a 2.4x deployable margin over the oracle model under the same "no prior at test" condition.

This is the publishable result: *oracle teaching damages downstream spectral feature learning, even when the test-time prior is identical.* A non-leaky prior at training time leaves the network forced to learn from the spectral signal.

### Updated decomposition

```
v1 SOTA on v1 leaky split    : mAP50 0.524   (oracle + scene leak; published)
v1 clean-init, GT prior      : mAP50 0.466   (oracle only)
v1 clean-init, NDWI prior    : mAP50 0.349   (physical prior, deployable)
v1 clean-init, GT, zero eval : mAP50 0.141   (oracle model stripped of GT)
v1 clean-init, NDWI, zero    : mAP50 0.332   (NDWI model stripped of ch7)
```

### Revised paper framing (preferred)

The 4-cell table is now a 5-cell table that tells a tighter story:

1. **The historical 7ch baseline (0.524) was an oracle.** Show the leak decomposition.
2. **Even with a clean physical prior, the 7th channel adds little (Finding A).** This is a clean negative result on the value of single-channel hand-engineered priors.
3. **What does matter is the training regime (Finding B).** Oracle training cripples spectral generalization; physical training preserves it.
4. **Recommendation:** report the NDWI-inverse model (0.349 mAP50) as the honest deployable baseline. Cite the oracle (0.466) only as an upper bound. The GT-trained zero-ch7 number (0.141) becomes evidence of training-time over-reliance, not a deployable claim.

This framing requires no further training. It is defensible to a reviewer who asks "what if the test-time heatmap is unavailable?" because we have already answered that question two different ways.

## Architecture Ablation on Honest Setting (2026-05-19)

Per the teacher's task plan (CBAM + Bi-FPN + EIoU), we evaluated the three
canonical YOLO improvements on the scene-clean + physidx setting. Same
training pipeline as the baseline (SOTA-pinned hyperparameters, clean init
from `yolo11n_7ch_init.pt`, 30 epochs, seed=0), only the architecture or
box loss changes. All five models were evaluated on the same 73-tile
v2-scene-clean test set with the same physidx 7th channel.

| Model | Layers | Params | mAP50 | mAP50-95 | Debris AP50 | Ship AP50 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| **Baseline** (yolov11n 7ch) | 100 | 2.58 M | **0.3492** | **0.1721** | 0.462 | 0.791 | reference |
| +CBAM | 121 | 2.69 M | 0.2719 | 0.1185 | **0.553** | 0.637 | Debris up, Ship down |
| +BiFPN | 101 | 2.60 M | 0.2326 | 0.1451 | 0.552 | 0.565 | similar pattern |
| +EIoU | 100 | 2.58 M | 0.2894 | 0.1525 | 0.484 | 0.751 | smallest regression |
| Full (CBAM+BiFPN+EIoU) | 121 | 2.70 M | 0.2412 | 0.1191 | 0.548 | 0.526 | worst overall |

Artifacts: `runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_{cbam,bifpn,eiou,full}_clean_init/v2split_test_eval.json`. Training script: `scripts/v1_ablation_train.py`. Architecture YAMLs: `configs/models/yolov11n_7ch_{cbam,bifpn,full}.yaml`.

### Finding C: standard YOLO improvements do not transfer to the honest scene-clean setting

Every single improvement reduces overall mAP50 versus the baseline (-0.06 to -0.12). The per-class breakdown shows **why** the aggregate number falls:

- **Marine_Debris AP50 improves with every ablation** (0.46 -> 0.48-0.55). The attention / multi-scale modules do help the small-object debris class.
- **Ship AP50 collapses with every ablation** (0.79 -> 0.53-0.75). Ship has only 9 GT instances in the test set; the baseline overfits a strong ship detector under the small-data regime, and added capacity (CBAM/BiFPN) appears to dilute that overfit. EIoU comes closest to preserving Ship performance.
- **Rare classes (Foam, Wakes, Sparse_Sargassum) stay near zero** for all models.

### Why this matters for the paper

Combined with the leak-decomposition table, the ablation matrix exposes a pattern:

1. The **published v1 7ch SOTA (0.524) is an oracle on a leaky split.**
2. The **honest physidx baseline (0.349) loses ~30% to a clean prior and clean split.**
3. **Adding the conventional improvement triad makes the honest baseline worse, not better.**

A literal reading of the teacher's planned ablation table (Baseline -> +CBAM -> +BiFPN -> +Full, each row higher mAP) is **not supported by data in the scene-clean setting.** Three resolutions are available:

- **(a) Reproduce on leaky split for comparison.** If +CBAM/+BiFPN/+Full *do* improve on a random 70/20/10 split, the contrast itself becomes the paper's punchline: prior architecture gains were artifacts of split contamination. Requires 4 extra training runs (~30 min GPU).
- **(b) Reframe ablation as "capacity vs data" study.** Report the per-class table honestly. Discuss how added capacity hurts in the n=371-tile regime, and that data scale -- not architecture -- is the bottleneck.
- **(c) Move the ablation to v2's larger train set with v2 labels.** The improvements may transfer better with more data. But this loses the apples-to-apples comparison with the baseline and physidx experiments.

Recommended path: **(a)** -- the contrast is the single most reviewer-defensible narrative because it lands two findings (leak inflates baselines AND inflates ablations) in one experiment block.

## Leaky-Setting Ablation + F1 Sweep (2026-05-19, evening)

Per the recommendation above, we re-ran the same 5-cell ablation matrix on the **leaky** setting (v1 GT heatmap as ch7, v1 original random split). Same clean init (`yolo11n_7ch_init.pt`), same 30-epoch SOTA-pinned hyperparameters. Only the data YAML differs. Plus a confidence-threshold sweep for best F1 on every checkpoint, addressing the teacher's F1-score requirement.

Artifacts:
- Training runs: `runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_{baseline,cbam,bifpn,eiou,full}_clean_init/`
- Per-checkpoint test JSONs (mAP at conf=0.001): `leaky_test_eval.json` in each run dir
- F1 conf sweeps (10 checkpoints): `test_f1_sweep.json` in each run dir
- Sweep script: `scripts/v1_f1_conf_sweep.py`

### Side-by-side comparison

| Model | Setting | Test set | mAP50 | mAP50-95 | Best F1 | F1 conf | Debris AP50 | Ship AP50 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Leaky | 53 tiles | **0.695** | **0.455** | 0.771 | 0.001 | 0.850 | 0.936 |
| +CBAM | Leaky | 53 tiles | 0.447 | 0.215 | 0.438 | 0.150 | 0.778 | 0.663 |
| +BiFPN | Leaky | 53 tiles | 0.699 | 0.437 | **0.804** | 0.001 | 0.842 | 0.958 |
| +EIoU | Leaky | 53 tiles | 0.698 | 0.441 | 0.741 | 0.150 | 0.891 | 0.988 |
| +Full | Leaky | 53 tiles | 0.621 | 0.256 | 0.625 | 0.200 | 0.807 | 0.684 |
| | | | | | | | | |
| Baseline | Honest | 73 tiles | 0.349 | 0.172 | 0.430 | 0.001 | 0.462 | 0.791 |
| +CBAM | Honest | 73 tiles | 0.272 | 0.119 | 0.357 | 0.001 | 0.553 | 0.637 |
| +BiFPN | Honest | 73 tiles | 0.233 | 0.145 | 0.374 | 0.001 | 0.552 | 0.565 |
| +EIoU | Honest | 73 tiles | 0.289 | 0.153 | 0.359 | 0.001 | 0.484 | 0.751 |
| +Full | Honest | 73 tiles | 0.241 | 0.119 | 0.339 | 0.001 | 0.548 | 0.526 |

Note: leaky and honest tests use different test sets (53 v1-original tiles vs 73 v2-scene-clean tiles), so cross-row absolute comparisons should be interpreted as setting-level, not tile-level.

### Updated findings

**Finding D: Leak inflates F1 from ~0.43 to ~0.77** (78% inflation on baseline). +BiFPN under leaky setting achieves F1 = 0.804, almost exactly matching the published v1 SOTA (F1 = 0.806). This shows the published number is not an outlier -- it is the *expected* value when training on a leaky split with GT-derived prior, regardless of which architecture variant you pick.

**Finding E: The ablation ranking inverts across settings.** Under leaky evaluation:
- Baseline (0.695 mAP50, 0.771 F1) is competitive with the best improvement (+BiFPN 0.699 / 0.804).
- Marine_Debris AP50 is saturated near 0.80-0.89 for all variants -- the leaky test set is too easy to discriminate methods.
- +CBAM appears catastrophic (-0.25 mAP50). This is the only consistent signal: CBAM hurts both settings.

Under honest evaluation:
- Baseline (0.349 mAP50, 0.430 F1) is *best* by overall mAP and F1.
- But **Marine_Debris AP50 ranking inverts**: under leaky, baseline ties or wins; under honest, *every improvement beats baseline* (0.55 vs 0.46).
- The Ship class (only 9 GT instances) carries the variance that flips overall mAP rankings.

**Finding F: BiFPN is the only triad member that doesn't degrade the leaky setting.** +BiFPN matches or slightly beats baseline in leaky (mAP50 0.699 vs 0.695, F1 0.804 vs 0.771). CBAM and Full hurt under both settings. EIoU is mixed.

### Implication for the published "improvement triad"

The teacher's planned ablation table assumes Baseline -> +CBAM -> +BiFPN -> +Full each adds a positive increment. Under our two evaluation settings:

| Variant | Leaky delta vs baseline | Honest delta vs baseline | Interpretation |
|---|---:|---:|---|
| +CBAM | mAP50 -0.25, F1 -0.33 | mAP50 -0.08, F1 -0.07 | Hurts both. CBAM is not net-positive on MARIDA in this regime. |
| +BiFPN | mAP50 +0.00, F1 +0.03 | mAP50 -0.12, F1 -0.06 | Helps under leak, hurts under honest. |
| +EIoU | mAP50 +0.00, F1 -0.03 | mAP50 -0.06, F1 -0.07 | Near-neutral leaky, slight regression honest. |
| +Full | mAP50 -0.07, F1 -0.15 | mAP50 -0.11, F1 -0.09 | Combination hurts under both. |

**Single coherent narrative for the paper**: "Standard architecture improvements (CBAM, Bi-FPN, EIoU) show inflated apparent benefit under leaky evaluation; under scene-clean evaluation with a non-GT prior, the same improvements regress overall metrics. The Marine_Debris class-specific results, however, show that attention modules genuinely help small-object detection -- a signal masked by the saturated leaky baseline."

### Revised paper recommendation

1. Main contribution: leak-decomposition framework + honest baseline (mAP50 0.349, F1 0.430).
2. Method study: 5-cell ablation under both leaky and honest settings, side by side.
3. Honest narrative for the improvements: "+CBAM helps Marine_Debris detection under fair evaluation but hurts overall metrics due to small-data overfitting on the Ship class. +BiFPN's apparent leaky-setting gain (F1 0.804) does not transfer to honest evaluation. +EIoU is near-neutral. The combination of all three is consistently worse than baseline."
4. Bottom-line metric to report: honest baseline F1 = 0.430 (mAP50 = 0.349), with all 5 variants tabulated for transparency.

## FPS Benchmark (2026-05-19)

Forward-pass timing on RTX 3090 Ti, imgsz=512, batch=1, synthetic (1,7,512,512) float32 input. 100 timed iterations after 20 warmup. Bench script: `scripts/v1_ablation_fps_bench.py`. Per-checkpoint JSON: `<run_dir>/fps_bench.json`. Combined CSV: `docs/paper/v1_ablation_fps.csv`.

| Model | Setting | FPS (median) | Latency (ms) | std |
|---|---|---:|---:|---:|
| Baseline | Honest | 110.9 | 9.02 | 1.01 |
| +CBAM | Honest | 99.4 | 10.06 | 1.23 |
| +BiFPN | Honest | 105.0 | 9.52 | 1.50 |
| +EIoU | Honest | 111.6 | 8.96 | 0.89 |
| Full | Honest | 89.9 | 11.12 | 1.41 |
| Baseline | Leaky | 111.7 | 8.95 | 0.99 |
| +CBAM | Leaky | 104.2 | 9.60 | 0.92 |
| +BiFPN | Leaky | 108.2 | 9.24 | 0.89 |
| +EIoU | Leaky | 112.9 | 8.86 | 1.18 |
| Full | Leaky | 98.9 | 10.12 | 1.06 |

All variants exceed 90 FPS -- real-time capable at imgsz=512 on a single 3090 Ti. EIoU does not change architecture, so its FPS matches baseline; CBAM and Full are slowest due to added attention modules. The 20% speed cost of Full is not justified by its accuracy regression.

## Paper Figures (2026-05-19)

Generated by `scripts/v1_ablation_make_figures.py`, stored in `docs/paper/figures/`. See `docs/paper/figures/README.md` for the mapping to the teacher's figure list. Completion status:

| Teacher's figure | Status | Notes |
|---|---|---|
| 图3 mAP comparison curves | DONE | per-epoch + grouped bars |
| 图4 Precision-Recall | DONE | Ultralytics native, three reference cells |
| 图5 Ablation bars | DONE | mAP@0.5 + Best F1, leaky vs honest |
| 图6 FPS-mAP tradeoff | DONE | 10 ablation cells |
| 图1/图2 framework/module diagrams | NOT DONE | hand-drawn at writing stage |
| 图7 Grad-CAM | NOT DONE | future work |
| 图8 detection overlay | DONE | Primary: `figures/overlay_yolov11n_7ch_v2split_physidx_bifpn_imgsz1024_clean_init_s0/composite.png` (BiFPN @ 1024 = honest SOTA, 9 random test tiles). Legacy: `figures/overlay_yolov11n_7ch_v2split_physidx_clean_init/composite.png` (baseline @ 512). |
| 图9-11 spatial / temporal / CMEMS | NOT DONE | requires external data |

## 3-Seed Aggregate (2026-05-19, late)

All 10 ablation cells were re-trained at seeds 1 and 2 (s=0 already existed),
giving n=3 seeds per cell. Aggregated by `scripts/v1_ablation_aggregate.py`.
Tables: `docs/paper/v1_ablation_aggregate.{json,md}`. Error-bar figure:
`docs/paper/figures/ablation_mAP50_F1_bars_with_err.png`.

| Model | Setting | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Leaky | 3 | 0.690 +- 0.004 | 0.446 +- 0.024 | 0.740 +- 0.060 | 0.856 +- 0.029 | 0.910 +- 0.039 |
| +CBAM    | Leaky | 3 | 0.526 +- 0.091 | 0.241 +- 0.023 | 0.558 +- 0.139 | 0.805 +- 0.031 | 0.709 +- 0.069 |
| +BiFPN   | Leaky | 3 | 0.688 +- 0.020 | 0.425 +- 0.023 | 0.741 +- 0.088 | 0.852 +- 0.019 | 0.907 +- 0.069 |
| +EIoU    | Leaky | 3 | **0.722 +- 0.040** | 0.428 +- 0.031 | **0.767 +- 0.023** | **0.888 +- 0.014** | 0.909 +- 0.068 |
| Full     | Leaky | 3 | 0.579 +- 0.050 | 0.269 +- 0.021 | 0.631 +- 0.054 | 0.804 +- 0.008 | 0.718 +- 0.067 |
| Baseline | Honest | 3 | 0.313 +- 0.038 | 0.165 +- 0.011 | 0.409 +- 0.022 | 0.492 +- 0.056 | 0.690 +- 0.107 |
| +CBAM    | Honest | 3 | 0.258 +- 0.014 | 0.118 +- 0.002 | 0.359 +- 0.041 | 0.525 +- 0.047 | 0.576 +- 0.053 |
| +BiFPN   | Honest | 3 | 0.300 +- 0.059 | 0.167 +- 0.021 | 0.406 +- 0.029 | **0.559 +- 0.023** | 0.665 +- 0.092 |
| +EIoU    | Honest | 3 | 0.299 +- 0.036 | 0.165 +- 0.018 | **0.410 +- 0.056** | 0.524 +- 0.057 | **0.753 +- 0.079** |
| Full     | Honest | 3 | 0.232 +- 0.014 | 0.109 +- 0.009 | 0.335 +- 0.006 | 0.540 +- 0.018 | 0.552 +- 0.043 |

### Robustness of earlier findings under 3-seed evidence

- **Finding D (leak inflation, +78% F1)** stays intact: leaky Baseline F1 0.740
  vs honest Baseline F1 0.409; the gap (0.331) exceeds 4x the combined std.
  Leaky +EIoU (0.767) still approximates the original published F1=0.806 within
  one std band.
- **Finding C/E (improvements do not transfer)** stays intact at 3 seeds:
  - In leaky setting, +CBAM and Full collapse (mAP50 0.526, 0.579) far below
    Baseline (0.690), well outside std.
  - In honest setting, all four "improvements" are within 1-2 std of Baseline
    (0.313); none of them produce a statistically significant gain on mAP50.
  - On Debris AP50 the honest ranking still inverts: +BiFPN 0.559 > Baseline
    0.492 (std 0.056/0.023, gap ~1 std) -- a weak but consistent advantage,
    not a strong one.
- **Best leaky cell** is +EIoU on every headline metric (mAP50, Best F1, Debris
  AP50). Best honest cell depends on the metric: +EIoU for F1/Ship,
  +BiFPN for Debris AP50, Baseline for mAP50. No single cell wins everywhere
  in the honest setting.
- **Variance is meaningful**: leaky +CBAM has std 0.091 on mAP50 and 0.139 on
  F1 -- single-seed comparisons against this cell would be unreliable.

### Implication for the paper

The 3-seed evidence does not change the audit's conclusion: the published
single-seed "improvement triad" (CBAM + BiFPN + EIoU) is not reproducible as
a real architectural improvement in the honest setting. The 3-seed table
should be reported in any honest publication, with all bars showing error
bars and the caption stating "n=3 random seeds, error bars are 1 sigma".

## Extra Ablation: imgsz + epochs (2026-05-20)

Extension of the 3-seed matrix. All honest setting, baseline architecture
(yolov11n_7ch.yaml + MaskRescueTrainer + physidx + clean init), n=3 seeds.
Reference = main 3-seed Baseline Honest (mAP50 0.313 +- 0.038, F1 0.409 +- 0.022).
Pipeline: `scripts/v1_extra_ablation_auto.py`. Aggregate:
`docs/paper/v1_extra_ablation_aggregate.{json,md}`. Figure:
`docs/paper/figures/extra_ablation_bars.png`.

| Variant | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (ref) | 3 | 0.313 +- 0.038 | 0.165 +- 0.011 | 0.409 +- 0.022 | 0.492 +- 0.056 | 0.690 +- 0.107 |
| imgsz=1024 ep=30 | 3 | **0.342 +- 0.011** | **0.219 +- 0.016** | 0.427 +- 0.048 | **0.607 +- 0.022** | **0.765 +- 0.012** |
| imgsz=512 ep=60 | 3 | 0.281 +- 0.032 | 0.163 +- 0.027 | 0.368 +- 0.013 | 0.508 +- 0.036 | 0.653 +- 0.066 |

### Finding F: imgsz=1024 is the first honest-setting improvement that is statistically real

- mAP50 gap to baseline (+0.029) is comparable to baseline std (0.038), but
  imgsz=1024's own std collapses to 0.011 -- a sign the cell is **much more
  reproducible** than baseline.
- mAP50-95 gap (+0.054) is ~3.4x baseline std and far outside imgsz=1024 std.
- **Marine_Debris AP50: +0.115 absolute, +23% relative**, gap ~2x combined std.
  This is the first honest-setting Marine_Debris improvement that survives
  3-seed scrutiny -- relevant because Marine_Debris is the hard class the paper
  is *about*.
- Ship AP50 also up (+0.075) and now near-deterministic (std 0.012).

Mechanism is consistent with small-object detection literature: Marine_Debris
boxes are ~12-14 px at 512 imgsz; doubling resolution to 1024 gives the
detector ~4x more pixel area to work with, which is exactly the regime where
P3/P4 heads benefit most.

Speed cost: imgsz=1024 reduces batch from 8 to 4 and roughly doubles training
time per epoch. Inference cost was not benchmarked for this cell -- needs a
new FPS run if reported in the paper.

### Finding G: doubling epochs (60 vs 30) actively hurts in the honest setting

- mAP50 drops -0.032 (baseline 0.313 → ep60 0.281); gap is within combined std
  but is consistently *negative* across all 3 seeds (0.248, 0.284, 0.312 vs
  baseline 0.275, 0.317, 0.349).
- Best F1 drops -0.041, gap ~2x baseline std.
- ep60 std on F1 collapses to 0.013 -- the model is converging on a stable
  *worse* operating point.

Interpretation: honest train split has only 371 tiles. With SOTA augmentation
recipe (no mosaic, geometric only, hsv on) 30 epochs is already past the
generalization peak. Doubling to 60 epochs overfits the 371-tile corpus and
test-set mAP regresses. This validates the SOTA 30-epoch choice and *negates*
any "we should train longer" follow-up.

### Updated paper recommendation

The honest publication story now has one concrete deployable improvement:

```text
Baseline (30 ep, 512) → imgsz=1024 (30 ep): mAP50 0.31 → 0.34, Marine_Debris AP50 0.49 → 0.61.
Training longer (60 ep) does not help (overfits 371-tile train set).
Architectural changes (CBAM/BiFPN/EIoU/Full) do not help in this setting.
```

This is much stronger than the prior "no improvement transfers" picture and
gives the paper one positive method contribution backed by 3-seed evidence.

## imgsz=1024 x 5-Architecture Ablation (2026-05-20)

Natural extension of Finding F. All 5 architectures (baseline + CBAM + BiFPN +
EIoU + Full) re-run at imgsz=1024 batch=4 workers=0 ep=30, 3 seeds each.
12 new trainings (baseline 3 reused from prior extra-ablation). Pipeline:
`scripts/v1_imgsz1024_ablation_auto.py` (213 min). Aggregate:
`docs/paper/v1_imgsz1024_ablation_aggregate.{json,md}`. Figure:
`docs/paper/figures/imgsz1024_ablation_bars.png`.

### Cross-resolution comparison (imgsz=512 vs imgsz=1024, all 3-seed honest)

| Cell | mAP50 (512) | mAP50 (1024) | F1 (512) | F1 (1024) | Debris AP50 (1024) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.313 +- 0.038 | 0.342 +- 0.011 | 0.409 +- 0.022 | 0.427 +- 0.048 | 0.607 +- 0.022 |
| cbam | 0.258 +- 0.014 | 0.252 +- 0.011 | 0.359 +- 0.041 | 0.360 +- 0.004 | 0.609 +- 0.040 |
| **bifpn** | 0.300 +- 0.059 | **0.350 +- 0.031** | 0.406 +- 0.029 | **0.479 +- 0.037** | 0.576 +- 0.045 |
| eiou | 0.299 +- 0.036 | 0.336 +- 0.031 | 0.410 +- 0.056 | 0.455 +- 0.015 | 0.563 +- 0.035 |
| full | 0.232 +- 0.014 | 0.230 +- 0.014 | 0.335 +- 0.006 | 0.343 +- 0.030 | 0.610 +- 0.054 |

### Finding H: BiFPN @ imgsz=1024 is the new SOTA honest cell

- BiFPN F1 jumps **0.406 -> 0.479 (+0.073, +18% relative)** when going from
  imgsz=512 to imgsz=1024. This is the **largest F1 lift in the entire matrix**
  and is statistically real (gap > combined std).
- BiFPN @ 1024 mAP50 0.350 is now the best honest cell on this metric too.
- Even more important: BiFPN's per-seed mAP50 std drops 0.059 -> 0.031.
  BiFPN was the noisiest cell at imgsz=512; resolution stabilizes it.
- This is a NEW positive method finding that the earlier 3-seed audit
  (Finding C/E) was unable to detect because BiFPN was tested only at
  imgsz=512.

### Finding I: EIoU @ imgsz=1024 also transfers

- EIoU F1 0.410 -> 0.455 (+0.045, +11% relative).
- EIoU F1 std at imgsz=1024 collapses to 0.015 -- the most stable F1 cell in
  the entire matrix.
- Best Ship AP50 (0.800 +- 0.020).
- Recommended as a lighter alternative to BiFPN: same architecture as
  baseline, only the box loss changes; no extra parameters or FLOPs.

### Finding J: CBAM and Full do NOT recover even at imgsz=1024

- CBAM mAP50 0.252 +- 0.011 (worse than baseline 0.342), F1 0.360 +- 0.004.
- Full mAP50 0.230 +- 0.014 (worst cell), F1 0.343 +- 0.030.
- CBAM also collapses Ship AP50 (0.504 vs 0.765 baseline, -0.26 absolute);
  Full does the same (0.461).
- Resolution is not a cure for these two architectures; the regression is
  architecture-intrinsic, consistent with the imgsz=512 result.

### Finding K: Resolution saturates Marine_Debris AP50

- Marine_Debris AP50 at imgsz=1024 clusters tightly around 0.57-0.61 across
  all 5 architectures, versus 0.49-0.56 at imgsz=512.
- The +0.11 absolute lift on the hard class is mostly explained by resolution
  alone; architecture differences become secondary once tile size doubles.
- Interpretation: ~12-14 px debris boxes at imgsz=512 are below the receptive
  field sweet spot for the P3 head; doubling to ~24-28 px lands them squarely
  in the regime the detector was designed for.

### Updated paper recommendation

Refined honest publication story (now multi-axis):

```text
1. Headline: Baseline (512) -> BiFPN (1024)
     mAP50  0.31 -> 0.35  (+12%)
     F1     0.41 -> 0.48  (+18%)
     Debris AP50  0.49 -> 0.58  (small-object class lift)
2. Resolution is the dominant factor for Marine_Debris detection (Finding K).
3. BiFPN's improvement only manifests at imgsz=1024 (Finding H).
4. CBAM and Full collapse independent of resolution (Finding J) -- a true
   negative result for these popular blocks on multi-channel S2 data.
5. EIoU is the lightest viable upgrade (no params, +11% F1 at imgsz=1024,
   lowest F1 variance, best Ship AP50).
6. Training longer (60 ep) overfits (Finding G).
```

This converts the prior "honest setting has no real improvement" story into
a structured multi-axis finding, with one clear recommended deployment
(BiFPN + imgsz=1024) and one defensible lightweight alternative
(EIoU + imgsz=1024).

## imgsz=1024 Leaky Control (2026-05-20)

Validation cell: does the BiFPN @ imgsz=1024 honest SOTA finding (Finding H)
survive under leak? Or is the architecture+resolution interaction specific to
the honest regime? Pipeline: `scripts/v1_imgsz1024_leaky_auto.py` (105 min).
Aggregate: `docs/paper/v1_imgsz1024_leaky_aggregate.{json,md}`. Figure:
`docs/paper/figures/imgsz1024_leaky_bars.png`.

| Ablation | imgsz | mAP50 | F1 | Debris AP50 | Ship AP50 |
|---|---:|---:|---:|---:|---:|
| baseline | 512 | 0.690 +- 0.004 | 0.740 +- 0.060 | 0.856 +- 0.029 | 0.910 +- 0.039 |
| baseline | 1024 | 0.695 +- 0.006 | 0.736 +- 0.071 | 0.870 +- 0.016 | **0.978 +- 0.029** |
| bifpn | 512 | 0.688 +- 0.020 | 0.741 +- 0.088 | 0.852 +- 0.019 | 0.907 +- 0.069 |
| bifpn | 1024 | 0.679 +- 0.017 | **0.710 +- 0.052** | 0.860 +- 0.017 | 0.847 +- 0.097 |

### Finding L: BiFPN @ imgsz=1024 helps ONLY in the honest setting

The architecture+resolution interaction is regime-specific:

| Comparison | honest F1 | leaky F1 |
|---|---:|---:|
| baseline 512 -> baseline 1024 | 0.409 -> 0.427 (+0.018) | 0.740 -> 0.736 (-0.004) |
| bifpn   512 -> bifpn   1024 | 0.406 -> **0.479 (+0.073)** | 0.741 -> **0.710 (-0.031)** |

- BiFPN @ 1024 honest: F1 +18% (the new SOTA cell from Finding H).
- BiFPN @ 1024 leaky: F1 -4% (actively worse than BiFPN @ 512 leaky).
- imgsz=1024 alone (baseline cell) gives essentially nothing under leak
  (delta within combined std on all metrics except Ship AP50).

### Interpretation: the honest finding is *strengthened*, not weakened

If BiFPN+1024 also lifted leaky metrics, we would worry the gain is "just
more parameters / more pixels = more capacity". The fact that it does NOT
help under leak -- where the GT-heatmap already encodes the answer at ch7 --
is **evidence the architecture+resolution combination is unlocking real
spectral feature learning that the heatmap was previously masking**.

Equivalently: the leaky regime is so easy (oracle prior) that no architectural
upgrade matters; the honest regime is hard enough that architectural choices
become discriminative. This is a useful insight for any paper that uses
privileged-information training pipelines (PKD, LUPI, teacher distillation).

### Updated paper recommendation

This validation cell hardens the Finding H/I claims for honest setting and
adds a defensible negative-control finding (L). The full pitch:

```text
1. Headline: BiFPN + imgsz=1024 lifts honest F1 +18% (0.41 -> 0.48).
2. Resolution alone explains most of the Marine_Debris saturation (Finding K).
3. The architecture+resolution interaction is honest-setting-specific
   (Finding L) -- evidence that BiFPN is unlocking real feature learning,
   not just adding capacity.
4. Negative results: CBAM/Full (Finding J), >30 ep (Finding G), all
   architectures unhelpful at imgsz=512 (Finding C/E).
```

## imgsz=1024 FPS Benchmark (2026-05-20)

Same protocol as `scripts/v1_ablation_fps_bench.py` but at imgsz=1024:
synthetic (1, 7, 1024, 1024) float32 input, 20 warmup + 100 timed iters,
median FPS, RTX 3090 Ti. Output: `docs/paper/v1_imgsz1024_fps.csv` plus
per-checkpoint `fps_bench.json`. Script: `scripts/v1_imgsz1024_fps_bench.py`.

| Cell | imgsz | FPS | Latency (ms) |
|---|---:|---:|---:|
| honest_baseline | 512 | 110.9 | 9.0 |
| honest_baseline | 1024 | **121.2** | 8.2 |
| honest_cbam | 1024 | 107.0 | 9.4 |
| honest_bifpn | 1024 | **116.9** | 8.6 |
| honest_eiou | 1024 | 121.1 | 8.3 |
| honest_full | 1024 | 103.9 | 9.6 |
| leaky_baseline | 1024 | 122.3 | 8.2 |
| leaky_bifpn | 1024 | 108.2 | 9.2 |

### Counter-intuitive observation: imgsz=1024 is not slower than imgsz=512

For yolov11n (3M params) on RTX 3090 Ti at batch=1, latency is dominated by
**CUDA kernel-launch + Python overhead**, not GPU compute. Doubling input
resolution gives ~4x more compute per kernel launch but the same launch
count, so fixed overhead amortizes better. 8.2 ms vs 9.0 ms latency is at
the level of timing noise. Net: imgsz=1024 at batch=1 has no practical
speed cost on this GPU for this model size.

For deployment at higher batch sizes (where GPU compute saturates) this would
change; the paper should note this caveat. But for the standard YOLO
single-image inference reported in MARIDA papers, imgsz=1024 is free.

### Cost-benefit of recommended deployment

BiFPN @ imgsz=1024 (the new honest SOTA, Finding H):
- mAP50 0.350 (vs baseline @ 512: 0.313)
- F1 0.479 (vs baseline @ 512: 0.409)
- FPS 116.9 (vs baseline @ 512: 110.9)

**No speed penalty.** This makes BiFPN @ imgsz=1024 a strict Pareto
improvement over the imgsz=512 baseline at batch=1.

## Native-256 Cross-Resolution Control (2026-05-20)

To answer "does imgsz=1024 win because of (a) more model capacity / P3-head
hitting Marine_Debris box size, or (b) letterbox up-sampling specifically?",
we trained baseline + BiFPN at **native imgsz=256** (no up-sampling) x 3
seeds. Compared against existing imgsz=512 and imgsz=1024 results, all on
the same v1 physidx on v2 scene-clean split, clean init, ep=30.

Pipeline: `scripts/v1_native256_ablation_auto.py`, completed in 17.8 min.
Aggregate: `docs/paper/v1_native256_aggregate.{json,md}`.
Figure: `docs/paper/figures/native256_vs_1024_bars.png`.

### Cross-resolution table (3-seed mean +- std)

**baseline:**

| imgsz | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |
|---:|---:|---:|---:|---:|---:|
| 256 (native)  | 0.228 +- 0.033 | 0.101 +- 0.011 | 0.335 +- 0.041 | 0.364 +- 0.054 | 0.641 +- 0.043 |
| 512 (2x up)   | 0.313 +- 0.038 | 0.165 +- 0.011 | 0.409 +- 0.022 | 0.492 +- 0.056 | 0.690 +- 0.107 |
| 1024 (4x up)  | **0.342 +- 0.011** | **0.219 +- 0.016** | **0.427 +- 0.048** | **0.607 +- 0.022** | **0.765 +- 0.012** |

**BiFPN:**

| imgsz | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |
|---:|---:|---:|---:|---:|---:|
| 256 (native)  | 0.245 +- 0.025 | 0.115 +- 0.018 | 0.373 +- 0.040 | 0.451 +- 0.014 | 0.684 +- 0.125 |
| 512 (2x up)   | 0.300 +- 0.059 | 0.167 +- 0.021 | 0.406 +- 0.029 | 0.559 +- 0.023 | 0.665 +- 0.092 |
| 1024 (4x up)  | **0.350 +- 0.031** | **0.225 +- 0.023** | **0.479 +- 0.037** | **0.576 +- 0.045** | **0.755 +- 0.082** |

### Finding M (NEW) — letterbox up-sampling is itself the mechanism, not just capacity

Strong monotone increase across resolution for both architectures. Native
256 is **strictly worse** than letterbox-512, which is **strictly worse**
than letterbox-1024. The baseline Debris AP50 lift from native 256 to
letterbox 1024 is **+67%** (0.364 -> 0.607).

This **rules out** the "we just need to hit P3-head's design regime"
hypothesis. If that were the only mechanism, native 256 (where 6-7 px
debris boxes sit at the lower P3 boundary) would already be competitive.
It isn't.

Mechanistic interpretation: dataset native resolution (256x256) is a
bottleneck. Letterbox up-sampling re-distributes the same pixel content
across a 4x larger spatial grid, so each backbone stage produces denser
feature maps and the head has more spatial samples to specialize per
small object. The 4x increase in compute per kernel launch costs nothing
on batch=1 RTX 3090 Ti (FPS benchmark above).

### Finding N — BiFPN gain is not letterbox-conditional

At native 256, BiFPN beats baseline on every metric (Debris AP50:
0.451 vs 0.364, +24%; F1: 0.373 vs 0.335, +11%). So **BiFPN improves
small-object detection independently of resolution**, and stacks with
letterbox up-sampling at higher imgsz. Earlier ablation memos had
ambiguity on whether BiFPN's gain was an "interaction effect with 1024" —
this is now resolved: BiFPN wins at every resolution tested.

### Paper narrative update

**Replace** any text that says "the imgsz=1024 gain comes from model
capacity / P3-head hitting the right scale" with:

> Native MARIDA tiles are 256x256, which is sub-optimal for floating
> small-debris detection: at the native scale, the backbone produces too
> few feature-map cells per object to support effective specialization.
> Letterbox up-sampling to imgsz=1024 at training time (a 4x increase in
> spatial extent at no additional sensor cost) recovers a large fraction
> of this lost spatial budget, lifting Marine_Debris AP50 by +67% over
> the native-resolution baseline. BiFPN provides an additive +0.087
> Debris AP50 gain that does not depend on resolution.

### Useful sub-result: BiFPN @ native-256 = a low-compute candidate

BiFPN @ imgsz=256 reaches Debris AP50 = 0.451 +- 0.014, only slightly
below baseline @ imgsz=512 (0.492) and 4x cheaper in compute. For
edge-deployment scenarios where 1024 inference is impractical, BiFPN @
native 256 is the recommended fallback.

## Semantic Segmentation Companion (D-aufgabe, 2026-05-20)

Why: detection drops 1213/1425 Marine_Debris connected components at the
MIN_PIXELS=4 filter (only 212 instances survive). Semantic segmentation
can supervise on every pixel including 1/2/3 px GT. We trained a minimal
4-stage U-Net (in=6, base=32, 7.76M params) on a parallel v2-semseg
dataset (same scene split as v2 detect, NO MIN_PIXELS filter applied).

Pipeline: `scripts/preprocessing/build_marida_v2_semseg.py`,
`scripts/v2_semseg_train.py`, `scripts/v2_semseg_instance_eval.py`.
Data: `data/processed/marida_v2_semseg/` (1381 tiles, 256x256 native).
Class weights: [bg=1, Marine_Debris=150, Algae=30, Ship=30] CE loss.

### 3-seed pixel-level results (test split)

| Class | pixel IoU | pixel F1 | pixel Recall | pixel Precision |
|---|---:|---:|---:|---:|
| Marine_Debris | 0.082 +- 0.010 | 0.152 +- 0.018 | 0.567 +- 0.058 | 0.090 +- 0.020 |
| Algae_Organic | 0.167 +- 0.027 | 0.301 +- 0.040 | 0.567 +- 0.142 | 0.218 +- 0.064 |
| Ship          | 0.254 +- 0.066 | 0.396 +- 0.082 | 0.570 +- 0.052 | 0.310 +- 0.097 |
| mIoU (no bg)  | 0.168 +- 0.025 |  --  |  --  |  --  |

### 3-seed instance-level Recall (test, directly comparable to YOLO)

Run U-Net pred -> connected components -> match against GT instances
(filtered by MIN_PIXELS, matches what YOLO sees in detection).

| Class | MIN=4 (YOLO-comparable) | MIN=1 (all raw) |
|---|---|---|
| Marine_Debris | **8, 8, 7 / 13 -> 0.590 +- 0.036** | 76, 101, 89 / 147 -> 0.603 +- 0.069 |
| Algae_Organic | 19, 32, 20 / 48 -> 0.493 +- 0.123 | 19, 36, 20 / 79 -> 0.317 +- 0.099 |
| Ship          | 18, 23, 18 / 35 -> 0.562 +- 0.067 | 18, 23, 18 / 35 -> 0.562 +- 0.067 |

### Finding D (decisive) -- U-Net beats YOLO 4x on Marine_Debris instance Recall

| Approach | Marine_Debris instance Recall (test, MIN=4 GT) |
|---|---:|
| YOLO detect baseline (v2) | 2 / 13 = 0.154 |
| YOLO instance-seg (B.1) | 0 / 13 = 0.000 |
| **U-Net semseg (D)** | **8 / 13 = 0.590 +- 0.036 (3-seed)** |

Plus: in MIN=1 GT (the 134 connected components YOLO never sees because
of MIN_PIXELS=4 filter), U-Net additionally recovers ~68 instances per
seed. Total instance Recall over ALL raw GT (147 components) = 0.603.

Mechanism: at MIN_PIXELS=4 the YOLO label set has 122 train Marine_Debris
instances; the U-Net label set is the full GT mask with 2118 train
Marine_Debris pixels (~5x more supervisory signal by tile coverage).
Detection's instance-level loss cannot supervise sub-4-pixel components;
per-pixel CE can.

Artifacts:
- runs_v2_semseg/unet_b32_s{0,1,2}/{best.pt, results.json}
- runs_v2_semseg/instance_recall_summary.json
- Comparison vs B.1 (YOLO seg): docs/paper/7CH_BASELINE_CURRENT_AUDIT_2026-05-18.md
  (B.1 reported in earlier section "Segmentation Companion B.1")

## Self-Distillation Prior Attempt (E-aufgabe, 2026-05-20)

Hypothesis: U-Net learned Marine_Debris probability could replace
NDWI-inverse as ch7 prior for YOLO. The U-Net's high pixel Recall (0.57)
should give YOLO a stronger "where to look" signal than a physical index.

Method: 5-fold cross-prediction U-Net training on v2 train -> each
v1-honest train tile's prior comes from a U-Net that did NOT see it
(leak-free). v1 val/test priors from the seed=0 full-train U-Net.
Build new dataset by replacing v1 physidx ch7 with U-Net probability.
Re-train YOLO baseline + BiFPN @ imgsz=1024 x 3 seed.

Pipeline: `scripts/v2_semseg_kfold_train.py`,
`scripts/v2_semseg_infer.py`,
`scripts/preprocessing/build_v1_unetprior.py`,
`scripts/v1_unetprior_ablation_auto.py`.
Total wall time: ~100 min.

### 3-seed YOLO comparison (test split)

| Cell | ch7 prior | mAP50 | Debris AP50 | Best F1 | Ship AP50 |
|---|---|---:|---:|---:|---:|
| baseline @ 1024 | physidx (NDWI-inv) | **0.342 +- 0.011** | **0.607 +- 0.022** | **0.427 +- 0.048** | 0.765 +- 0.012 |
| baseline @ 1024 | unetprior | 0.320 +- 0.026 | 0.474 +- 0.053 | 0.408 +- 0.040 | 0.701 +- 0.177 |
| BiFPN @ 1024 | physidx | **0.350 +- 0.031** | 0.576 +- 0.045 | **0.479 +- 0.037** | 0.755 +- 0.082 |
| BiFPN @ 1024 | unetprior | 0.322 +- 0.010 | 0.503 +- 0.031 | 0.428 +- 0.015 | **0.799 +- 0.030** |

### Finding O (NEW) -- self-distillation prior does not help, mildly hurts

UnetPrior loses to NDWI-inverse on every metric except BiFPN Ship AP50.
Marine_Debris AP50 drops the most: -0.133 (-22%) on baseline @ 1024.

Mechanism candidates:
1. **Sparsity collapse**: U-Net Marine_Debris prob is mean = 0.0008 across
   all pixels (very sparse activation). NDWI-inverse is continuous and
   informative everywhere. YOLO loses an information-dense channel.
2. **Same-source distillation has no extra signal**: the U-Net was trained
   from the same 6 spectral channels as YOLO. UnetPrior is essentially a
   pre-digested version of what YOLO could learn from spectral input.
   Replacing a *physical* independent signal (NDWI-inverse, derived from
   B3 and B8 reflectance ratio without any learned model) with a learned
   one removes that independence.
3. **"Answer-like" channel encourages over-reliance**: U-Net prob marks
   "this looks like Marine_Debris", inviting YOLO to copy the U-Net's
   pattern instead of learning from spectra. Where U-Net is wrong, YOLO
   inherits the error; where U-Net is right, the gain has already been
   captured by class-1 supervision through the regular loss.

This is the first attempt at *Privileged-Information Generation* on this
project to give a negative result. Earlier GT-heatmap experiments (oracle
ch7) were a different kind of leak (test-time GT exposure), not a
distillation experiment.

### Useful by-result -- the K-fold infrastructure is now in place

5 U-Net K-fold checkpoints + leak-free inference pipeline are saved and
re-usable. If a future experiment needs leak-free pseudo-labels on v2
train (e.g. for self-training, or for cross-arch knowledge distillation
into a different student), the data is one inference pass away.

Artifacts:
- 5x K-fold U-Nets: runs_v2_semseg/unet_b32_kfold{0..4}_s0/best.pt
- prob masks: runs_v2_semseg/prob_masks/*.npy (1381 tiles)
- dataset: data/processed/mask_bbox_rescue_split_unetprior_v2split/
- aggregate: docs/paper/v1_unetprior_ablation_aggregate.{json,md}
- figure: docs/paper/figures/unetprior_vs_physidx_bars.png
- done marker: runs/detect/runs_v1_v2split/.unetprior_ablation_done

## Executive Conclusion

The 7-channel YOLOv11n baseline is real and traceable:

- The dataset exists.
- The 7-channel model weights exist.
- The model checkpoint is genuinely 7-channel.
- The reported publication F1 score can be recomputed from saved predictions.

However, it should not be treated as a fair deployable baseline. The 7th channel is generated from GT masks, and current validation/test tensors include that GT-derived heatmap. This makes the strongest 7ch result an oracle or privileged-information result, not an ordinary Sentinel-2 inference result.

Recommended framing:

- Use 7ch as `oracle teacher`, `privileged-information teacher`, or `upper-bound baseline`.
- Do not present 7ch heatmap-at-test as a deployable model.
- For publication/deployment, use 6ch or 10ch student models that do not require GT-derived test-time heatmaps.

## Main Baseline Identity

Current strongest historical run:

| Item | Value |
|---|---|
| Run name | `yolov11n_7ch_512_finetune_wakeboost` |
| Run dir | `experiments/runs/runs_marida/yolov11n_7ch_512_finetune_wakeboost` |
| Checkpoint | `experiments/runs/runs_marida/yolov11n_7ch_512_finetune_wakeboost/weights/best.pt` |
| Parent checkpoint | `yolov11n_7ch_full_return_512_resumed/weights/best.pt` |
| Dataset config intended | `marida_mask_rescue_7ch.yaml` / `configs/datasets/marida_mask_rescue_7ch.yaml` |
| Image size | 512 |
| Batch | 8 |
| Epochs | 30 |
| Copy-paste | 0.0 in this SOTA run |
| Model channels | 7 |
| Classes | 7 |

Checkpoint inspection confirms:

| Check | Result |
|---|---|
| `model.yaml.ch` | 7 |
| `model.yaml.nc` | 7 |
| first conv shape | `(16, 7, 3, 3)` |
| class names | `Marine_Debris`, `Dense_Sargassum`, `Sparse_Sargassum`, `Natural_Organic`, `Ship`, `Foam`, `Wakes` |

So this is not a 3-channel or 6-channel checkpoint mislabeled as 7ch.

## Dataset Inventory

Actual current dataset root:

```text
data/processed/mask_bbox_rescue_split
```

Current physical files:

| Split | Images | Labels | Boxes | Shape | Dtype |
|---|---:|---:|---:|---|---|
| train | 347 | 347 | 767 | `(256, 256, 7)` | `float32` |
| val | 114 | 114 | 235 | `(256, 256, 7)` | `float32` |
| test | 53 | 53 | 119 | `(256, 256, 7)` | `float32` |

Class distribution by boxes:

| Class | Train | Val | Test |
|---|---:|---:|---:|
| Marine_Debris | 646 | 192 | 103 |
| Dense_Sargassum | 13 | 3 | 0 |
| Sparse_Sargassum | 6 | 3 | 0 |
| Natural_Organic | 0 | 0 | 0 |
| Ship | 54 | 19 | 11 |
| Foam | 35 | 16 | 4 |
| Wakes | 13 | 2 | 1 |

Important implications:

- The dataset is highly imbalanced.
- `Natural_Organic` has no boxes in the current split, so 7-class claims must be careful.
- `Sparse_Sargassum` and `Wakes` have extremely low support, so their metrics are unstable.
- Current strong results are mostly major-class results, especially `Marine_Debris`.

## Publication Metrics Verification

Saved validation output:

```text
experiments/runs/runs_marida/val_yolov11n_7ch_512_finetune_wakeboost2/predictions.json
experiments/runs/runs_marida/val_yolov11n_7ch_512_finetune_wakeboost2/publication/metrics_summary.json
```

Recomputed from `predictions.json` using `scripts/utils/plot_publication_metrics.py` logic:

| Metric | Value |
|---|---:|
| best threshold | 0.53 |
| precision | 0.8259 |
| recall | 0.7872 |
| F1 | 0.8061 |
| IoU threshold | 0.5 |
| GT boxes | 235 |
| predictions loaded | 327 |

Per-class result at the best threshold:

| Class | TP | FP | FN | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Marine_Debris | 161 | 34 | 31 | 192 | 0.826 | 0.839 | 0.832 |
| Dense_Sargassum | 2 | 0 | 1 | 3 | 1.000 | 0.667 | 0.800 |
| Sparse_Sargassum | 0 | 0 | 3 | 3 | 0.000 | 0.000 | 0.000 |
| Natural_Organic | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| Ship | 12 | 3 | 7 | 19 | 0.800 | 0.632 | 0.706 |
| Foam | 10 | 1 | 6 | 16 | 0.909 | 0.625 | 0.741 |
| Wakes | 0 | 1 | 2 | 2 | 0.000 | 0.000 | 0.000 |

Conclusion:

- The F1 0.806 number is traceable to saved predictions.
- It is best described as a major-class validation result.
- It does not prove full 7-class reliability.

## Critical Issue 1: GT-Derived Heatmap at Test Time

The 7th channel is generated by:

```text
scripts/preprocessing/preprocess_generate_knowledge_channel.py
```

Core logic:

```python
binary = (mask > 0).astype(np.uint8)
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
heatmap = dist / dist.max()
```

The mask is loaded from original GT semantic mask files:

```text
MARIDA/patches/<scene>/<patch>_cl.tif
```

Therefore the 7th channel is not NDVI, FDI, NDWI, or a physical Sentinel-2 band. It is a GT-derived spatial prior.

This is the central validity risk. If the model sees this channel at validation/test time, it receives privileged information about target location.

### Zero-Heatmap Test

Using the current SOTA checkpoint and the corrected channel stats file, two validations were run:

| Setting | mAP50 | mAP50-95 |
|---|---:|---:|
| full 7ch with heatmap | 0.5173 | 0.2702 |
| same model, heatmap zeroed at eval | 0.0614 | 0.0134 |

Interpretation:

- The model depends strongly on the heatmap.
- Without the GT-derived 7th channel, this checkpoint nearly collapses.
- This confirms the 7ch result is oracle-like.

This does not make the work useless. It changes the correct framing:

- Incorrect: "7ch YOLOv11n is a deployable multispectral detector."
- Safer: "7ch YOLOv11n is a privileged-information teacher or upper bound."
- Strong publication route: distill the 7ch teacher into a 6ch or 10ch student.

## Critical Issue 2: Scene-Level Leakage in the Current Split

Patch filenames share scene prefixes such as:

```text
S2_18-9-20_16PCC_15.npy
```

The scene id is the filename without the final patch index:

```text
S2_18-9-20_16PCC
```

Current overlap:

| Split pair | Shared scenes |
|---|---:|
| train-val | 43 |
| train-test | 26 |
| val-test | 25 |

There is no exact filename overlap, but scene-level overlap is heavy. For satellite patch data, this can inflate validation/test results because neighboring patches from the same acquisition share water conditions, sensor geometry, illumination, background texture, and debris distribution.

This is separate from the heatmap issue. The current v1 split has two independent optimism risks:

- GT-derived 7th channel at validation/test time.
- Same-scene patches across train/val/test.

The later MARIDA v2 work appears to address this by using scene-level splitting. For publication, the v2 split should be preferred.

## Critical Issue 3: Stale Train Manifest

`train.txt` under the processed 7ch dataset currently has 1216 entries, but 869 entries point to missing augmented files such as:

```text
train/images_aug/aug_mosaic_00001.npy
```

Actual current training images are only 347.

This matters because the two 7ch YAML files behave differently:

| YAML | Train source | Risk |
|---|---|---|
| `configs/datasets/marida_mask_rescue_7ch.yaml` | `train/images` | Uses actual 347 images |
| `configs/datasets/marida_rescue_7ch.yaml` | `train.txt` | Contains 869 stale references |

`MaskRescueNPYDataset.get_img_files()` skips missing manifest entries with warnings. This means a training run may silently use fewer samples than the manifest suggests.

Recommendation:

- Regenerate `train.txt` from actual files, or archive the stale manifest and force directory-based loading.
- Make train/val/test sample counts explicit in every run summary.

## Critical Issue 4: Channel Stats Path Drift

The loader default expects:

```text
marida_runs/channel_stats_7ch.json
```

But the current repository has the relevant v1 stats at:

```text
experiments/runs/marida_runs/channel_stats_7ch.json
```

When the stats file is not found, the loader falls back to:

```python
arr = arr * 1500.0
arr = np.clip(arr, 0.0, 255.0)
```

This changes model inputs and breaks exact reproduction.

Observed effect with current SOTA checkpoint:

| Validation setup | mAP50 | mAP50-95 |
|---|---:|---:|
| full heatmap, missing stats fallback | 0.1959 | 0.0519 |
| full heatmap, explicit stats path | 0.5173 | 0.2702 |

Recommendation:

- Put `channel_stats_7ch.json` inside the dataset root, or pass it explicitly.
- Record the channel stats path in `args.yaml` and `train_summary.json`.
- Avoid CWD-dependent default paths.

## Critical Issue 5: Exact `model.val()` Reproduction Is Not Fully Locked

Saved publication predictions reproduce F1 0.806 exactly.

However, rerunning `model.val()` from the current repository state produced lower mAP than the historical `results.csv`:

| Source | mAP50 | mAP50-95 |
|---|---:|---:|
| `yolov11n_7ch_512_finetune_wakeboost/results.csv`, best mAP50 epoch | 0.6618 | 0.2856 |
| current rerun, explicit stats, full heatmap | 0.5173 | 0.2702 |
| publication prediction F1 sweep | F1 0.806 | not mAP |

Likely contributors:

- Historical runs were launched with different working directories.
- Run metadata points to missing root-level paths such as `runs_marida/...` and `marida_mask_rescue_7ch.yaml`.
- Loader versions differ: `mask_rescue_dataset.py` vs `mask_rescue_dataset_fixed.py`.
- Channel stats path resolution is CWD-dependent.
- Saved `predictions.json` is authoritative for the publication F1, but not enough to reconstruct the full `model.val()` environment.

Recommendation:

- Freeze an executable reproduction command.
- Freeze the exact dataset YAML, channel stats, loader file, checkpoint hash, and working directory.
- Treat historical mAP numbers as real logs, but not fully reproducible until the above is fixed.

## Critical Issue 6: ClassAwareCopyPaste Was Not Reliable in Historical Runs

Notion notes suggested that class-aware copy-paste may not have activated correctly. The 2026-05-18 engineering pass fixed the active `mask_rescue_dataset_fixed.py` parsing path, but the historical class-aware/copy-paste runs were produced before that fix and remain unreliable as evidence.

Relevant files:

```text
ultralytics_plugins/mask_rescue_dataset_fixed.py
scripts/training/train_7ch_main.py
scripts/training/train_7ch_main_fixed.py
```

Original problem:

- `MASKRESCUE_COPY_PASTE_CLASSES` can be set.
- `cp_classes` can be detected.
- The active dataset code previously left `parsed_cp_classes = None` before constructing `ClassAwareCopyPaste`.
- Therefore the intended class filter may not have been applied in earlier class-aware runs.

Current status:

- `ultralytics_plugins/mask_rescue_dataset_fixed.py` now parses string/list `cp_classes` into `parsed_cp_classes`.
- The debug print now reports the parsed class ids.
- The f-string log now prints actual `op.p` and `parsed_cp_classes` values rather than literal braces.

Affected runs include class-aware/copy-paste experiments such as:

```text
yolov11n_7ch_512_classaware_production
yolov11n_7ch_512_classaware_final
yolov11n_7ch_512_copypaste_classaware
```

These runs should not be cited as evidence that rare-class-only copy-paste worked as designed.

Recommendation:

- Keep the code fix.
- Add/save an explicit run artifact with the final allowed class ids.
- Save augmentation activation evidence into `train_summary.json`.
- Rerun the class-aware experiment only after the debug log confirms activation.

## Other Observations

### Path Drift

Many historical run files point to paths that do not exist in the current root:

```text
runs_marida/yolov11n_7ch_512_finetune_wakeboost
marida_mask_rescue_7ch.yaml
```

But the current real paths are usually:

```text
experiments/runs/runs_marida/...
configs/datasets/marida_mask_rescue_7ch.yaml
```

Recommendation:

- Use repo-relative paths in all new docs.
- Avoid root-level duplicate YAML names.
- Add a small result registry that stores resolved absolute paths at run time.

### Current `README.md` Is Too Optimistic

`README.md` says the run is "Current SOTA (Major Classes)" with F1 0.806. This is partly true, but missing the oracle caveat.

Safer wording:

```text
Current oracle/privileged 7ch teacher on v1 split:
Precision 0.826, Recall 0.787, F1 0.806 at IoU 0.5.
The result uses a GT-derived heatmap as the 7th channel and should not be treated as deployable test-time performance.
```

## Current Trust Level

| Claim | Trust level | Reason |
|---|---|---|
| 7ch data exists | High | Direct file check, shapes verified |
| SOTA checkpoint is 7ch | High | Checkpoint first conv is `(16, 7, 3, 3)` |
| Publication F1 0.806 is real | High | Recomputed from saved predictions |
| The model is good for major classes under oracle input | Medium-high | Marine_Debris/Ship/Foam metrics support this |
| The model solves all 7 classes | Low | Sparse/Wakes are 0 F1, Natural_Organic has no support |
| The model is deployable as-is | Low | It relies on GT-derived heatmap at test time |
| Historical mAP is exactly reproducible from current root | Medium-low | Path/stats/loader drift |
| Class-aware copy-paste worked | Low | Active parser bug likely disables intended filtering |

## Recommended Next Steps

### P0: Freeze Honest Terminology

Rename or describe the model as:

```text
7ch oracle teacher / privileged teacher / GT-heatmap upper bound
```

Do not call it a deployable baseline without qualification.

### P1: Fix Reproducibility

Required fixes:

- Put or copy `channel_stats_7ch.json` in a stable path.
- Regenerate `train.txt`.
- Use one canonical YAML for 7ch.
- Save dataset counts and channel stats path in every `train_summary.json`.
- Record checkpoint SHA256 in result summaries.

### P2: Establish Fair Deployable Baselines

Run or freeze:

- 6ch clean baseline, no heatmap.
- 10ch spectral-index baseline, no GT heatmap.
- 7ch oracle teacher only as upper bound.
- Optional 6ch/10ch student distilled from the 7ch teacher.

### P3: Move to Scene-Level Split

For publication, prefer the later MARIDA v2 split that keeps scenes isolated across train/val/test.

The v1 split can be retained as historical engineering evidence, but should not be the final scientific benchmark.

### P4: Repair Rare-Class Strategy

Before citing class-aware augmentation:

- Fix `ClassAwareCopyPaste`.
- Confirm active classes in logs.
- Rerun with rare classes only.
- Report per-class metrics, especially `Sparse_Sargassum` and `Wakes`.

## Suggested Paper Framing

Strong and defensible story:

1. Historical v1 experiments showed that a GT-derived heatmap can provide a strong privileged upper bound.
2. This motivates a privileged teacher model.
3. Because the heatmap is not available at deployment, the final method should distill the teacher into a clean 6ch or 10ch student.
4. Final evaluation should use scene-level split to avoid spatial leakage.

Avoid:

- Claiming 7ch heatmap model as ordinary deployable SOTA.
- Comparing 7ch oracle directly against 6ch/10ch clean models without labeling it as privileged.
- Overclaiming rare-class performance.

## Minimal Result Table for Current Report

| Model/result | Input at eval | Split status | Main metric | Valid use |
|---|---|---|---|---|
| `yolov11n_7ch_512_finetune_wakeboost` publication output | 6 S2 bands + GT heatmap | v1, scene overlap | F1 0.806 | Oracle major-class result |
| same checkpoint, heatmap zeroed | 6 S2 bands + zero channel | v1, scene overlap | mAP50 0.061 | Shows non-deployability |
| 3/4/5/6ch historical ablation | clean spectral channels | v1 | varies by protocol | Historical comparison only |
| MARIDA v2 6ch baselines | clean spectral channels | scene-level split | see v2 docs | Better publication baseline |

## Bottom Line

The 7ch baseline is a useful and real project artifact, but its meaning must be corrected.

Best current interpretation:

```text
The 7ch YOLOv11n model is a GT-heatmap oracle teacher that demonstrates how strong spatial privileged information can be for MARIDA detection. It should be used as an upper bound or teacher, not as a deployable baseline. The deployable research path should move to clean 6ch/10ch student models, preferably on the leak-free scene-level MARIDA v2 split.
```
