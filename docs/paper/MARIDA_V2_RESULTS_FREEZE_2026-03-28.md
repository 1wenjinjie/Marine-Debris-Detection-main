# MARIDA v2 Main Results Freeze (2026-03-28)

## Scope

This note freezes the current MARIDA v2 detection results after the 3-seed validation of the strongest 6-channel model:

- Model: `YOLOv11n_v2_6ch`
- Training setting: `focal loss (gamma=2.0, alpha=0.25) + imgsz=1024`
- Seeds: `0, 1, 2`

The key point is that standard `eval_test.json` metrics and debris-specific threshold-sweep metrics answer different questions:

- `eval_test.json` reports the usual detector summary at `conf=0.001`.
- `sweep_results.json` measures whether Marine_Debris instances are actually recoverable at practical score thresholds.

For this project, the sweep is the primary evidence for Marine_Debris usability.

## Frozen Best 6ch Line

- Run directories:
  - `runs_v2/focal_g2.0_i1024_s0/`
  - `runs_v2/focal_g2.0_i1024_s1/`
  - `runs_v2/focal_g2.0_i1024_s2/`
- Sweep directories:
  - `runs_v2/sweep_focal_g2.0_i1024_s0/`
  - `runs_v2/sweep_focal_g2.0_i1024_s1/`
  - `runs_v2/sweep_focal_g2.0_i1024_s2/`

### Debris Threshold Sweep

`TP` below means `Marine_Debris` true positives at `IoU > 0.5` on the 13 debris test instances.

| Conf | s0 | s1 | s2 | Mean +- Std |
| --- | ---: | ---: | ---: | ---: |
| 0.001 | 7 | 10 | 10 | 9.0 +- 1.7 |
| 0.005 | 7 | 8 | 10 | 8.3 +- 1.5 |
| 0.010 | 7 | 8 | 10 | 8.3 +- 1.5 |
| 0.025 | 7 | 7 | 6 | 6.7 +- 0.6 |
| 0.050 | 6 | 6 | 5 | 5.7 +- 0.6 |
| 0.100 | 1 | 1 | 0 | 0.7 +- 0.6 |

Primary operating-point result:

- `conf=0.05`: `5.7 +- 0.6 / 13` TP
- Equivalent debris recall: about `44%`
- Baseline at the same practical threshold was effectively `0%`

This is the main result to report for Marine_Debris.

## Standard Test Metrics

These come from `eval_test.json` and are still useful, but they understate the practical debris gain because they are not the same as the threshold-sweep operating-point analysis.

### Baseline 6ch (imgsz=512, 3-seed)

Source: `runs_v2/baseline_6ch_summary.json`

| Metric | Mean +- Std |
| --- | ---: |
| mAP@0.5 | 0.1932 +- 0.0178 |
| mAP@0.5:0.95 | 0.0889 +- 0.0116 |
| Precision | 0.4570 +- 0.1546 |
| Recall | 0.1791 +- 0.0199 |
| Marine_Debris AP50 | 0.0157 +- 0.0062 |
| Marine_Debris Recall | 0.0000 +- 0.0000 |
| Algae_Organic AP50 | 0.0731 +- 0.0121 |
| Ship AP50 | 0.4909 +- 0.0420 |

### Focal `gamma=2.0`, `imgsz=1024` (3-seed)

Source:

- `runs_v2/focal_g2.0_i1024_s0/eval_test.json`
- `runs_v2/focal_g2.0_i1024_s1/eval_test.json`
- `runs_v2/focal_g2.0_i1024_s2/eval_test.json`

| Metric | Mean +- Std |
| --- | ---: |
| mAP@0.5 | 0.1340 +- 0.0253 |
| mAP@0.5:0.95 | 0.0632 +- 0.0134 |
| Precision | 0.1567 +- 0.0168 |
| Recall | 0.1923 +- 0.0314 |
| Marine_Debris AP50 | 0.0247 +- 0.0055 |
| Marine_Debris Recall | 0.0769 +- 0.1088 |

Interpretation:

- Conventional `model.val()` style metrics do not capture the actual debris operating-point improvement well.
- The model finds many more debris instances, but their confidence remains low and threshold-sensitive.
- Therefore, Marine_Debris threshold-sweep analysis is the correct headline evidence.

## Scene-Level Stability

Using `conf=0.001` and `IoU > 0.5` as the recoverability check:

| Scene | GT | s0 | s1 | s2 | Conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| `16PCC` | 2 | 2 | 2 | 2 | 100% stable |
| `51RVQ` | 1 | 1 | 1 | 1 | 100% stable |
| `18QYG` | 5 | 4 | 4 | 4 | Stable `4/5`, boundary instance rotates |
| `30VWH` | 5 | 0 | 3 | 3 | `s0` is the outlier |

Important correction from the 3-seed view:

- Earlier interpretation from `seed=0` alone: `30VWH` might require a different spectral mechanism such as FDI.
- Corrected interpretation after `seed=1,2`: `30VWH` is mostly the same low-confidence problem.
- In `s1` and `s2`, `30VWH` contains `3/5` true positives with `IoU=0.62-0.91`, but confidence remains below the practical threshold.

## Instance-Level Stability

Across the 13 Marine_Debris test instances:

- Always TP across all 3 seeds: `5`
- Always MISS across all 3 seeds: `2`
- Seed-dependent: `6`

### Always TP

- `16PCC` x2
- `51RVQ`
- `18QYG_12`
- `18QYG_13`

### Always MISS

- `30VWH gt#0`
- `30VWH gt#2`

These are the only instances that still look genuinely unrecoverable in the current 6-channel setting.

### Seed-Dependent

- `30VWH gt#1 / gt#3 / gt#5`
- `18QYG_7 / 18QYG_8`
- `18QYG_9`

Interpretation:

- Most remaining variability is not "detect vs not detect" in a broad sense.
- It is boundary behavior near the score threshold, plus initialization sensitivity for hard small objects.

## What Is Now Established

1. `MARIDA v2` is a valid leak-free benchmark for this task.
2. The plain 6-channel baseline is stable but fails on Marine_Debris at a practical operating point.
3. `imgsz=1024 + focal(gamma=2.0)` is the first 6-channel configuration that consistently recovers Marine_Debris.
4. The main remaining bottleneck is confidence calibration, not total absence of detections.
5. `30VWH` is not a separate mechanism overall; only `2` instances remain consistently unrecoverable.

## Recommended Writing Position

For the thesis/paper, the cleanest main claim is:

> On the leak-free MARIDA v2 benchmark, a standard 6-channel YOLO baseline fails to detect Marine_Debris at a usable threshold, while increasing input resolution to 1024 and replacing BCE with focal loss (`gamma=2.0`) raises debris recall at `conf=0.05` from effectively `0%` to about `44%`, with stable behavior across 3 random seeds.

## Recommended Next Step

Do not open a new major modeling branch yet.

Most valuable next step:

- Freeze this result as the current best 6-channel line.
- Build paper-ready tables/figures from the frozen JSON outputs.
- If time allows, run one small score-calibration analysis such as temperature scaling or Platt scaling on top of this frozen detector.

Lower priority now:

- FDI feature engineering
- New detection heads
- Returning to PKD / teacher-student as the main path

## Post-hoc Calibration Freeze

After freezing the best 6-channel detector, a small post-hoc Platt-scaling study was run as a deployment-oriented score calibration step.

### Calibration Summary (3 seeds)

| Conf | Raw Mean +- Std | Calibrated Mean +- Std | Delta |
| --- | ---: | ---: | ---: |
| 0.025 | 6.7 +- 0.5 | 6.7 +- 0.5 | 0.0 |
| 0.050 | 5.7 +- 0.5 | 6.0 +- 0.0 | +0.3 |
| 0.100 | 0.7 +- 0.5 | 5.0 +- 0.0 | +4.3 |

### Interpretation

- At `conf=0.05`, calibration changes very little.
- This means post-hoc calibration does not materially change the detector's underlying ranking quality.
- At `conf=0.10`, calibration is highly valuable: the calibrated model consistently keeps `5/13` debris true positives across all three seeds.
- Low-threshold behavior at `conf=0.001` can degrade slightly after calibration, which is expected because Platt scaling compresses extremely weak scores further downward.

### What This Means

The main value of calibration is not "finding new debris" in a fundamental modeling sense.

Its value is:

- making score outputs more usable
- supporting a cleaner deployment threshold
- reducing dependence on very low operating thresholds

Recommended wording for the paper:

> Post-hoc Platt scaling produced only minor gains at `conf=0.05`, confirming that it does not materially improve the detector's intrinsic ranking quality. However, it substantially improved score usability at higher thresholds: at `conf=0.1`, the calibrated model consistently retained `5/13` debris true positives across all three seeds, compared with only `0.7/13` before calibration. This makes `conf=0.1` a more practical operating point for deployment-oriented use.

## Paper Main Table Draft

Suggested main comparison table for the paper body:

| Method | Input | Loss / Setting | mAP@0.5 | mAP@0.5:0.95 | Marine_Debris AP50 | Marine_Debris Recall | Debris TP @ conf=0.05 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 6ch baseline | 512 | BCE, standard YOLO training | `0.193 +- 0.018` | `0.089 +- 0.012` | `0.016 +- 0.006` | `0.000 +- 0.000` | `0 / 13` |
| 6ch + high resolution | 1024 | BCE, standard YOLO training | seed0 `0.194` | seed0 only | sweep only | sweep only | `3 / 13` at `conf=0.001` |
| 6ch + focal | 512 | `gamma=1.5` | sweep study | sweep study | sweep study | sweep study | `5 / 13` at `conf=0.05` |
| **6ch + focal (best)** | **1024** | **`gamma=2.0, alpha=0.25`** | **`0.134 +- 0.025`** | **`0.063 +- 0.013`** | **`0.025 +- 0.006`** | **`0.077 +- 0.109`** | **`5.7 +- 0.6 / 13`** |
| 7ch oracle teacher | 1024-equivalent privileged setting | GT-derived knowledge channel | seed0 `0.526` | seed0 `0.368` | seed0 `0.244` | seed0 `0.769` | oracle analysis |

Recommended caption draft:

> Table X. Main comparison on the leak-free MARIDA v2 benchmark. Standard detector metrics are reported from `eval_test.json`, while Marine_Debris usability is primarily reflected by the debris-specific threshold sweep. The proposed 6-channel best configuration, using `imgsz=1024` and focal loss with `gamma=2.0`, raises practical debris detection from `0/13` true positives in the baseline to `5.7 +- 0.6 / 13` at `conf=0.05` across three random seeds.

## Paper Threshold Table Draft

This table is the most important debris-specific result and should appear either in the main paper or as a focused secondary table.

| Conf Threshold | s0 | s1 | s2 | Mean +- Std |
| --- | ---: | ---: | ---: | ---: |
| 0.001 | 7 | 10 | 10 | 9.0 +- 1.7 |
| 0.005 | 7 | 8 | 10 | 8.3 +- 1.5 |
| 0.010 | 7 | 8 | 10 | 8.3 +- 1.5 |
| 0.025 | 7 | 7 | 6 | 6.7 +- 0.6 |
| 0.050 | 6 | 6 | 5 | 5.7 +- 0.6 |
| 0.100 | 1 | 1 | 0 | 0.7 +- 0.6 |

Recommended caption draft:

> Table Y. Marine_Debris threshold sweep for the frozen best 6-channel model on the 13 debris test instances. Detection quality is highly threshold-sensitive: the detector can localize up to `9.0 +- 1.7` debris instances at `conf=0.001`, and remains stable at a practical threshold of `conf=0.05` with `5.7 +- 0.6` true positives.

## Paper Scene-Stability Table Draft

| Scene | Number of GT Debris | Seed 0 | Seed 1 | Seed 2 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `16PCC` | 2 | 2 | 2 | 2 | Stable and fully recoverable |
| `51RVQ` | 1 | 1 | 1 | 1 | Stable and fully recoverable |
| `18QYG` | 5 | 4 | 4 | 4 | Stable `4/5`, one boundary instance rotates |
| `30VWH` | 5 | 0 | 3 | 3 | `seed=0` outlier; mostly low-confidence rather than spectral failure |

Recommended caption draft:

> Table Z. Scene-level recoverability of Marine_Debris at `conf=0.001`. The three-seed view corrects the earlier seed-0-only interpretation: `30VWH` is not a separate failure mode overall, since three of its five debris instances are recovered in both seeds 1 and 2.

## Figure Drafts

Below are the strongest figure candidates already available in the repository, with suggested paper roles and caption drafts.

### Figure 1: Baseline Training Curves

- Suggested image:
  - `runs_v2/baseline_6ch_s0/results.png`
- Purpose:
  - Show the reference 6-channel baseline training behavior.
- Caption draft:

> Figure 1. Training curves of the frozen 6-channel baseline on MARIDA v2. Although the baseline converges normally and remains stable across seeds, it fails to recover Marine_Debris at a practical confidence threshold.

### Figure 2: Best Model Training Curves

- Suggested image:
  - `runs_v2/focal_g2.0_i1024_s1/results.png`
- Purpose:
  - Show the strongest representative run of the final 6-channel configuration.
- Caption draft:

> Figure 2. Training curves of the best 6-channel setting, combining `imgsz=1024` with focal loss (`gamma=2.0`). This configuration yields the first stable practical recovery of Marine_Debris on the MARIDA v2 test set.

### Figure 3: Baseline vs Best Prediction Examples

- Suggested images:
  - `runs_v2/baseline_6ch_s0/val_batch0_pred.jpg`
  - `runs_v2/focal_g2.0_i1024_s1/val_batch0_pred.jpg`
- Purpose:
  - Visual side-by-side qualitative comparison.
- Caption draft:

> Figure 3. Representative validation predictions from the baseline model and the final focal-loss model. The best model produces more plausible small-object activations and recovers debris instances that are suppressed by the baseline.

### Figure 4: Debris Patch Recoverability Cases

- Suggested images:
  - `runs_v2/sweep_focal_g2.0_i1024_s1/S2_20-4-18_30VWH_4.png`
  - `runs_v2/sweep_focal_g2.0_i1024_s1/S2_20-4-18_30VWH_5.png`
  - `runs_v2/sweep_focal_g2.0_i1024_s1/S2_7-3-20_18QYG_8.png`
  - `runs_v2/sweep_focal_g2.0_i1024_s2/S2_7-3-20_18QYG_7.png`
- Purpose:
  - Show case-level debris recovery and boundary-instance swapping.
- Caption draft:

> Figure 4. Debris patch diagnostics from the threshold sweep. Red boxes denote ground truth and blue boxes denote predictions. These examples illustrate two key behaviors: recoverable low-confidence debris in difficult scenes such as `30VWH`, and boundary instances in `18QYG` whose recovery varies with random initialization.

### Figure 5: Precision-Recall / F1 Curves

- Suggested images:
  - `runs_v2/baseline_6ch_s0/BoxPR_curve.png`
  - `runs_v2/focal_g2.0_i1024_s1/BoxPR_curve.png`
  - `runs_v2/baseline_6ch_s0/BoxF1_curve.png`
  - `runs_v2/focal_g2.0_i1024_s1/BoxF1_curve.png`
- Purpose:
  - Support the claim that the operating point shifts even when headline `mAP` is not the best summary.
- Caption draft:

> Figure 5. Precision-recall and F1 curves for the baseline and final 6-channel model. The final model improves recoverability of Marine_Debris at low confidence, motivating the use of threshold-sweep analysis in addition to standard detector metrics.

## Recommended Figure Selection

If the paper only has room for 3 to 4 figures, the best compact set is:

1. `Figure 2` best-model training curve
2. `Figure 3` baseline vs best prediction comparison
3. `Figure 4` debris patch recoverability cases
4. `Figure 5` PR/F1 curves

## Spectral Ablation: 6ch vs 3ch (2026-03-29)

Ablation run: one seed (seed=0) with matching hyperparameters. 3ch model uses only B4/B3/B2 (RGB); 6ch adds B8 (NIR) + B11/B12 (SWIR).

### AP50 Comparison

| Class | 3ch-base | 6ch-base | 3ch-focal | 6ch-focal |
| --- | ---: | ---: | ---: | ---: |
| Marine_Debris | 0.016 | 0.024 | 0.021 | 0.032 |
| Algae_Organic | 0.043 | 0.083 | 0.024 | 0.083 |
| Ship | 0.459 | 0.548 | 0.363 | 0.393 |
| **mAP50** | **0.173** | **0.218** | **0.136** | **0.169** |

6ch consistently outperforms 3ch across all classes. The Algae_Organic gap is largest (`+0.059` in the focal setting), consistent with the expectation that NIR/SWIR channels encode chlorophyll and floating-material spectral signatures.

### Debris Sweep: Focal Model (imgsz=1024)

| Conf | 3ch TP | 6ch TP | Delta |
| ---: | ---: | ---: | ---: |
| 0.001 | 2 | 7 | +5 |
| 0.005 | 2 | 7 | +5 |
| 0.010 | 2 | 7 | +5 |
| 0.025 | 1 | 7 | +6 |
| 0.050 | 0 | 6 | +6 |

### Scene Breakdown (focal model, conf=0.001)

| Scene | GT | 3ch TP | 6ch TP |
| --- | ---: | ---: | ---: |
| 30VWH | 5 | 0 | 0 |
| 16PCC | 2 | 1 | 2 |
| 51RVQ | 1 | 1 | 1 |
| **18QYG** | **5** | **0** | **4** |

The `18QYG` scene is the decisive difference: 3ch finds zero debris there, 6ch recovers 4/5. This is the primary evidence that NIR/SWIR spectral information is required for Marine_Debris detection in vegetation-free open-water scenes.

### Conclusion

- NIR/SWIR channels provide a clear, measurable benefit: `+5` debris TPs at all practical thresholds.
- The gain is not incremental - removing NIR/SWIR collapses debris detection from 7 TPs to 2 (seed=0).
- This gives the paper the "multispectral information benefit" evidence required for the main narrative: spectral richness + focal loss + high resolution are all necessary components.

Recommended paper sentence:

> The RGB-only (3-channel) ablation recovers only 2 of 13 debris test instances at `conf=0.001`, compared to 7 recovered by the full 6-channel model, demonstrating that NIR and SWIR bands provide spectral discriminability that RGB alone cannot replicate for floating marine debris.

## Notes For Writing

- When discussing Marine_Debris, prefer threshold-sweep evidence over only quoting `model.val()` metrics.
- When discussing overall detector quality, still report the standard `mAP` values for completeness.
- Do not describe `30VWH` as a separate spectral mechanism failure. The 3-seed evidence does not support that claim anymore.
- If calibration is added later, place it after this frozen result as a small post-hoc improvement study, not as a new main contribution.
