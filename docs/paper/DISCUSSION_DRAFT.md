# Discussion (draft, 2026-04-30)

> **⚠️ 状态 (2026-05-20)：本稿基于 v2 3-class 路径，已搁置。** 论文主线已迁移到 v1 7ch physidx on v2split。本稿留作 calibration / focal loss 讨论素材，新 Discussion 稿建立后归档。

This document drafts the Discussion section for the MARIDA v2 detection
paper. It is intentionally written for the author to revise rather than
to be pasted in directly: section headings match expected paper structure
and each point is backed by a specific table/figure produced by the
v2 pipeline scripts.

All numbers below are 3-seed mean ± std unless noted.

---

## 5.1 Why structural attention/fusion modules failed to stabilize

### Observation

The structure ablation (Table 1, Figure 5) is a *negative* result on the
3-seed evaluation: none of CBAM, BiFPN-lite, EIoU, or their full
combination produced a statistically meaningful and stable improvement
over the baseline. Specifically:

- **CBAM** raises Marine_Debris AP50 from `0.016 ± 0.008` (baseline) to
  `0.037 ± 0.025`, but the standard deviation exceeds the apparent
  improvement margin. Inspecting the per-seed values
  (`0.066 / 0.024 / 0.020`) shows the visible gain is driven entirely
  by `seed=0`. With only 13 debris test instances, a single
  initialization can flip `Debris TP@0.025` from `1` to `4`, which
  inflates AP50 disproportionately.

- **BiFPN-lite** marginally lowers mAP50 (`0.187 ± 0.021` vs
  `0.193 ± 0.022`) and does not move debris-specific recovery in any
  consistent direction. The weighted-concat addition appears to add
  capacity that is not productively used in this small-data regime.

- **EIoU** is the only structural change that reliably improves mAP50
  (`0.216 ± 0.010`, std actually smaller than baseline), but the gain
  comes from the easy classes - Algae_Organic (`+0.027`) and Ship
  (`+0.030`) - and does not transfer to Marine_Debris recovery
  (`Debris TP@0.025 = 1.0 ± 1.0`, indistinguishable from baseline).

- **Full** (CBAM + BiFPN + EIoU) is worse than each individual
  component, with mAP50 dropping to `0.161 ± 0.020`. We attribute this
  to capacity over-saturation: the 6-channel float32 input plus the
  small effective rare-class population (≤80 tiles with debris) is not
  enough signal for three additive structural modifications to all be
  jointly learned.

### Implication

The seed-stability requirement is the dominant filter on this dataset.
A `seed=0` gate (the original protocol) wrongly admits CBAM as an
improvement; a 3-seed gate correctly rejects it. We argue that any
future work on MARIDA-class small-rare detection benchmarks should make
3-seed stability a pre-registered acceptance criterion, not a
post-hoc check.

---

## 5.2 Confidence calibration is the actual bottleneck

### Observation

The engineering pipeline (Table 2, Figure 5b) shows a sharp dissociation
between aggregate mAP and practically usable detection:

- Standard mAP50 stays roughly constant for imgsz=1024 alone
  (`0.195 ± 0.013` vs baseline `0.193 ± 0.022`) and decreases when
  focal loss is added (`0.134 ± 0.031`).
- However, *practical* Marine_Debris recovery at `conf = 0.05` shows a
  more revealing pattern: baseline `1 / 13` → imgsz=1024 alone
  `0.0 ± 0.0 / 13` (statistically reproducible regression across all
  3 seeds) → focal+1024 `5.7 ± 0.6 / 13` (5.7× over baseline) → after
  Platt calibration, `6.0 ± 0.0 / 13` at `conf = 0.05` and
  `5.0 ± 0.0 / 13` at the more usable `conf = 0.1` threshold (compared
  to `0.7 ± 0.6` uncalibrated).

### Interpretation

The detector **localizes** debris well: in successful seeds, the box
IoU on hit instances exceeds 0.9. The failure mode is in the
*confidence head* - logit values for true positive debris are
systematically pushed below the practical operating threshold by the
abundance of negative pixels in marine scenes.

Focal loss addresses this at training time by re-weighting the loss
toward hard positives, lifting the raw debris score distribution.
Platt scaling addresses the residual gap at deployment time by
remapping the score scale so that `conf = 0.1` becomes a usable
operating point.

This view also explains why imgsz=1024 alone is *worse than baseline*
(Table 2, row 2: `Debris TP@0.05 = 0.0 ± 0.0` across 3 seeds, vs
baseline `1`). Doubling spatial resolution adds high-frequency content
to every tile, including the marine background; under BCE the
classifier has no incentive to focus on the rare positives, so the
extra negative-pixel detail simply pushes debris scores further into
the noise. Focal loss is the necessary partner that tells the network
which gradient signal to listen to.

---

## 5.3 The mAP-vs-utility paradox

### Observation

A reader scanning Table 2 is likely to flag that mAP50 *decreases* on
the proposed pipeline (`0.193 → 0.134`) and conclude that the change
hurts the model. This conclusion is wrong but the paradox is real and
must be addressed explicitly.

### Mechanism

mAP at IoU=0.5 averages AP across all classes and is dominated by the
Ship class, which the baseline already detects well (`AP50 ≈ 0.49`).
Focal loss reweights training toward rare positives; in practice it
slightly degrades the easy classes (Ship AP50 drops from `0.491` to
`0.299`) while massively improving recoverability of the rare debris
class. Because Ship contributes more weight to the macro mAP than
Marine_Debris does, the headline number moves down even as the more
operationally important metric moves up by an order of magnitude.

We therefore argue that any benchmark dealing with strongly imbalanced
rare classes should report **per-class AP at multiple confidence
thresholds**, not just a single macro mAP. The threshold sweep in
Table 2 is the minimum we found necessary to honestly describe model
behavior.

---

## 5.4 Multispectral information is necessary, not optional

### Observation

The 3ch (RGB-only, B4/B3/B2) ablation under the same focal+1024
pipeline recovers only `2 / 13` debris instances at `conf = 0.001`,
versus `7 / 13` for the 6ch (RGB+NIR+SWIR) model. The decisive scene
is `S2_7-3-20_18QYG`, where 3ch finds zero debris instances while 6ch
recovers 4 of 5.

### Interpretation

In open-water tiles without distinguishing surface texture, debris is
identified primarily through its spectral signature in the NIR (B8) and
SWIR (B11/B12) bands, not through visual contrast in RGB. This is
consistent with the marine-remote-sensing literature on plastic /
sargassum spectral indices, but to our knowledge has not been
demonstrated as a *detection* (vs segmentation) requirement on a
leak-free benchmark before.

This finding also constrains the deployment architecture: any future
operational system based on RGB drones or RGB satellites alone will
have a hard ceiling on debris recall determined by available spectral
discriminability, regardless of detector quality.

---

## 5.5 Limitations

### Test set scale

The MARIDA v2 test split contains exactly 13 Marine_Debris ground-truth
instances spread across 4 scenes. This is a fundamental statistical
power limit: a single instance moving across a confidence threshold
shifts reported recall by ~7.7 percentage points. We report std across
3 seeds, but a 5-seed or 10-seed re-run would be a stronger guarantee
of effect-size.

### Scene-level (not instance-level) splits

MARIDA v2 prevents image-level leakage between train/val/test, but
within-scene physical similarity (same satellite, same date, same
illumination) means recovering one debris instance in a scene is
strongly correlated with recovering its neighbors. The threshold sweep
respects this by reporting per-instance counts rather than tile-level
aggregates, but consumers of the result should be aware that the
"effective" sample count is closer to 4 (number of distinct test
scenes) than to 13.

### Fixed input size during training

We did not jointly explore training with multi-scale augmentation or
test-time multi-scale fusion. Both are likely to provide additional
debris recovery at non-trivial compute cost; we leave them to future
work.

### No domain transfer

The proposed pipeline is evaluated only on Sentinel-2 L2A. UAV-borne
or airborne RGB imagery (such as the Roboflow Ocean Waste corpus) lacks
the SWIR/NIR bands central to our findings, so direct transfer of the
final model is not possible without retraining on a different input
modality. We treat domain transfer as a parallel research track, not
a benchmark that this paper claims to solve.

---

## 5.6 What we did *not* find

For honest scientific record, the following were tried and abandoned:

- **Privileged Knowledge Distillation (PKD)** from a 7-channel oracle
  teacher: failed to transfer usable Marine_Debris ability to the 6ch
  student in any configuration we tried (alpha and temperature varied
  across `α ∈ {0.3, 0.5}`, `T ∈ {4, 8}`).
- **Class oversampling** (8× rare-class weight): hurt overall
  performance without helping debris recall.
- **Fourth (P2) detection head** at stride 4: did not improve debris
  detection despite the small object size.

These negative results are kept here so that follow-up work does not
re-investigate them under the impression they were unexplored.

---

## 5.7 Summary

The headline scientific contribution of this work is not a new
architectural module. It is the demonstration that, on a strict
leak-free benchmark, the bottleneck for rare floating-object detection
is **confidence calibration**, not **localization**, and that this
bottleneck is addressed by a small set of training/deployment changes
(focal loss, increased input resolution, post-hoc Platt scaling)
rather than by adding attention/fusion modules. The structure ablation
provides supporting evidence that the obvious architectural fixes do
not work on this scale of data.
