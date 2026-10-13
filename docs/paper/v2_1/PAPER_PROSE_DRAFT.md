# Paper prose draft — **v2.1 working copy** (zh + en mixed)

⚠️ **STATUS — 2026-05-XX (Phase 0)**: this file is a clone of the v2.0
prose, used as the **starting point** for the v2.1 rewrite. It still
contains v2.0 numbers and claims; these will be replaced once v2.1
experiments produce final results.

Tagging convention used through this file:

- 🔵 **OK**: still correct under v2.1 framing
- 🟡 **REWRITE**: needs full rewrite (new task: instance segmentation,
  new metric set, new ablations)
- 🔴 **WRONG**: contains false claims (notably "Sentinel-2 L2A"
  language); being fixed in Phase 0

Source of truth for the v2.1 protocol: `v2_1_DATASET_SPEC.md`.
Numbers below still refer to v2.0 frozen results unless explicitly
updated — do not cite into the final paper without replacing.

---

## ⚠️ Phase 0 corrections to the v2.0 wording

Below are the language fixes applied (or pending) to bring this draft
into v2.1 compliance. Search-and-replace in the prose downstream
must follow these rules.

### Atmospheric correction (🔴 WRONG → ✅ corrected text)

The product is **NOT** "Sentinel-2 L2A". Everywhere in the prose
below where it says "Sentinel-2 L2A" or "Sen2Cor", replace with:

> **ACOLITE Rayleigh-corrected reflectance from Sentinel-2 L1C, as
> released in MARIDA (Kikaki et al., 2022).**

Implication for §3.2 (preprocessing prose): the reflectance values
in the .tif files are **water-leaving reflectance after Rayleigh
correction**, in the range ~[0.005, 0.5], **not** Sen2Cor surface
reflectance and **not** scaled DN.

### Channel selection (🟡 REWRITE)

v2.0 prose claims "6 channels: B4, B3, B2, B8, B11, B12". v2.1 uses
**9 channels**: B4, B3, B2, **B6 (newly added)**, B8, B11, B12 +
**FDI, NDVI** (computed). This entire paragraph in §3.2 needs rewrite.

### Task definition (🟡 REWRITE)

v2.0 prose describes **detection** (bbox output, YOLO direct path).
v2.1 is **instance segmentation** (polygon + bbox, derived from
connected components of pixel masks). §3.3 needs partial rewrite to
clarify "we adopt YOLO instance segmentation to reduce the lossy
mask-to-bbox conversion required by detection-only models".

### Baseline comparison (🟡 REWRITE)

v2.0 prose has no same-split baseline rerun. v2.1 §4 must add tables
comparing **YOLO-seg (ours) / Kakogeorgiou RF SS+SI / Kikaki U-Net /
FDI threshold**, all on the v2.1 split.

### Confidence ablation (NEW for v2.1)

v2.0 had no confidence-aware label processing. v2.1 must add a new
§3.X subsection describing the highonly/highmod/all three-way label
generation and the resulting ablation (one of the v2.1
contributions).

### Calibration claim (🟡 REWRITE)

v2.0 prose claims "Platt scaling raises debris recovery at conf=0.1
from 0.7/13 to 5.0/13". This conflates **calibration** with
**recall**. v2.1 uses **F1-optimal threshold search on val set**
(not Platt scaling), and the claim must be reframed as "operational
threshold tuning", not "calibration improves detection".

---

This document holds the long-form prose for the sections that the
skeleton in `PAPER_DRAFT.md` left as stubs. Each section here is a
**single revision target**. Numbers below are **v2.0 numbers** and
must be replaced once v2.1 produces final results.

---

## Abstract

Marine floating debris poses a growing threat to coastal ecosystems,
yet automatic detection from satellite imagery remains hard: targets
are small (≈12-14 px in 10 m Sentinel-2 pixels), labeled instances are
rare (only 13 ground-truth Marine_Debris boxes in the test split of
the leak-free MARIDA v2 benchmark), and standard detector outputs are
systematically under-confident. We conduct a 3-seed ablation of three
popular YOLO improvements — CBAM channel-spatial attention, weighted
BiFPN feature fusion, and EIoU box regression — and find that **none
provides a stable improvement over the baseline once seed variability
is accounted for**. We argue that the practical bottleneck is not
localisation but **confidence calibration**, evidenced by a
threshold-sweep analysis in which the detector localises debris with
IoU > 0.9 yet assigns confidences below `0.01`. A simple combination
of focal loss (γ=2.0) at imgsz=1024 and post-hoc Platt scaling raises
practical Marine_Debris recovery at conf=0.05 from `1 / 13` (baseline)
to `5.7 ± 0.6 / 13` (a 5.7× improvement) and at conf=0.1 from
`0.7 ± 0.6` to `5.0 ± 0.0`, with statistically reproducible behaviour
across three random seeds. A 6-channel vs 3-channel ablation further
confirms that NIR and SWIR bands are necessary, not optional, for
open-water debris recovery: removing them collapses recovery from
`7 / 13` to `2 / 13` instances. Our results suggest that, on
small-data multispectral benchmarks dominated by rare classes,
unglamorous training-time and deployment-time choices outperform
fashionable architectural additions. **Code, splits, and per-seed
weights are released for full reproducibility.**

(~250 words; trim to journal limit at submission time.)

---

## §1. Introduction

Marine debris — primarily plastics and macro-plastics — is now found
in every major ocean basin, with measurable impacts on wildlife,
fisheries, and human food chains. Sentinel-2 multispectral imagery is
attractive for systematic monitoring because of its global coverage,
short revisit time, and free access; however, the spectral and
spatial signature of floating debris in 10 m resolution makes
automatic detection a challenging small, rare, and weakly contrasted
target problem. The MARIDA dataset (Kikaki et al., 2022) was the
first openly labelled benchmark for this task. Subsequent work has
shown that the original MARIDA splits suffer from scene-level
leakage between train and test partitions, biasing reported numbers
upward. We therefore work on **MARIDA v2**, a leak-free re-split
that enforces strict scene separation and treats one entire
Sentinel-2 acquisition as the unit of train/val/test assignment.
This re-split changes the empirical landscape considerably: previously
strong models often regress to baseline, and previously published
"improvements" disappear once spatial leakage is removed.

### What this paper contributes

We do **not** propose a new attention module, feature pyramid, or box
loss. Instead, we provide three contributions of a different kind.

**(1) A 3-seed audit of three popular structural improvements on
MARIDA v2.** We re-implement CBAM, BiFPN-lite, and EIoU on top of a
6-channel YOLOv11n baseline and run each variant — plus the full
combination — under three random seeds. Reported as 3-seed mean ± std,
none of the variants achieves a statistically meaningful and stable
improvement over the baseline. CBAM appears strong on `seed=0`
(`Marine_Debris AP@0.5 = 0.066`), but its standard deviation across
seeds is comparable to the apparent gain (`0.037 ± 0.025`), driven
largely by `seed=0` being an outlier. The full combination is
strictly worse than each individual component, which we attribute to
combined-capacity over-saturation in the small-data regime.

**(2) Identification of confidence calibration as the actual
bottleneck.** A debris-specific threshold sweep — counting how many
of the 13 test debris instances are localised with IoU > 0.5 at each
of several confidence thresholds — reveals that the detector
*localises* debris well (IoU often > 0.9) but assigns scores far
below any usable operating threshold. This decoupling explains why
mAP-style aggregates appear unchanged while practical detection
collapses, and is invisible to standard model.val() reporting.

**(3) A simple training-and-deployment recipe that stably recovers
practical Marine_Debris detection.** We replace BCE on the
classification branch with focal loss (γ=2.0, α=0.25) and train at
imgsz=1024. After training, we fit a one-parameter Platt scaling on
val-set TP/FP labels. The resulting pipeline raises Debris TP at
`conf=0.05` from `1 / 13` (baseline) to `5.7 ± 0.6 / 13` and at the
more usable `conf=0.1` from `0.7 ± 0.6` to `5.0 ± 0.0`, with
deterministic 3-seed reproducibility. This is, to our knowledge, the
first method that delivers stable practical debris recovery on the
leak-free MARIDA v2 benchmark.

We additionally provide a 6-channel vs 3-channel ablation under the
same training pipeline. The 3-channel (RGB-only) variant recovers
only `2 / 13` debris instances at `conf=0.001`, whereas the 6-channel
(RGB+NIR+SWIR) model recovers `7 / 13`. The decisive scene is
`S2_7-3-20_18QYG`, where 3-channel finds zero of five GT instances
while 6-channel finds four. We interpret this as direct empirical
evidence that NIR and SWIR bands carry information that is **necessary**,
not incidental, for floating-debris recovery in open water.

### Roadmap of the paper

Section 2 reviews related work in marine remote sensing, attention
modules, feature pyramids, IoU losses, and confidence calibration.
Section 3 describes the dataset, channel selection, preprocessing,
detector, and evaluation protocol. Section 4 presents the structure
ablation (Table 1), the engineering pipeline (Table 2), the spectral
ablation, the scene-level breakdown (Table 3), the speed-quality
trade-off (Figure 6), and qualitative comparisons (Figures 7-8).
Section 5 discusses why the structural changes failed, why
calibration is the actual bottleneck, the mAP-versus-utility paradox
that follows, and the limitations of the dataset scale. Section 6
concludes.

---

## §2. Related Work (skeletal prose; expand citations before submission)

**Marine debris detection from satellite imagery.** Early work was
mostly index-based: floating-debris-index (FDI) and the related
plastic-floats spectral signature have been used to flag suspicious
pixels (Biermann et al., 2020). Mature deep-learning approaches
appeared with the release of MARIDA (Kikaki et al., 2022), which
treated the task as semantic segmentation. The original MARIDA splits
have since been shown to contain scene-level leakage; we evaluate on
the leak-free MARIDA v2 re-split.

**Detection vs segmentation for marine debris.** MARIDA's reference
results are reported on a per-pixel segmentation backbone with
post-hoc connected-component conversion to bounding boxes. We instead
take the YOLO direct-detection route, motivated by the bbox-format
labels of MARIDA v2 and the operational simplicity of having a single
NMS step instead of threshold-and-morphology pipelines. Whether
direct detection or segmentation+post-processing is preferable on
MARIDA v2 is, to our knowledge, an open question; we treat it as
future work.

**Channel attention.** Squeeze-and-Excitation (Hu et al., 2018), CBAM
(Woo et al., 2018), and ECA (Wang et al., 2020) re-weight
intermediate features along channel and/or spatial axes. CBAM is the
default attention module in many YOLO papers since it can be inserted
without changing tensor shapes. Reported improvements in the original
literature were on COCO and ImageNet; results on small remote-sensing
benchmarks are mixed.

**Feature pyramid variants.** PANet (Liu et al., 2018) augments FPN
with a bottom-up path and remains the default neck for YOLOv8/v11.
BiFPN (Tan et al., 2020) adds bidirectional connections and learnable
fusion weights; "BiFPN-lite" variants common in YOLO improvement
papers replace the standard concat with weighted concat and add
cross-scale skip connections, without iterating the full BiFPN
multiple times.

**IoU-family box-regression losses.** Generalised IoU (Rezatofighi
et al., 2019), Distance IoU and Complete IoU (Zheng et al., 2020),
and Efficient IoU (Zhang et al., 2022) all aim to make bounding-box
regression more sample-efficient. CIoU is the default in YOLOv8/v11
and is replaced with EIoU in our ablation.

**Imbalanced-classification losses.** Focal loss (Lin et al., 2017)
is the canonical solution to extreme foreground/background imbalance
in dense detection; γ controls the focusing strength on hard
examples. We apply focal loss to the classification branch of YOLO,
following recent practice in YOLOv8 community implementations.

**Score calibration.** Platt scaling (Platt, 1999) and temperature
scaling (Guo et al., 2017) re-map raw classifier scores so that the
output looks like a probability. Calibration is well studied for
classification but is comparatively rare in the object-detection
literature; we apply it as a deployment-time fix on top of the
training-time focal-loss change.

---

## §3. Methodology (prose detailing what's already in the skeleton)

### 3.1 Dataset: MARIDA v2

MARIDA v2 is a re-split of the MARIDA collection (Kikaki et al., 2022)
of 1,381 image patches cropped to 256 × 256 pixels at 10 m ground
resolution. The radiometric product used by MARIDA is **ACOLITE
Rayleigh-corrected surface reflectance derived from Sentinel-2 L1C**,
not the more familiar Sen2Cor L2A product; ACOLITE is the water-colour
remote-sensing standard and preserves dim-water reflectance signal
that Sen2Cor (designed for terrestrial atmospheric correction) tends
to corrupt over dark water surfaces. Pixel values are 32-bit float
reflectance in the approximate range [0.005, 0.5]; we treat them as
band reflectance throughout.
Three classes are retained: Marine_Debris (cls=0), Algae_Organic
(cls=1), and Ship (cls=2). The decisive change versus the original
MARIDA is that train/validation/test partitioning is performed at
the **scene** level: every tile from a single Sentinel-2 acquisition
goes into exactly one of the three subsets, eliminating the
within-scene physical-similarity leakage that was previously possible.
The resulting split contains 1,381 tiles total; the test split
contains exactly **13 Marine_Debris GT instances** spread across 4
scenes, **80** tiles with at least one debris instance overall, and
**212 GT debris instances in total** (Figure 9).

### 3.2 Channel selection and preprocessing

Of the 13 Sentinel-2 L2A bands, we retain six: B4, B3, B2 (visible
RGB), B8 (NIR, ~842 nm), B11 (SWIR-1, ~1610 nm), and B12 (SWIR-2,
~2190 nm). NIR is included because of vegetation-style chlorophyll
contrast (which separates Algae from Ship); SWIR is included because
of water-absorption versus floating-material contrast (which is the
spectral lever for plastic detection). RGB alone is **insufficient**
on this benchmark — we report the empirical evidence in Section 4.3.

For each of the six retained bands we compute the 1st and 99th
percentile values across the training split and use these as fixed
clipping thresholds: `x_norm = clip((x - p01) / (p99 - p01), 0, 1)`.
We deliberately avoid per-tile histogram equalisation to preserve
inter-tile spectral consistency. We do not apply HSV jitter, mosaic,
mixup, or copy-paste augmentation: HSV is ill-defined on
non-RGB-semantic channels; mosaic/mixup combine multispectral float32
arrays in ways that violate the spectral consistency the model needs
to learn. Our augmentation is limited to **geometric transforms**:
rotation up to 180°, vertical flip 50%, horizontal flip 50%.

### 3.3 Detector

We build on YOLOv11n from Ultralytics. The first convolution is
modified to accept six input channels (we therefore train from
scratch — no COCO pretraining, since the 6-channel weight tensor
cannot be initialised from RGB). Backbone, neck (PANet), and detect
head (P3 / P4 / P5, strides 8 / 16 / 32) are otherwise unchanged in
our final pipeline. The backbone-feature production from MARIDA's
6-channel float32 `.npy` files required custom plugin code
(`ultralytics_plugins/`); the standard Ultralytics YOLO loader
assumes 3-channel uint8 RGB images and silently mishandles
multispectral inputs.

### 3.4 The proposed pipeline (focal + 1024 + Platt)

The final pipeline retains the unmodified YOLOv11n backbone and head
and changes three things relative to the baseline.

**(i) Increased input resolution.** We train at imgsz=1024 instead of
the default 512. The motivation is that a 12-14 pixel debris target
in 256 × 256 native is only 6-7 pixels at imgsz=512 after letterbox
scaling — below the resolution at which YOLOv11's stride-8 detection
head has any chance of producing a useful prediction. Doubling the
resolution makes the same target ~14 pixels at imgsz=1024 input and
restores the spatial signal. This also costs us batch size: we drop
batch from 8 to 4 to fit within 24 GB GPU memory.

**(ii) Focal loss on the classification branch.** We replace the
default binary cross-entropy on the YOLO classification head with
focal loss `FL(p_t) = -α (1 - p_t)^γ log(p_t)` with γ=2.0, α=0.25.
The motivation is that BCE on a class-imbalanced dense detector
accumulates orders of magnitude more gradient from background pixels
than from rare debris pixels, suppressing debris confidence
systematically. Focal loss down-weights the contribution of confident
negatives so that gradient flow on rare positives is no longer
drowned. Crucially, we apply focal loss only to the classification
branch — box regression continues to use CIoU, which performs fine
when the matching is correct.

**(iii) Post-hoc Platt scaling.** After training, we collect debris-
class predictions on the validation split, label each as TP or FP by
matching to GT at IoU > 0.5, and fit a single-parameter logistic
regression `p_cal = σ(a · logit(p_raw) + b)`. The fitted (a, b) is
applied at deployment time. Platt scaling does not change which
detections the model emits, only how confident the model claims to
be about each one. As we will see (Section 4.2), this is exactly
what is needed to make `conf = 0.1` a usable deployment threshold.

### 3.5 Structure-ablation modules (rejected after 3-seed audit)

For comparison and per the original benchmark protocol, we also
evaluated three structural variants. **CBAM** (Woo et al., 2018) is
inserted after each backbone C3k2 block at scales P2/P3/P4/P5; we
follow the standard `(channel attention) → (spatial attention)`
ordering. **BiFPN-lite** replaces the standard `Concat` operations
in the detection neck with weighted concat (using softmax-normalised
learnable weights) and adds direct skip connections from each
backbone scale to its corresponding head input. **EIoU** replaces
CIoU in the box regression branch. **Full** combines all three. All
four were trained at imgsz=512 (matching the original baseline) with
BCE classification, so that the only difference between rows in
Table 1 is the structural change of interest.

### 3.6 Evaluation protocol

We report two complementary metric families.

**Standard detector metrics** are obtained at `conf = 0.001`, IoU = 0.6
NMS, following YOLO defaults. We report mAP@0.5, mAP@0.5:0.95,
per-class AP, P, R, and F1. These are computed by
`scripts/v2_evaluate.py`, which monkey-patches the Ultralytics dataset
loader so that it does not silently re-encode our 6-channel float32
tiles.

**Marine_Debris threshold sweep.** Because the test split contains
only 13 debris instances, mAP at conf=0.001 over the whole class can
be misleading: a single instance moving across a confidence threshold
shifts reported recall by 7.7 pp. We therefore additionally report,
for each of a set of practical confidence thresholds
`conf ∈ {0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5}`, the
number of GT debris instances that are recovered at IoU > 0.5. This
is the "Debris TP@conf" column throughout Tables 1-3 and is, in our
view, the metric a deployer should look at when deciding what
threshold to ship a detector at.

### 3.7 Training details

All runs use AdamW with auto-tuned learning rate (≈ 1.4e-3 at start,
cosine schedule over 50 epochs), weight decay 1e-3, dropout 0.1, and
warmup over 5 epochs. Each variant is trained from scratch (no
external pretraining) for 50 epochs; loss curves (Figure 3) show
that all variants are well past the inflection point by epoch 30,
and 50 epochs is conservative. We run each variant under three
random seeds (`{0, 1, 2}`) and report mean ± std. All training was
performed on a single NVIDIA RTX 3090 Ti.

---

## §6. Conclusion

We presented a 3-seed empirical study of detector design choices on
the leak-free MARIDA v2 marine-debris benchmark. Three popular
structural additions to YOLOv11 — CBAM channel-spatial attention,
weighted BiFPN feature fusion, and EIoU box regression — failed to
deliver stable improvements over the baseline once seed variability
was accounted for; the full combination was strictly worse than each
individual component. We argued, supported by a debris-specific
threshold-sweep analysis, that the practical bottleneck on this
benchmark is **confidence calibration** rather than localisation:
the detector finds debris at IoU > 0.9 but assigns scores too low to
be operationally useful. A simple combination of focal loss
(γ=2.0) at imgsz=1024 and post-hoc Platt scaling raised stable
debris recovery at the practical operating point of conf=0.1 from
`0.7 ± 0.6 / 13` to `5.0 ± 0.0 / 13`, the first such recipe to be
3-seed reproducible on the leak-free split. A spectral-channel
ablation further demonstrated that NIR and SWIR bands are
**necessary**, not incidental, for open-water debris detection: the
3-channel RGB-only variant collapses to `2 / 13` recoverable
instances. We hope the negative finding on architectural additions
serves as a calibration signal for future work on similar
small-data, rare-class detection benchmarks: in this regime,
unglamorous training-time and deployment-time choices outperform
fashionable structural ones, and confidence-distribution diagnostics
should be the default diagnostic before any architectural intervention
is attempted.
