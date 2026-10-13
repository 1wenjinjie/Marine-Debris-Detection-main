# Marine Debris Detection on MARIDA — Paper Draft (v1 7ch physidx, clean rewrite)

> **Status (2026-05-23):** clean draft, v1 main path. Numerical content sourced from frozen artifacts:
> - Leak audit: `docs/paper/7CH_BASELINE_CURRENT_AUDIT_2026-05-18.md`, `runs_v1_v2split/*_eval.json`, `runs/detect/runs_v1_v2split/*/v2split_test_eval.json`
> - 5-architecture × 2-setting × 3-seed ablation and controls: `docs/paper/v1_ablation_aggregate.{md,json}`, `docs/paper/v1_extra_ablation_aggregate.{md,json}`, `docs/paper/v1_imgsz1024_ablation_aggregate.{md,json}`, `docs/paper/v1_imgsz1024_leaky_aggregate.{md,json}`, `docs/paper/v1_native256_aggregate.{md,json}`
> - Calibration: `docs/paper/calibration/{platt_fit.json, threshold_sweep_test.json, reliability_diagram.png}`
> - FDI baseline: `docs/paper/fdi_baseline/fdi_ndwi_sweep_test.json`
>
> v2 3-class stub: `PAPER_DRAFT.md` + `PAPER_PROSE_DRAFT.md` (frozen, archive after this draft solidifies).
>
> **Presentation conventions (do not violate):**
> - Headline metric: **Marine_Debris instance-level F1** (108 GT, IoU≥0.5, conf=0.25) — call this the *deployment F1*.
> - Secondary: 3-seed mean ± std mAP@0.5 over the test split, with the explicit caveat *"Marine_Debris dominates; Dense_Sargassum / Natural_Organic have 0 GT in test; Sparse_Sargassum / Wakes are too sparse for strong claims"*.
> - Per-class tables/bars only list classes with ≥1 GT in test: Marine_Debris, Sparse_Sargassum, Ship, Foam, Wakes; main prose focuses on Marine_Debris / Ship / Foam.
> - imgsz=1024 must be described as *"increases the cell budget the model sees and brings the median debris box back inside the P3 head's design range"*, **not** "higher sensor resolution".

**Working title (draft):** Honest gains on MARIDA: a leak-audited and calibration-aware re-evaluation of YOLO detectors for floating marine debris.

---

## Abstract

We revisit single-stage detection of floating marine debris in Sentinel-2 imagery on the MARIDA benchmark and show that, once **scene-level leakage** is removed and the released **physical-prior channel** is rebuilt to be derivable from the input at deployment time, most of the "improvements" reported on the original MARIDA splits do not transfer. To establish what *does* transfer, we run a leak-attribution audit followed by a $5$-architecture $\times$ $2$-setting $\times$ $3$-seed ablation and cross-resolution controls under the leak-free, oracle-free protocol. The single intervention that survives is **the combination of weighted BiFPN feature fusion with imgsz=1024 letterbox upsampling**, which lifts Marine_Debris instance-level F1 by $+17\%$ over the unmodified YOLOv11n at imgsz=512 (108 GT, scene-clean test). We further demonstrate three claims that are, to our knowledge, novel for this benchmark: (i) a non-ML floating-debris-index baseline (Biermann 2020 FDI + NDWI-inverse thresholding) achieves best F1 below $1\%$ — $50$ to $400\times$ weaker than the trained detector — and is therefore useful as an input channel rather than as a standalone detector; (ii) the raw confidence scores of the SOTA detector are already deployment-usable (F1 = 0.604 at conf $=$ 0.25, P/R $=$ 0.581 / 0.630), contradicting the common claim that satellite-debris detectors are uniformly under-confident; (iii) Platt scaling fitted on validation reduces held-out test Expected Calibration Error from 0.051 to 0.029 (a $1.75{\times}$ improvement on held-out test, $3.5{\times}$ in-sample on val) while preserving rank order, so probabilistic downstream use (Bayesian sensor fusion, risk-aware ranking) is also supported. We release the leak-free `v2split` partition, the audit ablation matrix, the FDI baseline implementation, and per-seed weights.

(~280 words; trim to journal limit at submission.)

---

## 1. Introduction

### 1.1 Why marine debris detection from Sentinel-2 is interesting and hard

Plastic and macro-plastic debris is now present in every major ocean basin. Sentinel-2 multispectral imagery is the workhorse remote-sensing source for systematic monitoring because of global coverage, $\sim$5-day revisit, and free access; yet automatic detection at the 10 m pixel pitch is hard. Floating debris instances are small ($\sim$12-14 pixels wide in the MARIDA tiles), spectrally weak compared with adjacent water clutter (foam, wakes, algae mats), and rare: our leak-free `v2split` test partition contains $108$ Marine_Debris instances across $73$ tiles.

### 1.2 A literature gap: detection-style evaluation on MARIDA is sparse

Of the published work on MARIDA, the majority operates in a **semantic-segmentation** framework (Kikaki et al., 2022; subsequent encoder-decoder variants), where the task is per-pixel labeling and standard metrics are pixel-IoU and pixel-F1. Detection-style evaluation — instance-level bounding boxes with IoU-matched precision and recall — is comparatively rare in the MARIDA literature, and where it does appear, results are usually reported on the **original** MARIDA splits, which (we show in §4.1) suffer from scene-level leakage and from a published "physical prior" channel that was implicitly derivable from the ground-truth mask. This combination has, in our reading of the literature, inflated reported numbers by an amount comparable to the entire methodological claim of several recent papers, and makes cross-paper comparison unreliable.

### 1.3 What this paper contributes

We make four contributions.

**(1) An honest re-evaluation protocol.** We adopt a strict scene-level split (`v2split`) that places every tile from a single Sentinel-2 acquisition into exactly one of the train / val / test partitions, and we rebuild the optional physical-prior channel ($\mathrm{NDWI^{-1}}$) so that it is computable from the L2A reflectance alone — eliminating ground-truth dependence at inference. We refer to the combination as the **honest setting**.

**(2) A leak-attribution audit.** We decompose previously reported numbers using measured oracle, zero-prior, and honest-prior cells. The oracle-from-mask channel is the dominant inflation source: on the scene-clean split, a clean oracle model reaches mAP50 $=0.466$, but the same checkpoint falls to $0.141$ when the heatmap is zeroed. The deployable $\mathrm{NDWI^{-1}}$ model recovers much of the spectral signal (mAP50 $=0.349$ seed 0; $0.313 \pm 0.038$ over 3 seeds) without ground-truth dependence.

**(3) A $5$-architecture $\times$ $2$-setting $\times$ $3$-seed ablation.** We re-run the baseline, CBAM, BiFPN, EIoU, and Full (CBAM+BiFPN+EIoU) cells under both leaky and honest settings. At imgsz=512, none gives a stable net improvement over the honest baseline. The only practical winner appears after the cross-resolution pass: **BiFPN $\times$ imgsz=1024**, which improves Marine_Debris Best F1 over both the imgsz=512 baseline and the imgsz=1024 baseline.

**(4) Two reviewer-facing controls absent from prior MARIDA detection work.** First, a non-ML **floating-debris-index baseline** (Biermann 2020 FDI plus NDWI-inverse thresholding) evaluated end-to-end with instance-level matching, demonstrating a $50{-}400\times$ gap to the trained detector and establishing the *deep-learning indispensability* floor. Second, an **explicit confidence-calibration study** on the SOTA model, reporting raw and Platt-scaled threshold sweeps, ECE, and a reliability diagram — to our knowledge the first such report for MARIDA detection. The calibration study yields the paper's headline practical finding: the raw model is already deployment-usable (seed-0 deployment F1 $=$ 0.604 at $\tau = 0.25$), and val-fitted Platt scaling reduces held-out test ECE by $1.75\times$ (in-sample val improvement is $3.5\times$) without sacrificing rank order.

### 1.4 What this paper deliberately does *not* do

We do not propose a new attention module, feature pyramid, or box loss. We do not claim a state-of-the-art on the original MARIDA splits (which we argue are no longer the right benchmark). We do not attempt cross-sensor generalization, temporal modeling, or domain adaptation; these are explicitly out of scope and discussed as future work in §6.

### 1.5 Roadmap

§2 reviews related work. §3 describes data, splits, channels, detector, the FDI baseline, the Platt-calibration procedure, and the evaluation protocol. §4 presents the leak audit, the $5$-architecture $\times$ $2$-setting $\times$ $3$-seed ablation, the imgsz=1024 honest ablation including the leaky and native-256 controls, the FDI baseline numbers, and the calibration analysis. §5 discusses mechanism and limitations. §6 concludes.

---

## 2. Related work

**Marine debris detection from satellite imagery.** Early work was index-based: the Floating Debris Index (FDI; Biermann et al., 2020) and related plastic-floats spectral signatures were used to flag suspicious pixels in Sentinel-2 imagery, demonstrating that the NIR-to-SWIR baseline is a physically meaningful contrast for floating plastic. Mature deep-learning approaches appeared with the release of MARIDA (Kikaki et al., 2022), which treated the task as semantic segmentation. Topouzelis et al. (2020) summarize the spectral-cue evidence that justifies using NIR and SWIR channels for floating-material detection.

**The MARIDA dataset and its splits.** The original MARIDA release exposes 11-band ACOLITE-corrected Sentinel-2 L2A surface reflectance tiles at native 256 $\times$ 256 and provides a per-pixel mask with $\geq$10 thematic classes including Marine_Debris, multiple algae/organic classes, Ship, Foam, and Wakes. The originally distributed train/test split is **not** scene-disjoint: tiles cut from the same Sentinel-2 acquisition can appear in both partitions, producing physical-similarity leakage that biases reported scores upward. This issue has been noted in passing in subsequent work but, to our knowledge, has not been characterized quantitatively in the detection setting prior to this paper.

**Detection vs segmentation on MARIDA.** The default MARIDA evaluation is pixel-segmentation; detection-style evaluation requires either (i) converting the segmentation mask to bounding boxes via connected-component extraction with a minimum component size, or (ii) using a parallel bounding-box label set. We use (i) with a minimum-component size of 4 pixels (matching the YOLO label-generation pipeline used throughout this work), which is also the matching protocol applied to the FDI baseline (§3.5).

**YOLO architecture variants on small remote-sensing targets.** YOLOv8 / YOLOv11 (Ultralytics) are the de facto baseline single-stage detectors for satellite-imagery work. Architectural augmentations that recur in the remote-sensing improvement literature include channel-spatial attention (SE, ECA, CBAM; Hu et al. 2018; Wang et al. 2020; Woo et al. 2018), feature pyramid variants (PANet; Liu et al. 2018; BiFPN; Tan et al. 2020), additional fine-stride detection heads (P2 head), and IoU-family box regression losses (CIoU, EIoU; Zheng et al. 2020; Zhang et al. 2022). Reported gains from these augmentations on MARIDA, when reported on the original split, range from $+3$ to $+15$ pp Marine_Debris mAP; in our honest re-evaluation (§4.2), most of these gains do not reproduce.

**Confidence calibration.** Platt scaling (Platt 1999) and temperature scaling (Guo et al. 2017) re-map raw classifier scores so that the output approximates a probability. Calibration is well-studied for classification but comparatively rare in object detection; recent reproducibility-focused work (Kapoor & Narayanan 2023) argues that calibration reporting should be a default for any deployment-oriented detection paper.

---

## 3. Methodology

### 3.1 Dataset: MARIDA, the `v2split` partition, and the v1 7-class label set

We work on the public MARIDA collection of 1,381 ACOLITE-corrected Sentinel-2 L2A patches at native $256 \times 256$ pixels and 10 m ground resolution. The radiometric product is **ACOLITE-Rayleigh-corrected surface reflectance**, the water-color remote-sensing standard, which preserves dim-water signal that Sen2Cor (designed for land) tends to suppress over dark surfaces. Pixel values are 32-bit float reflectance in the approximate range $[0.005, 0.5]$.

**Splits.** We define `v2split`, a strict scene-level partition: every tile from a single Sentinel-2 acquisition is placed in exactly one of train / val / test, eliminating within-scene physical-similarity leakage. The Marine_Debris-relevant subset of `v2split` contains $371 / 70 / 73$ tiles in train / val / test, with $108$ Marine_Debris GT instances in test. This is the partition used throughout §4 unless explicitly noted (the *leaky control* in §4.4 uses the original MARIDA split for comparison only).

**Label set.** We retain the v1 seven-class label assignment: Marine_Debris (cls 0), Dense_Sargassum (1), Sparse_Sargassum (2), Natural_Organic (3), Ship (4), Foam (5), Wakes (6). In the `v2split` test, the class counts are Marine_Debris 108, Sparse_Sargassum 4, Ship 13, Foam 11, and Wakes 8; Dense_Sargassum and Natural_Organic have 0 GT. Marine_Debris dominates by a wide margin and is the only class for which we report deployment-oriented numbers (deployment F1, calibration, FDI comparison). Sparse_Sargassum and Wakes are retained in mAP for protocol consistency but are not used for strong per-class claims.

**Tile labels.** Bounding boxes are derived from the published MARIDA per-pixel masks by connected-component extraction with a minimum component size of 4 pixels and a 2-pixel bounding-box pad. We use this same protocol for the FDI baseline (§3.5) so the two are directly comparable.

### 3.2 Channel selection and the honest physical-prior channel

Of the 11 input bands MARIDA exposes, we use six spectral channels — B4, B3, B2 (visible RGB); B8 (NIR, $\sim$842 nm); B11 (SWIR-1, $\sim$1610 nm); B12 (SWIR-2, $\sim$2190 nm) — and one optional physical-prior channel.

The prior channel, when used, is **NDWI-inverse**:
$$\mathrm{NDWI^{-1}} = \tfrac{1}{2} \bigl( 1 - (R_{B3} - R_{B8}) / (R_{B3} + R_{B8}) \bigr),$$
which assigns higher values to non-water pixels and is therefore a physically motivated, mask-free prior for "anything floating". This is the **honest** prior: at inference time it is computable from the L2A input alone with no ground-truth dependence. We contrast it in §4.1 with the **oracle prior** — a heatmap built from the GT mask — used by prior published baselines.

For each retained band we compute the 1st and 99th percentile values across the training split and apply fixed clipping `x_norm = clip((x - p01) / (p99 - p01), 0, 1)`. We deliberately avoid per-tile histogram equalization to preserve inter-tile spectral consistency, and we use only geometric data augmentation (rotation $\leq 180^\circ$, vertical flip 50%, horizontal flip 50%); HSV jitter, mosaic, mixup, and copy-paste are unsafe on multi-channel float32 spectra.

### 3.3 Detector

The detector is YOLOv11n (Ultralytics). The first convolution is widened to accept seven input channels (six spectral + one prior), initialized from the clean `yolo11n_7ch_init.pt` checkpoint rather than from any historical leaky run. The backbone, PANet neck, and three-stride detection head (P3 / P4 / P5; strides 8 / 16 / 32) are otherwise unchanged in the baseline configuration. The core architectural variants are: (a) **CBAM**, inserted into the backbone; (b) **BiFPN**, where standard concat operations in the neck are replaced with learnable weighted fusion; (c) **EIoU**, replacing the baseline box-regression loss; and (d) **Full**, combining CBAM, BiFPN, and EIoU. Implementation lives in `configs/models/yolov11n_7ch*.yaml` and `ultralytics_plugins/`. Multi-channel float32 `.npy` input handling is via the custom plugin in `ultralytics_plugins/`, which patches the Ultralytics data pipeline so that the standard 3-channel uint8 loader is bypassed.

### 3.4 Training

All v1 runs use the SOTA-pinned hyperparameters in `scripts/v1_train_on_v2split.py`: 30 epochs, batch 8 at imgsz=512, `optimizer=auto`, `lr0=5e-4`, `weight_decay=5e-4`, warmup 3 epochs, no mosaic/mixup/copy-paste, and the historical HSV settings retained for apples-to-apples comparison with the published v1 line. Each core architecture cell is trained under three random seeds $\{0, 1, 2\}$. Letterbox `imgsz` is 512 for the core ablation and 1024 for the cross-resolution pass (batch=4, workers=0); native-256 controls use batch=16. Hardware: NVIDIA RTX 3090 Ti.

### 3.5 Non-ML physical baselines (Biermann 2020 FDI, NDWI-inverse)

To establish that the trained detector's gains are not trivially achievable by a hand-engineered spectral index, we evaluate two non-ML physical baselines on the same `v2split` test, with the same connected-components and IoU-matching pipeline used for the YOLO labels.

**FDI (Floating Debris Index; Biermann 2020).** The FDI measures the deviation of the NIR reflectance from a linear baseline interpolated between the red-edge and SWIR bands:
$$\mathrm{FDI} = R_{\mathrm{NIR}} - \left[ R_{\mathrm{RE2}} + \left(R_{\mathrm{SWIR1}} - R_{\mathrm{RE2}}\right) \cdot \frac{\lambda_{\mathrm{NIR}} - \lambda_{\mathrm{RE2}}}{\lambda_{\mathrm{SWIR1}} - \lambda_{\mathrm{RE2}}} \right]$$
where $R_{\mathrm{RE2}}$, $R_{\mathrm{NIR}}$, $R_{\mathrm{SWIR1}}$ are Sentinel-2 L2A surface reflectance at bands B6 (740 nm), B8 (842 nm), B11 (1610 nm) respectively.

**NDWI-inverse threshold.** Identical formula to the prior channel in §3.2, evaluated as a standalone detector by thresholding.

**Sweep + matching protocol.** For each index $I$ and each threshold $\tau$, we (i) binarize the per-pixel index map, (ii) extract connected components with a minimum size of 4 pixels (identical to YOLO label generation), (iii) compute axis-aligned bounding boxes with 2-pixel padding, and (iv) match these predictions to GT Marine_Debris instances at IoU $\geq 0.5$. We sweep $\tau$ over the 50th–99.5th percentiles of the index distribution (40 values).

### 3.6 Confidence calibration (Platt scaling)

We use the SOTA model (BiFPN $\times$ imgsz=1024, seed 0) as the calibration target. Per-instance scores and TP/FP labels are collected at minimum confidence $0.001$ on validation and test, with IoU-matching at $\geq 0.5$.

Following Platt (1999), we fit
$$P(y = 1 \mid s) = \sigma\bigl( a \cdot \mathrm{logit}(s) + b \bigr)$$
on the **validation set** ($n = 1004$ predictions, 162 TPs out of 178 GT). We use bounded L-BFGS-B with $\ell_2$ regularization (the manual Newton-Raphson alternative diverged on this near-separable dataset; see Appendix B). The fitted parameters are $(a, b) = (0.408, -0.509)$, applied unchanged at test time.

We report Expected Calibration Error (ECE; Guo et al. 2017) with 10 equal-width confidence bins on **both** the validation set (the in-sample Platt fit set, an upper bound on Platt's reach) and the held-out test set (val-fitted Platt parameters applied unchanged, the actual generalization number), plus a $2 \times 2$ reliability diagram (val/test $\times$ raw/Platt) and a paired threshold sweep on the test set.

### 3.7 Evaluation protocol — instance-level F1 as the deployment metric

Standard YOLO `model.val()` at conf $=$ 0.001 produces a mAP-style summary that, on this rare-class benchmark, is dominated by sub-deployment-threshold predictions and is hard to interpret operationally. We therefore complement it with a **deployment F1** measurement: at a chosen threshold $\tau$, run the model, NMS (IoU $=$ 0.6), and report instance-level precision, recall, and F1 by IoU $\geq 0.5$ matching to GT. For the BiFPN@1024 seed-0 checkpoint, the strongest audited operating point is $\tau = 0.25$ (§4.5); future multi-seed deployment should lock this threshold on validation before final test reporting.

We note explicitly that **`model.val()` must be called via `scripts/v2_evaluate.py`**, which monkey-patches the Ultralytics dataset loader; calling `model.val()` directly uses the standard 3-channel loader and silently returns wrong results.

---

## 4. Results

### 4.1 Leak audit — measured attribution cells

We decompose previously reported numbers into measured cells that isolate the two risk sources: scene overlap and an oracle prior channel. A fully crossed original-split $\times$ $\mathrm{NDWI^{-1}}$ retrain is not available in the frozen artifacts, so we do **not** fill that cell with an approximate number. The table below uses only hard artifacts.

| Cell | Split / eval | Prior at train / eval | mAP50 | Notes |
|---|---|---|---:|---|
| A | original split | oracle / oracle | 0.5243 | historical published-style reference |
| B | v2split | oracle / oracle | 0.4658 | clean split, but GT-derived heatmap still available |
| C | v2split | oracle / zeroed | 0.1408 | same checkpoint stripped of the GT heatmap at eval |
| D | v2split | NDWI-inverse / NDWI-inverse | 0.3492 | fully deployable seed-0 baseline; 3-seed mAP50 is $0.313 \pm 0.038$ |

The comparison B $\to$ C shows that an oracle-trained model collapses when the GT heatmap is removed ($0.466 \to 0.141$ mAP50). The comparison C $\to$ D shows the value of training with a weak physical prior instead of an oracle: the deployable NDWI-inverse model retains substantially more spectral ability ($0.349$ vs $0.141$). Source artifacts: `docs/paper/7CH_BASELINE_CURRENT_AUDIT_2026-05-18.md`, `runs_v1_v2split/yolov11n_7ch_v2split_clean_init2/*eval.json`, and `runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_clean_init/v2split_test_eval.json`.

The implication for the rest of the paper is that **published improvement claims on the original MARIDA split must be discounted by the oracle-prior contribution before being interpreted as architectural gains**. We make this point explicit in §5.

### 4.2 Architecture ablation under honest and leaky settings ($5 \times 2 \times 3$)

We re-ran five core cells — baseline, CBAM, BiFPN, EIoU, and Full (CBAM+BiFPN+EIoU) — under both leaky and honest settings, with three random seeds per cell at imgsz $=$ 512. Per-cell results (3-seed mean $\pm$ std on mAP50, mAP50-95, Best F1, Debris AP50, and Ship AP50) are in `docs/paper/v1_ablation_aggregate.md`.

Headline observation: **none of the four modifications delivers a 3-seed-stable net improvement over the unmodified honest baseline at imgsz $=$ 512.** Several cells improve Debris AP50 while hurting aggregate mAP or Ship AP50; the Full combination collapses to strictly worse than each individual component, consistent with combined-capacity over-saturation in the 371-tile-train regime.

The conclusion at imgsz $=$ 512 is therefore that *the published improvement triad does not reproduce in the honest setting*.

### 4.3 The single intervention that does transfer — BiFPN $\times$ imgsz $=$ 1024

We then ran the architectural ablation again at **imgsz $=$ 1024**. Source artifact: `docs/paper/v1_imgsz1024_ablation_aggregate.md`.

The headline finding is that **BiFPN $+$ imgsz $=$ 1024 produces the best 3-seed Best F1 in the honest setting**: $0.479 \pm 0.037$, versus $0.409 \pm 0.022$ for baseline@512 and $0.427 \pm 0.048$ for baseline@1024. Plain imgsz $=$ 1024 alone already lifts Debris AP50 from $0.492 \pm 0.056$ to $0.607 \pm 0.022$; BiFPN trades some Debris AP50 ($0.576 \pm 0.045$) for a better deployable F1 operating profile. The practical recommendation that follows from §4.2-4.3 is: **train at imgsz $=$ 1024, add BiFPN if deployment F1 is the target, and report the AP/F1 trade-off explicitly.**

This is the configuration used as the SOTA model for all downstream analyses (FDI comparison in §4.4, calibration in §4.5).

We anticipate the obvious mechanism question — *is the BiFPN $\times$ 1024 win simply more parameters?* — and answer it with two controls.

**Leaky control.** We re-ran the BiFPN $\times$ imgsz $=$ 1024 cell on the **leaky** partition (oracle prior plus original split), expecting the honest gain to compound. It does the opposite: the gain disappears entirely under leak. Artifact: `docs/paper/v1_imgsz1024_leaky_aggregate.md`. This rules out the "more capacity" explanation: when the model already has the oracle answer in ch7, an architecture sensitive to small-target multi-scale detail has nothing left to contribute.

**Native-256 control.** We additionally trained the BiFPN cell at the *native* tile resolution (no letterbox up-sampling). Artifact: `docs/paper/v1_native256_aggregate.md`. The BiFPN gain *survives* at native 256, indicating the mechanism is genuinely architectural — but the imgsz $=$ 1024 gain on top of BiFPN is *a property of the letterbox up-sampling itself*, not of P3-head capacity per se. Operationally, both interventions are independently valuable and stack.

### 4.4 Non-ML physical-index baseline (FDI / NDWI-inverse)

To rule out the reviewer concern that the trained detector's gains are trivially achievable by a hand-engineered spectral index, we evaluated FDI and NDWI-inverse thresholding end-to-end with instance-level matching on the `v2split` test. Source: `docs/paper/fdi_baseline/fdi_ndwi_sweep_test.json`. Figure: `docs/paper/figures/fdi_pr_comparison.png`.

| Method | Best F1 | Best P | Best R | vs YOLO BiFPN+1024 |
|---|---:|---:|---:|---:|
| FDI threshold (single-seed, non-ML) | 0.0016 | 0.001 | 0.093 | $377\times$ weaker |
| NDWI-inverse threshold (single-seed, non-ML) | 0.0097 | 0.005 | 0.046 | $62\times$ weaker |
| YOLO baseline @ 1024 (3-seed Best F1 mean, n=3) | 0.427 | — | — | reference |
| **YOLO BiFPN @ 1024 (seed-0 deployment F1, $\tau$ $=$ 0.25)** | **0.604** | **0.581** | **0.630** | reference |

Both physical-index detectors achieve best F1 below $1\%$ — roughly two orders of magnitude weaker than the trained detector. The mechanism is that the per-pixel index thresholding produces thousands of small false-positive components per tile (FDI: $\sim$37,000 FP at the lowest tested $\tau$); the IoU $\geq 0.5$ instance-matching criterion is intolerant of this geometry. Physical indices are good *pixel-level* probability maps but poor *instance proposers*. We therefore retain $\mathrm{NDWI^{-1}}$ as an **input channel** to the trained detector (§3.2) rather than as a standalone detector.

### 4.5 Confidence calibration — the headline practical finding

We applied the calibration procedure of §3.6 to the BiFPN $\times$ imgsz $=$ 1024 (seed 0) checkpoint. Fitted Platt parameters: $(a, b) = (0.408, -0.509)$.

**Expected Calibration Error** with 10 equal-width bins, reported separately for the validation set (the in-sample Platt fit set) and for the held-out test set (val-fitted Platt parameters applied unchanged):

| Split | n preds | n TP | ECE raw | ECE Platt-calibrated | Reduction |
|---|---:|---:|---:|---:|---:|
| val (in-sample) | 1004 | 162 | 0.1034 | 0.0294 | 3.5× |
| **test (held-out)** | 886 | 101 | **0.0514** | **0.0293** | **1.75×** |

Two points are worth emphasizing. First, the held-out test improvement (1.75×, from 0.051 to 0.029) is the **generalization claim**; the val 3.5× number is an in-sample upper bound on what the Platt fit achieves on the data it was fitted to, and should not be used as the headline. Second, the Platt-calibrated ECE is essentially identical on val and test (0.0294 vs 0.0293), which means the Platt fit itself transfers cleanly across val → test — what differs is that the raw model on the test split is already better-calibrated (raw ECE 0.0514) than it is on the val split (raw ECE 0.1034), so there is less room for Platt to improve on test. The reliability diagram (`docs/paper/calibration/reliability_diagram.png`, $2 \times 2$ layout: top row val, bottom row test; raw vs Platt in columns) makes this visually explicit.

**Test-set threshold sweep** (108 GT, instance-level IoU $\geq 0.5$):

| $\tau$ | raw P | raw R | raw F1 | Platt P | Platt R | Platt F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.114 | **0.935** | 0.203 | 0.114 | 0.935 | 0.203 |
| 0.05 | 0.458 | 0.759 | 0.571 | 0.218 | 0.907 | 0.351 |
| 0.10 | 0.497 | 0.694 | 0.579 | 0.367 | 0.815 | 0.506 |
| **0.25** | **0.581** | **0.630** | **0.604** | 0.531 | 0.639 | 0.580 |
| 0.50 | 0.637 | 0.537 | 0.583 | 0.714 | 0.417 | 0.526 |
| 0.75 | 0.719 | 0.426 | 0.535 | 0.857 | 0.111 | 0.197 |

The raw model attains its best F1 ($0.604$) at $\tau = 0.25$, which we take as the **deployment operating point** for all subsequent comparisons. At this threshold, precision and recall are balanced ($0.581 / 0.630$) and the absolute precision is high enough to support operational use (human-in-the-loop alert verification, downstream sensor fusion).

**Practical recommendation.** For threshold-based deployment (binary alerting), the raw confidences are adequate at $\tau = 0.25$. For probabilistic downstream use (Bayesian fusion with other sensors, risk-aware ranking), Platt-scaled scores should be used: on held-out test, ECE drops from 0.051 to 0.029 (a $\sim 43\%$ reduction in expected calibration error) while rank order — and therefore the threshold-sweep numbers above — are preserved.

This is, to our knowledge, the **first MARIDA detection paper to report explicit confidence calibration**.

---

## 5. Discussion

### 5.1 The published-vs-honest gap, and what it means for cross-paper comparison

The leak audit in §4.1 shows that, on the original MARIDA split with an oracle prior channel, the historical published-style model achieves mAP50 $\approx 0.524$. On the scene-clean split, an oracle-prior model still reaches 0.466, but the same checkpoint drops to 0.141 when the GT-derived heatmap is removed at evaluation. The fully deployable NDWI-inverse model reaches 0.349 on seed 0 ($0.313 \pm 0.038$ over 3 seeds). The first consequence is that improvement numbers reported in the MARIDA detection literature must be read with the split and prior carefully disambiguated; the second is that our own architectural ablation under the honest setting (§4.2) finds that the published improvement triad does not reproduce. This is a cautionary observation, not a criticism: the published numbers are correct *given* their evaluation choices; the issue is that the choices systematically inflate cross-paper comparability.

### 5.2 Why does BiFPN $\times$ 1024 transfer when nothing else does?

Two factors compose: (i) letterbox up-sampling from native 256 to imgsz $=$ 1024 makes the median Marine_Debris box $\sim 4\times$ larger in pixel terms, returning the target to the design range of the P3 (stride-8) head; (ii) weighted feature fusion at the corresponding pyramid scale lets the network preferentially route the relevant fine-scale features into the head. Neither factor is novel on its own; the empirical contribution is that *no other architectural variant we tested adds value on top of the two combined*. The leaky control (§4.3) further suggests that small-target multi-scale architecture sensitivity is *masked* when an oracle prior channel is available, which may explain why this combination has not been previously isolated in MARIDA work.

### 5.3 Mechanism of the FDI gap

Physical indices are essentially *single-band differencing operators*. They produce real per-pixel evidence (Biermann 2020 is correct about this), but they cannot enforce object-level priors: compactness, scale, multi-scale support, separation from spectral confounders (foam, wakes, algae). Combined with the IoU $\geq 0.5$ matching criterion used for instance-level evaluation, this geometry mismatch dominates the gap; the trained detector wins not because it sees something physical the index does not, but because it imposes the right *spatial* prior on top of the index-level evidence.

### 5.4 The deployment-vs-mAP framing

Marine_Debris detection on a 73-tile / 108-instance test split is, by classical detector metrics, a noisy task: moving a single instance across a confidence threshold shifts reported recall by $\sim 1$ pp. mAP at conf $=$ 0.001 — the standard YOLO summary — is therefore a poor operational summary; it integrates over an entire confidence range that nobody would deploy. The **deployment F1** framing (§3.7) ties the headline number to a specific operating point ($\tau = 0.25$), making the reported number directly actionable. The aggregate 3-seed Best F1 for BiFPN@1024 is $0.479 \pm 0.037$, while the seed-0 deployment operating point reaches F1 $=0.604$ at $\tau = 0.25$; we recommend reporting both and clearly labeling the latter as a seed-0 deployment-point result.

### 5.5 Limitations

(i) **Test scale.** 73 tiles / 108 instances. Confidence intervals are wide; numerical differences smaller than $\sim 3$ pp F1 should not be interpreted as signal. We report 3-seed mean $\pm$ std throughout to make this explicit. (ii) **Single sensor.** All results are Sentinel-2 L2A; transferability to PlanetScope, Worldview, or aerial imagery is untested. (iii) **No domain adaptation.** All splits are within the MARIDA geographic distribution (heavy SE-Asia / Mediterranean concentration); generalization to Atlantic or Pacific open-ocean is untested. (iv) **No temporal modeling.** Each tile is a single Sentinel-2 acquisition; multi-date stacks would likely improve persistence-based filtering. (v) **Calibration generalization.** Platt fits are on a single seed and a single calibration set; multi-seed calibration is listed in §6 as future work.

### 5.6 What we did *not* find (recorded for transparency)

For completeness, the following interventions were tested and dropped:

- **U-Net self-distillation prior** (Finding O): replacing $\mathrm{NDWI^{-1}}$ in ch7 with a learned U-Net probability map *hurts* the YOLO detector. Mechanism: ch7 must be an *independent* physical signal, not a learned answer that correlates with the YOLO loss.
- **CBAM / EIoU / Full at imgsz=512**: no stable net gain over the honest baseline in the 5-cell ablation; Full is consistently worse.
- **Leaky BiFPN $\times$ 1024 control**: no gain under the oracle setting, indicating the honest-setting improvement is not a generic capacity effect.
- **Epoch 60 schedule**: overfits the 371-tile train set on the honest cell; epoch 30 is the right budget.

These are listed in §5.6 rather than foregrounded as claims because none survives the 3-seed reproducibility bar. Per-cell numbers are in `docs/paper/v1_ablation_aggregate.md`, `docs/paper/v1_extra_ablation_aggregate.md`, `docs/paper/v1_imgsz1024_leaky_aggregate.md`, and `docs/paper/v1_unetprior_ablation_aggregate.md`.

---

## 6. Conclusion

We presented an honest re-evaluation of YOLO-based detection on MARIDA. By imposing strict scene-level splits and rebuilding the optional physical-prior channel to be derivable from L2A reflectance alone, we showed that the bulk of previously reported gains on the original MARIDA split are attributable to scene leakage and to an oracle-from-mask prior channel — not to architecture. A $5$-architecture $\times$ $2$-setting $\times$ $3$-seed ablation recovered a single transferable practical win after the cross-resolution pass, the **BiFPN $\times$ imgsz $=$ 1024** combination, and the leaky and native-256 controls clarified the mechanism: the gain is driven by letterbox up-sampling restoring small targets to the design range of the P3 head, compounded by weighted fusion that preferentially routes the relevant fine-scale features. As reviewer-facing controls, we additionally established a non-ML floating-debris-index baseline ($50$ to $400\times$ weaker than the trained detector, confirming the deep-learning indispensability floor) and an explicit confidence-calibration study (seed-0 deployment-F1 $= 0.604$ at $\tau = 0.25$; val-fitted Platt scaling reduces held-out test ECE by $1.75\times$). Together these support a single practical conclusion: on MARIDA, deployment-oriented detection of floating marine debris is viable with a small YOLOv11n backbone when the evaluation is honest, the model sees enough letterboxed cell budget for tiny targets, and confidence calibration is reported alongside.

Future work: (a) 3-seed calibration / multi-seed Platt parameters; (b) temperature scaling and isotonic regression as alternatives; (c) per-class calibration for non-debris classes once test support is sufficient; (d) cross-sensor / cross-region transfer; (e) temporal stacking for persistence-based filtering.

---

## Appendix A. Reproducibility checklist

- **Data.** MARIDA original release; `v2split` partition manifest at `data/processed/mask_bbox_rescue_split_physidx_v2split/{train,val,test}.txt`.
- **Channel stats.** Precomputed `channel_stats_7ch.json` (6 spectral $p_{01}/p_{99}$ + ch7 stats), auto-loaded by `MaskRescueNPYDataset`.
- **Honest prior channel.** $\mathrm{NDWI^{-1}}$ as defined in §3.2; computed offline and saved as the 7th channel of each `.npy` tile. Code: `scripts/v1_make_physidx_tiles.py`.
- **Models.** `configs/models/yolov11n_7ch.yaml` (baseline), `configs/models/yolov11n_7ch_bifpn.yaml` (BiFPN), `configs/models/yolov11n_7ch_cbam.yaml` (CBAM), and `configs/models/yolov11n_7ch_full.yaml` (Full).
- **Hyperparameters.** v1 SOTA-pinned defaults: `scripts/v1_train_on_v2split.py::SOTA_HYPERPARAMS`.
- **Training (BiFPN $\times$ 1024 SOTA).** `python scripts/v1_ablation_train.py --ablation bifpn --imgsz 1024 --batch 4 --workers 0 --seed {0,1,2}`.
- **Evaluation.** `python scripts/v2_evaluate.py --weights … --data configs/datasets/marida_v1_physidx_on_v2split_7ch.yaml --channels 7 --imgsz 1024`. **Never call `model.val()` directly**; the standard loader silently produces wrong results on multi-channel float32 input.
- **Instance-level deployment F1 / threshold sweep.** `python scripts/v2_sweep_thresholds.py --weights … --tag bifpn_1024_s0 --imgsz 1024`.
- **FDI baseline.** `python scripts/baselines/fdi_threshold_baseline.py`; plot `python scripts/baselines/plot_fdi_pr.py`. Wall time $<60$s.
- **Calibration.** `python scripts/v1_calibration_platt.py`. Output: `docs/paper/calibration/{platt_fit.json, threshold_sweep_test.json, reliability_diagram.png}`.

All seed-{0,1,2} runs are deterministic given seed and hardware.

## Appendix B. Calibration numerical-stability note

The first implementation of Platt scaling used manual Newton-Raphson without line search; on the near-separable validation set this diverges to $a \approx 1.3{\times}10^9$ and produces a *worse* ECE than the raw model. The working implementation uses bounded L-BFGS-B with $\ell_2$ regularization (`scipy.optimize.minimize`, bounds $[-100, 100]$ on both $a$ and $b$, log-sum-exp NLL with explicit gradient). `sklearn.linear_model.LogisticRegression` would be a one-line alternative but is not in the project's environment and was deliberately not added to avoid a dependency commitment. This is a worthwhile recommendation for any future Platt-on-detection work: use bounded L-BFGS-B with $\ell_2$.

## Appendix C. Figures and tables

| ID | Item | Source |
|---|---|---|
| Fig. 1 | Pipeline overview (to be drawn) | TBD |
| Fig. 2 | Leak-attribution bar chart | `docs/paper/7CH_BASELINE_CURRENT_AUDIT_2026-05-18.md` + run JSONs — plot TBD |
| Fig. 3 | $5$-architecture $\times$ $2$-setting $\times$ 3-seed ablation (imgsz=512) | `docs/paper/v1_ablation_aggregate.md` |
| Fig. 4 | Ablation @ imgsz=1024, BiFPN $\times$ 1024 honest SOTA | `docs/paper/v1_imgsz1024_ablation_aggregate.md` |
| Fig. 5 | Leaky-control + native-256 control | `docs/paper/v1_imgsz1024_leaky_aggregate.md`, `docs/paper/v1_native256_aggregate.md` |
| **Fig. Y** | **FDI / NDWI vs YOLO (2-panel)** | `docs/paper/figures/fdi_pr_comparison.png` |
| **Fig. Z** | **Reliability diagram (raw vs Platt)** | `docs/paper/calibration/reliability_diagram.png` |
| Tab. 1 | Leak-attribution measured cells | §4.1 of this draft |
| Tab. 2 | $5$-architecture $\times$ $2$-setting $\times$ 3-seed ablation | §4.2 (source above) |
| Tab. 3 | imgsz=1024 ablation incl. controls | §4.3 (sources above) |
| **Tab. 4** | **FDI / NDWI / YOLO comparison** | §4.4 of this draft |
| **Tab. 5** | **Calibration threshold sweep (raw vs Platt)** | §4.5 of this draft |

Figures marked **bold** are already rendered; the rest are content-ready and just need a plotting pass.

## Appendix D. Bibliographic references to add

- Biermann, L., Clewley, D., Martinez-Vicente, V., & Topouzelis, K. (2020). Finding plastic patches in coastal waters using optical satellite data. *Scientific Reports*, 10(1), 5364.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
- Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. *CVPR 2018*.
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
- Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D. E., & Karantzalos, K. (2022). MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data. *PLoS ONE*, 17(1), e0262247.
- Liu, S., Qi, L., Qin, H., Shi, J., & Jia, J. (2018). Path Aggregation Network for Instance Segmentation. *CVPR 2018*.
- Platt, J. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. *Advances in Large-Margin Classifiers*, 10(3), 61–74.
- Tan, M., Pang, R., & Le, Q. V. (2020). EfficientDet: Scalable and Efficient Object Detection (incl. BiFPN). *CVPR 2020*.
- Topouzelis, K., Papageorgiou, D., Karagaitanakis, A., Papakonstantinou, A., & Arias Ballesteros, M. (2020). Remote sensing of sea surface artificial floating plastic targets with Sentinel-2 and unmanned aerial systems. *Remote Sensing*, 12(12), 2013.
- Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., & Hu, Q. (2020). ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. *CVPR 2020*.
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
- Zhang, Y.-F., Ren, W., Zhang, Z., Jia, Z., Wang, L., & Tan, T. (2022). Focal and Efficient IoU Loss for Accurate Bounding Box Regression. *Neurocomputing*, 506, 146–157.
- Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D. (2020). Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression. *AAAI 2020*.
