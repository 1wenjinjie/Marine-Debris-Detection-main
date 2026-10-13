# Figure Design Notes — fig1_pipeline.svg and fig2_modules.svg

Reference for the paper author when drafting captions and cross-referencing sections.

---

## Global Style

- **Palette:** navy primary `#1f3a5f`, orange accent `#e88a3a`, neutral grays.
- **Fonts:** Inter/Helvetica/Arial for labels; JetBrains Mono/Menlo for tensor shapes, formulas, and code-like symbols.
- **No gradients, no drop shadows.** Fills are flat. Dashed borders signal optional or de-emphasized elements.

---

## Figure 1 — `fig1_pipeline.svg` (1700 × 560)

End-to-end inference pipeline, reading left-to-right across six stages.

### Stage 1: Input
- **Visual:** stacked colored band strips representing a MARIDA tile.
- **Label:** "MARIDA Tile · 256×256 px · 11 bands, L2A surface reflectance"
- **Small note:** "ACOLITE-corrected, native resolution"
- **Caption guidance:** Emphasize that the tile is *native* 256×256 — no super-resolution preprocessing.

### Stage 2: Channel Construction
- **Visual:** six colored squares (B4–B12) + one orange-bordered box for ch7.
- **ch7 formula:** `(1 − (B3−B8)/(B3+B8)) / 2`
- **Key claim shown:** "deterministic from L2A; no GT dependency at inference" — this differentiates from leaky baselines (see Fig. 2b).
- **Tensor annotation:** `[7, 256, 256]`
- **Caption guidance:** Cite §3.2 for channel selection rationale. Stress the "physical prior" framing — ch7 is a water-sensitive index, not learned.

### Stage 3: Preprocessing
- **Two sub-steps:** (1) p01/p99 percentile normalization using train-split statistics; (2) letterbox up-sampling to 1024×1024.
- **Important note inside box:** "sensor resolution unchanged; model sees more P3 cells" — this is the correct mechanistic explanation for why imgsz=1024 helps. Do NOT write "higher resolution imagery" in the caption.
- **Tensor annotation:** `[7, 1024, 1024]`
- **Caption guidance:** Cite the letterbox up-sampling explanation from §3.2 / CLAUDE.md note on imgsz.

### Stage 4: Detector (largest box)
Three sub-boxes inside the detector:

#### 4a. Backbone
- 7-ch stem conv + standard YOLOv11n C2f stages (P3/P4/P5).
- **Note shown:** "no COCO pretrain (channel mismatch with ImageNet RGB)" — trained from scratch.
- Caption: mention this as a limitation; no transfer learning benefit.

#### 4b. BiFPN Neck (orange-bordered)
- Replaces PANet concat with weighted bidirectional feature pyramid fusion.
- Shows P5→P4→P3 top-down and P3→P4→P5 bottom-up passes with weight labels w₁, w₂.
- **Caption guidance:** "See Fig. 2a for full fusion formula."

#### 4c. Detect Heads
- Three parallel heads: P5 (stride 32, 32×32), P4 (stride 16, 64×64), **P3 (stride 8, 128×128, orange emphasis)**.
- P3 is highlighted with a star (★) because Marine_Debris targets (~12 px) fall within the P3 receptive field at imgsz=1024.
- **Caption guidance:** Explicitly note "P3 is the operative head for Marine_Debris; at imgsz=512 the target occupies fewer than two cells."

#### Stage 4 subtitle
- "~2.6M parameters · trained from scratch · 30 epochs · 3-seed protocol"

### Stage 5: Post-processing
Three sequential elements:

1. **NMS** — IoU threshold = 0.6 (solid box)
2. **Confidence threshold** — τ = 0.25 (deployment operating point; conf=0.001 for AP evaluation) (solid box)
3. **Platt scaling** — dashed border, optional branch. Logistic model fit on validation set. ECE: 0.103 → 0.029 (3.5× improvement).
   - **Caption guidance:** "Dashed box indicates optional calibration step used for probabilistic downstream applications."

### Stage 6: Output
- Simulated Sentinel-2 patch with three red bounding boxes labeled "MD" (Marine_Debris).
- **Performance metrics (seed 0, BiFPN+imgsz=1024, scene-clean test, τ=0.25):**
  - P = 0.581, R = 0.630, F1 = 0.604
- **Caption guidance:** Specify "seed 0, BiFPN+imgsz=1024, τ=0.25, scene-clean test set" — do not omit seed qualifier.

### Non-ML Baseline Branch (bottom, gray, dashed)
- Two gray boxes: FDI threshold (Biermann 2020 formula) and NDWI-inverse threshold.
- **Numbers shown:**
  - FDI: instance-level F1 = 0.0016
  - NDWI-inv: instance-level F1 = 0.0097
  - Note: "50–400× weaker than YOLO"
- **Visual treatment:** gray fill, dashed container, no orange accents — deliberately de-emphasized.
- **Caption guidance:** "Bottom branch provides a non-ML physical reference (§4.4), confirming deep learning is indispensable for this detection task."

---

## Figure 2 — `fig2_modules.svg` (1660 × 530)

Two-panel figure, divided by a vertical gray line at x=830.

---

### Panel (a): BiFPN Weighted Bidirectional Feature Fusion

**Layout:** three feature level boxes on the left spine (P5/P4/P3), two fusion passes.

#### Feature level boxes
| Box | Color | Size label | Note |
|-----|-------|-----------|------|
| P5 | light blue fill | 32×32 cells | high semantic |
| P4 | medium blue fill | 64×64 cells | intermediate |
| P3 | darker blue + orange border | 128×128 cells | ★ key for Marine_Debris |

#### Top-down pass (navy arrows)
- P5 → fusion node → P4ᵗᵈ → fusion node → P3ᵒᵘᵗ
- Arrow color: navy `#1f3a5f`
- Weight labels: w₁P₅ shown at P4 node

#### Bottom-up pass (orange arrows)
- P3ᵒᵘᵗ → bu-P4 fusion node → P4ᵒᵘᵗ → bu-P5 fusion node → P5ᵒᵘᵗ
- Arrow color: orange `#e88a3a`
- Weight labels: w₁, w₂, w₃ at P4 node

#### Fusion formula box (center)
```
         w₁·Resize(P5) + w₂·P4ᵗᵈ + w₃·Resize(P3)
P4ᵒᵘᵗ = ─────────────────────────────────────────
                    w₁ + w₂ + w₃ + ε
```
- ε = 10⁻⁴ for numerical stability
- Weights softmax-normalized, one set per fusion node per level

#### PANet comparison (gray dashed)
- Small gray box: "standard PANet: simple concat, fixed, no learnable weights"
- **Caption guidance:** "Unlike PANet, BiFPN assigns learned importance to each scale's contribution."

#### Output heads (right edge)
- Three orange boxes: "→ Detect Head P5/P4/P3"
- P3 box is slightly more prominent.

#### Caption text (below panel)
"Top-down pass (navy): higher semantic features flow to coarser levels. Bottom-up pass (orange): fine spatial detail flows back upward. P3★ is critical for detecting ~12 px debris."

---

### Panel (b): Physical-Prior Channel — Honest vs. Leaky

Two columns separated by a vertical light gray line.

#### Left column: Honest Physical Prior (green)
Flow: B3 (Green) + B8 (NIR) → NDWI = (B3−B8)/(B3+B8) → ch7 = (1−NDWI)/2 → "ch7 ∈ [0,1]; high on non-water"

- **Green annotation box:** "Deterministic from L2A input. No GT dependency at inference."
- **Green badge:** "Used in this paper"
- **Caption guidance:** This is the channel used throughout the honest ablation (§3.2, §4.1). Available in any real deployment scenario.

#### Right column: Oracle Prior / Leaky (red)
Flow: GT Segmentation Mask (dashed, GT-ONLY badge) → Gaussian Blur → Heatmap Channel (ch7) → "Inflated mAP50 ≈ 0.52"

- **Red annotation box:** "Requires GT at inference — leaks dataset. Contributes ~70% of published F1 to dataset construction artifact (§4.1)."
- **Red badge:** "Invalid at Inference"
- **Caption guidance:** Cite §4.1 and the leak decomposition result. The oracle prior is what earlier published baselines used.

#### Delta panel (center, between columns)
Three numbers stacked:
- Honest mAP50: 0.264
- Leaky mAP50: 0.524
- Difference: +0.26 (~2× inflation from data construction)

- **Caption guidance:** "The ~0.26 mAP50 gap (orange) quantifies the data construction inflation reported in §4.1."

---

## Numbers to Double-Check Before Submission

These appear in the figures — verify against final experimental logs:

| Figure | Location | Value |
|--------|----------|-------|
| Fig. 1 Stage 6 | P, R, F1 | 0.581, 0.630, 0.604 (seed 0, BiFPN+imgsz=1024, τ=0.25) |
| Fig. 1 Stage 5 | ECE improvement | 0.103 → 0.029 |
| Fig. 1 non-ML | FDI F1 | 0.0016 |
| Fig. 1 non-ML | NDWI-inv F1 | 0.0097 |
| Fig. 1 detector | parameters | ~2.6M |
| Fig. 1 detector | epochs | 30 |
| Fig. 2b delta | Honest mAP50 | 0.264 |
| Fig. 2b delta | Leaky mAP50 | 0.524 |
| Fig. 2 detector | Marine_Debris target size | ~12 px |

---

## Residual Reviewer Concerns

1. **BiFPN diagram (Fig. 2a) simplification.** The top-down pass shown uses one intermediate node per level, but actual BiFPN implementations may differ in the exact number of fusion nodes depending on the depth configuration. Verify against `configs/models/yolov11n_v2_6ch_bifpn.yaml` before finalizing. If the architecture uses a different node count, update the diagram.

2. **P3 emphasis claim.** The statement "Marine_Debris ≈ 12 px → falls in P3 at imgsz=1024" assumes standard YOLO anchor-free assignment. Confirm that at imgsz=1024 the 12 px box does map to P3 (stride 8, 128 grid cells → each cell = 8 px; a 12 px box spans ~1.5 cells, which is reasonable for P3). This is correct but worth confirming in writing with a sentence in §3.3.

3. **Fig. 1 non-ML branch formula.** The FDI formula uses band labels (R_RE2, λ_RE2) that differ from the primary six-channel set (B4/B3/B2/B8/B11/B12). This is correct — FDI uses Red Edge 2 (B6/B7) which is NOT in the YOLO 6-channel set. The figure should note this in the caption or §4.4: "FDI requires Sentinel-2 B6/B7, outside the detector's spectral scope."

4. **Seed labeling.** Fig. 1 Stage 6 shows seed 0 results only. If the final paper uses 3-seed mean ± std as the primary number, replace the Stage 6 metrics with the 3-seed average and note the change here.

5. **Platt scaling ECE numbers.** The ECE 0.103 → 0.029 values are from the calibration analysis (docs/paper/calibration/). Confirm these match the final calibration experiment for BiFPN+imgsz=1024 specifically (not the baseline model).
