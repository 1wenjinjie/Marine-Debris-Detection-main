# Multispectral Extension Experiments

This note tracks the follow-up experiments we outlined (NDVI/FDI/GLCM variants and knowledge-weight tweaks). Each subsection acts as a lightweight contract: it summarizes the hypothesis, concrete actions, success criteria, and validation hooks so we can iterate fast without derailing the mainline training.

## 1. Spectral Index Augmentation (NDVI/FDI)

- **Hypothesis**: Explicit vegetation-related indices (NDVI / Floating Debris Index) will help the network disambiguate Sargassum vs marine debris without needing a deeper backbone.
- **Implementation sketch**:
  1. Extend `preprocess_generate_knowledge_channel.py` (or a sibling script) to derive NDVI = (B8 − B4)/(B8 + B4) and FDI = B8 − (B6 + B11)/2 from the original Sentinel-2 11-band stacks before channel clipping.
  2. Append the indices into the `.npy` tensors as channels 8 and 9 (keeping YAML `channels` in sync). Optionally keep a version that replaces the knowledge heatmap so we can isolate the effect.
  3. Regenerate `channel_stats_9ch.json` via `tools/compute_channel_stats.py --channels 9` and update the dataset loader map.
- **Validation plan**:
  - Train a slim 9-channel run mirroring `yolov11n_7ch_full_return_cb` hyperparameters.
  - Use `tools/run_val_with_outputs.py` + `plot_publication_metrics.py --category-offset 1` for apples-to-apples metrics.
  - Primary success criteria: +3 mAP50 for `Dense/Sparse Sargassum`, no regression on `Marine_Debris`.

## 2. Texture-aware Channels (GLCM)

- **Hypothesis**: Garbage and foam exhibit higher local texture variance than sargassum mats; injecting Grey-Level Co-occurrence Matrix (GLCM) features will boost precision on rare foam/wakes classes.
- **Implementation sketch**:
  1. Build a small utility (e.g. `tools/derive_glcm_channels.py`) that loads cached `.npy`, computes per-pixel contrast/entropy over a 7×7 window on the visible bands, and appends 1–2 texture channels.
  2. Cache outputs alongside existing `.npy` (use deterministic naming such as `_glcm.npy`).
  3. Update `MaskRescueNPYDataset` to prefer the extended tensors when present so back-compat is preserved.
- **Validation plan**:
  - Keep existing knowledge channel and indices disabled to isolate the impact of texture across two ablation runs (texture-only vs texture+NDVI/FDI).
  - Track PR curves for `Foam` and `Wakes`—success is >+5% recall while holding precision >0.3.

## 3. Adaptive Knowledge Weights

- **Hypothesis**: Confidence weights derived from masks underplay the rare categories. Combining them with class-frequency priors (new `class_balance_weights.json`) or entropy of knowledge heatmaps should yield fairer sampling.
- **Implementation sketch**:
  1. Leverage the newly-added class balance hook inside `MaskRescueNPYDataset` (already loading `class_balance_weights.json`).
  2. Expose CLI knobs in `train_7ch_main.py` (`--class-balance-power`, `--class-balance-floor`) once Ultralytics 8.4 lands so we can sweep power ∈ {0.5, 1.0, 1.5} without touching code.
  3. Prototype an entropy-aware scaler: compute per-sample knowledge entropy (available in `knowledge_generation_summary.json`) and add it as a multiplier (higher entropy ⇒ more sampling).
- **Validation plan**:
  - Run paired trainings (confidence-only vs confidence×class-balance vs confidence×class-balance×entropy).
  - Compare PR/F1 at operating point (best F1 from publication plots). Goal: +0.05 recall on `Dense/Sparse Sargassum` with ≤1% drop in `Marine_Debris` precision.

## 4. Execution Checklist

- [ ] Finish current class-balanced baseline (`yolov11n_7ch_full_return_cb`).
- [ ] Cut a short experiment log (table with run name → features → metrics) in `docs/experiments.md` once figures are in.
- [ ] Prioritize NDVI/FDI branch first (low engineering cost), then texture, then knowledge entropy.
- [ ] Use `run_val_with_outputs.py --save_json` + `plot_publication_metrics.py` for every run so plots stay consistent for the paper.

This file should evolve alongside the experiment backlog—append new sections when we spin up an idea, then link results/plots back to `docs/experiments.md` for traceability.
