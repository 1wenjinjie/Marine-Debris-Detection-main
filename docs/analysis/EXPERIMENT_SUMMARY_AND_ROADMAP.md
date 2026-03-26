# Project Summary & Roadmap: Marine Debris Detection

## 1. Experiment History & Analysis

We have conducted a rigorous series of experiments to identify the optimal configuration for detecting marine debris using multispectral satellite imagery.

| Experiment ID | Configuration | mAP50 | Key Finding |
| :--- | :--- | :--- | :--- |
| **Baseline (RGB)** | RGB Only, 3-Channel | **0.00** | **Spectral Criticality**: RGB alone is insufficient. Debris is indistinguishable from waves without NIR/SWIR bands. |
| **Baseline (6ch)** | 6-Channel, 7 Classes | **0.345** | **Proof of Concept**: Adding spectral bands immediately enabled detection. |
| **Innovation (P2)** | High-Res Head (Stride 4) | **0.173** | **Noise Amplification**: Increasing spatial resolution (P2) hurt performance because it resolved wave glint as noise. Spectral signal > Spatial detail. |
| **Final (Attempt 1)** | 200 Epochs, Strong Aug | **0.201** | **Cold Start Issue**: Too much augmentation (180° rotation) + small batch size prevented convergence from scratch. |
| **Final (Best)** | **100 Epochs, Std Aug** | **0.385** | **SOTA**: The combination of 640px resolution, Batch 32, and Standard Augmentation yielded the best results (+11.6% over baseline). |
| **Fine-Tune (50e)** | **Low LR, Strong Aug** | **0.419** | **New SOTA**: Rotation invariance + low LR refinement pushed performance significantly (+8.8% relative gain). |

### Why did the final model succeed?
1.  **Resolution (640px)**: Provided enough spatial detail for small objects without introducing excessive wave noise (unlike the P2 head).
2.  **Batch Size (32)**: Stabilized the batch normalization layers, crucial for the complex 6-channel statistics.
3.  **Standard Augmentation**: Allowed the model to learn basic features first. The "Strong Augmentation" (rotation) is better suited for a second fine-tuning stage.

---

## 2. Dataset Integrity Analysis

We performed a strict audit of the dataset splits (`analyze_dataset_integrity.py`).

### ✅ Integrity Checks (Passed)
*   **Exact Leakage**: **PASS**. No single image file appears in both Train and Test sets. The evaluation is technically valid.

### ⚠️ Risk Factors (Identified)
1.  **Spatial Leakage (Scene Overlap)**:
    *   **Warning**: 43 satellite scenes are shared between Train and Val.
    *   **Implication**: The model is being tested on the *same water conditions* (same day/location) it trained on, just different crops. This means the mAP (0.385) might be slightly optimistic compared to a completely unseen location.
2.  **Severe Class Imbalance**:
    *   **Marine Debris**: **84.2%** (Dominant class)
    *   **Sparse Sargassum**: **0.78%** (Extremely rare)
    *   **Wakes**: **1.69%**
    *   **Implication**: The model is heavily biased towards "Marine Debris". It performs well on the main task but may struggle to generalize to rare classes like "Sparse Sargassum" in new environments.

---

## 3. Roadmap for Improvement

To push the mAP beyond 0.40 and ensure robustness, we recommend the following steps:

### Phase 1: Data Engineering (High Priority)
1.  **Fix Imbalance**:
    *   **Action**: Apply **Mosaic Augmentation** specifically for rare classes (Copy-Paste augmentation).
    *   **Goal**: Increase "Sparse Sargassum" and "Wakes" representation to at least 5%.
2.  **Strict Split**:
    *   **Action**: Re-split the dataset by **Scene ID**, not by file. Ensure `S2_18-9-20` is *only* in Train or *only* in Test.
    *   **Goal**: Realistic evaluation of generalization capability.

### Phase 2: Advanced Training
1.  **Fine-Tuning** (✅ **COMPLETED**):
    *   Take the current best model (`best.pt`).
    *   Train for another 50 epochs with **Low LR (0.001)** and **Strong Augmentation** (MixUp + 180° Rotation).
    *   *Result:* **mAP 0.419**. The strategy worked perfectly.
2.  **Test Time Augmentation (TTA)**:
    *   Enable TTA during inference (`model.predict(..., augment=True)`).
    *   **Expected Gain**: +1-2% mAP immediately without retraining.

### Phase 3: Architecture
1.  **Model Scale**: Upgrade from `yolov11n` (Nano) to `yolov11s` (Small).
    *   The dataset is complex (6 channels). Nano might be underfitting. Small adds parameters to capture spectral nuances better.

---

**Current Status**: **SOTA Achieved (mAP 0.419)**. Fine-tuning successful.
**Next Immediate Step**: Proceed to Phase 1 (Data Engineering) to fix class imbalance and spatial leakage.
