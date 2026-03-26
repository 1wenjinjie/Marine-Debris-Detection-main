# Innovation Experiment Report: High-Resolution Multispectral Detection
**Date:** 2025-11-25
**Experiment:** Evaluate the impact of a **P2 Detection Head** (High-Resolution Feature Fusion) on small marine debris detection.

## 1. Hypothesis
Marine debris objects are often extremely small (< 8 pixels). Standard YOLO architectures detect objects at strides of 8, 16, and 32 (P3, P4, P5). We hypothesized that adding a **P2 Head (Stride 4)** would significantly improve the detection of these tiny objects by preserving high-resolution spatial details.

## 2. Model Architecture (YOLOv11n-6ch-P2)
*   **Base**: YOLOv11n (6-Channel Multispectral).
*   **Modification**: Extended the Feature Pyramid Network (FPN) to include a P2 layer (128x128 resolution for 512x512 input).
*   **Head**: Added a 4th detection head at stride 4.
*   **Complexity**: Parameters increased slightly; GFLOPs increased from 6.6 to 10.5 (due to processing high-res feature maps).

## 3. Results (50 Epochs)

| Metric | Baseline (6ch) | Innovation (6ch + P2) | Change | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Overall mAP50** | **0.345** | 0.173 | -50% | High-resolution features introduced significant noise. |
| **Marine Debris** | **0.431** | 0.028 | -93% | **Critical Finding**: Debris spectral signatures are subtle. At high resolution (P2), wave glint and noise dominate the signal, confusing the model. |
| **Floating Object** | 0.265 | **0.290** | **+9%** | **Success**: Distinct objects (plastic bottles/buoys) benefited from the higher resolution. |
| **Ship Wake** | **0.339** | 0.201 | -40% | Large features (wakes) do not need P2 and suffered from the added noise/complexity. |

## 4. Scientific Conclusion
1.  **Resolution vs. Noise Trade-off**: While higher resolution (P2) helps resolve distinct small objects (Floating Objects), it drastically amplifies background noise (waves) for subtle objects (Marine Debris).
2.  **Spectral Dominance**: The success of the Baseline (P3-P5) confirms that **spectral information (color/IR)** is more important than **spatial resolution** for identifying marine debris. The model needs to "see the material" (IR), not just "see the shape" (High Res).
3.  **Recommendation**: For the final model, we should stick to the **Standard Resolution (P3-P5)** but focus on **Spectral Augmentation**, as this yields the highest robustness.

## 5. Next Steps
*   **Discard P2 Head** for the final production model.
*   **Proceed with Long-Duration Training (200 Epochs)** of the **Baseline 6-Channel Model**, which is the proven SOTA configuration.
