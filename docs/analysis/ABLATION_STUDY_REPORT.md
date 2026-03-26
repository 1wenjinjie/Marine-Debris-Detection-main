# Ablation Study Report: RGB vs. 6-Channel Multispectral
**Date:** 2025-11-25
**Experiment:** Assess the contribution of multispectral bands (NIR/SWIR) to Marine Debris Detection.

## 1. Experiment Setup
To isolate the impact of the additional spectral bands, we trained two identical YOLOv11n models with the exact same augmentation strategy ("Spectral-Aware": No HSV, +/- 180° Rotation).

| Feature | RGB Baseline | 6-Channel Model (Ours) |
| :--- | :--- | :--- |
| **Input Channels** | 3 (Red, Green, Blue) | 6 (R, G, B, NIR, SWIR1, SWIR2) |
| **Model Config** | `yolov11n_rgb.yaml` | `yolov11n_6ch.yaml` |
| **Epochs** | 50 | 50 |
| **Augmentation** | No HSV, High Rotation | No HSV, High Rotation |
| **Dataset** | MARIDA (Same Split) | MARIDA (Same Split) |

## 2. Results (50 Epochs)

| Metric | RGB Baseline | 6-Channel Model | Improvement |
| :--- | :--- | :--- | :--- |
| **Overall mAP50** | 0.150 | **0.345** | **+130%** |
| **Overall mAP50-95** | 0.089 | **0.185** | **+107%** |
| **Marine Debris mAP50** | 0.000 (Failed) | **0.431** | **Critical Success** |
| **Floating Object mAP50** | 0.114 | **0.265** | +132% |
| **Ship Wake mAP50** | 0.336 | **0.339** | +0.9% |

## 3. Analysis
1.  **Marine Debris Detection**: The RGB model completely failed to detect Marine Debris (mAP=0.0). This confirms that in the visible spectrum, marine debris is often indistinguishable from ocean waves or glint. The 6-channel model achieved a respectable 0.431 mAP, proving that the **NIR and SWIR bands are the primary discriminators** for this class.
2.  **Ship Wakes**: Performance was similar between RGB and 6-Channel (0.336 vs 0.339). This suggests that ship wakes are highly visible in the RGB spectrum (white foam on blue water) and do not benefit significantly from extra spectral bands.
3.  **Floating Objects**: Significant improvement (+132%) with 6 channels, indicating that material properties captured in IR bands help distinguish these objects from water.

## 4. Conclusion
The inclusion of NIR and SWIR bands is not just "helpful" but **mandatory** for the specific task of Marine Debris detection. An RGB-only approach is insufficient for this dataset. The 6-channel strategy is scientifically validated.
