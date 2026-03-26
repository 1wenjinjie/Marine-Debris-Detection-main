# Fine-Tuning Experiment Report

## Experiment Details
- **Base Model**: `runs_marida/yolov11n_6ch_7class_best_100e/weights/best.pt` (mAP 0.385)
- **Configuration**:
    - **Epochs**: 50
    - **Learning Rate**: Low (`lr0=0.002`, `lrf=0.01`)
    - **Optimizer**: AdamW
    - **Augmentation**: **Strong**
        - Rotation: +/- 180 degrees (Full invariance)
        - Shear: 2.0
        - Perspective: 0.0005
        - Flip: Up/Down + Left/Right
    - **Disabled**: Mosaic/Mixup (to preserve spectral purity)

## Results
| Metric | Base Model | Fine-Tuned Model | Change |
| :--- | :--- | :--- | :--- |
| **mAP50 (All)** | 0.385 | **0.419** | **+8.8%** |
| **mAP50-95** | 0.201 | **0.227** | +12.9% |

### Class-wise Performance (mAP50)
| Class | Base | Fine-Tuned | Change |
| :--- | :--- | :--- | :--- |
| **Marine Debris** | 0.412 | **0.452** | 🔼 Improved |
| **Dense Sargassum** | 0.791 | **0.830** | 🔼 Improved |
| **Sparse Sargassum** | 0.421 | **0.510** | 🚀 **Major Gain** |
| **Ship** | 0.432 | **0.465** | 🔼 Improved |
| **Foam** | 0.215 | **0.256** | 🔼 Improved |
| **Wakes** | 0.000 | 0.000 | ➖ No Change |

## Analysis
The fine-tuning strategy was highly successful. By lowering the learning rate and introducing strong geometric augmentations (especially 180-degree rotation), the model learned to be invariant to the orientation of debris and sargassum. This is critical for aerial/satellite imagery where "up" is arbitrary.

The most significant gain was in **Sparse Sargassum**, which jumped from 0.42 to 0.51. This suggests that rotation augmentation helped the model generalize better on this rare class.

## Next Steps
The "Short Term" phase of the roadmap is complete.
We are now ready to move to the **Medium Term** phase:
1.  **Data Rebalancing**: Address the "Wakes" (0.00 mAP) and class imbalance issues using Mosaic/Copy-Paste.
2.  **Model Upgrade**: Consider moving to YOLOv11s (Small) if rebalancing doesn't fix the Wakes issue.
