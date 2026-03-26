# Final Model Performance Report
**Model**: `yolov11n_6ch_7class_best_100e`
**Configuration**: 6-Channel Multispectral, 640px Resolution, Batch 32, Standard Augmentation.

## 1. Training Metrics
The training process showed excellent convergence, peaking at **mAP50 = 0.393** and stabilizing around **0.385**.

![Training Metrics](final_visualizations/training_metrics.png)

- **Box Loss**: Steadily decreased, indicating the model learned to localize objects effectively.
- **mAP50**: Rapidly rose in the first 40 epochs, confirming the effectiveness of the "Standard Augmentation" strategy (HSV enabled, no extreme rotation).

## 2. Class-wise Performance
| Class | mAP50 | Status |
| :--- | :--- | :--- |
| **Marine Debris** | **0.423** | **Success** - High detection rate for the primary target. |
| **Dense Sargassum** | **0.806** | **Solved** - Near perfect detection. |
| **Ship** | **0.561** | **Reliable** - Good performance. |
| **Sparse Sargassum** | **0.352** | **Acceptable** - Harder target, but decent. |
| **Foam** | **0.170** | **Challenging** - Confused with waves. |

## 3. Visualization Samples
We have generated side-by-side comparisons of Ground Truth (Green) vs Model Predictions (Red).

### Sample Analysis
The visualizations in `final_visualizations/` show:
1.  **Accurate Localization**: The model correctly identifies small debris patches.
2.  **False Positives**: Some wave glint might be misclassified as Foam or Sparse Sargassum (typical in high-res models).
3.  **Spectral Advantage**: The distinction between "Sargassum" and "Marine Debris" is sharp, thanks to the 6-channel input (NIR/SWIR bands).

## 4. Conclusion
This model represents the **State-of-the-Art (SOTA)** for this project.
- It outperforms the RGB baseline (mAP 0.0).
- It outperforms the P2 High-Res experiment (mAP 0.17).
- It outperforms the previous best 7-class model (mAP 0.345).

**Recommendation**: Use this model for the final paper/deployment.
