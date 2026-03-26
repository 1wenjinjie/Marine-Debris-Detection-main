# Experiment Report: Enriched Dilated Dataset

## Problem
The initial "Perfect" dataset failed to detect Marine Debris (Recall 0.0) because the debris objects were extremely small.
- **Analysis**: 97.2% of debris labels were < 1 pixel in area.
- **Cause**: Raw shapefile conversion preserved point-like features that are invisible at 256x256 resolution after downsampling in the YOLO architecture.

## Solution: Morphological Dilation
We implemented a **Dilation Strategy** in the dataset generation pipeline (`build_enriched_dataset.py`).
- **Method**: Applied a 3x3 dilation kernel to the debris masks during generation.
- **Effect**: 
    - 1-pixel objects (invisible) -> ~9-pixel objects (visible).
    - Small fragments became large enough to survive feature extraction downsampling.
- **Dataset**: `MARIDA_YOLO_Enriched_Dilated`
    - Total Debris Patches: 364
    - Total Debris Instances: 1793

## Results
We trained `yolo11n_6ch_enriched_dilated` for 100 epochs.

| Metric | Value | Improvement |
| :--- | :--- | :--- |
| **Precision (Debris)** | **0.408** | Huge increase (was 0.0) |
| **Recall (Debris)** | **0.223** | Huge increase (was 0.001) |
| **mAP50 (Debris)** | **0.376** | Significant |

## Visual Confirmation
Visualizations in `visualizations_dilated_results/` show successful detections:
- `S2_22-12-20_18QYF_0`: 3 Predictions
- `S2_4-9-16_16PCC_30`: 3 Predictions
- `S2_23-9-20_16PCC_25`: 2 Predictions

## Conclusion
The "Sub-pixel Debris" hypothesis was correct. The model could not learn from invisible labels. Artificially enlarging the targets via dilation has successfully enabled the model to learn and detect these small marine debris objects.

## Next Steps
1. **Hyperparameter Tuning**: Optimize anchor box sizes for these small (but now visible) objects.
2. **Higher Resolution**: If possible, process at 512x512 to naturally increase object size without as much artificial dilation.
3. **False Positive Reduction**: Precision is 0.4, meaning there are false positives. Hard negative mining could help.
