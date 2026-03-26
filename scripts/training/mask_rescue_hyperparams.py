"""Shared training hyperparameters for all MaskRescue experiments.

Import this dict and merge run-specific overrides on top. Centralising
the defaults here prevents silent divergence between training scripts.
"""

MASK_RESCUE_TRAIN_DEFAULTS: dict = {
    "epochs": 50,
    "batch": 8,
    "imgsz": 512,
    "workers": 2,
    "task": "detect",
    "dropout": 0.1,
    "weight_decay": 0.001,
    "warmup_epochs": 5.0,
    # Spectral-safe augmentation: no colour jitter, strong geometric transforms.
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 180.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    # Mosaic/mixup break float32 multi-channel inputs.
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}
