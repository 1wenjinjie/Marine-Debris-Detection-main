import numpy as np
from pathlib import Path

# Check 10ch dataset
sample_10ch = np.load('mask_bbox_rescue_10ch/train/images/S2A_MSIL2A_20160810T144732_65_3168.npy')
print(f"10ch Dataset Sample Shape: {sample_10ch.shape}")
print(f"Expected: (H, W, 10), Got channels: {sample_10ch.shape[-1]}")

# Check if data was actually used during training
print(f"\n10ch dataset exists: {Path('mask_bbox_rescue_10ch').exists()}")
print(f"Train samples: {len(list(Path('mask_bbox_rescue_10ch/train/images').glob('*.npy')))}")
print(f"Val samples: {len(list(Path('mask_bbox_rescue_10ch/val/images').glob('*.npy')))}")

# Check model input layer
import torch
model = torch.load('runs_marida/yolov11n_10ch_spectral_v2/weights/best.pt')
print(f"\nModel first conv input channels: {model['model'].model[0].conv.weight.shape[1]}")
print(f"Expected 10 channels, got: {model['model'].model[0].conv.weight.shape[1]}")
