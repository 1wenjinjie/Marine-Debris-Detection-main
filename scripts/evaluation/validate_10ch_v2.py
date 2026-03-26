"""Validate 10ch model using MaskRescueValidator"""
from pathlib import Path
from ultralytics_plugins.mask_rescue_validator import MaskRescueValidator
from ultralytics.utils import DEFAULT_CFG

# Setup args
args = DEFAULT_CFG.copy()
args.update({
    'data': 'marida_mask_rescue_10ch.yaml',
    'model': 'runs_marida/yolov11n_10ch_spectral_v2/weights/best.pt',
    'batch': 16,
    'imgsz': 256,
    'device': 0,
    'split': 'val',
    'save_json': False,
    'plots': True,
    'verbose': True
})

# Run validation
validator = MaskRescueValidator(args=args)
results = validator()

# Print results
print(f"\n{'='*60}")
print(f"{'10-Channel Spectral Model Validation Results':^60}")
print(f"{'='*60}")
print(f"mAP@50:      {results['metrics/mAP50(B)']:.4f}")
print(f"mAP@50-95:   {results['metrics/mAP50-95(B)']:.4f}")
print(f"Precision:   {results['metrics/precision(B)']:.4f}")
print(f"Recall:      {results['metrics/recall(B)']:.4f}")

print(f"\n{'Per-Class AP@50:':^60}")
print(f"{'-'*60}")
class_names = ['Marine_Debris', 'Dense_Sargassum', 'Sparse_Sargassum',
               'Natural_Organic_Material', 'Ship', 'Clouds', 'Marine_Water']
for i, name in enumerate(class_names):
    key = f'metrics/{name}(B)'
    if key in results:
        print(f"{name:35s}: {results[key]:.4f}")
print(f"{'='*60}\n")
