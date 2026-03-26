"""Validate 10ch spectral model with custom NPY dataset loader"""
from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer
import yaml

# Load model
model = YOLO('runs_marida/yolov11n_10ch_spectral_v2/weights/best.pt')

# Validate with custom trainer
results = model.val(
    data='marida_mask_rescue_10ch.yaml',
    split='val',
    device=0,
    trainer=MaskRescueTrainer
)

print(f"\n{'='*50}")
print(f"=== 10-Channel Spectral Model Validation ===")
print(f"{'='*50}")
print(f"mAP@50:    {results.box.map50:.4f}")
print(f"mAP@50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall:    {results.box.mr:.4f}")
print(f"\nPer-class AP@50:")
class_names = ['Marine_Debris', 'Dense_Sargassum', 'Sparse_Sargassum', 
               'Natural_Organic_Material', 'Ship', 'Clouds', 'Marine_Water']
for i, (name, ap) in enumerate(zip(class_names, results.box.ap50)):
    print(f"  {name:30s}: {ap:.4f}")
