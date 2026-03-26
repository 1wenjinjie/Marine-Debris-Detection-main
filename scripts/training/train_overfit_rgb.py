import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset
import torch
import ultralytics.data.dataset
import ultralytics.data.build

# Monkey patch the dataset class into Ultralytics
print("Monkey patching YOLODataset...")
ultralytics.data.dataset.YOLODataset = MaskRescueNPYDataset
ultralytics.data.build.YOLODataset = MaskRescueNPYDataset
print("Monkey patching complete.")

def train():
    # Load a standard YOLOv11n model (pretrained on COCO)
    model = YOLO('yolo11n.pt')

    # Train the model on the overfit subset
    results = model.train(
        data='overfit_rgb.yaml',
        epochs=100,
        imgsz=256,
        batch=16,
        project='marida_overfit_runs',
        name='yolo11n_rgb_overfit',
        device='0',
        workers=0,
        exist_ok=True,
        amp=True,
        optimizer='AdamW',
        lr0=0.001,
        patience=0, # Disable early stopping
        augment=False, # Disable augmentation for overfitting check
        hsv_h=0.0, # Disable color jitter
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )

if __name__ == '__main__':
    train()
