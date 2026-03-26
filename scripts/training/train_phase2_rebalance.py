"""
Phase 2 Training: Rebalanced Dataset.
Starting from: yolov11n_6ch_finetune_50e2 (mAP 0.419)
Goal: Improve rare classes (Wakes, Sparse Sargassum) using Mosaic augmented data.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

def main():
    # Previous best model
    model_path = "runs_marida/yolov11n_6ch_finetune_50e2/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    overrides = {
        "data": "data_npy_6ch_rebalanced.yaml",
        "epochs": 50,
        "batch": 32,
        "imgsz": 640,
        "device": "0",
        "project": "runs_marida",
        "name": "yolov11n_6ch_rebalance_v2_50e",
        "task": "detect",
        "workers": 0,  # Windows fix
        
        # --- Training Strategy ---
        "lr0": 0.002,           # Low LR to preserve features
        "lrf": 0.01,
        "warmup_epochs": 3.0,
        "optimizer": "AdamW",
        
        # --- Augmentation ---
        # Keep the strong augmentation from Phase 1, as it worked well
        "degrees": 180.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "shear": 2.0,
        "perspective": 0.0005,
        
        # Disable internal mosaic since we pre-generated mosaic images
        # We want the model to see our specific rebalanced mosaics
        "mosaic": 0.0,
        "mixup": 0.0,
    }
    
    print(f"Starting Phase 2 Training (Rebalanced) from {model_path}...")
    model = YOLO(model_path)
    model.train(trainer=MaskRescueTrainer, **overrides)

if __name__ == "__main__":
    main()
