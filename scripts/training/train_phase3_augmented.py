"""
Phase 3 Training: Augmented Dataset.
Starting from: yolov11n_6ch_rebalance_50e (Phase 2)
Goal: Fix Wakes and Sparse Sargassum using heavily oversampled augmented data.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

def main():
    # Previous best model (Phase 2)
    model_path = "runs_marida/yolov11n_6ch_rebalance_50e/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        # Fallback to Phase 1 if Phase 2 missing
        model_path = "runs_marida/yolov11n_6ch_finetune_50e2/weights/best.pt"
        if not os.path.exists(model_path):
             print(f"Error: Model not found at {model_path}")
             return

    print(f"Starting training from {model_path}")

    overrides = {
        "data": "data_augmented_6ch.yaml",
        "epochs": 50,
        "batch": 16, # Smaller batch size for stability
        "imgsz": 640,
        "device": "0",
        "project": "runs_marida",
        "name": "yolov11n_6ch_augmented_50e",
        "task": "detect",
        "workers": 0,
        
        # --- Training Strategy ---
        "lr0": 0.001,           
        "lrf": 0.01,
        "warmup_epochs": 3.0,
        "optimizer": "AdamW",
        
        # --- Augmentation ---
        # We have offline augmentation, so we can be lighter on online augmentation
        "degrees": 90.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.5, # Enable mosaic but not 1.0
        "mixup": 0.1,
    }

    # Initialize model
    model = YOLO(model_path)

    # Train
    # Pass the custom trainer class, not an instance
    results = model.train(trainer=MaskRescueTrainer, **overrides)
    print("Training complete.")

if __name__ == "__main__":
    main()
