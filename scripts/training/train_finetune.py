"""Fine-tuning script for YOLOv11n 6-channel model.
Phase 2: Strong Augmentation & Low LR.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO with Strong Augmentation")
    parser.add_argument("--model", default="runs_marida/yolov11n_6ch_7class_best_100e/weights/best.pt", help="Pretrained checkpoint")
    parser.add_argument("--data", default="data_npy_6ch.yaml", help="Dataset YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Fine-tuning duration")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Training device")
    parser.add_argument("--project", default="runs_marida", help="Project directory")
    parser.add_argument("--name", default="yolov11n_6ch_finetune_50e", help="Run name")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    overrides = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "task": "detect",
        
        # --- System ---
        "workers": 0,           # Fix for Windows shared memory error (error 1455)
        
        # --- Fine-Tuning Strategy ---
        "lr0": 0.002,           # Lower initial learning rate (20% of standard)
        "lrf": 0.01,            # Final learning rate fraction
        "warmup_epochs": 3.0,
        "optimizer": "AdamW",   # Explicitly use AdamW
        
        # --- Strong Geometric Augmentation ---
        # Now that the model knows what objects look like, we force it to be invariant to orientation.
        "degrees": 180.0,       # +/- 180 degree rotation (Crucial for aerial views)
        "flipud": 0.5,          # Vertical flip
        "fliplr": 0.5,          # Horizontal flip
        "shear": 2.0,           # Slight shear
        "perspective": 0.0005,  # Slight perspective
        
        # --- Disable Color/Spectral Augmentation ---
        # We want to preserve the exact spectral signatures learned.
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        
        # --- Disable Complex Mixing ---
        # Mosaic/Mixup can be risky with float32 inputs and small objects.
        # We rely on pure geometric transforms of the original data.
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    
    print(f"Starting Fine-Tuning from {args.model}...")
    model = YOLO(args.model)
    model.train(trainer=MaskRescueTrainer, **overrides)

if __name__ == "__main__":
    main()
