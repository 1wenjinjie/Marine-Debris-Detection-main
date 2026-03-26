"""Fine-tuning script for YOLOv11n 6-channel model on REBALANCED dataset.
Phase 3: Data Rebalancing (Mosaic/Copy-Paste).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on Rebalanced Dataset")
    # Start from the best fine-tuned model
    parser.add_argument("--model", default="runs_marida/yolov11n_6ch_finetune_50e2/weights/best.pt", help="Pretrained checkpoint")
    # Use the NEW rebalanced dataset
    parser.add_argument("--data", default="data_npy_6ch_rebalanced.yaml", help="Dataset YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Fine-tuning duration")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Training device")
    parser.add_argument("--project", default="runs_marida", help="Project directory")
    parser.add_argument("--name", default="yolov11n_6ch_rebalanced_50e", help="Run name")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
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
        "workers": 0,           # Fix for Windows shared memory error
        
        # --- Fine-Tuning Strategy ---
        # We continue with the successful Low LR + Strong Aug strategy
        "lr0": 0.002,
        "lrf": 0.01,
        "warmup_epochs": 3.0,
        "optimizer": "AdamW",
        
        # --- Strong Geometric Augmentation ---
        "degrees": 180.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "shear": 2.0,
        "perspective": 0.0005,
        
        # --- Disable Color/Spectral Augmentation ---
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        
        # --- Disable Built-in Mosaic ---
        # We have already applied "offline" Mosaic/Copy-Paste in the dataset itself.
        # Adding more online Mosaic might be overkill or distort the synthetic images too much.
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    
    print(f"Starting Rebalanced Training from {args.model}...")
    model = YOLO(args.model)
    model.train(trainer=MaskRescueTrainer, **overrides)

if __name__ == "__main__":
    main()
