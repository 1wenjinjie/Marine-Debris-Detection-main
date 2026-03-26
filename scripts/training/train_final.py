"""Entry point for the FINAL Long-Duration Training (200 Epochs).
Configuration: 6-Channel Multispectral YOLOv11n with Spectral-Aware Augmentation.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO

from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Final SOTA Model")
    parser.add_argument("--model", default="yolov11n_6ch.yaml", help="Model YAML")
    parser.add_argument("--data", default="data_npy_6ch.yaml", help="Dataset YAML (7 classes)")
    parser.add_argument("--epochs", type=int, default=100, help="Standard duration training")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Training device string, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs_marida", help="Project directory for runs")
    parser.add_argument("--name", default="yolov11n_6ch_7class_best_100e", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--cache", default="", help="Ultralytics cache mode")
    parser.add_argument("--rect", action="store_true", help="Enable rectangular training")
    parser.add_argument("--amp", action="store_true", help="Enable AMP")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "cache": args.cache or None,
        "rect": args.rect,
        "amp": args.amp,
        "task": "detect",
        
        # --- Optimization for Long Training ---
        "patience": 50,        # Early stopping patience
        "dropout": 0.1,
        "weight_decay": 0.001,
        "warmup_epochs": 5.0,
        "cos_lr": True,        # Cosine learning rate scheduler for better convergence
        
        # --- Standard Augmentation (Proven Convergence) ---
        # Reverting to standard settings that achieved the initial high performance.
        # The "Spectral-Aware" (No HSV, High Rot) strategy works best for FINE-TUNING, 
        # not for training from scratch.
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,    # Standard rotation (0.0) is safer for convergence from scratch
        "flipud": 0.0,     # Disable vertical flip (standard YOLO setting)
        "fliplr": 0.5,     # Horizontal flip
        
        # Disable mosaic/mixup because they assume uint8 images and break float32 inputs
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    model = YOLO(args.model)
    model.train(trainer=MaskRescueTrainer, **overrides)


if __name__ == "__main__":
    main()
