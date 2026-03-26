"""Entry point for training YOLOv11 RGB Baseline (Ablation Study)."""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO

from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO RGB Baseline")
    parser.add_argument("--model", default="yolov11n_rgb.yaml", help="Model YAML or checkpoint")
    parser.add_argument("--data", default="marida_rgb_baseline.yaml", help="Dataset YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0", help="Training device string, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs_marida", help="Project directory for runs")
    parser.add_argument("--name", default="yolov11n_rgb_baseline", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in project/name")
    parser.add_argument("--cache", default="", help="Ultralytics cache mode (ram/disk/true/false)")
    parser.add_argument("--rect", action="store_true", help="Enable rectangular training")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (Automatic Mixed Precision)")
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
        # Regularization to prevent overfitting/collapse
        "dropout": 0.1,
        "weight_decay": 0.001,
        "warmup_epochs": 5.0,
        
        # --- Augmentation Strategy (Identical to 6-channel experiment) ---
        # 1. Disable Color Augmentation (Protect Spectral Signatures)
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        # 2. Enable Strong Geometric Augmentation (Satellite images are rotation invariant)
        "degrees": 180.0,  # +/- 180 degree rotation
        "flipud": 0.5,     # Vertical flip
        "fliplr": 0.5,     # Horizontal flip (default is usually 0.5)
        
        # Disable mosaic/mixup because they assume uint8 images and break float32 inputs
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    model = YOLO(args.model)
    # Use the same custom trainer that handles .npy loading
    model.train(trainer=MaskRescueTrainer, **overrides)


if __name__ == "__main__":
    main()
