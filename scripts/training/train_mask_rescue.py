"""Entry point for training YOLOv11 with the 7-channel mask rescue dataset."""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO

from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer
from scripts.training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO with 7-channel knowledge tensors")
    parser.add_argument("--model", default="yolov11n_7ch.yaml", help="Model YAML or checkpoint")
    parser.add_argument("--data", default="marida_rescue_7ch.yaml", help="Dataset YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="cpu", help="Training device string, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs_marida", help="Project directory for runs")
    parser.add_argument("--name", default="yolov11n_rescue_knowledge", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in project/name")
    parser.add_argument("--cache", default="", help="Ultralytics cache mode (ram/disk/true/false)")
    parser.add_argument("--rect", action="store_true", help="Enable rectangular training")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (Automatic Mixed Precision)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument("--lr0", type=float, default=None, help="Optional override for initial learning rate")
    parser.add_argument("--lrf", type=float, default=None, help="Optional override for final LR fraction")
    parser.add_argument("--cos-lr", action="store_true", help="Enable cosine LR schedule")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = dict(MASK_RESCUE_TRAIN_DEFAULTS)
    overrides.update({
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
    })
    if args.lr0 is not None:
        overrides["lr0"] = args.lr0
    if args.lrf is not None:
        overrides["lrf"] = args.lrf
    if args.cos_lr:
        overrides["cos_lr"] = True
    model = YOLO(args.model)
    model.train(trainer=MaskRescueTrainer, **overrides)


if __name__ == "__main__":
    main()
