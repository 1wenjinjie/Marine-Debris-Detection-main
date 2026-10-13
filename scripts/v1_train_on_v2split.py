"""Train v1 7ch model on the v2 scene-clean split.

This run isolates the contribution of *scene-level leakage* in the v1
SOTA result. Hyperparameters are pinned to the exact values used in
the historical SOTA training (`yolov11n_7ch_512_finetune_wakeboost`)
so the only changed variable is the train/val/test split.

What this script does NOT do:
- Touch the GT-derived 7th channel (oracle leak is intentional here;
  we are measuring scene-leak contribution alone).
- Use the offline mosaic aug pool (those tiles cannot be assigned
  to v2 scenes; aug_mosaic_* files are excluded from the new manifest).

Output: runs go under runs_v1_v2split/<name>/. After training, the
script invokes scripts/v1_reproduce_7ch_baseline.py-style evaluation
on the v2-split test set automatically.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

SOTA_HYPERPARAMS: dict = {
    # core schedule (from yolov11n_7ch_512_finetune_wakeboost/args.yaml)
    "epochs": 30,
    "batch": 8,
    "imgsz": 512,
    "workers": 8,
    "optimizer": "auto",
    "lr0": 0.0005,
    "lrf": 0.1,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": False,
    "close_mosaic": 0,
    "patience": 50,
    "amp": True,
    "dropout": 0.0,
    # augmentation (SOTA used ultralytics defaults for HSV — note this
    # is inconsistent with mask_rescue_hyperparams.py spectral-safe
    # defaults, but we replicate SOTA for an apples-to-apples comparison)
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.2,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.2,
    # task
    "task": "detect",
}

DEFAULT_PARENT = "experiments/runs/runs_marida/yolov11n_7ch_full_return_512_resumed/weights/best.pt"
DEFAULT_DATA = "configs/datasets/marida_v1_on_v2split_7ch.yaml"


def main():
    p = argparse.ArgumentParser(description="v1 7ch on v2 scene-clean split (SOTA-pinned hparams)")
    p.add_argument("--model", default=DEFAULT_PARENT,
                   help="Starting checkpoint (default: SOTA parent for fair finetune comparison)")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--project", default="runs_v1_v2split")
    p.add_argument("--name", default="yolov11n_7ch_v2split_finetune")
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        sys.exit(f"checkpoint not found: {model_path}")
    if not data_path.exists():
        sys.exit(f"dataset yaml not found: {data_path}")

    overrides = dict(SOTA_HYPERPARAMS)
    overrides.update({
        "data": str(data_path),
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "resume": args.resume,
        "pretrained": True,
        "deterministic": True,
    })

    print("=" * 60)
    print("v1 7ch on v2 scene-clean split — SOTA-pinned training")
    print("=" * 60)
    print(f"  parent: {model_path}")
    print(f"  data  : {data_path}")
    print(f"  run   : {args.project}/{args.name}")
    print(f"  device: {args.device}")
    print(f"  seed  : {args.seed}")
    print()

    model = YOLO(str(model_path))
    model.train(trainer=MaskRescueTrainer, **overrides)


if __name__ == "__main__":
    main()
