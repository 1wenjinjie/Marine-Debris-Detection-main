"""
v2.1 Phase 2 smoke test — 1-epoch end-to-end run on the binary
Marine_Debris instance-segmentation task.

Goal: prove the entire v2.1 pipeline lights up before launching real
training runs. Specifically:
  - 9-channel .npy loader (MaskRescueSegDataset_v2_1)
  - YOLOv11n-seg COCO weights inflated 3 -> 9 ch
  - WaterAwareSegLoss is the loss class
  - Forward + backward + optimizer step + epoch-end eval all run
    without exceptions
  - Output written to runs_v2_1/smoke_v2_1/ (separate from real runs)

Usage:
    python scripts/v2_1/training/v2_1_smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Triggers ultralytics_plugins_v2_1 side effects (IMG_FORMATS patch, etc.)
import ultralytics_plugins_v2_1  # noqa: F401
from ultralytics_plugins_v2_1 import MaskRescueSegTrainer_v2_1


def main() -> int:
    print("=" * 60)
    print("v2.1 Phase 2 smoke test")
    print("=" * 60)

    overrides = {
        "model": "yolo11n-seg.pt",     # COCO-pretrained, will be inflated to 9ch
        "data": str(ROOT / "configs" / "datasets" / "v2_1" /
                     "marida_v2_1_binary_highmod.yaml"),
        "project": str(ROOT / "runs_v2_1"),
        "name": "smoke_v2_1",
        "exist_ok": True,
        "epochs": 1,
        "batch": 4,
        "imgsz": 512,
        "workers": 0,
        "device": "0",
        "task": "segment",
        # Explicit zero per SPEC §20 — no spectral-incompatible aug
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 180.0,
        "fliplr": 0.5,
        "flipud": 0.5,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        # Other
        "patience": 0,        # disable early stopping for smoke test
        "save_period": -1,
        "plots": False,
        "val": True,
    }

    print("\nOverrides:")
    for k, v in overrides.items():
        print(f"  {k:14s} = {v}")

    trainer = MaskRescueSegTrainer_v2_1(overrides=overrides)
    trainer.train()
    print("\n[OK] smoke test completed without exception")
    return 0


if __name__ == "__main__":
    sys.exit(main())
