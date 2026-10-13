"""Ablation training matrix for v1 7ch physidx on v2 scene-clean split.

Five cells, one shared training pipeline:
- baseline : yolov11n_7ch.yaml         + MaskRescueTrainer
- cbam     : yolov11n_7ch_cbam.yaml    + MaskRescueTrainer
- bifpn    : yolov11n_7ch_bifpn.yaml   + MaskRescueTrainer + install_bifpn_patch
- eiou     : yolov11n_7ch.yaml         + EIoUTrainer
- full     : yolov11n_7ch_full.yaml    + EIoUTrainer       + install_bifpn_patch

Hyperparameters are pinned to SOTA_HYPERPARAMS (same as v1_train_on_v2split.py)
so the only deltas across the matrix are model architecture and box loss.

Init: yolo11n_7ch_init.pt (clean, no leak from any prior v1 run).
Data: configs/datasets/marida_v1_physidx_on_v2split_7ch.yaml.
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
# ultralytics_plugins.__init__ auto-registers CBAM via register_extra_modules
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer
from ultralytics_plugins.eiou_trainer import EIoUTrainer
from ultralytics_plugins.bifpn_modules import install_bifpn_patch

# Reuse the exact SOTA-pinned hyperparameters used in v1_train_on_v2split.py
from scripts.v1_train_on_v2split import SOTA_HYPERPARAMS

ABLATIONS = {
    "baseline": {
        "model_yaml": "configs/models/yolov11n_7ch.yaml",
        "trainer": MaskRescueTrainer,
        "needs_bifpn_patch": False,
    },
    "cbam": {
        "model_yaml": "configs/models/yolov11n_7ch_cbam.yaml",
        "trainer": MaskRescueTrainer,
        "needs_bifpn_patch": False,
    },
    "bifpn": {
        "model_yaml": "configs/models/yolov11n_7ch_bifpn.yaml",
        "trainer": MaskRescueTrainer,
        "needs_bifpn_patch": True,
    },
    "eiou": {
        "model_yaml": "configs/models/yolov11n_7ch.yaml",
        "trainer": EIoUTrainer,
        "needs_bifpn_patch": False,
    },
    "full": {
        "model_yaml": "configs/models/yolov11n_7ch_full.yaml",
        "trainer": EIoUTrainer,
        "needs_bifpn_patch": True,
    },
}

DEFAULT_INIT = "yolo11n_7ch_init.pt"
DEFAULT_DATA = "configs/datasets/marida_v1_physidx_on_v2split_7ch.yaml"


def main():
    p = argparse.ArgumentParser(description="v1 7ch physidx ablation matrix on v2 scene-clean split")
    p.add_argument("--ablation", required=True, choices=list(ABLATIONS.keys()))
    p.add_argument("--init", default=DEFAULT_INIT, help="Starting checkpoint (default: clean 7ch init)")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--project", default="runs_v1_v2split")
    p.add_argument("--name", default=None, help="Default: yolov11n_7ch_v2split_physidx_<ablation>_clean_init")
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=None, help="Override imgsz")
    p.add_argument("--batch", type=int, default=None, help="Override batch size")
    p.add_argument("--workers", type=int, default=None, help="Override DataLoader workers")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = p.parse_args()

    spec = ABLATIONS[args.ablation]
    init = Path(args.init)
    data = Path(args.data)
    if not init.exists():
        sys.exit(f"init checkpoint not found: {init}")
    if not data.exists():
        sys.exit(f"data yaml not found: {data}")
    if not Path(spec["model_yaml"]).exists():
        sys.exit(f"model yaml not found: {spec['model_yaml']}")

    if spec["needs_bifpn_patch"]:
        install_bifpn_patch()

    name = args.name or f"yolov11n_7ch_v2split_physidx_{args.ablation}_clean_init"
    overrides = dict(SOTA_HYPERPARAMS)
    overrides.update({
        "data": str(data),
        "device": args.device,
        "project": args.project,
        "name": name,
        "seed": args.seed,
        "pretrained": True,
        "deterministic": True,
    })
    for k in ("imgsz", "batch", "workers", "epochs"):
        v = getattr(args, k)
        if v is not None:
            overrides[k] = v

    print("=" * 60)
    print(f"v1 7ch physidx ablation: {args.ablation.upper()}")
    print("=" * 60)
    print(f"  model_yaml: {spec['model_yaml']}")
    print(f"  trainer   : {spec['trainer'].__name__}")
    print(f"  bifpn_patch: {spec['needs_bifpn_patch']}")
    print(f"  init      : {init}")
    print(f"  data      : {data}")
    print(f"  run       : {args.project}/{name}")
    print(f"  seed      : {args.seed}")
    print()

    # Build a fresh model from the architecture YAML so new modules (CBAM/BiFPN)
    # get instantiated, then partial-load init weights for matching layers.
    model = YOLO(spec["model_yaml"])
    try:
        model.load(str(init))
        print(f"  loaded init weights via partial-load: {init}")
    except Exception as e:
        print(f"  WARNING: partial-load of {init} failed ({e}); training from scratch")
    model.train(trainer=spec["trainer"], **overrides)


if __name__ == "__main__":
    main()
