"""
Unified evaluation script for MARIDA v2 experiments.

Evaluates a trained model on the v2 test set with comprehensive metrics:
  - mAP@0.5, mAP@0.5:0.95
  - Per-class AP, Precision, Recall
  - Saves results to standardized JSON

Usage:
  python scripts/v2_evaluate.py --weights runs_v2/baseline_6ch_s0/weights/best.pt \
                                --data configs/datasets/marida_v2_6ch.yaml \
                                --channels 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from ultralytics import YOLO
from ultralytics.data import build as yolo_build
from ultralytics.models.yolo.detect import val as detect_val

from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset
from ultralytics.utils import colorstr

CLASS_NAMES = ["Marine_Debris", "Algae_Organic", "Ship"]


def _make_patched_build(dataset_root: Path, channels: int, channel_stats: str | None):
    """Create a patched build_yolo_dataset that uses MaskRescueNPYDataset."""
    def _patched(cfg, img_path, batch, data, mode="train", rect=False, stride=32, **kw):
        return MaskRescueNPYDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=False,
            hyp=cfg,
            rect=rect,
            stride=stride,
            pad=0.5,
            prefix=colorstr(f"{mode}: "),
            dataset_root=dataset_root,
            channels=channels,
            channel_stats=channel_stats,
            task=cfg.task,
            data=data,
        )
    return _patched


def evaluate(weights: str, data_yaml: str, channels: int, split: str = "test",
             imgsz: int = 512, batch: int = 8, device: str = "0",
             conf: float = 0.001, iou: float = 0.6) -> dict:
    """Run evaluation and return metrics dict."""

    import yaml
    data_cfg = yaml.safe_load(Path(data_yaml).read_text())
    dataset_root = Path(data_cfg.get("path", "")).resolve()

    # Find channel stats
    stats_path = dataset_root / f"channel_stats_{channels}ch.json"
    channel_stats = str(stats_path) if stats_path.exists() else None

    # Monkey-patch dataset builder
    _orig_build = yolo_build.build_yolo_dataset
    _orig_det = detect_val.build_yolo_dataset
    patched = _make_patched_build(dataset_root, channels, channel_stats)
    yolo_build.build_yolo_dataset = patched
    detect_val.build_yolo_dataset = patched

    try:
        model = YOLO(weights)
        results = model.val(
            data=data_yaml,
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            conf=conf,
            iou=iou,
            verbose=True,
        )
    finally:
        yolo_build.build_yolo_dataset = _orig_build
        detect_val.build_yolo_dataset = _orig_det

    # Extract metrics
    nc = len(CLASS_NAMES)
    metrics = {
        "weights": str(weights),
        "data": str(data_yaml),
        "channels": channels,
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        # Overall
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        # Per-class
        "per_class": {},
    }

    for i in range(nc):
        metrics["per_class"][CLASS_NAMES[i]] = {
            "AP50": float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
            "AP50_95": float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
            "precision": float(results.box.p[i]) if i < len(results.box.p) else 0.0,
            "recall": float(results.box.r[i]) if i < len(results.box.r) else 0.0,
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="MARIDA v2 unified evaluation")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default=str(ROOT / "configs/datasets/marida_v2_6ch.yaml"))
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    metrics = evaluate(
        weights=args.weights, data_yaml=args.data, channels=args.channels,
        split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, conf=args.conf, iou=args.iou,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print()
    print("  Per-class AP@0.5:")
    for name, cls_m in metrics["per_class"].items():
        print(f"    {name:20s}: AP50={cls_m['AP50']:.4f}  P={cls_m['precision']:.4f}  R={cls_m['recall']:.4f}")

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.weights).parent.parent / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
