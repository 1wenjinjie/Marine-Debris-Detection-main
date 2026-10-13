"""Evaluation for MARIDA v2-seg instance segmentation models.

Same monkey-patch strategy as v2_evaluate.py but for segmentation:
- Patches build_yolo_dataset with MaskRescueSegNPYDataset
- Reports box AP + mask AP (mask is the main signal we care about)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.data import build as yolo_build
from ultralytics.models.yolo.detect import val as detect_val
from ultralytics.utils import colorstr

from ultralytics_plugins.mask_rescue_seg_dataset import MaskRescueSegNPYDataset

CLASS_NAMES = ["Marine_Debris", "Algae_Organic", "Ship"]


def _make_patched_build(dataset_root: Path, channels: int, channel_stats: str | None):
    def _patched(cfg, img_path, batch, data, mode="train", rect=False, stride=32, **kw):
        return MaskRescueSegNPYDataset(
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
            task="segment",
            data=data,
        )
    return _patched


def evaluate(weights: str, data_yaml: str, channels: int, split: str = "test",
             imgsz: int = 512, batch: int = 8, device: str = "0",
             conf: float = 0.001, iou: float = 0.6, workers: int = 0) -> dict:
    import yaml
    data_cfg = yaml.safe_load(Path(data_yaml).read_text())
    dataset_root = Path(data_cfg.get("path", "")).resolve()

    stats_path = dataset_root / f"channel_stats_{channels}ch.json"
    channel_stats = str(stats_path) if stats_path.exists() else None

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
            workers=workers,
            verbose=True,
        )
    finally:
        yolo_build.build_yolo_dataset = _orig_build
        detect_val.build_yolo_dataset = _orig_det

    nc = len(CLASS_NAMES)
    metrics = {
        "weights": str(weights),
        "data": str(data_yaml),
        "channels": channels,
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "workers": workers,
        "box_mAP50": float(results.box.map50),
        "box_mAP50_95": float(results.box.map),
        "box_precision": float(results.box.mp),
        "box_recall": float(results.box.mr),
        "mask_mAP50": float(results.seg.map50),
        "mask_mAP50_95": float(results.seg.map),
        "mask_precision": float(results.seg.mp),
        "mask_recall": float(results.seg.mr),
        "per_class": {},
    }
    for i in range(nc):
        metrics["per_class"][CLASS_NAMES[i]] = {
            "box_AP50": float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
            "box_AP50_95": float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
            "mask_AP50": float(results.seg.ap50[i]) if i < len(results.seg.ap50) else 0.0,
            "mask_AP50_95": float(results.seg.ap[i]) if i < len(results.seg.ap) else 0.0,
            "box_precision": float(results.box.p[i]) if i < len(results.box.p) else 0.0,
            "box_recall": float(results.box.r[i]) if i < len(results.box.r) else 0.0,
        }
    return metrics


def main():
    p = argparse.ArgumentParser(description="MARIDA v2-seg evaluation")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--data", type=str, default=str(ROOT / "configs/datasets/marida_v2_seg_6ch.yaml"))
    p.add_argument("--channels", type=int, default=6)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    m = evaluate(
        weights=args.weights, data_yaml=args.data, channels=args.channels,
        split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, conf=args.conf, iou=args.iou, workers=args.workers,
    )
    print("\n" + "=" * 60)
    print("SEG EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Box  mAP50: {m['box_mAP50']:.4f}  mAP50-95: {m['box_mAP50_95']:.4f}")
    print(f"  Mask mAP50: {m['mask_mAP50']:.4f}  mAP50-95: {m['mask_mAP50_95']:.4f}")
    print(f"  Box  P/R:   {m['box_precision']:.4f} / {m['box_recall']:.4f}")
    print(f"  Mask P/R:   {m['mask_precision']:.4f} / {m['mask_recall']:.4f}")
    print()
    print("  Per-class:")
    for name, c in m["per_class"].items():
        print(f"    {name:18s}: box_AP50={c['box_AP50']:.4f}  mask_AP50={c['mask_AP50']:.4f}")

    out_path = Path(args.output) if args.output else (Path(args.weights).parent.parent / f"seg_eval_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(m, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
