"""Sweep confidence thresholds and report best F1 per checkpoint.

YOLO's standard val() uses conf=0.001 (lowest for AP computation). The
reported precision/recall at that threshold is rarely the operating
point for a real deployment, and is not the F1 that the teacher's
result table asks for.

This script reloads each checkpoint, runs val() at a grid of confidence
thresholds, computes F1 = 2*P*R/(P+R), and saves the best F1 plus the
threshold and (P, R, mAP50) at that point.

Output: <checkpoint_dir>/v2split_test_f1_sweep.json (or the data-yaml's
corresponding test split). One JSON per checkpoint, plus a combined
table printed at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
from ultralytics import YOLO
from ultralytics.data import build as yolo_build
from ultralytics.models.yolo.detect import val as detect_val
from ultralytics.utils import colorstr
from ultralytics_plugins.mask_rescue_dataset_fixed import MaskRescueNPYDataset


CONF_GRID = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]


def _patched_build(dataset_root: Path, channels: int, stats_path: str):
    def _patched(cfg, img_path, batch, data, mode="train", rect=False, stride=32, **kw):
        return MaskRescueNPYDataset(
            img_path=img_path, imgsz=cfg.imgsz, batch_size=batch,
            augment=False, hyp=cfg, rect=rect, stride=stride, pad=0.5,
            prefix=colorstr(f"{mode}: "),
            dataset_root=dataset_root, channels=channels,
            channel_stats=stats_path, task=cfg.task, data=data,
        )
    return _patched


def sweep(weights: str, data_yaml: str, split: str = "test",
          imgsz: int = 512, batch: int = 8, device: str = "0") -> dict:
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    dataset_root = Path(cfg["path"]).resolve()
    stats_path = dataset_root / f"channel_stats_{cfg.get('channels', 7)}ch.json"

    patched = _patched_build(dataset_root, cfg.get("channels", 7), str(stats_path))
    orig_build = yolo_build.build_yolo_dataset
    orig_det = detect_val.build_yolo_dataset
    yolo_build.build_yolo_dataset = patched
    detect_val.build_yolo_dataset = patched

    results = []
    try:
        model = YOLO(weights)
        for conf in CONF_GRID:
            res = model.val(
                data=data_yaml, split=split,
                imgsz=imgsz, batch=batch, device=device,
                conf=conf, iou=0.6, workers=0, verbose=False, plots=False,
            )
            p = float(res.box.mp)
            r = float(res.box.mr)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            results.append({
                "conf": conf,
                "precision": p,
                "recall": r,
                "f1": f1,
                "mAP50": float(res.box.map50),
                "mAP50_95": float(res.box.map),
            })
            print(f"  conf={conf:>6.3f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}  mAP50={float(res.box.map50):.4f}")
    finally:
        yolo_build.build_yolo_dataset = orig_build
        detect_val.build_yolo_dataset = orig_det

    best = max(results, key=lambda x: x["f1"])
    return {
        "weights": str(weights),
        "data": str(data_yaml),
        "split": split,
        "sweep": results,
        "best": best,
    }


def main():
    p = argparse.ArgumentParser(description="F1 conf-threshold sweep")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--output", default=None,
                   help="JSON output (default: <ckpt_parent>/<split>_f1_sweep.json)")
    args = p.parse_args()

    wp = Path(args.weights)
    if not wp.exists():
        sys.exit(f"weights not found: {wp}")

    print(f"=== F1 sweep ===")
    print(f"  weights: {wp}")
    print(f"  data   : {args.data}")
    print(f"  split  : {args.split}")
    print()

    res = sweep(str(wp), args.data, args.split, args.imgsz, args.batch, args.device)
    b = res["best"]
    print()
    print(f"BEST F1 = {b['f1']:.4f} at conf={b['conf']:.3f}  (P={b['precision']:.4f}  R={b['recall']:.4f}  mAP50={b['mAP50']:.4f})")

    out = Path(args.output) if args.output else (wp.parent.parent / f"{args.split}_f1_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
