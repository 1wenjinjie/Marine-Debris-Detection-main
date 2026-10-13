"""
Locked reproduction script for the v1 7ch oracle baseline.

Purpose
-------
Re-run ``model.val()`` on the historical SOTA checkpoint with a fully pinned
environment so the reported numbers can be regenerated from any clean checkout.
This is the answer to Audit Issue 5 ("model.val() reproduction is not fully
locked") in docs/paper/7CH_BASELINE_CURRENT_AUDIT_2026-05-18.md.

What it pins
------------
- checkpoint path + SHA256
- dataset YAML + dataset root + channel_stats path
- loader version (mask_rescue_dataset_fixed by default)
- working directory (changes to repo root)
- imgsz/batch/conf/iou from the original training args.yaml

Reminder
--------
This baseline is a GT-derived-heatmap *oracle*, not a deployable detector.
See the audit doc before quoting its numbers. The script is for reproducing
the historical result so the v1 line stays auditable, not for shipping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import yaml
from ultralytics import YOLO
from ultralytics.data import build as yolo_build
from ultralytics.models.yolo.detect import val as detect_val
from ultralytics.utils import colorstr


DEFAULTS = {
    "weights": "experiments/runs/runs_marida/yolov11n_7ch_512_finetune_wakeboost/weights/best.pt",
    "data": "configs/datasets/marida_mask_rescue_7ch.yaml",
    "channels": 7,
    "imgsz": 512,
    "batch": 8,
    "conf": 0.001,
    "iou": 0.6,
}

EXPECTED = {
    "mAP50": 0.5173,
    "mAP50_95": 0.2702,
    "note": "from audit doc 2026-05-18, full heatmap + explicit channel_stats",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_patched_build(dataset_root: Path, channels: int, channel_stats: str, loader: str):
    if loader == "fixed":
        from ultralytics_plugins.mask_rescue_dataset_fixed import MaskRescueNPYDataset
    else:
        from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset

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


def run(args: argparse.Namespace) -> dict:
    weights = Path(args.weights).resolve()
    data_yaml = Path(args.data).resolve()
    if not weights.exists():
        sys.exit(f"weights not found: {weights}")
    if not data_yaml.exists():
        sys.exit(f"data yaml not found: {data_yaml}")

    data_cfg = yaml.safe_load(data_yaml.read_text())
    dataset_root = Path(data_cfg.get("path", "")).resolve()
    if not dataset_root.exists():
        sys.exit(f"dataset_root from yaml not found: {dataset_root}")

    stats_path = dataset_root / f"channel_stats_{args.channels}ch.json"
    if not stats_path.exists():
        sys.exit(
            f"channel_stats not found at dataset root: {stats_path}\n"
            "Audit fix 4 expects it copied there. Re-run the fix step."
        )

    pin = {
        "repo_root": str(ROOT),
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "channel_stats": str(stats_path),
        "loader": args.loader,
        "channels": args.channels,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "split": args.split,
        "device": args.device,
    }
    print("=" * 60)
    print("REPRODUCTION PIN")
    print("=" * 60)
    for k, v in pin.items():
        print(f"  {k}: {v}")
    print()

    patched = make_patched_build(dataset_root, args.channels, str(stats_path), args.loader)
    orig_build = yolo_build.build_yolo_dataset
    orig_det = detect_val.build_yolo_dataset
    yolo_build.build_yolo_dataset = patched
    detect_val.build_yolo_dataset = patched
    try:
        model = YOLO(str(weights))
        results = model.val(
            data=str(data_yaml),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            workers=0,
            verbose=True,
        )
    finally:
        yolo_build.build_yolo_dataset = orig_build
        detect_val.build_yolo_dataset = orig_det

    names = data_cfg.get("names", [])
    metrics = {
        "pin": pin,
        "expected": EXPECTED,
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class": {n: None for n in names},
    }
    # results.box.ap_class_index maps row -> original class id (skips classes with 0 GT)
    ap_idx = list(getattr(results.box, "ap_class_index", []))
    for row, cls_id in enumerate(ap_idx):
        if cls_id >= len(names):
            continue
        metrics["per_class"][names[cls_id]] = {
            "AP50": float(results.box.ap50[row]),
            "AP50_95": float(results.box.ap[row]),
            "precision": float(results.box.p[row]),
            "recall": float(results.box.r[row]),
        }

    delta_map50 = metrics["mAP50"] - EXPECTED["mAP50"]
    delta_map = metrics["mAP50_95"] - EXPECTED["mAP50_95"]
    metrics["delta_vs_expected"] = {
        "mAP50": round(delta_map50, 4),
        "mAP50_95": round(delta_map, 4),
    }

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  mAP50      : {metrics['mAP50']:.4f}   (expected {EXPECTED['mAP50']:.4f}, delta {delta_map50:+.4f})")
    print(f"  mAP50_95   : {metrics['mAP50_95']:.4f}   (expected {EXPECTED['mAP50_95']:.4f}, delta {delta_map:+.4f})")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print()
    for name, m in metrics["per_class"].items():
        if m is None:
            print(f"  {name:20s} no support / no AP row")
            continue
        print(f"  {name:20s} AP50={m['AP50']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    out = Path(args.output) if args.output else (
        weights.parent.parent / f"v1_reproduce_{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\n  Saved: {out}")
    return metrics


def main():
    p = argparse.ArgumentParser(description="Locked reproduction of v1 7ch oracle baseline")
    p.add_argument("--weights", default=DEFAULTS["weights"])
    p.add_argument("--data", default=DEFAULTS["data"])
    p.add_argument("--channels", type=int, default=DEFAULTS["channels"])
    p.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    p.add_argument("--conf", type=float, default=DEFAULTS["conf"])
    p.add_argument("--iou", type=float, default=DEFAULTS["iou"])
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="0")
    p.add_argument("--loader", choices=["fixed", "legacy"], default="fixed",
                   help="which MaskRescueNPYDataset to use (default: fixed)")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
