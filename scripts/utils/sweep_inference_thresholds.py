"""Grid-search confidence/IoU thresholds for MaskRescue YOLO weights."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
import sys
from typing import Dict, List

import yaml
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.data import build as yolo_build
from ultralytics.models.yolo.detect import val as detect_val
from ultralytics.utils import LOGGER

from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset


def _parse_float_list(spec: str) -> List[float]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty threshold specification")
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Range spec '{spec}' must be start:stop:step")
        start, stop, step = map(float, parts)
        if step <= 0:
            raise ValueError("Range step must be positive")
        values: List[float] = []
        current = start
        while current <= stop + 1e-9:
            values.append(round(current, 6))
            current += step
        return values
    return [float(item) for item in spec.split(",") if item]


def _parse_names(raw_names) -> Dict[int, str]:
    if raw_names is None:
        return {}
    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}
    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}
    raise TypeError(f"Unsupported names format: {type(raw_names)}")


def _make_builder(channels: int, dataset_root: Path):
    def _builder(cfg, img_path, batch, data, mode="val", rect=False, stride=32):
        return MaskRescueNPYDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=cfg,
            rect=rect,
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            use_segments=cfg.task == "segment",
            use_keypoints=cfg.task == "pose",
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
            dataset_root=dataset_root,
            channels=channels,
        )

    return _builder


def _extract_metrics(val_results, class_names: Dict[int, str]) -> Dict[str, object]:
    metrics_obj = getattr(val_results, "metrics", None)
    if metrics_obj is None:
        return {}
    box = getattr(metrics_obj, "box", None)
    if box is None:
        return {}

    def _mean(value):
        if value is None:
            return None
        try:
            import numpy as np  # type: ignore

            arr = np.asarray(value, dtype=float)
            if arr.size == 0:
                return None
            return float(arr.mean())
        except Exception:
            if isinstance(value, (list, tuple)) and value:
                return float(sum(value) / len(value))
            if isinstance(value, (int, float)):
                return float(value)
            return None

    per_class = {}
    maps = getattr(box, "maps", None)
    prec = getattr(box, "p", None)
    rec = getattr(box, "r", None)
    if maps is not None:
        for idx, map_val in enumerate(maps):
            name = class_names.get(idx, str(idx))
            cls_metrics = {"map50-95": float(map_val) if map_val is not None else None}
            if prec is not None and idx < len(prec):
                cls_metrics["precision"] = float(prec[idx])
            if rec is not None and idx < len(rec):
                cls_metrics["recall"] = float(rec[idx])
            per_class[name] = cls_metrics

    summary = {
        "precision": _mean(getattr(box, "p", None)),
        "recall": _mean(getattr(box, "r", None)),
        "map50": float(getattr(box, "map50", 0.0)),
        "map": float(getattr(box, "map", 0.0)),
        "per_class": per_class,
    }

    speed = getattr(val_results, "speed", None)
    if isinstance(speed, dict):
        summary["speed"] = {k: float(v) for k, v in speed.items()}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Path to YOLO weights (.pt)")
    parser.add_argument("--data", type=Path, default=Path("marida_rescue_7ch.yaml"))
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--conf-list", type=str, default="0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--iou-list", type=str, default="0.45,0.5,0.55")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--project", type=str, default="runs_marida")
    parser.add_argument("--name", type=str, default="threshold_sweeps")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep YOLO val artifacts for each combo")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path for JSON output")
    parser.add_argument("--save-csv", type=Path, default=None, help="Optional path for CSV summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = args.weights.resolve()
    data_path = args.data.resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    declared_channels = data_cfg.get("channels")
    channels = args.channels or declared_channels or 3
    dataset_root = args.dataset_root or data_cfg.get("path") or data_path.parent
    dataset_root = Path(dataset_root).resolve()
    class_names = _parse_names(data_cfg.get("names"))

    conf_values = _parse_float_list(args.conf_list)
    iou_values = _parse_float_list(args.iou_list)
    combos = list(product(conf_values, iou_values))
    if not combos:
        raise RuntimeError("No conf/iou combinations parsed")

    LOGGER.info(
        f"Sweeping {len(conf_values)} conf * {len(iou_values)} iou = {len(combos)} runs "
        f"(channels={channels}, dataset_root={dataset_root})"
    )

    builder = _make_builder(channels, dataset_root)
    yolo_build.build_yolo_dataset = builder
    detect_val.build_yolo_dataset = builder

    model = YOLO(str(weights_path))

    rows: List[Dict[str, object]] = []
    for idx, (conf, iou) in enumerate(combos, start=1):
        run_name = args.name if not args.keep_artifacts else f"{args.name}_c{conf:.2f}_i{iou:.2f}"
        LOGGER.info(f"[{idx}/{len(combos)}] conf={conf:.3f}, iou={iou:.3f}, run={run_name}")
        val_results = model.val(
            data=str(data_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=conf,
            iou=iou,
            max_det=args.max_det,
            project=args.project,
            name=run_name,
            exist_ok=True,
            plots=args.keep_artifacts,
            save=args.keep_artifacts,
            save_txt=False,
            save_json=False,
        )
        metrics = _extract_metrics(val_results, class_names)
        row = {
            "conf": conf,
            "iou": iou,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "map50": metrics.get("map50"),
            "map": metrics.get("map"),
            "per_class": metrics.get("per_class", {}),
            "speed": metrics.get("speed"),
        }
        rows.append(row)

    if not rows:
        LOGGER.warning("No rows collected from sweep")
        return

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.save_json or (save_dir / "threshold_sweep.json")
    json_path.write_text(json.dumps({"weights": str(weights_path), "results": rows}, indent=2), encoding="utf-8")
    LOGGER.info(f"Saved JSON summary to {json_path}")

    csv_path = args.save_csv or (save_dir / "threshold_sweep.csv")
    fieldnames = ["conf", "iou", "precision", "recall", "map50", "map"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    LOGGER.info(f"Saved CSV summary to {csv_path}")


if __name__ == "__main__":
    main()
