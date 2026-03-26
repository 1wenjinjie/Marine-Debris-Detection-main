"""Run YOLO validation (MaskRescue-friendly) with optional TXT/conf outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from ultralytics import YOLO
from ultralytics.data import build as yolo_build
from ultralytics.models.yolo.detect import val as detect_val
from ultralytics.utils import colorstr

from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset


def _make_builder(channels: int, dataset_root: Path):
    def _builder(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
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
            prefix=colorstr(f"{mode}: "),
            use_segments=cfg.task == "segment",
            use_keypoints=cfg.task == "pose",
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
            dataset_root=dataset_root,
            channels=channels,
        )

    return _builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Path to model weights (pt)")
    parser.add_argument("--data", type=Path, default=Path("marida_rescue_7ch.yaml"))
    parser.add_argument("--channels", type=int, default=None, help="Override channel count if needed")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=str, default="runs_marida")
    parser.add_argument("--name", type=str, default="val_with_outputs")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--save_txt", action="store_true")
    parser.add_argument("--save_conf", action="store_true")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--save_hybrid", action="store_true")
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
    dataset_root = Path(data_cfg.get("path", data_path.parent)).resolve()

    builder = _make_builder(channels, dataset_root)
    yolo_build.build_yolo_dataset = builder
    detect_val.build_yolo_dataset = builder

    model = YOLO(str(weights_path))
    model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name=args.name,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_json=args.save_json,
        save_hybrid=args.save_hybrid,
    )


if __name__ == "__main__":
    main()
