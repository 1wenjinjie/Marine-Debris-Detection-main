import argparse
import json
import os
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.data import build as yolo_build
from ultralytics.models.yolo.detect import val as detect_val
from ultralytics.utils import colorstr

from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO model on MaskRescue NPY dataset")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default="val_mask_rescue")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--run-label", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    declared_channels = data_cfg.get("channels")
    channels = args.channels or declared_channels or 3
    if declared_channels is not None and declared_channels != channels:
        raise ValueError(
            f"Channel mismatch: YAML declares {declared_channels}, but CLI requested {channels}."
        )

    dataset_root = Path(data_cfg.get("path", data_path.parent)).resolve()
    builder = _make_builder(channels, dataset_root)

    # Apply monkey patch so model.val uses our dataset
    yolo_build.build_yolo_dataset = builder
    detect_val.build_yolo_dataset = builder

    model = YOLO(str(weights_path))
    results = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    metrics = getattr(results, "results_dict", None)
    if metrics is None:
        raise RuntimeError("Ultralytics result missing results_dict; upgrade/downgrade may be needed.")

    payload = {
        "run": args.run_label or weights_path.parent.parent.name,
        "data": str(data_path),
        "weights": str(weights_path),
        "channels": channels,
        "metrics": metrics,
        "speed": getattr(results, "speed", {}),
    }

    out_path = Path(args.output_json) if args.output_json else weights_path.parent.parent / "eval" / "val_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Validation metrics saved to {out_path}")


if __name__ == "__main__":
    main()
