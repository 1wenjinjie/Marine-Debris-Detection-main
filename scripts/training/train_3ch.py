import argparse
import os
from pathlib import Path

import yaml

# Force allow KMP duplicate lib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset
import ultralytics.data.build
from ultralytics.models.yolo.detect import train as detect_train
from ultralytics.utils import colorstr
# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

EXPECTED_CHANNELS = 3


# --------------------------------------------------------------------------------
# Monkey-patch build_yolo_dataset to force channels=3
# --------------------------------------------------------------------------------
_original_build_yolo_dataset = ultralytics.data.build.build_yolo_dataset

def _custom_build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    print(f"DEBUG: _custom_build_yolo_dataset called for mode={mode} (3ch)")
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
        channels=3  # <--- FORCE 3 CHANNELS (RGB)
    )

# Apply the patch
ultralytics.data.build.build_yolo_dataset = _custom_build_yolo_dataset
detect_train.build_yolo_dataset = _custom_build_yolo_dataset

# --------------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11n on MARIDA 3-channel inputs")
    parser.add_argument("--data", default="marida_rescue_split_3ch.yaml", help="Dataset YAML")
    parser.add_argument("--weights", default="yolo11n.pt", help="Initial weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="marida_runs")
    parser.add_argument("--name", default="yolo11n_3ch_rgb")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data to use")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    return parser.parse_args()


def _assert_yaml_channels(data_path: str | os.PathLike[str], expected: int) -> None:
    yaml_path = Path(data_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    declared = payload.get("channels")
    if declared is None:
        raise ValueError(f"Dataset YAML {yaml_path} is missing a 'channels' field")
    if declared != expected:
        raise ValueError(
            f"Dataset YAML {yaml_path} declares channels={declared}, expected {expected}."
        )
    print(f"[Config] {yaml_path} declares channels={declared} (expected {expected}).")


def main(args):
    _assert_yaml_channels(args.data, EXPECTED_CHANNELS)
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        device=args.device,
        exist_ok=True,
        amp=not args.no_amp,
        verbose=True,
        fraction=args.fraction,
    )
    print("3ch training finished.")


if __name__ == "__main__":
    main(parse_args())
