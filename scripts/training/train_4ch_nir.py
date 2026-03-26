import argparse
import os
import warnings

warnings.filterwarnings('ignore')

from pathlib import Path

import yaml

from ultralytics import YOLO

# Monkey patch
from ultralytics.data import build
from ultralytics.models.yolo.detect import train as detect_train
from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset
from ultralytics.utils import colorstr

def _patched_build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    # Extract channels from dataset YAML
    # Ultralytics parses YAML into 'data' dict
    ch = data.get('channels') or data.get('ch', 3)
    print(f"[MonkeyPatch] Intercepted build_yolo_dataset for mode={mode}. Using channels={ch}")
    
    return MaskRescueNPYDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',
        hyp=cfg,
        rect=rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0,
        prefix=colorstr(f'{mode}: ') if mode in ('train', 'val') else '',
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0,
        channels=ch, # Pass the channels explicitly
    )

# Apply patch
build.build_yolo_dataset = _patched_build_yolo_dataset
detect_train.build_yolo_dataset = _patched_build_yolo_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO11n on MARIDA 4-channel (RGB+NIR) inputs')
    parser.add_argument('--data', default='marida_rescue_split_4ch_nir.yaml')
    parser.add_argument('--weights', default='yolo11n.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', default='marida_runs')
    parser.add_argument('--name', default='yolo11n_4ch_nir')
    parser.add_argument('--no-amp', action='store_true')
    return parser.parse_args()

EXPECTED_CHANNELS = 4


def _assert_yaml_channels(data_path: str | os.PathLike[str], expected: int) -> None:
    yaml_path = Path(data_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    with yaml_path.open('r', encoding='utf-8') as f:
        payload = yaml.safe_load(f) or {}
    declared = payload.get('channels')
    if declared is None:
        raise ValueError(f"Dataset YAML {yaml_path} is missing a 'channels' field")
    if declared != expected:
        raise ValueError(
            f"Dataset YAML {yaml_path} declares channels={declared}, expected {expected}."
        )
    print(f"[Config] {yaml_path} declares channels={declared} (expected {expected}).")


def train_4ch(args):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    _assert_yaml_channels(args.data, EXPECTED_CHANNELS)

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        exist_ok=True,
        amp=not args.no_amp,
        verbose=True,
    )
    print('4ch training finished.')


if __name__ == '__main__':
    args = parse_args()
    train_4ch(args)
