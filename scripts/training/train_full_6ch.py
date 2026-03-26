import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
import argparse

# Monkey patch (same approach as RGB) to force custom dataset
from ultralytics.data import build
from ultralytics.models.yolo.detect import train as detect_train
from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset
from ultralytics.utils import colorstr

_original_build_yolo_dataset = build.build_yolo_dataset

def _patched_build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    print(f"[MonkeyPatch 6ch] Intercepted build_yolo_dataset for mode={mode}")
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
    )

# Apply patch
build.build_yolo_dataset = _patched_build_yolo_dataset
detect_train.build_yolo_dataset = _patched_build_yolo_dataset
print("DEBUG: Applied monkey patch for 6ch training")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='marida_rescue_split_6ch.yaml')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--dryrun', action='store_true', help='Only instantiate dataset and report found images')
    p.add_argument('--fraction', type=float, default=1.0, help='Fraction of data to use (useful for smoke tests)')
    return p.parse_args()


def dry_run_check(data_yaml, fraction=0.01):
    # Try to instantiate the custom MaskRescueNPYDataset directly and report how many images found
    from pathlib import Path

    print('[dryrun] Building dataset (no training) ...')
    try:
        # Resolve dataset root and train image path from the YAML/ dict if provided
        if isinstance(data_yaml, dict):
            dataset_root = Path(data_yaml.get('path', '.'))
            img_path = str(dataset_root / data_yaml.get('train', 'train/images'))
        else:
            dataset_root = Path('.')
            img_path = data_yaml

        ds = MaskRescueNPYDataset(
            img_path=img_path,
            imgsz=640,
            batch_size=1,
            augment=False,
            hyp=None,
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix='dryrun: ',
            use_segments=False,
            use_keypoints=False,
            classes=None,
            data=data_yaml,
            fraction=fraction,
            dataset_root=dataset_root,
            channels=6,
        )

        print(f"[dryrun] Dataset built. Found {len(ds.im_files)} images (fraction={fraction}).")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[dryrun] Failed to build dataset: {e}")


def train_6ch(args):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    model = YOLO('yolo11n.pt')
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project='marida_runs',
        name='yolo11n_6ch_full',
        device='0',
        workers=4,
        patience=20,
        save=True,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        cos_lr=True,
        close_mosaic=10,
        fraction=args.fraction,
    )
    print('6ch training finished.')


if __name__ == '__main__':
    args = parse_args()
    if args.dryrun:
        # Load YAML to dict if possible
        import yaml
        with open(args.data, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
        dry_run_check(data_yaml, fraction=args.fraction)
    else:
        train_6ch(args)
