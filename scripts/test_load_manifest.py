"""Quick loader test: read manifest and load the first .npy + labels using helper.

Run from repository root:
  python scripts/test_load_manifest.py --manifest mask_bbox_rescue_split/train.txt
"""

from pathlib import Path
import argparse
from datasets.multispectral_yolo_dataset import load_npy_and_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("mask_bbox_rescue_split/train.txt"))
    p.add_argument("--dataset-root", type=Path, default=Path("mask_bbox_rescue_split"))
    args = p.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    lines = [l.strip() for l in args.manifest.read_text().splitlines() if l.strip()]
    if not lines:
        print("Manifest is empty")
        return

    first = args.dataset_root / Path(lines[0]) if not Path(lines[0]).is_absolute() else Path(lines[0])
    print("Loading:", first)
    d = load_npy_and_labels(first, args.dataset_root)
    img = d['image']
    labels = d['labels']
    print(f"image shape: {img.shape}, dtype: {img.dtype}")
    print(f"channels: {img.shape[2] if img.ndim==3 else 'N/A'}")
    print(f"labels count: {len(labels)}")
    if labels:
        print("first labels:")
        for l in labels[:5]:
            print(" ", l)
    print("image path:", d['image_path'])
    print("label path:", d['label_path'])


if __name__ == '__main__':
    main()
