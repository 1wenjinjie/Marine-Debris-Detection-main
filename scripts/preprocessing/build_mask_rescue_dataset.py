"""Build a YOLO-style dataset from curated mask_bbox_preview labels.

This utility copies the selected label TXT files into a fresh
train/val/test directory layout while reusing the original
multispectral imagery (GeoTIFF + .npy cache) from MARIDA_YOLO_6band_split.

It relies on ``MARIDA_YOLO_split/split_info.json`` to determine the split
assignment of each patch (train/val/test).
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict

from tqdm import tqdm

DEFAULT_LABELS_DIR = Path("mask_bbox_preview/labels")
DEFAULT_CLASSES_FILE = Path("mask_bbox_preview/classes.txt")
DEFAULT_SPLIT_INFO = Path("MARIDA_YOLO_split/split_info.json")
DEFAULT_SOURCE_DATASET = Path("MARIDA_YOLO_6band_split")
DEFAULT_OUTPUT_DATASET = Path("mask_bbox_rescue_split")


def load_split_mapping(split_info_path: Path) -> Dict[str, str]:
    with split_info_path.open("r", encoding="utf-8") as fh:
        split_info = json.load(fh)

    mapping: Dict[str, str] = {}
    for split_name, items in split_info.items():
        if not isinstance(items, list):
            continue
        for patch in items:
            mapping[patch] = split_name
    return mapping


def copy_file(src: Path, dst: Path, overwrite: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_dataset(
    labels_dir: Path,
    classes_file: Path,
    split_info_path: Path,
    source_dataset: Path,
    output_dataset: Path,
    *,
    copy_images: bool,
    copy_npy: bool,
    overwrite: bool,
) -> dict:
    mapping = load_split_mapping(split_info_path)
    stats = {
        "total_labels": 0,
        "copied_labels": 0,
        "missing_split": 0,
        "missing_image": 0,
        "missing_npy": 0,
        "per_split": Counter(),
    }

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"❌ No TXT files found under {labels_dir}")

    for label_path in tqdm(label_files, desc="Copying labels"):
        base = label_path.stem
        stats["total_labels"] += 1

        split = mapping.get(base)
        if split is None:
            stats["missing_split"] += 1
            tqdm.write(f"⚠️  Split info not found for {base}, skipping")
            continue

        split_dir = output_dataset / split
        labels_output = split_dir / "labels" / label_path.name
        labels_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_path, labels_output)
        stats["copied_labels"] += 1
        stats["per_split"][split] += 1

        if copy_images:
            src_image = source_dataset / split / "images" / f"{base}.tif"
            dst_image = split_dir / "images" / src_image.name
            if not copy_file(src_image, dst_image, overwrite):
                stats["missing_image"] += 1
                tqdm.write(f"⚠️  Missing source image for {base}: {src_image}")

        if copy_npy:
            src_npy = source_dataset / split / "images" / f"{base}.npy"
            dst_npy = split_dir / "images" / src_npy.name
            if not copy_file(src_npy, dst_npy, overwrite):
                stats["missing_npy"] += 1
                tqdm.write(f"⚠️  Missing source npy for {base}: {src_npy}")

    # Copy classes.txt for reference
    if classes_file.exists():
        shutil.copy2(classes_file, output_dataset / "classes.txt")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rescue dataset from curated labels")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--classes-file", type=Path, default=DEFAULT_CLASSES_FILE)
    parser.add_argument("--split-info", type=Path, default=DEFAULT_SPLIT_INFO)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--output-dataset", type=Path, default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--no-copy-images", action="store_true", help="Only copy labels")
    parser.add_argument("--no-copy-npy", action="store_true", help="Skip .npy cache copy")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset(
        labels_dir=args.labels_dir,
        classes_file=args.classes_file,
        split_info_path=args.split_info,
        source_dataset=args.source_dataset,
        output_dataset=args.output_dataset,
        copy_images=not args.no_copy_images,
        copy_npy=not args.no_copy_npy,
        overwrite=args.overwrite,
    )

    print("\n========== Summary ==========")
    print(f"Labels processed: {stats['total_labels']}")
    print(f"Copied labels:   {stats['copied_labels']}")
    print(f"Missing splits:  {stats['missing_split']}")
    print(f"Missing images:  {stats['missing_image']}")
    print(f"Missing npy:     {stats['missing_npy']}")
    print("Per split counts:")
    for split, count in stats["per_split"].items():
        print(f"  {split}: {count}")


if __name__ == "__main__":
    main()
