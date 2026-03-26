"""Utilities to audit samples containing a specific class and optionally render visualizations."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from visualize_dataset import load_labels, visualize_sample, load_class_names


@dataclass
class SampleInfo:
    stem: str
    label_path: Path
    image_path: Path
    total_boxes: int
    target_boxes: int


def find_samples(dataset_dir: Path, class_id: int) -> list[SampleInfo]:
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    samples: list[SampleInfo] = []

    for label_path in labels_dir.glob("*.txt"):
        labels = load_labels(label_path)
        if not labels:
            continue

        total_boxes = len(labels)
        target_boxes = sum(1 for row in labels if int(row[0]) == class_id)
        if target_boxes == 0:
            continue

        stem = label_path.stem
        # Prefer GeoTIFF, then PNG/JPG fallback
        image_path = images_dir / f"{stem}.tif"
        if not image_path.exists():
            for ext in (".png", ".jpg", ".jpeg", ".npy"):
                alt = images_dir / f"{stem}{ext}"
                if alt.exists():
                    image_path = alt
                    break
        if not image_path.exists():
            raise FileNotFoundError(f"Image for {stem} not found in {images_dir}")

        samples.append(
            SampleInfo(
                stem=stem,
                label_path=label_path,
                image_path=image_path,
                total_boxes=total_boxes,
                target_boxes=target_boxes,
            )
        )

    return sorted(samples, key=lambda s: s.stem)


def save_visuals(samples: Iterable[SampleInfo], class_names: list[str] | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        save_path = out_dir / f"{sample.stem}.png"
        visualize_sample(
            sample.image_path,
            sample.label_path,
            class_names=class_names,
            save_path=save_path,
            show=False,
        )


def summarize(samples: list[SampleInfo], split: str) -> str:
    total_imgs = len(samples)
    total_boxes = sum(s.total_boxes for s in samples)
    target_boxes = sum(s.target_boxes for s in samples)
    return (
        f"Split {split}: {total_imgs} images with class, "
        f"{target_boxes} target boxes out of {total_boxes} total boxes"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("mask_bbox_rescue_split"))
    parser.add_argument("--splits", nargs="*", default=("train", "val", "test"))
    parser.add_argument("--class-id", type=int, required=True, help="Target class id to audit")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples per split (0 = all)")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory to store rendered PNGs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.dataset_root)

    for split in args.splits:
        split_dir = args.dataset_root / split
        if not split_dir.exists():
            print(f"⚠️ Split not found: {split_dir}")
            continue

        samples = find_samples(split_dir, args.class_id)
        if not samples:
            print(f"Split {split}: no samples with class {args.class_id}")
            continue

        if args.limit > 0:
            samples = samples[: args.limit]

        print(summarize(samples, split))
        for sample in samples:
            print(
                f"  - {sample.stem}: target={sample.target_boxes}, total={sample.total_boxes}, "
                f"image={sample.image_path.name}"
            )

        if args.save_dir:
            out_dir = args.save_dir / split
            save_visuals(samples, class_names, out_dir)
            print(f"  Saved {len(samples)} visuals to {out_dir}")


if __name__ == "__main__":
    main()
