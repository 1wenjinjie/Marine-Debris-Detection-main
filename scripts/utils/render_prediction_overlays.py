"""Render visualizations for selected prediction TXT files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from visualize_dataset import load_class_names, visualize_sample


def find_image_path(dataset_root: Path, stem: str) -> Path:
    for split in ("train", "val", "test"):
        img_dir = dataset_root / split / "images"
        for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Image for {stem} not found under {dataset_root}")


def sanitize_predictions(pred_file: Path) -> list[str]:
    cleaned: list[str] = []
    with pred_file.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cleaned.append(" ".join(parts[:5]))
    return cleaned


def write_temp_labels(lines: Iterable[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pred_dir", type=Path, help="Directory containing prediction txt files")
    parser.add_argument("--dataset-root", type=Path, default=Path("mask_bbox_rescue_split"))
    parser.add_argument("--stems", nargs="*", help="Specific sample stems to render (default: all in pred_dir)")
    parser.add_argument("--out-dir", type=Path, default=Path("visualizations_check/predictions"))
    args = parser.parse_args()

    class_names = load_class_names(args.dataset_root) or []
    stems = args.stems or [p.stem for p in args.pred_dir.glob("*.txt")]

    tmp_dir = args.out_dir / "tmp_labels"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for stem in stems:
        pred_file = args.pred_dir / f"{stem}.txt"
        if not pred_file.exists():
            print(f"Skipping missing prediction file: {pred_file}")
            continue
        lines = sanitize_predictions(pred_file)
        if not lines:
            continue
        tmp_label = tmp_dir / f"{stem}.txt"
        write_temp_labels(lines, tmp_label)
        image_path = find_image_path(args.dataset_root, stem)
        save_path = args.out_dir / f"{stem}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_sample(
            image_path,
            tmp_label,
            class_names=class_names,
            save_path=save_path,
            show=False,
        )
        count += 1
        print(f"Saved visualization to {save_path}")

    print(f"Rendered {count} prediction overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
