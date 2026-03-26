"""Compare prediction files against ground-truth presence for a target class."""
from __future__ import annotations

import argparse
from pathlib import Path

from visualize_dataset import load_labels


def stems_with_class(labels_dir: Path, class_id: int) -> set[str]:
    stems: set[str] = set()
    if not labels_dir.exists():
        return stems
    for txt in labels_dir.glob("*.txt"):
        labels = load_labels(txt)
        if any(int(row[0]) == class_id for row in labels):
            stems.add(txt.stem)
    return stems


def collect_dataset_stems(dataset_root: Path, class_id: int) -> set[str]:
    stems: set[str] = set()
    for split in ("train", "val", "test"):
        stems |= stems_with_class(dataset_root / split / "labels", class_id)
    return stems


def stems_from_predictions(pred_dir: Path, class_id: int) -> set[str]:
    hits: set[str] = set()
    for txt in pred_dir.glob("*.txt"):
        with txt.open() as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(float(parts[0]))
                except ValueError:
                    continue
                if cls == class_id:
                    hits.add(txt.stem)
                    break
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pred_dir", type=Path, help="Directory containing prediction txt files")
    parser.add_argument("--dataset-root", type=Path, default=Path("mask_bbox_rescue_split"))
    parser.add_argument("--class-id", type=int, required=True)
    args = parser.parse_args()

    gt_stems = collect_dataset_stems(args.dataset_root, args.class_id)
    pred_stems = stems_from_predictions(args.pred_dir, args.class_id)

    tp = sorted(gt_stems & pred_stems)
    fp = sorted(pred_stems - gt_stems)

    print(f"GT images with class {args.class_id}: {len(gt_stems)}")
    print(f"Predicted images with class {args.class_id}: {len(pred_stems)}")
    print(f"Overlap (potential TP): {len(tp)} -> {tp}")
    print(f"Likely FP (no GT class): {len(fp)} -> {fp}")


if __name__ == "__main__":
    main()
