"""Utility to count YOLO label occurrences per class in mask_bbox_preview/labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count YOLO boxes per class in a labels directory.")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("mask_bbox_preview/labels"),
        help="Directory containing YOLO TXT files (default: mask_bbox_preview/labels)",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=Path("mask_bbox_preview/classes.txt"),
        help="Optional class names file (one per line) to map IDs to names",
    )
    return parser.parse_args()


def load_class_names(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def count_labels(labels_dir: Path) -> Counter:
    counts: Counter[int] = Counter()
    for label_file in labels_dir.glob("*.txt"):
        with label_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(parts[0])
                except ValueError:
                    continue
                counts[cls] += 1
    return counts


def main() -> None:
    args = parse_args()
    counts = count_labels(args.labels)
    class_names = load_class_names(args.class_names)

    summary = {
        "total_boxes": sum(counts.values()),
        "per_class": {}
    }
    for cls_id, num in sorted(counts.items()):
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class_{cls_id}"
        summary["per_class"][name] = num

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
