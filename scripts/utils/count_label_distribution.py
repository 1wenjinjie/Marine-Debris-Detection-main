from __future__ import annotations

from collections import Counter
from pathlib import Path


def load_classes(classes_path: Path) -> list[str]:
    with classes_path.open() as f:
        return [line.strip() for line in f if line.strip()]


def summarize_labels(labels_dir: Path, classes: list[str]) -> tuple[int, Counter[int]]:
    counts: Counter[int] = Counter()
    file_count = 0
    for txt_path in sorted(labels_dir.glob("*.txt")):
        file_count += 1
        with txt_path.open() as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_idx = int(parts[0])
                except ValueError:
                    continue
                counts[cls_idx] += 1
    return file_count, counts


def main() -> None:
    root = Path("mask_bbox_preview")
    labels_dir = root / "labels"
    classes_path = root / "classes.txt"

    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)
    if not classes_path.exists():
        raise FileNotFoundError(classes_path)

    classes = load_classes(classes_path)
    file_count, counts = summarize_labels(labels_dir, classes)

    total_boxes = sum(counts.values())
    print(f"Label files: {file_count}")
    print(f"Total boxes: {total_boxes}")
    print("Per-class counts:")
    for idx, name in enumerate(classes):
        print(f"  {idx} - {name}: {counts.get(idx, 0)}")


if __name__ == "__main__":
    main()
