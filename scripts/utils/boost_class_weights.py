"""Utility to multiply confidence weights for samples containing a target class.

Usage example:
    python tools/boost_class_weights.py \
        --summary mask_bbox_rescue_split/knowledge_generation_summary.json \
        --dataset-root mask_bbox_rescue_split \
        --class-id 2 \
        --multiplier 8.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to knowledge_generation_summary.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root that contains train/val image + label manifests",
    )
    parser.add_argument("--class-id", type=int, required=True, help="Target class id to boost")
    parser.add_argument(
        "--multiplier",
        type=float,
        default=5.0,
        help="Multiply confidence_weight by this factor when the sample contains the class.",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=1.0,
        help="Only scale entries whose existing weight is >= this value (to avoid scaling zeroes).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report stats without writing the summary file",
    )
    return parser.parse_args()


def infer_label_path(dataset_root: Path, relative_path: str) -> Path:
    data_path = dataset_root / relative_path
    parts = list(data_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def load_labels(label_path: Path) -> Iterable[int]:
    if not label_path.exists():
        return []
    result = []
    for line in label_path.read_text().strip().splitlines():
        if not line:
            continue
        parts = line.split()
        try:
            cls_id = int(float(parts[0]))
        except (ValueError, IndexError):
            continue
        result.append(cls_id)
    return result


def main() -> None:
    args = parse_args()
    summary_path = args.summary.resolve()
    dataset_root = args.dataset_root.resolve()

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    data = json.loads(summary_path.read_text())

    details = data.get("details") or []
    total = len(details)
    hit = 0
    updated = 0
    for entry in details:
        rel = entry.get("relative_path")
        if not rel:
            continue
        label_path = infer_label_path(dataset_root, rel)
        labels = load_labels(label_path)
        if args.class_id not in labels:
            continue
        hit += 1
        current = float(entry.get("confidence_weight", 1.0))
        if current < args.min_weight:
            continue
        entry["confidence_weight"] = current * args.multiplier
        updated += 1

    if args.dry_run:
        print(
            f"dry-run: samples containing class {args.class_id}: {hit} / {total},"
            f" would update {updated} entries"
        )
        return

    summary_path.write_text(json.dumps(data, indent=2))
    print(
        f"Updated {updated} entries (found {hit} samples with class {args.class_id})"
        f" out of {total} total entries."
    )


if __name__ == "__main__":
    main()
