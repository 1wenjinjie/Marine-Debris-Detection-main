"""Audit the current MaskRescue detection dataset and report split/class issues."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CLASS_NAMES = [
    "Marine_Debris",
    "Dense_Sargassum",
    "Sparse_Sargassum",
    "Natural_Organic",
    "Ship",
    "Foam",
    "Wakes",
]


def _scene_id_from_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("aug_"):
        return stem
    return stem.rsplit("_", 1)[0]


def _read_label_rows(label_path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    if not label_path.exists():
        return rows
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        rows.append([float(x) for x in parts[:5]])
    return rows


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _audit_split(split_dir: Path, class_names: list[str]) -> dict[str, Any]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    image_files = sorted(images_dir.glob("*.npy"))
    label_files = sorted(labels_dir.glob("*.txt"))

    class_counts = Counter()
    image_presence = Counter()
    area_by_class: dict[int, list[float]] = defaultdict(list)
    objects_per_image: list[int] = []
    areas: list[float] = []
    scenes = set()
    labeled_images = 0
    malformed_rows = 0

    for image_path in image_files:
        scenes.add(_scene_id_from_name(image_path))
        label_path = labels_dir / f"{image_path.stem}.txt"
        rows = _read_label_rows(label_path)
        object_count = 0
        present_classes = set()

        if rows:
            labeled_images += 1

        for row in rows:
            try:
                cls = int(row[0])
                w = float(row[3])
                h = float(row[4])
            except (TypeError, ValueError):
                malformed_rows += 1
                continue
            area_px = w * h * 256 * 256
            class_counts[cls] += 1
            area_by_class[cls].append(area_px)
            areas.append(area_px)
            present_classes.add(cls)
            object_count += 1

        for cls in present_classes:
            image_presence[cls] += 1
        objects_per_image.append(object_count)

    total_objects = sum(class_counts.values())
    return {
        "images": len(image_files),
        "label_files": len(label_files),
        "labeled_images": labeled_images,
        "empty_or_missing_labels": len(image_files) - labeled_images,
        "objects": total_objects,
        "objects_per_image_mean": round(total_objects / len(image_files), 4) if image_files else 0.0,
        "objects_per_labeled_image_mean": round(total_objects / labeled_images, 4) if labeled_images else 0.0,
        "scenes": len(scenes),
        "scene_ids": sorted(scenes),
        "class_counts": {class_names[i]: class_counts.get(i, 0) for i in range(len(class_names))},
        "images_with_class": {class_names[i]: image_presence.get(i, 0) for i in range(len(class_names))},
        "area_px": {
            "min": min(areas) if areas else None,
            "median": _quantile(areas, 0.5),
            "mean": round(sum(areas) / len(areas), 4) if areas else None,
            "p95": _quantile(areas, 0.95),
        },
        "mean_area_by_class": {
            class_names[i]: round(sum(area_by_class[i]) / len(area_by_class[i]), 4) if area_by_class.get(i) else None
            for i in range(len(class_names))
        },
        "malformed_rows": malformed_rows,
    }


def _resolve_manifest_entry(manifest_path: Path, dataset_root: Path, raw: str) -> Path | None:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (manifest_path.parent / candidate).resolve()
    if candidate.exists():
        return candidate
    alt = (dataset_root / raw).resolve()
    if alt.exists():
        return alt
    return None


def _audit_manifest(dataset_root: Path, split: str) -> dict[str, Any]:
    manifest_path = dataset_root / f"{split}.txt"
    if not manifest_path.exists():
        return {"exists": False}

    lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing: list[str] = []
    resolved = 0
    aug_entries = 0

    for raw in lines:
        if "images_aug" in raw:
            aug_entries += 1
        if _resolve_manifest_entry(manifest_path, dataset_root, raw):
            resolved += 1
        else:
            missing.append(raw)

    return {
        "exists": True,
        "manifest_lines": len(lines),
        "resolved_entries": resolved,
        "missing_entries": len(missing),
        "aug_entries": aug_entries,
        "missing_examples": missing[:5],
    }


def audit_dataset(dataset_root: Path, class_names: list[str]) -> dict[str, Any]:
    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = _audit_split(dataset_root / split, class_names)

    totals = Counter()
    for split in splits.values():
        for class_name, count in split["class_counts"].items():
            totals[class_name] += count
    total_objects = sum(totals.values())

    train_scenes = set(splits["train"]["scene_ids"])
    val_scenes = set(splits["val"]["scene_ids"])
    test_scenes = set(splits["test"]["scene_ids"])

    return {
        "dataset_root": str(dataset_root.resolve()),
        "splits": splits,
        "total_unique_scenes": len(train_scenes | val_scenes | test_scenes),
        "total_class_counts": dict(totals),
        "total_class_percent": {
            class_name: round(count / total_objects * 100, 4) if total_objects else 0.0
            for class_name, count in totals.items()
        },
        "scene_overlap": {
            "train_val": {
                "count": len(train_scenes & val_scenes),
                "examples": sorted(train_scenes & val_scenes)[:10],
            },
            "train_test": {
                "count": len(train_scenes & test_scenes),
                "examples": sorted(train_scenes & test_scenes)[:10],
            },
            "val_test": {
                "count": len(val_scenes & test_scenes),
                "examples": sorted(val_scenes & test_scenes)[:10],
            },
        },
        "manifest_audit": {
            split: _audit_manifest(dataset_root, split) for split in ("train", "val", "test")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the current MaskRescue detection dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/processed/mask_bbox_rescue_split"),
        help="Dataset root containing train/val/test folders and optional manifests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    report = audit_dataset(args.dataset_root.resolve(), DEFAULT_CLASS_NAMES)
    payload = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"Saved audit report to {args.output.resolve()}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
