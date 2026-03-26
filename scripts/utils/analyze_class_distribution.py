"""Analyze MARIDA YOLO label distribution and emit recommended class-balance weights."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("marida_mask_rescue_7ch.yaml"),
        help="YOLO data YAML that defines dataset root and class names",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the class balance JSON (defaults to <dataset_root>/class_balance_weights.json)",
    )
    parser.add_argument("--power", type=float, default=1.0, help="Recommended default power for weighting")
    parser.add_argument(
        "--floor",
        type=float,
        default=0.25,
        help="Lower clamp for per-sample multipliers before power transform",
    )
    parser.add_argument(
        "--ceiling",
        type=float,
        default=3.0,
        help="Upper clamp for per-sample multipliers before power transform",
    )
    return parser.parse_args()


def load_data_yaml(path: Path) -> Tuple[Path, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    dataset_root = Path(payload.get("path") or path.parent).expanduser().resolve()
    names = payload.get("names")
    if isinstance(names, dict):
        ordered = [name for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
        class_names = [str(name) for name in ordered]
    elif isinstance(names, (list, tuple)):
        class_names = [str(name) for name in names]
    else:
        raise ValueError(f"Unable to parse class names from {path}")
    return dataset_root, class_names


def parse_label_file(label_path: Path) -> List[int]:
    cls_ids: List[int] = []
    if not label_path.exists():
        return cls_ids
    for raw in label_path.read_text(encoding="utf-8").strip().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        cls_ids.append(cls)
    return cls_ids


def accumulate_counts(dataset_root: Path, class_names: List[str]) -> Tuple[Dict[str, Dict[str, int]], Counter]:
    splits = {"train", "val", "test"}
    per_split: Dict[str, Dict[str, int]] = {split: {name: 0 for name in class_names} for split in splits}
    totals: Counter = Counter()
    for split in splits:
        label_dir = dataset_root / split / "labels"
        if not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            cls_ids = parse_label_file(label_path)
            for cls in cls_ids:
                if 0 <= cls < len(class_names):
                    name = class_names[cls]
                    per_split[split][name] += 1
                    totals[name] += 1
    return per_split, totals


def compute_weights(totals: Counter, class_names: List[str]) -> List[float]:
    counts = [max(totals.get(name, 0), 0) for name in class_names]
    nonzero = [count for count in counts if count > 0]
    if not nonzero:
        return [1.0 for _ in class_names]
    max_count = max(nonzero)
    min_nonzero = min(nonzero)
    weights: List[float] = []
    for count in counts:
        if count <= 0:
            ratio = max_count / min_nonzero
        else:
            ratio = max_count / count
        weights.append(ratio)
    # Normalize to mean ~1 for stability
    mean = sum(weights) / len(weights)
    if mean > 0:
        weights = [w / mean for w in weights]
    return weights


def main() -> None:
    args = parse_args()
    dataset_root, class_names = load_data_yaml(args.data.resolve())
    output = args.output or (dataset_root / "class_balance_weights.json")
    per_split, totals = accumulate_counts(dataset_root, class_names)
    weights = compute_weights(totals, class_names)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_yaml": str(args.data.resolve()),
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "class_counts": {
            split: counts for split, counts in sorted(per_split.items())
        },
        "total_class_counts": {name: totals.get(name, 0) for name in class_names},
        "recommended_weights": {name: weight for name, weight in zip(class_names, weights)},
        "recommended_power": args.power,
        "min_multiplier": args.floor,
        "max_multiplier": args.ceiling,
        "notes": {
            "weight_formula": "weights = (max_count / class_count) normalized to mean=1; zero-count classes receive max boost",
            "instructions": "MaskRescueNPYDataset automatically consumes this file to amplify rare classes via sample weights.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved class balance weights to {output}")


if __name__ == "__main__":
    main()
