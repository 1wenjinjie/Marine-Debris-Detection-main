"""Generate per-file channel counts for the mask_bbox_rescue_split dataset.

This script reads the manifest files (train/val/test .txt) created earlier and
records the raw numpy shape, interpreted HWC shape, and channel count for every
sample. Results are stored as JSON for further inspection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def interpret_array(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    """Return array in HWC format plus a note about orientation."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (HWC or CHW). Got shape {arr.shape}")

    # If first dimension is small (<=8) and likely channel-first, transpose it
    if arr.shape[0] <= 8 and arr.shape[0] <= arr.shape[2] and arr.shape[0] not in (arr.shape[1], arr.shape[2]):
        arr = np.transpose(arr, (1, 2, 0))
        orientation = "CHW->HWC"
    else:
        orientation = "HWC"

    return arr, orientation


def load_manifest(manifest_path: Path) -> List[Path]:
    lines = [l.strip() for l in manifest_path.read_text().splitlines() if l.strip()]
    return [Path(l) for l in lines]


def main():
    dataset_root = Path("mask_bbox_rescue_split").resolve()
    manifests = {
        split: dataset_root / f"{split}.txt"
        for split in ("train", "val", "test")
    }

    for split, path in manifests.items():
        if not path.exists():
            raise FileNotFoundError(f"Manifest missing: {path}")

    records = []
    histogram: Dict[int, int] = {}
    by_split: Dict[str, Dict[str, int]] = {split: {"files": 0, "channels": {}} for split in manifests}

    for split, manifest in manifests.items():
        rel_paths = load_manifest(manifest)
        for rel in rel_paths:
            npy_path = rel if rel.is_absolute() else dataset_root / rel
            raw_shape = None
            orientation = None
            channel_count = None
            interpreted_shape = None
            error = None
            try:
                arr = np.load(npy_path)
                raw_shape = arr.shape
                arr, orientation = interpret_array(arr)
                interpreted_shape = arr.shape
                channel_count = arr.shape[2]
                histogram[channel_count] = histogram.get(channel_count, 0) + 1
                by_split[split]["files"] += 1
                by_split[split]["channels"].setdefault(channel_count, 0)
                by_split[split]["channels"][channel_count] += 1
            except Exception as exc:
                error = str(exc)
            records.append(
                {
                    "split": split,
                    "relative_path": rel.as_posix(),
                    "absolute_path": str(npy_path),
                    "raw_shape": raw_shape,
                    "interpreted_shape": interpreted_shape,
                    "orientation": orientation,
                    "channels": channel_count,
                    "error": error,
                }
            )

    summary = {
        "dataset_root": dataset_root.as_posix(),
        "total_files": sum(v["files"] for v in by_split.values()),
        "channel_histogram": histogram,
        "by_split": by_split,
        "records": records,
    }

    output_path = dataset_root / "channel_report.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Report saved to {output_path}")
    print("Channel histogram:")
    for ch in sorted(histogram):
        print(f"  {ch} channels: {histogram[ch]} files")


if __name__ == "__main__":
    main()
