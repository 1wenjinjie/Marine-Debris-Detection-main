"""Quickly compare per-channel stats between source 6-band caches and 7-channel target files."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio


@dataclass
class SampleStats:
    path: Path
    exists: bool
    shape: Tuple[int, ...] | None = None
    dtype: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    per_channel_min: np.ndarray | None = None
    per_channel_max: np.ndarray | None = None
    per_channel_mean: np.ndarray | None = None

    def to_dict(self) -> Dict:
        return {
            "path": self.path.as_posix(),
            "exists": self.exists,
            "shape": self.shape,
            "dtype": self.dtype,
            "min": None if self.min_value is None else float(self.min_value),
            "max": None if self.max_value is None else float(self.max_value),
            "mean": None if self.mean_value is None else float(self.mean_value),
            "per_channel_min": None
            if self.per_channel_min is None
            else [float(x) for x in self.per_channel_min],
            "per_channel_max": None
            if self.per_channel_max is None
            else [float(x) for x in self.per_channel_max],
            "per_channel_mean": None
            if self.per_channel_mean is None
            else [float(x) for x in self.per_channel_mean],
        }


def summarize(path: Path) -> SampleStats:
    if not path.exists():
        return SampleStats(path=path, exists=False)

    arr = np.load(path)
    stats = SampleStats(
        path=path,
        exists=True,
        shape=tuple(arr.shape),
        dtype=str(arr.dtype),
        min_value=float(arr.min()),
        max_value=float(arr.max()),
        mean_value=float(arr.mean()),
    )
    if arr.ndim == 3:
        flat = arr.reshape(-1, arr.shape[-1]).astype("float64")
        stats.per_channel_min = flat.min(axis=0)
        stats.per_channel_max = flat.max(axis=0)
        stats.per_channel_mean = flat.mean(axis=0)
    return stats


def pick_sample(images_dir: Path, sample: str | None, seed: int) -> str:
    if sample:
        return sample
    files = sorted(images_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"❌ No .npy files found under {images_dir}")
    random.seed(seed)
    return files[seed % len(files)].stem


def scene_from_patch(patch_name: str) -> str:
    parts = patch_name.split("_")
    if len(parts) <= 1:
        return patch_name
    return "_".join(parts[:-1])


def describe_raw_patch(raw_root: Path, sample_name: str) -> None:
    scene = scene_from_patch(sample_name)
    tif_path = raw_root / scene / f"{sample_name}.tif"
    print(f"\n[raw] {tif_path}")
    if not tif_path.exists():
        print("  ⚠️  Missing raw GeoTIFF")
        return
    with rasterio.open(tif_path) as src:
        data = src.read().astype("float32")
        print(f"  shape: {data.shape} (C, H, W)")
        print(f"  dtype: {data.dtype}")
        print(f"  value range: {data.min():.2f} .. {data.max():.2f}")
        if data.shape[0] >= 6:
            selected = [0, 1, 2, 3, 10, 11]
            selected = [idx for idx in selected if idx < data.shape[0]]
            for idx in selected:
                band = data[idx]
                print(
                    f"    band{idx+1}: min={band.min():.2f}, max={band.max():.2f}, mean={band.mean():.2f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare source vs target .npy stats")
    parser.add_argument("--base-root", type=Path, default=Path("MARIDA_YOLO_6band_split"))
    parser.add_argument("--target-root", type=Path, default=Path("mask_bbox_rescue_split"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample", type=str, help="Patch name without extension")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random sample when --sample not set")
    parser.add_argument("--raw-root", type=Path, default=Path("MARIDA/patches"))
    parser.add_argument("--inspect-raw", action="store_true", help="Also show source GeoTIFF stats")
    args = parser.parse_args()

    base_images = args.base_root / args.split / "images"
    target_images = args.target_root / args.split / "images"

    sample_name = pick_sample(target_images, args.sample, args.seed)
    print(f"Comparing sample '{sample_name}'")

    base_path = base_images / f"{sample_name}.npy"
    target_path = target_images / f"{sample_name}.npy"

    base_stats = summarize(base_path)
    target_stats = summarize(target_path)

    for label, stats in (("base", base_stats), ("target", target_stats)):
        print(f"\n[{label}] {stats.path}")
        if not stats.exists:
            print("  ⚠️  Missing file")
            continue
        print(f"  shape: {stats.shape}")
        print(f"  dtype: {stats.dtype}")
        print(f"  value range: {stats.min_value:.6f} .. {stats.max_value:.6f}")
        print(f"  mean: {stats.mean_value:.6f}")
        if stats.per_channel_mean is not None:
            rows = zip(stats.per_channel_min, stats.per_channel_max, stats.per_channel_mean)
            for idx, (ch_min, ch_max, ch_mean) in enumerate(rows):
                print(
                    f"    ch{idx}: min={ch_min:.6f}, max={ch_max:.6f}, mean={ch_mean:.6f}"
                )

    if args.inspect_raw:
        describe_raw_patch(args.raw_root, sample_name)


if __name__ == "__main__":
    main()
