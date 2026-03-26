"""Sample channel statistics for multispectral .npy caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-channel stats from .npy tiles")
    parser.add_argument("root", type=Path, help="Dataset root that contains split folders (train/val/test)")
    parser.add_argument("--channels", type=int, default=7, help="Number of channels to retain when sampling")
    parser.add_argument("--sample-per-file", type=int, default=4096, help="Pixels per file to sample")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated split names to scan (default: train,val)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marida_runs/channel_stats_7ch.json"),
        help="Where to write the JSON stats",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".npy",
        help="Image file extension to search for",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="images",
        help="Subfolder to glob inside each split (default: images)",
    )
    return parser.parse_args()


def collect_files(root: Path, splits: Iterable[str], subdir: str, suffix: str) -> list[Path]:
    files: list[Path] = []
    for split in splits:
        base = root / split / subdir
        if not base.exists():
            continue
        files.extend(sorted(base.glob(f"*{suffix}")))
    return files


def main() -> None:
    args = parse_args()
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    files = collect_files(args.root, splits, args.glob_pattern, args.extension)
    if not files:
        raise SystemExit(f"No files ending with {args.extension} found under {args.root}")

    rng = np.random.default_rng(42)
    sample_accum: list[np.ndarray] = []

    for path in tqdm(files, desc="Sampling pixels"):
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 3:
            continue
        if arr.shape[0] == args.channels and arr.shape[-1] != args.channels:
            arr = arr.transpose(1, 2, 0)
        if arr.shape[-1] < args.channels:
            continue
        data = arr[..., : args.channels].reshape(-1, args.channels)
        n = data.shape[0]
        k = min(args.sample_per_file, n)
        idx = rng.choice(n, size=k, replace=False)
        sample_accum.append(data[idx])

    if not sample_accum:
        raise SystemExit("No channel samples collected; check dataset layout")

    samples = np.concatenate(sample_accum, axis=0)
    percentiles = np.percentile(samples, [1, 25, 50, 75, 99], axis=0)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)

    stats = {
        "channels": args.channels,
        "sampled_pixels": int(samples.shape[0]),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "percentiles": {
            "p01": percentiles[0].tolist(),
            "p25": percentiles[1].tolist(),
            "p50": percentiles[2].tolist(),
            "p75": percentiles[3].tolist(),
            "p99": percentiles[4].tolist(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2))
    print(f"Saved stats to {args.output} (pixels={stats['sampled_pixels']})")


if __name__ == "__main__":
    main()
