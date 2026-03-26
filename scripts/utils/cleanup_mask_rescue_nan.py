"""Utility to sanitize NaN/Inf values within mask-bbox rescue .npy caches."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace NaN/Inf pixels in knowledge-augmented tensors")
    parser.add_argument(
        "--root",
        default="mask_bbox_rescue_split",
        help="Dataset root containing train/val/test manifests",
    )
    parser.add_argument(
        "--manifests",
        nargs="*",
        default=["train.txt", "val.txt", "test.txt"],
        help="Manifest files (relative to root) listing .npy samples",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan only; report counts without modifying files",
    )
    return parser.parse_args()


def gather_files(root: Path, manifest_names: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for name in manifest_names:
        manifest = root / name
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest {manifest} was not found")
        for line in manifest.read_text().splitlines():
            rel = line.strip()
            if not rel or rel.startswith("#"):
                continue
            files.append((root / rel).resolve())
    if not files:
        raise RuntimeError("No .npy samples discovered from the provided manifests")
    return files


def sanitize_file(path: Path, dry_run: bool) -> tuple[int, int]:
    arr = np.load(path, allow_pickle=False)
    mask = ~np.isfinite(arr)
    if not mask.any():
        return 0, arr.size

    invalid = int(mask.sum())
    if not dry_run:
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(arr, 0.0, 1.0, out=arr)
        np.save(path, arr.astype(np.float32))
    return invalid, arr.size


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root {root} does not exist")

    files = gather_files(root, args.manifests)
    total_files = len(files)
    cleaned_files = 0
    replaced_values = 0

    for path in files:
        invalid, _ = sanitize_file(path, args.dry_run)
        if invalid:
            cleaned_files += 1
            replaced_values += invalid
            print(f"CLEAN {'(dry-run)' if args.dry_run else ''}: {path} -> replaced {invalid} values")

    print("--- summary ---")
    print(f"files_scanned={total_files}")
    print(f"files_cleaned={cleaned_files}")
    print(f"values_replaced={replaced_values}")
    if args.dry_run:
        print("(dry-run) no files were modified")


if __name__ == "__main__":
    main()
