"""Create a quick contact sheet comparing RGB, knowledge heatmap, and mask overlays."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual sanity check for knowledge channel")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("mask_bbox_rescue_split"),
        help="Root folder that contains split manifests and data",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path("MARIDA/patches"),
        help="Directory containing original MARIDA masks",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        help="Relative path inside dataset root to the .npy sample (e.g. train/images/foo.npy)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("mask_bbox_rescue_split/train.txt"),
        help="Fallback manifest to pick the first sample when --sample is omitted",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit output path",
    )
    return parser.parse_args()


def derive_scene_name(patch_name: str) -> str:
    parts = patch_name.split("_")
    if len(parts) <= 1:
        return patch_name
    return "_".join(parts[:-1])


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    arr -= arr.min(axis=(0, 1), keepdims=True)
    maxv = arr.max(axis=(0, 1), keepdims=True)
    arr = np.divide(arr, maxv + 1e-6)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def normalize_scalar(channel: np.ndarray) -> np.ndarray:
    arr = channel.astype(np.float32)
    arr -= arr.min()
    maxv = arr.max()
    if maxv > 1e-6:
        arr /= maxv
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def load_mask(mask_root: Path, patch_stem: str) -> np.ndarray | None:
    scene = derive_scene_name(patch_stem)
    mask_path = mask_root / scene / f"{patch_stem}_cl.tif"
    if not mask_path.exists():
        return None
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return mask


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if args.sample is None:
        manifest = args.manifest.resolve()
        lines = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
        if not lines:
            raise SystemExit(f"Manifest is empty: {manifest}")
        rel_sample = Path(lines[0])
    else:
        rel_sample = args.sample

    sample_path = (dataset_root / rel_sample).resolve()
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_path}")

    arr = np.load(sample_path)
    if arr.ndim != 3 or arr.shape[2] < 4:
        raise ValueError(f"Unexpected array shape: {arr.shape}")

    rgb = arr[:, :, :3]
    heatmap = arr[:, :, -1]
    rgb_uint8 = normalize_rgb(rgb)
    heat_uint8 = normalize_scalar(heatmap)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(rgb_uint8, 0.7, heat_color, 0.3, 0)

    mask = load_mask(args.mask_root.resolve(), Path(sample_path.stem).stem)
    if mask is not None:
        mask_uint8 = normalize_scalar(mask)
        mask_edges = cv2.Canny(mask_uint8, 50, 150)
        mask_overlay = rgb_uint8.copy()
        mask_overlay[mask_edges > 0] = (0, 255, 0)
    else:
        mask_overlay = rgb_uint8.copy()

    contact_sheet = np.concatenate([rgb_uint8, heat_color, overlay, mask_overlay], axis=1)

    output_dir = Path("visualizations_check")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = args.output
    else:
        out_path = output_dir / f"knowledge_overlay_{sample_path.stem}.png"
    cv2.imwrite(str(out_path), contact_sheet[:, :, ::-1])  # convert RGB->BGR for saving

    stats = {
        "sample": sample_path.as_posix(),
        "shape": tuple(int(dim) for dim in arr.shape),
        "rgb_range": (
            float(rgb.min()),
            float(rgb.max()),
        ),
        "heatmap_range": (
            float(heatmap.min()),
            float(heatmap.max()),
        ),
        "mask_found": mask is not None,
        "output": out_path.as_posix(),
    }
    print(stats)


if __name__ == "__main__":
    main()
