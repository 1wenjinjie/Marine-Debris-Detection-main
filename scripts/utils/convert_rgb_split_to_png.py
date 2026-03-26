"""Convert float32 RGB GeoTIFF tiles in the split MARIDA dataset to uint8 PNGs.

The current YOLO data split stores 3-channel images as float32 GeoTIFF files
with values normalised by 1/10000. Ultralytics (PIL/OpenCV) cannot load these
so this utility rescales them back to reflectance space and exports RGB PNGs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

# Reasonable upper bound for surface reflectance after reversing the 1/10000 scale.
DEFAULT_REFLECTANCE_CLIP = 1.5


def convert_image(tif_path: Path, clip_value: float) -> None:
    """Convert a single float32 GeoTIFF into an 8-bit PNG alongside the source."""
    with rasterio.open(tif_path) as src:
        data = src.read(out_dtype="float32")  # shape: (C, H, W)

    # Undo the 1/10000 scaling applied during preprocessing.
    data *= 10_000.0
    data = np.nan_to_num(data, nan=0.0, neginf=0.0, posinf=clip_value)
    data = np.clip(data, 0.0, clip_value)
    data /= clip_value  # now in [0, 1]

    # Convert to uint8 RGB.
    data_uint8 = np.clip(data * 255.0, 0.0, 255.0).astype(np.uint8)
    image = np.transpose(data_uint8, (1, 2, 0))  # to (H, W, C)

    stem = tif_path.stem
    if stem.endswith("_rgb"):
        stem = stem[:-4]
    png_path = tif_path.with_name(f"{stem}.png")
    Image.fromarray(image, mode="RGB").save(png_path, format="PNG")


def convert_split(base_dir: Path, clip_value: float, remove_source: bool) -> None:
    """Convert every .tif file under train/val/test images directories."""
    for split in ("train", "val", "test"):
        image_dir = base_dir / split / "images"
        if not image_dir.exists():
            continue

        tif_files = sorted(image_dir.glob("*.tif"))
        if not tif_files:
            continue

        for tif_path in tqdm(tif_files, desc=f"{split} images", unit="img"):
            convert_image(tif_path, clip_value)
            if remove_source:
                tif_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("D:/OceanEdge_Group/Marine-Debris-Detection-main/MARIDA_YOLO_RGB_split"),
        help="Path to the split RGB dataset root (containing train/val/test).",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=DEFAULT_REFLECTANCE_CLIP,
        help="Maximum reflectance value before scaling to 8-bit (default: 1.5).",
    )
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Delete each source .tif after its PNG has been created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_split(args.dataset_dir, args.clip, args.remove_source)


if __name__ == "__main__":
    main()
