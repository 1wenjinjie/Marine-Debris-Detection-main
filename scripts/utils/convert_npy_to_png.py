"""Convert `.npy` image arrays to 8-bit PNGs using the same basic preprocessing used in training.

This script mirrors the `_load_array` scaling used by `MaskRescueNPYDataset` so the saved PNGs are visually consistent
with the images the model saw during training.

Usage (PowerShell):
    python convert_npy_to_png.py --src mask_bbox_rescue_split/test/images --dst marida_runs/yolo11n_rgb_full/visualize_test_png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def normalize_and_select_rgb(arr: np.ndarray, channels: int = 3) -> np.ndarray:
    # If array is stored as (C, H, W) with small C, transpose to (H, W, C)
    if arr.ndim == 3 and arr.shape[0] in (6, 7) and arr.shape[0] <= arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)

    # If there are many bands (e.g., 11), pick sensible RGB indices like training code does
    if arr.ndim == 3 and arr.shape[-1] >= 11:
        # Training used indices [3,2,1] for RGB (B4, B3, B2)
        if channels == 3:
            arr = arr[..., [3, 2, 1]]
        elif channels == 6:
            # For 6-ch preview, select [3,2,1,7,9,10] but we only save RGB so take first 3
            arr = arr[..., [3, 2, 1, 7, 9, 10]][..., :3]
    elif arr.ndim == 3 and arr.shape[-1] > channels:
        arr = arr[..., :channels]

    # Ensure we have at least 3 channels for RGB saving
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 2:
        arr = np.concatenate([arr, arr[..., :1]], axis=-1)

    # The dataset loader multiplies by 1500 and clips to [0,255]. We reproduce that mapping here.
    arr = arr.astype(np.float32)
    arr = arr * 1500.0
    arr = np.clip(arr, 0.0, 255.0)

    # Convert to uint8 0-255
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    return arr_uint8[..., :3]


def convert_folder(src: Path, dst: Path, channels: int = 3) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    npy_files = sorted([p for p in src.glob("*.npy")])
    for p in tqdm(npy_files, desc="converting .npy -> png"):
        try:
            arr = np.load(p, allow_pickle=False)
            img = normalize_and_select_rgb(arr, channels=channels)
            img_pil = Image.fromarray(img, mode="RGB")
            out_name = p.stem
            out_path = dst / (out_name + ".png")
            img_pil.save(out_path)
        except Exception as e:
            print(f"Skipping {p}: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Source folder containing .npy files")
    p.add_argument("--dst", type=Path, required=True, help="Destination folder for PNGs")
    p.add_argument("--channels", type=int, default=3, help="Number of channels to expect (3 or 6)")
    return p.parse_args()


def main():
    args = parse_args()
    convert_folder(args.src, args.dst, channels=args.channels)


if __name__ == '__main__':
    main()
