import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import rasterio


def load_six_channel_image(path: Path) -> np.ndarray:
    """读取六通道 GeoTIFF，返回 HWC 排列的 float32 numpy 数组。"""
    with rasterio.open(path) as src:
        data = src.read(out_dtype="float32")  # C,H,W
    return np.transpose(data, (1, 2, 0))


def channel_to_uint8(channel: np.ndarray) -> np.ndarray:
    """将单通道数据映射到 0-255，使用分位数拉伸增强对比度。"""
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros(channel.shape, dtype=np.uint8)

    lo, hi = np.percentile(finite, (2, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            return np.zeros(channel.shape, dtype=np.uint8)

    scaled = (channel - lo) / (hi - lo + 1e-6)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def channels_to_grid(image: np.ndarray) -> np.ndarray:
    """将 6 通道图像放入 3x2 的网格，生成单张可视化 PNG。"""
    h, w, c = image.shape
    if c != 6:
        raise ValueError(f"期望 6 通道，但得到 {c}")

    grid = np.zeros((h * 2, w * 3), dtype=np.uint8)
    for idx in range(6):
        row, col = divmod(idx, 3)
        ch = channel_to_uint8(image[:, :, idx])
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = ch
    return grid


def convert_split(split_dir: Path, output_dir: Path) -> Tuple[int, Path]:
    images_dir = split_dir / "images"
    output_images_dir = output_dir / split_dir.name / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for tif_path in tqdm(sorted(images_dir.glob("*.tif")), desc=f"转换 {split_dir.name}"):
        image = load_six_channel_image(tif_path)
        grid = channels_to_grid(image)
        out_path = output_images_dir / f"{tif_path.stem}.png"
        Image.fromarray(grid).save(out_path)
        count += 1
    return count, output_images_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="将六通道 GeoTIFF 转换为 PNG 网格供人工审查")
    parser.add_argument(
        "--input",
        default="MARIDA_YOLO_6band_split",
        type=Path,
        help="输入数据根目录，包含 train/val/test 等子目录",
    )
    parser.add_argument(
        "--output",
        default="MARIDA_YOLO_6band_png",
        type=Path,
        help="输出 PNG 根目录",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"输入目录不存在: {args.input}")

    total_counts: Dict[str, int] = {}
    for split_dir in sorted(args.input.iterdir()):
        if not split_dir.is_dir():
            continue
        if not (split_dir / "images").exists():
            continue
        count, out_dir = convert_split(split_dir, args.output)
        total_counts[split_dir.name] = count
        print(f"{split_dir.name} 转换完成，共 {count} 张，输出目录: {out_dir}")

    grand_total = sum(total_counts.values())
    print("=== 转换统计 ===")
    for split, count in total_counts.items():
        print(f"{split}: {count} 张")
    print(f"总计: {grand_total} 张")


if __name__ == "__main__":
    main()
