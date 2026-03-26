"""
将 MARIDA 多通道 GeoTIFF 影像转换为与 Ultralytics 兼容的 .npy 缓存。
Ultralytics 的 YOLODataset 会在存在同名 .npy 文件时优先读取，从而绕过
OpenCV 对多通道 float32 GeoTIFF 的限制。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

DEFAULT_SPLITS = ["train", "val", "test"]


def convert_image(tif_path: Path, force: bool = False) -> bool:
    """将单个 .tif 转换为 HWC 的 float32 .npy 文件。"""
    npy_path = tif_path.with_suffix(".npy")
    if npy_path.exists() and not force:
        return False

    with rasterio.open(tif_path) as src:
        data = src.read().astype("float32")  # (C, H, W)
    arr = np.transpose(data, (1, 2, 0))  # -> (H, W, C)
    np.save(npy_path, arr)
    return True


def convert_split(images_dir: Path, force: bool = False) -> tuple[int, int]:
    """为某个 split/images 目录批量生成 .npy 文件。"""
    tifs = sorted(images_dir.glob("*.tif"))
    created, skipped = 0, 0

    for tif in tqdm(tifs, desc=f"Processing {images_dir.name}"):
        try:
            if convert_image(tif, force=force):
                created += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"⚠️ 转换 {tif} 失败: {exc}")
    return created, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="为 Ultralytics 生成 .npy 缓存")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("MARIDA_YOLO_6band_split"),
        help="包含 train/val/test 子目录的根路径",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=DEFAULT_SPLITS,
        help="需要处理的子集 (默认: train val test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若已存在 .npy 仍强制重建",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise SystemExit(f"❌ 数据集目录不存在: {dataset_root}")

    total_created = total_skipped = 0
    for split in args.splits:
        images_dir = dataset_root / split / "images"
        if not images_dir.exists():
            print(f"⚠️ 跳过 {split}: 未找到 {images_dir}")
            continue
        created, skipped = convert_split(images_dir, force=args.force)
        total_created += created
        total_skipped += skipped
        print(f"完成 {split}: 新建 {created} 个 .npy，跳过 {skipped} 个已存在文件")

    print("\n================ 总结 ================")
    print(f"新建 .npy: {total_created}")
    print(f"已存在跳过: {total_skipped}")
    print("====================================")


if __name__ == "__main__":
    main()
