import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 3 类配色（可按需调整）
CLASS_COLORS = {
    0: (255, 0, 0),      # Marine_Debris → 红色
    1: (0, 255, 0),      # Floating_Object → 绿色
    2: (0, 128, 255),    # Ship_Wake → 蓝色
}

CLASS_NAMES = {
    0: "Marine_Debris",
    1: "Floating_Object",
    2: "Ship_Wake",
}

PANEL_GRID = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]  # 3×2 网格


def load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    text = label_path.read_text().strip()
    if not text:
        return []
    labels = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        labels.append((int(float(cls)), float(xc), float(yc), float(w), float(h)))
    return labels


def draw_boxes_on_grid(image: Image.Image, labels: List[Tuple[int, float, float, float, float]],
                       panel_size: int = 256) -> Image.Image:
    if not labels:
        return image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for cls, xc, yc, bw, bh in labels:
        x_center = xc * panel_size
        y_center = yc * panel_size
        box_width = bw * panel_size
        box_height = bh * panel_size
        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2
        color = CLASS_COLORS.get(cls, (255, 255, 0))
        name = CLASS_NAMES.get(cls, str(cls))
        for col_index, row_index in PANEL_GRID:
            offset_x = col_index * panel_size
            offset_y = row_index * panel_size
            draw.rectangle([(x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y)], outline=color, width=2)
            label_pos = (x1 + offset_x + 2, y1 + offset_y + 2)
            draw.text(label_pos, name, fill=color, font=font)
    return image


def annotate_split(split: str, png_root: Path, label_root: Path, output_root: Path) -> Dict[str, int]:
    png_dir = png_root / split / "images"
    label_dir = label_root / split / "labels"
    output_dir = output_root / split / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    stats["total_images"] = 0
    stats["images_with_boxes"] = 0
    stats["total_boxes"] = 0

    for png_path in sorted(png_dir.glob("*.png")):
        stats["total_images"] += 1
        name = png_path.stem
        label_path = label_dir / f"{name}.txt"
        labels = load_yolo_labels(label_path)
        if labels:
            stats["images_with_boxes"] += 1
            stats["total_boxes"] += len(labels)
            for cls, *_ in labels:
                stats[f"class_{cls}"] += 1
        img = Image.open(png_path).convert("RGB")
        img = draw_boxes_on_grid(img, labels)
        img.save(output_dir / png_path.name)
    return stats


def summarize_stats(split: str, stats: Dict[str, int]) -> None:
    total = stats.get("total_images", 0)
    with_boxes = stats.get("images_with_boxes", 0)
    ratio = (with_boxes / total * 100) if total else 0.0
    print(f"[{split}] 图像总数: {total}")
    print(f"[{split}] 含目标图像: {with_boxes} ({ratio:.2f}%)")
    print(f"[{split}] 目标框数量: {stats.get('total_boxes', 0)}")
    for cls_id, name in CLASS_NAMES.items():
        count = stats.get(f"class_{cls_id}", 0)
        print(f"    类别 {cls_id} ({name}): {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="在六通道 PNG 上叠加 YOLO 标注框并统计占比")
    parser.add_argument("--png-root", type=Path, default=Path("MARIDA_YOLO_6band_png"), help="输入的 6 通道 PNG 根目录")
    parser.add_argument("--dataset-root", type=Path, default=Path("MARIDA_YOLO_6band_split"), help="包含 labels 的 YOLO 数据根目录")
    parser.add_argument("--output-root", type=Path, default=Path("MARIDA_YOLO_6band_png_annotated"), help="输出标注后的 PNG 目录")
    args = parser.parse_args()

    if not args.png_root.exists():
        raise FileNotFoundError(f"未找到 PNG 根目录: {args.png_root}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"未找到 YOLO 数据目录: {args.dataset_root}")

    splits = [d.name for d in args.png_root.iterdir() if d.is_dir()]
    splits = sorted(s for s in splits if (args.dataset_root / s / "labels").exists())

    overall_boxes = 0
    overall_images = 0
    overall_with_boxes = 0

    print("开始叠加标注框并统计……")
    for split in splits:
        stats = annotate_split(split, args.png_root, args.dataset_root, args.output_root)
        summarize_stats(split, stats)
        overall_images += stats.get("total_images", 0)
        overall_with_boxes += stats.get("images_with_boxes", 0)
        overall_boxes += stats.get("total_boxes", 0)

    overall_ratio = (overall_with_boxes / overall_images * 100) if overall_images else 0.0
    print("=== 总体统计 ===")
    print(f"图像总数: {overall_images}")
    print(f"含目标图像: {overall_with_boxes} ({overall_ratio:.2f}%)")
    print(f"目标框总数: {overall_boxes}")


if __name__ == "__main__":
    main()
