from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

DATASET_ROOT = Path("MARIDA")
PATCHES_DIR = DATASET_ROOT / "patches"
OUTPUT_DIR = Path("mask_bbox_preview")

# Runtime overrides (populated via CLI)
LIMIT_SAMPLES: int | None = 50
COVERAGE_THRESHOLDS: dict[int, tuple[float, float]] = {}

# Original MARIDA class -> YOLO detection class
TARGET_CLASSES = [1, 2, 3, 4, 5, 9, 14]
CLASS_INDEX = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
CLASS_LABELS = {
    1: "Marine Debris",
    2: "Dense Sargassum",
    3: "Sparse Sargassum",
    4: "Natural Organic",
    5: "Ship",
    9: "Foam",
    14: "Wakes",
}
YOLO_CLASS_NAMES = [CLASS_LABELS[c] for c in TARGET_CLASSES]

# Minimum connected component area (pixels) per original class
MIN_AREA = {
    1: 3,
    2: 8,
    3: 8,
    4: 6,
    5: 18,
    9: 3,
    14: 64,
}

# Light padding (pixels) around each bounding box per original class
PADDING = {
    1: 2,
    2: 2,
    3: 2,
    4: 2,
    5: 2,
    9: 2,
    14: 3,
}

# Morphological operation per class: (operation, iterations)
# Supported operations: "close", "dilate", "none"
CLASS_MORPHOLOGY: dict[int, tuple[str, int]] = {
    1: ("dilate", 1),
    2: ("close", 1),
    3: ("close", 1),
    4: ("close", 1),
    5: ("close", 1),
    9: ("dilate", 1),
    14: ("close", 2),
}

# Colour map for mask visualisation
MASK_COLORS = {
    0: (0, 0, 80),
    1: (255, 0, 0),
    2: (0, 150, 0),
    3: (144, 238, 144),
    4: (255, 215, 0),
    5: (255, 255, 0),
    6: (255, 255, 255),
    7: (0, 0, 139),
    8: (70, 130, 180),
    9: (255, 0, 255),
    10: (72, 209, 204),
    11: (30, 144, 255),
    12: (100, 149, 237),
    13: (105, 105, 105),
    14: (255, 140, 0),
    15: (139, 69, 19),
}

MASK_ORDER = list(range(16))
MASK_CMAP = ListedColormap(
    np.array([MASK_COLORS.get(i, (0, 0, 0)) for i in MASK_ORDER], dtype=np.float32) / 255.0
)
LEGEND_PATCHES = [
    Patch(facecolor=MASK_CMAP(i / (len(MASK_ORDER) - 1)), edgecolor="k", label=f"{i}: {name}")
    for i, name in enumerate([
        "Background/Sea",
        "Marine Debris",
        "Dense Sargassum",
        "Sparse Sargassum",
        "Natural Organic",
        "Ship",
        "Clouds",
        "Marine Water",
        "Sediment Water",
        "Foam",
        "Turbid Water",
        "Shallow Water",
        "Waves",
        "Cloud Shadows",
        "Wakes",
        "Land Mask",
    ])
]

DEFAULT_COVERAGE_THRESHOLDS = {
    1: (0.12, 0.30),  # Marine Debris
    2: (0.15, 0.32),  # Dense Sargassum
    3: (0.15, 0.32),  # Sparse Sargassum
    4: (0.18, 0.35),  # Natural Organic
    5: (0.22, 0.40),  # Ship
    9: (0.12, 0.28),  # Foam
    14: (0.05, 0.15),  # Wakes
}


def ensure_dirs() -> None:
    (OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels_ignore").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "previews").mkdir(parents=True, exist_ok=True)


def stretch_rgb(image_path: Path) -> np.ndarray:
    with rasterio.open(image_path) as src:
        data = src.read([4, 3, 2]).astype(np.float32)
    rgb = np.transpose(data, (1, 2, 0))
    for c in range(3):
        band = np.nan_to_num(rgb[:, :, c], nan=0.0)
        p2, p98 = np.percentile(band, (2, 98))
        if p98 <= p2:
            rgb[:, :, c] = np.clip(band, 0, 1)
        else:
            rgb[:, :, c] = np.clip((band - p2) / (p98 - p2), 0, 1)
    return rgb


def extract_components(binary: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return num_labels, labels, stats


def iter_patches() -> Iterable[tuple[str, str]]:
    for scene_dir in sorted(PATCHES_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        for mask_path in sorted(scene_dir.glob("*_cl.tif")):
            base = mask_path.stem[:-3]
            yield scene_dir.name, base


def clamp_bbox(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def component_to_bbox(
    stats_row: np.ndarray,
    padding: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h, _ = stats_row
    x0 = x - padding
    y0 = y - padding
    x1 = x + w + padding - 1
    y1 = y + h + padding - 1
    return clamp_bbox(x0, y0, x1, y1, width, height)


def yolo_from_bbox(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    x_center = (x0 + box_w / 2.0) / width
    y_center = (y0 + box_h / 2.0) / height
    return x_center, y_center, box_w / width, box_h / height


def draw_boxes(rgb: np.ndarray, boxes: list[dict]) -> np.ndarray:
    canvas = (rgb * 255.0).astype(np.uint8)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for box in boxes:
        x0, y0, x1, y1 = box["pixel_bbox"]
        orig_cls = box["original_class"]
        color_rgb = MASK_COLORS.get(orig_cls, (255, 255, 255))
        color = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(canvas_bgr, (x0, y0), (x1, y1), color, 2)
        label = CLASS_LABELS.get(orig_cls, f"Class {orig_cls}")
        cv2.putText(
            canvas_bgr,
            label,
            (x0, max(0, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = mask.astype(np.float32) / max(len(MASK_ORDER) - 1, 1)
    color_mask = MASK_CMAP(normalized)[:, :, :3]
    blended = 0.6 * rgb + 0.4 * color_mask
    return np.clip(blended, 0.0, 1.0)


def get_coverage_thresholds(cls: int) -> tuple[float, float]:
    low, high = COVERAGE_THRESHOLDS.get(cls, DEFAULT_COVERAGE_THRESHOLDS.get(cls, (0.1, 0.3)))
    if high < low:
        high = low
    return float(low), float(high)


def process_patch(scene: str, patch: str, stats) -> tuple[list[dict], list[dict]]:
    image_path = PATCHES_DIR / scene / f"{patch}.tif"
    mask_path = PATCHES_DIR / scene / f"{patch}_cl.tif"

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        height, width = mask.shape

    boxes: list[dict] = []
    ignored_boxes: list[dict] = []
    coverage_counts: list[float] = []

    for original_class in TARGET_CLASSES:
        yolo_class = CLASS_INDEX[original_class]
        binary = (mask == original_class).astype(np.uint8)
        if binary.sum() == 0:
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        op, iters = CLASS_MORPHOLOGY.get(original_class, ("none", 0))
        if op == "close":
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=max(1, iters))
        elif op == "dilate":
            binary = cv2.dilate(binary, kernel, iterations=max(1, iters))
        elif op == "erode":
            binary = cv2.erode(binary, kernel, iterations=max(1, iters))

        num_labels, labels, stats_arr = extract_components(binary)
        if num_labels <= 1:
            continue

        min_area = MIN_AREA.get(original_class, 9)
        pad = PADDING.get(original_class, 2)
        cov_low, cov_high = get_coverage_thresholds(original_class)

        for idx in range(1, num_labels):
            component_area = stats_arr[idx, cv2.CC_STAT_AREA]
            if component_area < min_area:
                stats["filtered_small"] += 1
                continue
            x0, y0, x1, y1 = component_to_bbox(stats_arr[idx], pad, width, height)
            if x1 <= x0 or y1 <= y0:
                stats["invalid_bbox"] += 1
                continue

            submask = mask[y0 : y1 + 1, x0 : x1 + 1]
            positive = np.count_nonzero(submask == original_class)
            total = submask.size
            coverage = positive / float(total) if total else 0.0
            coverage_counts.append(coverage)

            x_center, y_center, box_w, box_h = yolo_from_bbox(x0, y0, x1, y1, width, height)
            if not (0.0 < box_w <= 1.0 and 0.0 < box_h <= 1.0):
                stats["invalid_bbox"] += 1
                continue

            entry = {
                "class_id": yolo_class,
                "original_class": original_class,
                "x_center": x_center,
                "y_center": y_center,
                "width": box_w,
                "height": box_h,
                "pixel_bbox": (int(x0), int(y0), int(x1), int(y1)),
                "coverage": coverage,
            }

            if coverage < cov_low:
                stats["filtered_coverage_low"] += 1
                continue
            if coverage < cov_high:
                ignored_boxes.append(entry)
                stats["ignored_boxes"] += 1
                stats["ignore_class_counts"][original_class] += 1
                continue

            boxes.append(entry)
            stats["class_counts"][yolo_class] += 1
            stats["original_class_counts"][original_class] += 1
            stats["coverage_values"][original_class].append(coverage)

    stats["boxes"] += len(boxes)
    stats["per_patch"].append(len(boxes))
    stats["coverage_patch"].extend(coverage_counts)
    return boxes, ignored_boxes


def save_labels(patch: str, boxes: list[dict], *, ignore: bool = False) -> None:
    label_path = OUTPUT_DIR / "labels" / f"{patch}.txt"
    if ignore:
        label_path = OUTPUT_DIR / "labels_ignore" / f"{patch}.txt"
    with label_path.open("w", encoding="utf-8") as fh:
        for box in boxes:
            fh.write(
                f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} "
                f"{box['width']:.6f} {box['height']:.6f}\n"
            )


def save_preview(scene: str, patch: str, boxes: list[dict]) -> None:
    image_path = PATCHES_DIR / scene / f"{patch}.tif"
    mask_path = PATCHES_DIR / scene / f"{patch}_cl.tif"

    rgb = stretch_rgb(image_path)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    mask_overlay = overlay_mask(rgb, mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(patch, fontsize=14, fontweight="bold")

    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap=MASK_CMAP, vmin=0, vmax=len(MASK_ORDER) - 1)
    axes[1].set_title("Mask", fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(mask_overlay)
    axes[2].set_title(f"YOLO Boxes ({len(boxes)})", fontweight="bold")
    axes[2].axis("off")

    for box in boxes:
        x0, y0, x1, y1 = box["pixel_bbox"]
        orig_cls = box["original_class"]
        color = np.array(MASK_COLORS.get(orig_cls, (255, 255, 255)), dtype=np.float32) / 255.0
        rect = Rectangle(
            (x0, y0),
            max(1, x1 - x0 + 1),
            max(1, y1 - y0 + 1),
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        axes[2].add_patch(rect)
        label = f"{CLASS_LABELS.get(orig_cls, orig_cls)} | cov {box['coverage']:.2f}"
        axes[2].text(
            x0,
            max(0, y0 - 6),
            label,
            color=color,
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55, edgecolor=color),
        )

    fig.legend(handles=LEGEND_PATCHES, loc="center right", bbox_to_anchor=(1.15, 0.5), frameon=False, fontsize=7)
    fig.tight_layout()
    save_path = OUTPUT_DIR / "previews" / f"{patch}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_summary(stats) -> None:
    summary = {
        "total_patches_considered": stats["patch_total"],
        "patches_with_boxes": stats["patch_with_boxes"],
        "patches_with_ignored_only": stats["patch_with_ignored_only"],
        "total_boxes": stats["boxes"],
        "ignored_boxes": stats["ignored_boxes"],
        "filtered_small_components": stats["filtered_small"],
        "filtered_low_coverage": stats["filtered_coverage_low"],
        "per_class_counts": {
            YOLO_CLASS_NAMES[idx]: stats["class_counts"].get(idx, 0)
            for idx in range(len(TARGET_CLASSES))
        },
        "per_original_class_counts": {
            str(k): v for k, v in stats["original_class_counts"].items()
        },
        "ignored_class_counts": {
            str(k): v for k, v in stats["ignore_class_counts"].items()
        },
        "coverage_mean": float(np.mean(stats["coverage_patch"])) if stats["coverage_patch"] else 0.0,
        "coverage_median": float(np.median(stats["coverage_patch"])) if stats["coverage_patch"] else 0.0,
        "coverage_by_class": {
            CLASS_LABELS.get(cls, f"Class {cls}"): {
                "mean": float(np.mean(stats["coverage_values"][cls])) if stats["coverage_values"][cls] else 0.0,
                "median": float(np.median(stats["coverage_values"][cls])) if stats["coverage_values"][cls] else 0.0,
            }
            for cls in TARGET_CLASSES
        },
        "coverage_thresholds": {
            str(cls): {
                "low": get_coverage_thresholds(cls)[0],
                "high": get_coverage_thresholds(cls)[1],
            }
            for cls in TARGET_CLASSES
        },
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    classes_txt = OUTPUT_DIR / "classes.txt"
    with classes_txt.open("w", encoding="utf-8") as fh:
        for name in YOLO_CLASS_NAMES:
            fh.write(f"{name}\n")

    mapping = {
        "target_classes": TARGET_CLASSES,
        "class_names": YOLO_CLASS_NAMES,
        "original_to_yolo": {str(cls): CLASS_INDEX[cls] for cls in TARGET_CLASSES},
    }
    with (OUTPUT_DIR / "class_mapping.json").open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO bboxes from semantic masks with preview panels.")
    parser.add_argument("--patches-dir", type=Path, default=PATCHES_DIR, help="Directory containing scene patch folders")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory for labels/previews")
    parser.add_argument("--max-samples", type=int, default=50, help="Limit number of patches exported (<=0 for all)")
    parser.add_argument(
        "--coverage-config",
        type=Path,
        help="Optional JSON mapping of original class id to {low, high} coverage thresholds",
    )
    return parser.parse_args()


def load_coverage_thresholds(config_path: Path | None) -> dict[int, tuple[float, float]]:
    if config_path is None:
        return DEFAULT_COVERAGE_THRESHOLDS.copy()
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    thresholds: dict[int, tuple[float, float]] = {}
    for key, value in data.items():
        try:
            cls = int(key)
            low = float(value["low"] if isinstance(value, dict) else value[0])
            high = float(value["high"] if isinstance(value, dict) else value[1])
            thresholds[cls] = (low, high)
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return thresholds


def main() -> None:
    global PATCHES_DIR, OUTPUT_DIR, LIMIT_SAMPLES, COVERAGE_THRESHOLDS

    args = parse_args()
    PATCHES_DIR = args.patches_dir
    OUTPUT_DIR = args.output_dir
    LIMIT_SAMPLES = None if args.max_samples <= 0 else args.max_samples
    COVERAGE_THRESHOLDS = load_coverage_thresholds(args.coverage_config)

    ensure_dirs()
    stats = {
        "patch_total": 0,
        "patch_with_boxes": 0,
        "patch_with_ignored_only": 0,
        "boxes": 0,
        "class_counts": Counter(),
        "original_class_counts": Counter(),
        "ignore_class_counts": Counter(),
        "coverage_values": defaultdict(list),
        "coverage_patch": [],
        "per_patch": [],
        "filtered_small": 0,
        "filtered_coverage_low": 0,
        "invalid_bbox": 0,
        "ignored_boxes": 0,
    }

    exported = 0
    for scene, patch in iter_patches():
        stats["patch_total"] += 1
        boxes, ignored = process_patch(scene, patch, stats)
        if not boxes and not ignored:
            continue
        if boxes:
            stats["patch_with_boxes"] += 1
            save_labels(patch, boxes)
        elif ignored:
            stats["patch_with_ignored_only"] += 1
        if ignored:
            save_labels(patch, ignored, ignore=True)
        save_preview(scene, patch, boxes + ignored)
        exported += 1
        if LIMIT_SAMPLES is not None and exported >= LIMIT_SAMPLES:
            break

    export_summary(stats)
    print(f"Exported {exported} preview images to {OUTPUT_DIR / 'previews'}")
    print(f"Total boxes: {stats['boxes']}")
    print(f"Filtered small components: {stats['filtered_small']}")
    print(f"Invalid boxes: {stats['invalid_bbox']}")


if __name__ == "__main__":
    main()
