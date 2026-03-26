import os
import cv2
import numpy as np
import rasterio

# 配置
DATASET_ROOT = 'MARIDA'
PATCHES_DIR = os.path.join(DATASET_ROOT, 'patches')
SHAPEFILES_DIR = os.path.join(DATASET_ROOT, 'shapefiles')
import os
import struct
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import array_bounds

DATASET_ROOT = Path('MARIDA')
PATCHES_DIR = DATASET_ROOT / 'patches'
SHAPEFILES_DIR = DATASET_ROOT / 'shapefiles'
SPLITS_DIR = DATASET_ROOT / 'splits'
OUTPUT_DIR = Path('shapefile_yolo_export')

YOLO_CLASS_NAMES = ['Marine_Debris', 'Floating_Object', 'Ship_Wake']
ORIGINAL_TO_YOLO_MAP = {
    1: 0,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    9: 1,
    14: 2,
}

YOLO_COLORS = {
    0: (0, 0, 255),   # Red (BGR)
    1: (0, 255, 0),   # Green
    2: (255, 0, 0),   # Blue
}

MAX_VISUALIZATIONS = 120


def ensure_dirs() -> None:
    (OUTPUT_DIR / 'images').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'labels').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'visualizations').mkdir(parents=True, exist_ok=True)


def read_dbf(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open('rb') as fh:
        header = fh.read(32)
        num_records = int.from_bytes(header[4:8], 'little')
        record_length = int.from_bytes(header[10:12], 'little')

        fields: list[tuple[str, str, int, int]] = []
        while True:
            field_desc = fh.read(32)
            if not field_desc or field_desc[0] == 0x0D:
                break
            name = field_desc[:11].split(b'\x00', 1)[0].decode('latin-1', errors='ignore').strip()
            field_type = chr(field_desc[11])
            size = field_desc[16]
            decimal = field_desc[17]
            fields.append((name, field_type, size, decimal))

        fh.read(1)  # header terminator

        for _ in range(num_records):
            data = fh.read(record_length)
            if not data or data[0] == 0x1A:
                break
            if data[0:1] == b'*':
                continue
            offset = 1
            record: dict[str, int | float | str | None] = {}
            for name, field_type, size, decimal in fields:
                raw = data[offset:offset + size]
                offset += size
                text = raw.decode('latin-1', errors='ignore').replace('\x1a', '').strip()
                if field_type == 'N':
                    if text:
                        record[name] = float(text) if ('.' in text or decimal > 0) else int(text)
                    else:
                        record[name] = None
                else:
                    record[name] = text
            records.append(record)
    return records


def read_polygons(path: Path) -> list[dict]:
    polygons: list[dict] = []
    with path.open('rb') as fh:
        fh.read(100)  # skip file header
        while True:
            record_header = fh.read(8)
            if len(record_header) < 8:
                break
            _, content_length_words = int.from_bytes(record_header[:4], 'big'), int.from_bytes(record_header[4:], 'big')
            content = fh.read(content_length_words * 2)
            if len(content) < 4:
                break
            shape_type = int.from_bytes(content[:4], 'little')
            if shape_type != 5:
                continue
            bbox = struct.unpack('<4d', content[4:36])
            num_parts = int.from_bytes(content[36:40], 'little')
            num_points = int.from_bytes(content[40:44], 'little')
            parts = struct.unpack('<' + 'i' * num_parts, content[44:44 + num_parts * 4]) if num_parts else ()
            points_data = struct.unpack('<' + 'd' * num_points * 2, content[44 + num_parts * 4:44 + num_parts * 4 + num_points * 16])
            coords = [(points_data[i], points_data[i + 1]) for i in range(0, len(points_data), 2)]
            polygons.append({'bbox': bbox, 'coords': coords, 'parts': parts})
    return polygons


def load_target_patches() -> set[str]:
    patches: set[str] = set()
    if not SPLITS_DIR.exists():
        return patches
    for txt in SPLITS_DIR.glob('*_X.txt'):
        with txt.open('r', encoding='utf-8') as fh:
            for line in fh:
                name = line.strip()
                if name:
                    patches.add(f'S2_{name}')
    return patches


def load_scene_patches(scene: str, allowed: set[str]) -> dict[str, dict]:
    scene_dir = PATCHES_DIR / scene
    if not scene_dir.exists():
        return {}
    patches: dict[str, dict] = {}
    for patch_name in allowed:
        image_path = scene_dir / f'{patch_name}.tif'
        mask_path = scene_dir / f'{patch_name}_cl.tif'
        if not image_path.exists() or not mask_path.exists():
            continue
        with rasterio.open(mask_path) as src:
            patches[patch_name] = {
                'name': patch_name,
                'image_path': image_path,
                'mask_path': mask_path,
                'transform': src.transform,
                'bounds': array_bounds(src.height, src.width, src.transform),
                'width': src.width,
                'height': src.height,
                'mask_array': None,
            }
    return patches


def get_patch_mask(patch_info: dict) -> np.ndarray:
    if patch_info['mask_array'] is None:
        with rasterio.open(patch_info['mask_path']) as src:
            patch_info['mask_array'] = src.read(1)
    return patch_info['mask_array']


def find_patch_for_point(scene_patches: dict[str, dict], x: float, y: float) -> dict | None:
    for patch in scene_patches.values():
        minx, miny, maxx, maxy = patch['bounds']
        if minx <= x <= maxx and miny <= y <= maxy:
            return patch
    return None


def clip_bbox_to_patch(bbox: tuple[float, float, float, float], patch_bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    poly_minx, poly_miny, poly_maxx, poly_maxy = bbox
    patch_minx, patch_miny, patch_maxx, patch_maxy = patch_bounds
    clip_minx = max(poly_minx, patch_minx)
    clip_maxx = min(poly_maxx, patch_maxx)
    clip_miny = max(poly_miny, patch_miny)
    clip_maxy = min(poly_maxy, patch_maxy)
    if clip_minx >= clip_maxx or clip_miny >= clip_maxy:
        return None
    return clip_minx, clip_miny, clip_maxx, clip_maxy


def convert_polygon_to_bbox(patch: dict, clipped_bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float, tuple[float, float, float, float]] | None:
    clip_minx, clip_miny, clip_maxx, clip_maxy = clipped_bbox
    inv = ~patch['transform']
    col_min, row_min = inv * (clip_minx, clip_maxy)
    col_max, row_max = inv * (clip_maxx, clip_miny)
    col_min = float(max(0.0, min(col_min, patch['width'])))
    col_max = float(max(0.0, min(col_max, patch['width'])))
    row_min = float(max(0.0, min(row_min, patch['height'])))
    row_max = float(max(0.0, min(row_max, patch['height'])))
    if col_max <= col_min or row_max <= row_min:
        return None
    width = patch['width']
    height = patch['height']
    x_center = ((col_min + col_max) / 2.0) / width
    y_center = ((row_min + row_max) / 2.0) / height
    box_w = (col_max - col_min) / width
    box_h = (row_max - row_min) / height
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return None
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    pixel_bbox = (col_min, row_min, col_max, row_max)
    return x_center, y_center, box_w, box_h, pixel_bbox


def evaluate_bbox_overlap(patch: dict, pixel_bbox: tuple[float, float, float, float], original_class: int) -> float:
    mask = get_patch_mask(patch)
    col_min, row_min, col_max, row_max = pixel_bbox
    r0 = max(0, int(np.floor(row_min)))
    r1 = min(patch['height'], int(np.ceil(row_max)))
    c0 = max(0, int(np.floor(col_min)))
    c1 = min(patch['width'], int(np.ceil(col_max)))
    if r1 <= r0 or c1 <= c0:
        return 0.0
    crop = mask[r0:r1, c0:c1]
    total_pixels = crop.size
    if total_pixels == 0:
        return 0.0
    positive = np.count_nonzero(crop == original_class)
    return positive / float(total_pixels)


def read_rgb_image(image_path: Path) -> np.ndarray:
    with rasterio.open(image_path) as src:
        data = src.read([4, 3, 2]).astype(np.float32)
    rgb = np.transpose(data, (1, 2, 0))
    for channel in range(3):
        band = rgb[:, :, channel]
        p2, p98 = np.percentile(band, (2, 98))
        if p98 <= p2:
            rgb[:, :, channel] = np.clip(band, 0, 1)
        else:
            scaled = (band - p2) / (p98 - p2)
            rgb[:, :, channel] = np.clip(scaled, 0, 1)
    return (rgb * 255.0).astype(np.uint8)


def export_visualization(patch: dict, boxes: list[dict], index: int) -> None:
    rgb = read_rgb_image(patch['image_path'])
    base = rgb.copy()
    for box in boxes:
        col_min, row_min, col_max, row_max = box['pixel_bbox']
        pt1 = (int(round(col_min)), int(round(row_min)))
        pt2 = (int(round(col_max)), int(round(row_max)))
        color = YOLO_COLORS.get(box['class_id'], (255, 255, 255))
        cv2.rectangle(base, pt1, pt2, color, 2)
        label = YOLO_CLASS_NAMES[box['class_id']]
        cv2.putText(base, label, (pt1[0], max(0, pt1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    image_name = f"{patch['name']}.png"
    vis_name = f"{patch['name']}_bbox.png"
    cv2.imwrite(str(OUTPUT_DIR / 'images' / image_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUTPUT_DIR / 'visualizations' / vis_name), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))


def write_yolo_labels(labels: dict[str, list[dict]], all_targets: set[str]) -> None:
    for patch_name in sorted(all_targets):
        entries = labels.get(patch_name, [])
        path = OUTPUT_DIR / 'labels' / f'{patch_name}.txt'
        with path.open('w', encoding='utf-8') as fh:
            for entry in entries:
                fh.write(
                    f"{entry['class_id']} "
                    f"{entry['x_center']:.6f} {entry['y_center']:.6f} "
                    f"{entry['width']:.6f} {entry['height']:.6f}\n"
                )


def process_scene(scene: str, target_patches: set[str], labels: dict[str, list[dict]], stats: dict) -> None:
    relevant = {p for p in target_patches if p.startswith(scene)}
    if not relevant:
        return
    scene_patches = load_scene_patches(scene, relevant)
    if not scene_patches:
        stats['skipped_scenes'] += 1
        return

    shp_path = SHAPEFILES_DIR / f'{scene}.shp'
    dbf_path = SHAPEFILES_DIR / f'{scene}.dbf'
    if not shp_path.exists() or not dbf_path.exists():
        stats['missing_shapefiles'] += 1
        return

    records = read_dbf(dbf_path)
    polygons = read_polygons(shp_path)
    stats['scenes'] += 1
    stats['polygons_total'] += len(polygons)

    for record, poly in zip(records, polygons):
        original_class = record.get('report')
        if not isinstance(original_class, (int, float)):
            stats['skipped_class'] += 1
            continue
        original_class = int(original_class)
        if original_class not in ORIGINAL_TO_YOLO_MAP:
            stats['skipped_class'] += 1
            continue

        coords = poly['coords']
        if not coords:
            stats['skipped_geometry'] += 1
            continue
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))

        patch = find_patch_for_point(scene_patches, centroid[0], centroid[1])
        if patch is None:
            stats['skipped_no_patch'] += 1
            continue

        clipped_bbox = clip_bbox_to_patch(poly['bbox'], patch['bounds'])
        if clipped_bbox is None:
            stats['skipped_bbox'] += 1
            continue

        bbox_values = convert_polygon_to_bbox(patch, clipped_bbox)
        if bbox_values is None:
            stats['skipped_bbox'] += 1
            continue

        x_center, y_center, width, height, pixel_bbox = bbox_values
        class_id = ORIGINAL_TO_YOLO_MAP[original_class]
        overlap = evaluate_bbox_overlap(patch, pixel_bbox, original_class)

        entry = {
            'class_id': class_id,
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height,
            'pixel_bbox': pixel_bbox,
            'original_class': original_class,
            'overlap': overlap,
        }
        labels[patch['name']].append(entry)
        stats['boxes_written'] += 1
        stats['class_counts'][class_id] += 1
        stats['overlaps'].append(overlap)
        stats['class_overlaps'][class_id].append(overlap)


def summarize_results(stats: dict, labels: dict[str, list[dict]], target_patches: set[str]) -> None:
    print('\n================ Shapefile → YOLO 统计 ================')
    print(f"处理场景: {stats['scenes']}")
    print(f"原始多边形: {stats['polygons_total']}")
    print(f"生成边框: {stats['boxes_written']}")
    print(f"目标补丁总数: {len(target_patches)}")
    print(f"含标注补丁: {len([p for p in labels if labels[p]])}")
    print(f"缺少场景/文件: {stats['missing_shapefiles']}")
    print(f"跳过(无匹配补丁): {stats['skipped_no_patch']}")
    print(f"跳过(类别不在映射内): {stats['skipped_class']}")
    print(f"跳过(几何/边界异常): {stats['skipped_bbox'] + stats['skipped_geometry']}")

    print('\n类别统计:')
    for class_id, count in stats['class_counts'].most_common():
        print(f"- {YOLO_CLASS_NAMES[class_id]}: {count}")

    if stats['overlaps']:
        overlaps = np.array(stats['overlaps'], dtype=np.float32)
        print('\n像素覆盖率概览 (Shapefile vs Mask):')
        print(f"- 平均覆盖率: {overlaps.mean():.3f}")
        print(f"- 中位覆盖率: {np.median(overlaps):.3f}")
        print(f"- ≥0.25 覆盖率: {(overlaps >= 0.25).mean():.3f}")
        print(f"- ≥0.50 覆盖率: {(overlaps >= 0.50).mean():.3f}")
        print('\n按类别分布:')
        for class_id, values in stats['class_overlaps'].items():
            arr = np.array(values, dtype=np.float32)
            print(f"  · {YOLO_CLASS_NAMES[class_id]} → 均值 {arr.mean():.3f}, 中位 {np.median(arr):.3f}, ≥0.25 {(arr >= 0.25).mean():.3f}")
    else:
        print('\n未计算到任何覆盖率统计。')


def create_outputs() -> None:
    ensure_dirs()
    target_patches = load_target_patches()
    if not target_patches:
        print('未找到 splits 中的补丁列表，终止。')
        return

    stats = {
        'scenes': 0,
        'polygons_total': 0,
        'boxes_written': 0,
        'missing_shapefiles': 0,
        'skipped_scenes': 0,
        'skipped_no_patch': 0,
        'skipped_class': 0,
        'skipped_bbox': 0,
        'skipped_geometry': 0,
        'class_counts': Counter(),
        'overlaps': [],
        'class_overlaps': defaultdict(list),
    }

    labels: dict[str, list[dict]] = defaultdict(list)

    for shp_file in sorted(SHAPEFILES_DIR.glob('*.shp')):
        scene = shp_file.stem
        process_scene(scene, target_patches, labels, stats)

    write_yolo_labels(labels, target_patches)

    # 导出有限数量的可视化以便抽检
    exported = 0
    for patch_name in sorted(labels.keys()):
        if not labels[patch_name]:
            continue
        patch_scene = patch_name.rsplit('_', 1)[0]
        scene_patches = load_scene_patches(patch_scene, {patch_name})
        patch_info = scene_patches.get(patch_name)
        if patch_info is None:
            continue
        export_visualization(patch_info, labels[patch_name], exported)
        exported += 1
        if exported >= MAX_VISUALIZATIONS:
            break

    summarize_results(stats, labels, target_patches)
    print(f"\n输出目录: {OUTPUT_DIR.resolve()}\n")


if __name__ == '__main__':
    import struct

    print('开始将 Shapefile 几何转换为 YOLO 边界框...')
    create_outputs()