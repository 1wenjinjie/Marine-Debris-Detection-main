from pathlib import Path
import rasterio
import json

root = Path('MARIDA_YOLO_Enriched_Dilated_split')
splits = ['train', 'val', 'test']
im_cache = {}
summary = {}

for split in splits:
    labels_dir = root / split / 'labels'
    stats = {
        'objects': 0,
        'class_hist': [0] * 7,
        'areas_px': [],
        'width_px': [],
        'height_px': [],
        'tiny_lt4px': 0,
        'tiny_lt9px': 0,
        'tiny_lt16px': 0,
        'samples': 0
    }
    for label_file in labels_dir.glob('*.txt'):
        stem = label_file.stem
        img_file = root / split / 'images' / f'{stem}.tif'
        if not img_file.exists():
            imgs = list((root / split / 'images').glob(f'{stem}.*'))
            img_file = imgs[0] if imgs else None
        if img_file and stem not in im_cache:
            with rasterio.open(img_file) as src:
                h, w = src.height, src.width
                im_cache[stem] = (h, w)
        H, W = im_cache.get(stem, (256, 256))
        stats['samples'] += 1
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                stats['objects'] += 1
                stats['class_hist'][cls] += 1
                width_px = w * W
                height_px = h * H
                area_px = width_px * height_px
                stats['width_px'].append(width_px)
                stats['height_px'].append(height_px)
                stats['areas_px'].append(area_px)
                if area_px < 4:
                    stats['tiny_lt4px'] += 1
                if area_px < 9:
                    stats['tiny_lt9px'] += 1
                if area_px < 16:
                    stats['tiny_lt16px'] += 1
    summary[split] = stats

    out_path = Path('dataset_stats.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved stats to {out_path.resolve()}')
