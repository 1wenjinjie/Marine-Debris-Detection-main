import json
from pathlib import Path
import numpy as np

root = Path('MARIDA_YOLO_Enriched_Dilated_split')
splits = ['train', 'val', 'test']
results = {}

for split in splits:
    labels_dir = root / split / 'labels'
    areas = []
    widths = []
    heights = []
    total_objs = 0
    class_hist = [0] * 7
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls, xc, yc, w, h = map(float, line.split())
                class_hist[int(cls)] += 1
                total_objs += 1
                widths.append(w * 256)
                heights.append(h * 256)
                areas.append(w * h * 256 * 256)
    arr = np.array(areas)
    if arr.size:
        stats = {
            'objects': total_objs,
            'class_hist': class_hist,
            'area_px_min': float(arr.min()),
            'area_px_mean': float(arr.mean()),
            'area_px_median': float(np.median(arr)),
            'area_px_pct1': float(np.percentile(arr, 1)),
            'area_px_pct5': float(np.percentile(arr, 5)),
            'area_px_pct25': float(np.percentile(arr, 25)),
            'area_px_pct75': float(np.percentile(arr, 75)),
            'area_px_pct95': float(np.percentile(arr, 95)),
            'area_px_pct99': float(np.percentile(arr, 99)),
            'tiny_lt4px': int((arr < 4).sum()),
            'tiny_lt9px': int((arr < 9).sum()),
            'tiny_lt16px': int((arr < 16).sum()),
        }
    else:
        stats = {}
    results[split] = stats

out_path = Path('dataset_stats_summary.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved summary to {out_path.resolve()}')
