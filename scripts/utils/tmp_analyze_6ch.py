import csv
import json
from pathlib import Path

csv_path = Path('marida_runs/yolo11n_6ch_full_tuned/results.csv')
rows = []
with csv_path.open(newline='') as f:
       reader = csv.DictReader(f)
       for row in reader:
              rows.append(row)

if not rows:
       raise SystemExit('No rows found in results.csv')

def to_float(row, key):
       return float(row[key])

best = max(rows, key=lambda r: to_float(r, 'metrics/mAP50-95(B)'))
best_epoch = int(float(best['epoch']))
result = {
       'best_epoch': best_epoch,
       'precision': to_float(best, 'metrics/precision(B)'),
       'recall': to_float(best, 'metrics/recall(B)'),
       'mAP50': to_float(best, 'metrics/mAP50(B)'),
       'mAP50_95': to_float(best, 'metrics/mAP50-95(B)'),
       'val_box_loss': to_float(best, 'val/box_loss'),
       'val_cls_loss': to_float(best, 'val/cls_loss'),
       'val_dfl_loss': to_float(best, 'val/dfl_loss'),
       'max_mAP50': max(to_float(r, 'metrics/mAP50(B)') for r in rows)
}
print(json.dumps(result, indent=2))
out_path = csv_path.with_name('best_metrics.json')
out_path.write_text(json.dumps(result, indent=2))
print(f'Wrote {out_path}')
