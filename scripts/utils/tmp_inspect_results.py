import pandas as pd
from pathlib import Path
run = Path('runs_marida/yolov11n_7ch_full_return_512_resumed')
res = run / 'results.csv'
if not res.exists():
    raise SystemExit('results.csv missing: ' + str(res))
df = pd.read_csv(res)
best_idx = df['metrics/mAP50(B)'].idxmax()
best = df.iloc[int(best_idx)]
last = df.iloc[-1]
print('epochs', len(df))
print('best_epoch', int(best['epoch']))
print('best_map50', float(best['metrics/mAP50(B)']))
print('best_map95', float(best['metrics/mAP50-95(B)']))
print('best_precision', float(best['metrics/precision(B)']))
print('best_recall', float(best['metrics/recall(B)']))
print('final_map50', float(last['metrics/mAP50(B)']))
print('final_map95', float(last['metrics/mAP50-95(B)']))
