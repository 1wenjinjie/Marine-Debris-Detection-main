import pandas as pd
import numpy as np

# Read training results
df = pd.read_csv('runs_marida/yolov11n_10ch_spectral_v2/results.csv')
df.columns = df.columns.str.strip()

# Get final epoch (best based on mAP)
best_idx = df['metrics/mAP50(B)'].idxmax()
best_epoch = df.loc[best_idx]

print("\n" + "="*70)
print(" "*15 + "10-Channel Spectral Model - Best Epoch Results")
print("="*70)
print(f"Best Epoch: {int(best_epoch['epoch'])}")
print(f"\n{'Metric':<30} {'Value':>10}")
print("-"*70)
print(f"{'mAP@50':<30} {best_epoch['metrics/mAP50(B)']:>10.4f}")
print(f"{'mAP@50-95':<30} {best_epoch['metrics/mAP50-95(B)']:>10.4f}")
print(f"{'Precision':<30} {best_epoch['metrics/precision(B)']:>10.4f}")
print(f"{'Recall':<30} {best_epoch['metrics/recall(B)']:>10.4f}")

# Compare with baseline
print(f"\n{'='*70}")
print(" "*20 + "Comparison with 7ch Baseline")
print(f"{'='*70}")
print(f"{'Metric':<25} {'7ch Baseline':>15} {'10ch Spectral':>15} {'Gain':>10}")
print("-"*70)

baseline_map50 = 0.881  # From previous experiments
spectral_map50 = best_epoch['metrics/mAP50(B)']
print(f"{'mAP@50':<25} {baseline_map50:>15.4f} {spectral_map50:>15.4f} {(spectral_map50-baseline_map50):>10.4f}")

# Final epoch metrics
final = df.iloc[-1]
print(f"\n{'='*70}")
print(" "*25 + "Final Epoch (50) Metrics")
print(f"{'='*70}")
print(f"mAP@50:    {final['metrics/mAP50(B)']:.4f}")
print(f"mAP@50-95: {final['metrics/mAP50-95(B)']:.4f}")
print(f"Precision: {final['metrics/precision(B)']:.4f}")
print(f"Recall:    {final['metrics/recall(B)']:.4f}")
print("="*70 + "\n")
