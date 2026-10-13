# imgsz=1024 leaky ablation, 3-seed aggregate (mean +- std)

Leaky data (marida_mask_rescue_7ch.yaml, 1216 train tiles, GT-derived heatmap as ch7).
imgsz=512 cells reused from prior 3-seed leaky aggregate.
imgsz=1024 cells new (batch=4 workers=0 ep=30).

| Ablation | imgsz | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 512 | 3 | 0.690 +- 0.004 | 0.446 +- 0.024 | 0.740 +- 0.060 | 0.856 +- 0.029 | 0.910 +- 0.039 |
| baseline | 1024 | 3 | 0.695 +- 0.006 | 0.456 +- 0.018 | 0.736 +- 0.071 | 0.870 +- 0.016 | 0.978 +- 0.029 |
| bifpn | 512 | 3 | 0.688 +- 0.020 | 0.425 +- 0.023 | 0.741 +- 0.088 | 0.852 +- 0.019 | 0.907 +- 0.069 |
| bifpn | 1024 | 3 | 0.679 +- 0.017 | 0.461 +- 0.021 | 0.710 +- 0.052 | 0.860 +- 0.017 | 0.847 +- 0.097 |
