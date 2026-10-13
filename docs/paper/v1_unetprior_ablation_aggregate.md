# U-Net prior vs NDWI-inverse prior, YOLO @ imgsz=1024 (3-seed)

All honest setting (v1 7ch on v2 scene-clean split), clean init, ep=30, batch=4, workers=0.
Only ch7 changes: NDWI-inverse (physidx) vs U-Net Marine_Debris probability (unetprior, K-fold leak-free).

## baseline

| ch7 prior | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |
|---|---:|---:|---:|---:|---:|---:|
| physidx | 3 | 0.342 +- 0.011 | 0.219 +- 0.016 | 0.427 +- 0.048 | 0.607 +- 0.022 | 0.765 +- 0.012 |
| unetprior | 3 | 0.320 +- 0.026 | 0.192 +- 0.010 | 0.408 +- 0.040 | 0.474 +- 0.053 | 0.701 +- 0.177 |

## bifpn

| ch7 prior | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |
|---|---:|---:|---:|---:|---:|---:|
| physidx | 3 | 0.350 +- 0.031 | 0.225 +- 0.023 | 0.479 +- 0.037 | 0.576 +- 0.045 | 0.755 +- 0.082 |
| unetprior | 3 | 0.322 +- 0.010 | 0.197 +- 0.001 | 0.428 +- 0.015 | 0.503 +- 0.031 | 0.799 +- 0.030 |

