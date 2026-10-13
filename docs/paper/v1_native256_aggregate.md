# Native-256 vs imgsz=1024 cross-resolution control (3-seed)

All honest setting, v1 physidx on v2 scene-clean split, clean init, ep=30.
imgsz=256: batch=16, workers=0 (native, no letterbox up-sampling).
imgsz=512: batch=8, workers=0 (2x letterbox up-sampling).
imgsz=1024: batch=4, workers=0 (4x letterbox up-sampling).

Question: does the imgsz=1024 gain come from (a) more model capacity / P3-head
landing on debris box size, or (b) letterbox up-sampling specifically?
If imgsz=256 already beats imgsz=512 -> (a) dominates. If only imgsz=1024 wins -> (b).

## Per-architecture cross-resolution table

### baseline

| imgsz | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 3 | 0.228 +- 0.033 | 0.101 +- 0.011 | 0.335 +- 0.041 | 0.364 +- 0.054 | 0.641 +- 0.043 |
| 512 | 3 | 0.313 +- 0.038 | 0.165 +- 0.011 | 0.409 +- 0.022 | 0.492 +- 0.056 | 0.690 +- 0.107 |
| 1024 | 3 | 0.342 +- 0.011 | 0.219 +- 0.016 | 0.427 +- 0.048 | 0.607 +- 0.022 | 0.765 +- 0.012 |

### bifpn

| imgsz | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 3 | 0.245 +- 0.025 | 0.115 +- 0.018 | 0.373 +- 0.040 | 0.451 +- 0.014 | 0.684 +- 0.125 |
| 512 | 3 | 0.300 +- 0.059 | 0.167 +- 0.021 | 0.406 +- 0.029 | 0.559 +- 0.023 | 0.665 +- 0.092 |
| 1024 | 3 | 0.350 +- 0.031 | 0.225 +- 0.023 | 0.479 +- 0.037 | 0.576 +- 0.045 | 0.755 +- 0.082 |

