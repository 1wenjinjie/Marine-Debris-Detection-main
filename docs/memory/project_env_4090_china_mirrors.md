---
name: project-env-4090-china-mirrors
description: "New machine (RTX 4090) Python env setup, China network mirror requirements, and how to run the pipeline"
metadata: 
  node_type: memory
  type: project
  originSessionId: 034bdf7c-ee65-44e8-abd5-81311ac7c1b8
---

2026-06-17: Migrated to a new machine (RTX 4090 24GB, driver 591.74, CUDA 13.1 max). Data, 464 `.pt` weights, and configs all carried over; dataset YAML hardcoded paths are still `D:/OceanEdge_Group/Marine-Debris-Detection-main/...` and match this machine, so no path edits needed.

**Conda env:** `marine` (Python 3.11) at `C:\Users\18453\miniconda3\envs\marine\python.exe`. Miniconda base at `C:\Users\18453\miniconda3`. Installed: torch 2.6.0+cu124, torchvision 0.21.0+cu124, ultralytics 8.4.69, rasterio 1.4.4, opencv-python, scipy, pandas, scikit-learn, scikit-image, albumentations, matplotlib. Verified: `torch.cuda.is_available()` True, device = RTX 4090. NMS import `from ultralytics.utils.nms import non_max_suppression` works on 8.4.69; plugins import fine.

**Why:** China network — direct TLS to `anaconda.org` and `pypi.org` fails with SSLEOFError; the Miniconda `.exe` only downloaded via PowerShell `Invoke-WebRequest` (Windows schannel), not curl/conda.
**How to apply (mirrors that WORK here):**
- conda env create: `-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ --override-channels` (also avoids the conda 26.x ToS error on default channels).
- pip general packages: `-i https://pypi.tuna.tsinghua.edu.cn/simple`
- PyTorch CUDA wheels: SJTU mirror `--index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu124` (official download.pytorch.org STALLS — only got 0.1MB). Pin exact versions (`torch==2.6.0 torchvision==0.21.0`) or pip's resolver churns metadata forever.
- Big wheels are most reliable downloaded via browser (Edge/Chrome use schannel) then `pip install <local.whl> -i <tsinghua>`.

**Command sandbox gotcha:** the Bash/PowerShell tools sandbox discards NSIS-installer writes — Miniconda had to be installed by the user double-clicking the GUI installer; `Start-Process` silent installs exited 0 but wrote nothing. Mirror-host downloads/installs DO work through the sandbox.

**Run pattern:** set `KMP_DUPLICATE_LIB_OK=TRUE`, call the marine python by full path. Verified smoke test: `python scripts/v2_evaluate.py --weights runs_v2/baseline_6ch_s0/weights/best.pt --data configs/datasets/marida_v2_6ch.yaml --channels 6 --device 0` → mAP50 0.209 (matches baseline), 1.9ms/img.

**Not yet installed (geo/figure-only, needed for paper figures):** cartopy, geopandas, gdal/osgeo, pyproj, shapely (use conda-forge), plus pip `grad-cam` and `datasets`. See [[project-v1-revival-2026-05-18]] for the active paper line.
