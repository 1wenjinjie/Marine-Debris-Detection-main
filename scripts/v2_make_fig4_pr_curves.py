"""
Figure 4 (dedicated PR curves): Marine_Debris precision-recall on the
test split, multi-model comparison.

Two sources of data are combined:

1. Per-tile inference (one shot at conf=0.001) over all 9 debris test
   tiles, giving the full set of (conf, max_iou) pairs needed to build
   a smooth PR curve.

2. Per-instance-best from each model's sweep_results.json, used as a
   sanity-check overlay (sparse points).

Output:
  runs_v2/figures/fig4_pr_curves_marine_debris.png

Usage:
  python scripts/v2_make_fig4_pr_curves.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ultralytics_plugins  # noqa: F401
from v2_sweep_thresholds import (
    DATA_ROOT, STATS_PATH, DEBRIS_CLS,
    _load_channel_stats, _preprocess_npy, _arr_to_tensor,
    _load_gt_boxes, _xywh_norm_to_xyxy_pixel, _iou_matrix, _raw_predict,
)

FIG_DIR = ROOT / "runs_v2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# (label, weights, imgsz, color, linestyle, linewidth)
MODELS = [
    ("Baseline (BCE, 512)",       "baseline_6ch_s0",     512,  "#1f77b4", "-",  2.2),
    ("+CBAM",                     "cbam_v2_s0",          512,  "#ff7f0e", "-",  1.6),
    ("+BiFPN-lite",               "bifpn_v2_s0",         512,  "#2ca02c", "-",  1.6),
    ("+EIoU",                     "eiou_v2_s0",          512,  "#9467bd", "-",  1.6),
    ("Full (CBAM+BiFPN+EIoU)",    "full_v2_s0",          512,  "#8c564b", "-",  1.6),
    ("+ imgsz=1024 (BCE)",        "imgsz1024_s0",        1024, "#17becf", "--", 1.8),
    ("+ focal γ=2.0 (i1024)",     "focal_g2.0_i1024_s0", 1024, "#d62728", "-",  2.5),
]


def collect_predictions(weights_path: Path, imgsz: int, channels: int = 6):
    """
    Run inference on every debris-containing test tile at conf=0.001.
    Return list of (conf, is_tp) where is_tp = 1 if this prediction
    matches some GT debris box with IoU > 0.5 (in pixel coords).

    Also returns total GT count across all tiles (for recall denominator).
    """
    from ultralytics import YOLO

    label_dir = DATA_ROOT / "test" / "labels"
    image_dir = DATA_ROOT / "test" / "images"
    low, high = _load_channel_stats(STATS_PATH)
    low_c = low[:, :, :channels]
    high_c = high[:, :, :channels]

    nn_model = YOLO(str(weights_path)).model.cuda().eval()
    nc = nn_model.yaml.get("nc", 3)

    debris_label_files = sorted([
        lf for lf in label_dir.glob("*.txt")
        if any(int(line.split()[0]) == DEBRIS_CLS
               for line in lf.read_text().splitlines() if line.strip())
    ])

    pred_records = []  # (conf, is_tp)
    total_gt = 0

    for label_path in debris_label_files:
        stem = label_path.stem
        npy_path = image_dir / f"{stem}.npy"
        if not npy_path.exists():
            continue

        arr, _, _ = _preprocess_npy(npy_path, low_c, high_c,
                                    channels=channels, imgsz=imgsz)
        tensor = _arr_to_tensor(arr)
        preds = _raw_predict(nn_model, tensor, conf_thr=0.001, iou_thr=0.45, nc=nc)

        gt_xywh = _load_gt_boxes(label_path)
        gt_xyxy = _xywh_norm_to_xyxy_pixel(gt_xywh, imgsz)
        total_gt += len(gt_xyxy)

        if not len(preds):
            continue
        debris_preds = preds[preds[:, 5].astype(int) == DEBRIS_CLS]
        if not len(debris_preds):
            continue

        # Match each prediction greedily to GT for TP/FP labelling
        if len(gt_xyxy):
            iou = _iou_matrix(debris_preds[:, :4], gt_xyxy)
            # Greedy: for each prediction (sorted by conf desc), assign to best
            # unused GT if IoU > 0.5
            order = np.argsort(-debris_preds[:, 4])
            gt_used = np.zeros(len(gt_xyxy), dtype=bool)
            for pi in order:
                ji_sorted = np.argsort(-iou[pi])
                is_tp = 0
                for ji in ji_sorted:
                    if gt_used[ji]:
                        continue
                    if iou[pi, ji] > 0.5:
                        is_tp = 1
                        gt_used[ji] = True
                    break
                pred_records.append((float(debris_preds[pi, 4]), int(is_tp)))
        else:
            for c in debris_preds[:, 4]:
                pred_records.append((float(c), 0))

    del nn_model
    torch.cuda.empty_cache()
    return pred_records, total_gt


def pr_curve_from_records(records, total_gt: int):
    """Compute precision-recall curve points by sorting predictions by conf desc."""
    if not records:
        return np.array([0.0]), np.array([0.0]), 0.0
    arr = np.array(records, dtype=np.float64)  # (N, 2): conf, is_tp
    order = np.argsort(-arr[:, 0])
    arr = arr[order]
    tp_cum = np.cumsum(arr[:, 1])
    fp_cum = np.cumsum(1 - arr[:, 1])
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / max(total_gt, 1)

    # AP @ IoU=0.5 via average of precision over recall points
    # (use 11-point interpolation for stability with few GT instances)
    ap = 0.0
    for r in np.linspace(0, 1, 11):
        mask = recall >= r
        ap += (np.max(precision[mask]) if mask.any() else 0.0) / 11.0
    return precision, recall, ap


def main() -> int:
    print("Computing PR curves for Marine_Debris (test split)...\n")
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)

    for label, run_name, imgsz, color, ls, lw in MODELS:
        weights = ROOT / "runs_v2" / run_name / "weights" / "best.pt"
        if not weights.exists():
            print(f"  [WARN] missing weights for {run_name}")
            continue
        print(f"  {label:32s}  imgsz={imgsz}")
        records, n_gt = collect_predictions(weights, imgsz)
        precision, recall, ap = pr_curve_from_records(records, n_gt)
        ax.plot(recall, precision, color=color, linestyle=ls, linewidth=lw,
                alpha=0.92, label=f"{label}  (AP={ap:.3f})")
        print(f"    n_preds={len(records)}, n_gt={n_gt}, AP@0.5={ap:.3f}")

    ax.set_xlabel("Recall (Marine_Debris)", fontsize=11)
    ax.set_ylabel("Precision (Marine_Debris)", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.set_title("Figure 4. Marine_Debris precision-recall, test split (seed=0)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = FIG_DIR / "fig4_pr_curves_marine_debris.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
