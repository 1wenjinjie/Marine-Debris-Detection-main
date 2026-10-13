"""
Figure 8: side-by-side detection visualization, baseline vs final model.

For each Marine_Debris test tile, runs both models and renders:
  red box     = ground truth
  blue dashed = model prediction (alpha by confidence)

Output is a single Figure-8 grid PNG, suitable for the paper body.

Defaults to comparing:
  baseline_6ch_s0  (imgsz=512, BCE)        ↔ final = focal_g2.0_i1024_s0
                                              (imgsz=1024, focal γ=2.0)

Usage:
  python scripts/v2_make_fig8_detection_overlay.py
  python scripts/v2_make_fig8_detection_overlay.py \\
      --tiles S2_23-9-20_16PCC_25 S2_29-8-17_51RVQ_4 S2_7-3-20_18QYG_12

The script imports preprocessing helpers from v2_sweep_thresholds to
keep behaviour aligned with the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ultralytics_plugins  # noqa: F401  (registers CBAM etc.)
from v2_sweep_thresholds import (
    DATA_ROOT, STATS_PATH, DEBRIS_CLS, IOU_STRICT,
    _load_channel_stats, _preprocess_npy, _arr_to_tensor,
    _load_gt_boxes, _xywh_norm_to_xyxy_pixel, _iou_matrix, _raw_predict,
)

FIG_DIR = ROOT / "runs_v2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TILES = [
    "S2_23-9-20_16PCC_25",        # 2 GT, stable winner
    "S2_29-8-17_51RVQ_4",         # 1 GT, stable winner
    "S2_7-3-20_18QYG_12",         # 1 GT, always-TP
    "S2_7-3-20_18QYG_9",          # 1 GT, seed-dependent
    "S2_7-3-20_18QYG_8",          # 1 GT, seed-dependent
    "S2_20-4-18_30VWH_4",         # 4 GT, hardest scene
]


def _build_rgb(arr_hwc: np.ndarray) -> np.ndarray:
    """B4/B3/B2 -> stretched RGB image."""
    rgb = arr_hwc[:, :, :3].astype(np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        rgb[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-6) * 255, 0, 255)
    return rgb.astype(np.uint8)


def _draw_panel(ax, rgb: np.ndarray, gt_xyxy: np.ndarray,
                preds_debris: np.ndarray, *, title: str) -> None:
    ax.imshow(rgb)
    # GT in red
    for box in gt_xyxy:
        x1, y1, x2, y2 = box
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.6, edgecolor="red", facecolor="none",
        ))
    # Predictions in blue, alpha scales with confidence
    for pred in preds_debris:
        x1, y1, x2, y2, conf, _ = pred
        alpha = float(min(1.0, 0.3 + conf * 4))
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.0, edgecolor="blue", facecolor="none",
            linestyle="--", alpha=alpha,
        ))
        ax.text(x1, max(2, y1 - 2), f"{conf:.3f}",
                color="cyan", fontsize=6,
                bbox={"facecolor": "black", "alpha": 0.4, "pad": 0.5, "edgecolor": "none"})
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _summarize(preds_debris: np.ndarray, gt_xyxy: np.ndarray) -> str:
    """Return short caption: '2/2 TP @ conf=0.05, max IoU=0.92' style."""
    if len(gt_xyxy) == 0:
        return "no GT"
    if len(preds_debris) == 0:
        return f"0/{len(gt_xyxy)} matched (no preds)"
    iou = _iou_matrix(preds_debris[:, :4], gt_xyxy)
    n_tp = 0
    best_conf = 0.0
    max_iou = 0.0
    for j in range(len(gt_xyxy)):
        col = iou[:, j]
        if not len(col):
            continue
        i_best = int(np.argmax(col))
        max_iou = max(max_iou, float(col[i_best]))
        if col[i_best] > IOU_STRICT:
            n_tp += 1
            best_conf = max(best_conf, float(preds_debris[i_best, 4]))
    return f"{n_tp}/{len(gt_xyxy)} TP, max IoU={max_iou:.2f}, top conf={best_conf:.3f}"


def predict_tile(model_nn, npy_path: Path, low, high, channels: int, imgsz: int):
    arr, _, _ = _preprocess_npy(npy_path, low[:, :, :channels], high[:, :, :channels],
                                channels=channels, imgsz=imgsz)
    tensor = _arr_to_tensor(arr)
    preds = _raw_predict(model_nn, tensor, conf_thr=0.001, iou_thr=0.45,
                          nc=model_nn.yaml.get("nc", 3))
    debris_preds = preds[preds[:, 5].astype(int) == DEBRIS_CLS] if len(preds) else preds
    return arr, debris_preds


def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 8: detection overlay comparison")
    ap.add_argument("--baseline-weights",
                    default=str(ROOT / "runs_v2" / "baseline_6ch_s0" / "weights" / "best.pt"))
    ap.add_argument("--baseline-imgsz", type=int, default=512)
    ap.add_argument("--baseline-name", default="Baseline (BCE, 512)")
    ap.add_argument("--final-weights",
                    default=str(ROOT / "runs_v2" / "focal_g2.0_i1024_s0" / "weights" / "best.pt"))
    ap.add_argument("--final-imgsz", type=int, default=1024)
    ap.add_argument("--final-name", default="Proposed (focal γ=2.0, 1024)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tiles", nargs="+", default=DEFAULT_TILES)
    ap.add_argument("--out", default=str(FIG_DIR / "fig8_detection_overlay.png"))
    args = ap.parse_args()

    image_dir = DATA_ROOT / args.split / "images"
    label_dir = DATA_ROOT / args.split / "labels"
    low, high = _load_channel_stats(STATS_PATH)

    from ultralytics import YOLO
    print(f"Loading baseline:  {args.baseline_weights}")
    base_nn = YOLO(args.baseline_weights).model.cuda().eval()
    print(f"Loading final:     {args.final_weights}")
    final_nn = YOLO(args.final_weights).model.cuda().eval()

    n = len(args.tiles)
    fig, axes = plt.subplots(n, 2, figsize=(11, 5.0 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])

    for row_i, tile_name in enumerate(args.tiles):
        npy_path = image_dir / f"{tile_name}.npy"
        label_path = label_dir / f"{tile_name}.txt"
        if not npy_path.exists():
            print(f"  [WARN] missing tile: {tile_name}")
            continue

        gt_xywh = _load_gt_boxes(label_path)

        # baseline at its own imgsz
        arr_b, preds_b = predict_tile(base_nn, npy_path, low, high, 6, args.baseline_imgsz)
        gt_b = _xywh_norm_to_xyxy_pixel(gt_xywh, args.baseline_imgsz)
        rgb_b = _build_rgb(arr_b)

        # final at its own imgsz
        arr_f, preds_f = predict_tile(final_nn, npy_path, low, high, 6, args.final_imgsz)
        gt_f = _xywh_norm_to_xyxy_pixel(gt_xywh, args.final_imgsz)
        rgb_f = _build_rgb(arr_f)

        sum_b = _summarize(preds_b, gt_b)
        sum_f = _summarize(preds_f, gt_f)

        _draw_panel(axes[row_i, 0], rgb_b, gt_b, preds_b,
                    title=f"{args.baseline_name}\n{tile_name} | {sum_b}")
        _draw_panel(axes[row_i, 1], rgb_f, gt_f, preds_f,
                    title=f"{args.final_name}\n{tile_name} | {sum_f}")

    # Legend at the very top of the figure
    legend_handles = [
        mpatches.Patch(edgecolor="red", facecolor="none", label="Ground truth (debris)"),
        mpatches.Patch(edgecolor="blue", facecolor="none", linestyle="--",
                       label="Prediction (alpha = confidence)"),
    ]
    fig.suptitle("Figure 8. Detection overlay on Marine_Debris test tiles",
                 fontsize=13, fontweight="bold")
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.005))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
