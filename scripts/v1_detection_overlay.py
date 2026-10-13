"""Render detection-overlay figures (teacher's 图8) for a trained checkpoint.

For each tile in the test split, plot:
- RGB composite (B4, B3, B2 of the multispectral input, p2-p98 stretched)
- GT bounding boxes (green dashed)
- Predicted bounding boxes (red solid, with class label + confidence)

Output: one PNG per tile, plus a paper-grade composite image showing
6-12 randomly sampled tiles in a grid.

Usage:
  python scripts/v1_detection_overlay.py \\
      --weights runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_clean_init/weights/best.pt \\
      --data configs/datasets/marida_v1_physidx_on_v2split_7ch.yaml \\
      --split test --conf 0.001 --n-grid 9 --seed 0

Output dir: docs/paper/figures/overlay_<run_name>/
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import ultralytics_plugins  # noqa: F401
from ultralytics_plugins.bifpn_modules import install_bifpn_patch
install_bifpn_patch()
from ultralytics import YOLO

CLASS_NAMES = ['Marine_Debris', 'Dense_Sargassum', 'Sparse_Sargassum',
               'Natural_Organic', 'Ship', 'Foam', 'Wakes']
CLASS_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                '#ff7f00', '#ffff33', '#a65628']


def stretch_rgb(rgb: np.ndarray, low_pct: float = 2, high_pct: float = 98) -> np.ndarray:
    """Percentile stretch each channel of an (H,W,3) array to [0,1]."""
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        lo = np.percentile(rgb[..., c], low_pct)
        hi = np.percentile(rgb[..., c], high_pct)
        out[..., c] = np.clip((rgb[..., c] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def load_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Return list of (cls, x, y, w, h) in normalized YOLO format."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        s = line.strip().split()
        if len(s) >= 5:
            out.append((int(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4])))
    return out


def draw_box_norm(ax, xn, yn, wn, hn, h_img, w_img, color, label=None, linestyle="-", linewidth=1.5):
    x = (xn - wn / 2) * w_img
    y = (yn - hn / 2) * h_img
    w = wn * w_img
    h = hn * h_img
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=color,
                              linewidth=linewidth, linestyle=linestyle)
    ax.add_patch(rect)
    if label:
        ax.text(x, max(y - 2, 0), label, color=color, fontsize=6,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5))


def draw_box_xyxy(ax, x1, y1, x2, y2, color, label=None, linestyle="-", linewidth=1.5):
    rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                              edgecolor=color, linewidth=linewidth, linestyle=linestyle)
    ax.add_patch(rect)
    if label:
        ax.text(x1, max(y1 - 2, 0), label, color=color, fontsize=6,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5))


def main():
    p = argparse.ArgumentParser(description="Detection overlay figures for teacher's 图8")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--n-grid", type=int, default=9, help="how many tiles in composite grid")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=512, help="must match training resolution")
    p.add_argument("--out-dir", default=None,
                   help="default: docs/paper/figures/overlay_<run_name>/")
    args = p.parse_args()

    wp = Path(args.weights)
    if not wp.exists():
        sys.exit(f"weights not found: {wp}")
    cfg = yaml.safe_load(Path(args.data).read_text())
    root = Path(cfg["path"])
    split_field = cfg.get(args.split, args.split)
    if isinstance(split_field, str) and Path(split_field).is_absolute():
        manifest_or_dir = Path(split_field)
    else:
        manifest_or_dir = root / split_field

    # Resolve tile list
    if manifest_or_dir.is_file():
        rels = [l.strip() for l in manifest_or_dir.read_text().splitlines() if l.strip()]
        tile_paths = [root / r for r in rels]
    else:
        tile_paths = sorted(manifest_or_dir.glob("*.npy"))

    random.seed(args.seed)
    sample = random.sample(tile_paths, min(args.n_grid, len(tile_paths)))

    run_name = wp.parent.parent.name
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / f"docs/paper/figures/overlay_{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(wp))

    # Composite grid
    cols = int(np.ceil(np.sqrt(args.n_grid)))
    rows = int(np.ceil(args.n_grid / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_2d(axes).flatten()

    for i, tp in enumerate(sample):
        arr = np.load(tp)
        # arr shape (H, W, 7); first 3 are B4(R), B3(G), B2(B)
        rgb = stretch_rgb(arr[..., [0, 1, 2]])
        # Label path
        lp = Path(str(tp).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        gt = load_labels(lp)
        # Predict on the (H, W, 7) directly via numpy
        # YOLO.predict expects either a path or image; for npy we feed a tensor.
        # Easiest: write a temp npy/tensor input.
        # Use model.predict with source as numpy array of shape (H,W,7); ultralytics will throw if not 3ch.
        # Instead: use model.model(...) on a preprocessed tensor.
        H, W = arr.shape[:2]
        x = torch.from_numpy(arr.transpose(2, 0, 1)[None].astype(np.float32))
        # Quick p01-p99 norm to match training; if dataset stats exist, use them, else use min/max.
        stats_path = root / "channel_stats_7ch.json"
        if stats_path.exists():
            import json
            s = json.loads(stats_path.read_text())
            p01 = np.array(s["percentiles"]["p01"], dtype=np.float32).reshape(1, -1, 1, 1)
            p99 = np.array(s["percentiles"]["p99"], dtype=np.float32).reshape(1, -1, 1, 1)
            xn = np.clip(arr.transpose(2, 0, 1)[None].astype(np.float32), p01, p99)
            xn = (xn - p01) / (p99 - p01 + 1e-6) * 255.0
            x = torch.from_numpy(xn)
        else:
            x = (x - x.min()) / (x.max() - x.min() + 1e-6) * 255.0
        x = x.to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            out = model.predict(x, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False, device=0 if torch.cuda.is_available() else "cpu")
        pred = out[0].boxes
        # pred.xyxy is in image pixels (already mapped back)
        ax = axes[i]
        ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(tp.name, fontsize=8)
        for cls, xn, yn, wn, hn in gt:
            draw_box_norm(ax, xn, yn, wn, hn, H, W,
                          color="lime", label=f"GT:{CLASS_NAMES[cls]}",
                          linestyle="--", linewidth=1.2)
        if pred is not None and len(pred) > 0:
            xyxy = pred.xyxy.cpu().numpy()
            conf = pred.conf.cpu().numpy()
            cls = pred.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cf, c in zip(xyxy, conf, cls):
                if c < len(CLASS_NAMES):
                    draw_box_xyxy(ax, x1, y1, x2, y2, color="red",
                                  label=f"{CLASS_NAMES[c]} {cf:.2f}",
                                  linestyle="-", linewidth=1.4)

    # Hide unused axes
    for j in range(len(sample), len(axes)):
        axes[j].axis("off")

    handles = [mpatches.Patch(facecolor="none", edgecolor="lime", linestyle="--", label="Ground truth"),
               mpatches.Patch(facecolor="none", edgecolor="red", linestyle="-", label="Prediction")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True)
    fig.suptitle(f"Detection overlay - {run_name}  (conf={args.conf}, {len(sample)} random tiles)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path = out_dir / "composite.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
