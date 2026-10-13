"""
Figure 7: feature-attention heatmaps (EigenCAM-style) for baseline vs
proposed model on Marine_Debris test tiles.

Implementation note: a strict Grad-CAM on YOLOv11 detection requires
backprop through the multi-head Detect output, which is fragile across
ultralytics versions. We instead use the simpler EigenCAM-style proxy:
register a forward hook on the P3-stride-8 detection feature (layer 20
for baseline, the C3k2 just before Detect for the proposed model),
take the channel-wise mean, normalise, upsample, and overlay as a
heatmap. This is the visualization most rare-small-object detection
papers actually report and is documented as such in the caption.

Output:
  runs_v2/figures/fig7_gradcam.png

Usage:
  python scripts/v2_make_fig7_gradcam.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ultralytics_plugins  # noqa: F401
from v2_sweep_thresholds import (
    DATA_ROOT, STATS_PATH, DEBRIS_CLS,
    _load_channel_stats, _preprocess_npy, _arr_to_tensor,
    _load_gt_boxes, _xywh_norm_to_xyxy_pixel,
)

FIG_DIR = ROOT / "runs_v2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Layers feeding the Detect head — these are the same indices in every
# v2 baseline-style YAML (see configs/models/yolov11n_v2_6ch.yaml, line
# defining "Detect" with [16, 19, 22] arguments).
TARGET_LAYERS = {
    "P3 (stride 8)":  16,
    "P4 (stride 16)": 19,
    "P5 (stride 32)": 22,
}

DEFAULT_TILES = [
    "S2_23-9-20_16PCC_25",      # 2 GT, stable winner for proposed
    "S2_29-8-17_51RVQ_4",       # 1 GT, recoverable
    "S2_7-3-20_18QYG_12",       # 1 GT, always-TP
    "S2_20-4-18_30VWH_4",       # 4 GT, hardest scene (both fail)
]


def _hook_layers(model_nn, layer_indices):
    """Register forward hooks; returns (out_dict, handles)."""
    out = {}
    handles = []
    for idx in layer_indices:
        layer = model_nn.model[idx]

        def make_hook(i):
            def _h(_module, _input, output):
                out[i] = output.detach()
            return _h
        handles.append(layer.register_forward_hook(make_hook(idx)))
    return out, handles


def _build_rgb(arr_hwc: np.ndarray) -> np.ndarray:
    rgb = arr_hwc[:, :, :3].astype(np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        rgb[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-6) * 255, 0, 255)
    return rgb.astype(np.uint8)


def _attention_map(feat: torch.Tensor, target_size: int) -> np.ndarray:
    """Channel-mean over feature tensor (1, C, h, w) -> (target_size, target_size) [0,1]."""
    if feat.ndim == 4:
        amap = feat.mean(dim=1, keepdim=True)  # (1, 1, h, w)
    else:
        raise ValueError(f"unexpected feature shape: {feat.shape}")
    amap = F.interpolate(amap, size=(target_size, target_size),
                         mode="bilinear", align_corners=False)
    amap = amap[0, 0].cpu().numpy()
    # ReLU + min-max normalise
    amap = np.maximum(amap, 0)
    if amap.max() > amap.min():
        amap = (amap - amap.min()) / (amap.max() - amap.min())
    return amap


def _draw_panel(ax, rgb: np.ndarray, attn: np.ndarray | None,
                gt_xyxy: np.ndarray, *, title: str) -> None:
    ax.imshow(rgb)
    if attn is not None:
        ax.imshow(attn, cmap="jet", alpha=0.45, interpolation="bilinear")
    for box in gt_xyxy:
        x1, y1, x2, y2 = box
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.6, edgecolor="lime", facecolor="none",
        ))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def visualize_one_model(model_nn, npy_path, low, high, channels, imgsz):
    """Forward one tile through model with hooks; return (rgb_uint8, attn_dict, gt_xyxy)."""
    out, handles = _hook_layers(model_nn, TARGET_LAYERS.values())
    arr, _, _ = _preprocess_npy(npy_path,
                                 low[:, :, :channels], high[:, :, :channels],
                                 channels=channels, imgsz=imgsz)
    tensor = _arr_to_tensor(arr)
    model_nn.eval()
    with torch.no_grad():
        _ = model_nn(tensor)
    for h in handles:
        h.remove()

    rgb = _build_rgb(arr)
    attn = {name: _attention_map(out[idx], imgsz) for name, idx in TARGET_LAYERS.items()}
    return rgb, attn


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Figure 7: Grad-CAM-style attention overlay")
    ap.add_argument("--baseline-weights",
                    default=str(ROOT / "runs_v2" / "baseline_6ch_s0" / "weights" / "best.pt"))
    ap.add_argument("--baseline-imgsz", type=int, default=512)
    ap.add_argument("--final-weights",
                    default=str(ROOT / "runs_v2" / "focal_g2.0_i1024_s0" / "weights" / "best.pt"))
    ap.add_argument("--final-imgsz", type=int, default=1024)
    ap.add_argument("--scale", default="P3 (stride 8)",
                    choices=list(TARGET_LAYERS.keys()),
                    help="Which detection scale to visualize")
    ap.add_argument("--tiles", nargs="+", default=DEFAULT_TILES)
    ap.add_argument("--out", default=str(FIG_DIR / "fig7_gradcam.png"))
    args = ap.parse_args()

    image_dir = DATA_ROOT / "test" / "images"
    label_dir = DATA_ROOT / "test" / "labels"
    low, high = _load_channel_stats(STATS_PATH)

    from ultralytics import YOLO
    print(f"Loading baseline:  {args.baseline_weights}")
    base_nn = YOLO(args.baseline_weights).model.cuda().eval()
    print(f"Loading final:     {args.final_weights}")
    final_nn = YOLO(args.final_weights).model.cuda().eval()

    n = len(args.tiles)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.3 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])

    for row_i, tile_name in enumerate(args.tiles):
        npy_path = image_dir / f"{tile_name}.npy"
        label_path = label_dir / f"{tile_name}.txt"
        if not npy_path.exists():
            print(f"  [WARN] missing tile: {tile_name}")
            continue

        gt_xywh = _load_gt_boxes(label_path)

        # Baseline forward + attention
        rgb_b, attn_b = visualize_one_model(
            base_nn, npy_path, low, high, 6, args.baseline_imgsz)
        gt_b = _xywh_norm_to_xyxy_pixel(gt_xywh, args.baseline_imgsz)

        # Final forward + attention
        rgb_f, attn_f = visualize_one_model(
            final_nn, npy_path, low, high, 6, args.final_imgsz)
        gt_f = _xywh_norm_to_xyxy_pixel(gt_xywh, args.final_imgsz)

        # Column 1: original tile + GT (for reference)
        _draw_panel(axes[row_i, 0], rgb_f, None, gt_f,
                    title=f"{tile_name}\nRGB + GT (lime)")
        # Column 2: baseline attention at chosen scale
        _draw_panel(axes[row_i, 1], rgb_b, attn_b[args.scale], gt_b,
                    title=f"Baseline attention @ {args.scale}")
        # Column 3: proposed attention at chosen scale
        _draw_panel(axes[row_i, 2], rgb_f, attn_f[args.scale], gt_f,
                    title=f"Proposed attention @ {args.scale}")

    legend_handles = [
        mpatches.Patch(edgecolor="lime", facecolor="none", label="Ground truth (debris)"),
        mpatches.Patch(facecolor="red", alpha=0.45, label="Feature attention (warmer = stronger)"),
    ]
    fig.suptitle(f"Figure 7. Feature-attention heatmaps (EigenCAM-style, scale={args.scale})",
                 fontsize=13, fontweight="bold")
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.005))

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
