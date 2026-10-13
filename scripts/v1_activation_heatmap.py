"""图 7 (Grad-CAM feature heatmap) for the v1 7ch BiFPN @ 1024 SOTA.

Uses EigenCAM (pytorch-grad-cam) on the deepest P3-stride layer (BiFPN-fused
feature map feeding the small-target detect head). EigenCAM is gradient-free
(SVD of activation map), so it works cleanly with YOLO's complex output without
class-conditional backprop through the detect head.

For 6 selected test tiles (chosen by largest n_tp from v1_test_per_tile.csv):
  col 1: RGB rendering (B4/B3/B2, p2-p98 stretch)
  col 2: RGB + EigenCAM heatmap overlay (P3-stride BiFPN feature)
  col 3: RGB + GT (green dashed) + Pred (red solid) overlay
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import zoom
from pytorch_grad_cam import EigenCAM

import ultralytics_plugins  # noqa: F401
from ultralytics_plugins.bifpn_modules import install_bifpn_patch

install_bifpn_patch()
from ultralytics import YOLO

WEIGHTS = "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_bifpn_imgsz1024_clean_init_s0/weights/best.pt"
DATA_YAML = "configs/datasets/marida_v1_physidx_on_v2split_7ch.yaml"
PER_TILE_CSV = ROOT / "docs/paper/v1_test_per_tile.csv"
OUT_FIG = ROOT / "docs/paper/figures/v1_activation_heatmap.png"
IMGSZ = 1024
CHANNELS = 7
TARGET_STRIDE = 8  # P3 head — best scale for small Marine_Debris
N_TILES = 6
CONF_THR = 0.25


def stretch_rgb(rgb: np.ndarray, low_pct: float = 2, high_pct: float = 98) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        lo = np.percentile(rgb[..., c], low_pct)
        hi = np.percentile(rgb[..., c], high_pct)
        out[..., c] = np.clip((rgb[..., c] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def load_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    out = []
    for L in label_path.read_text().splitlines():
        s = L.strip().split()
        if len(s) >= 5:
            out.append((int(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4])))
    return out


def preprocess_tensor(arr_hwc: np.ndarray, p01: np.ndarray, p99: np.ndarray) -> torch.Tensor:
    """Normalize + scale to 0..255 to match training pipeline."""
    x = arr_hwc.transpose(2, 0, 1)[None].astype(np.float32)
    x = np.clip(x, p01, p99)
    x = (x - p01) / (p99 - p01 + 1e-6) * 255.0
    return torch.from_numpy(x)


def find_p3_layer(model_nn):
    """Forward a synthetic (1, 7, IMGSZ, IMGSZ) tensor with hooks on every layer,
    return the deepest layer whose output spatial size matches IMGSZ/TARGET_STRIDE."""
    target_hw = IMGSZ // TARGET_STRIDE
    seen = []

    def make_hook(idx):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                seen.append((idx, tuple(out.shape)))
        return hook

    handles = []
    for i, layer in enumerate(model_nn.model):
        handles.append(layer.register_forward_hook(make_hook(i)))
    synth = torch.randn(1, CHANNELS, IMGSZ, IMGSZ).cuda()
    with torch.no_grad():
        _ = model_nn(synth)
    for h in handles:
        h.remove()

    candidates = [(i, sh) for i, sh in seen if sh[2] == target_hw and sh[3] == target_hw]
    if not candidates:
        print(f"[warn] no layer with H=W={target_hw}; falling back to deepest 4D layer")
        candidates = seen
    # Prefer DEEPEST layer at the target stride (i.e., max layer index), not max channels:
    # the deepest P3-stride layer is the one feeding the small-target detection head.
    best = max(candidates, key=lambda t: t[0])
    print(f"[hook] candidates at H=W={target_hw}:", [(i, sh) for i, sh in candidates])
    print(f"[hook] selected deepest layer index {best[0]} with shape {best[1]}")
    return best[0]


def main():
    df = pd.read_csv(PER_TILE_CSV)
    df = df[df["n_gt"] >= 1].sort_values(["n_tp", "n_gt"], ascending=False).head(N_TILES)
    selected_tiles = df["tile"].tolist()
    print(f"[select] {len(selected_tiles)} tiles: {selected_tiles}")

    cfg = yaml.safe_load(Path(DATA_YAML).read_text())
    dataset_root = Path(cfg["path"])
    stats_path = dataset_root / f"channel_stats_{CHANNELS}ch.json"
    stats = json.loads(stats_path.read_text())
    p01 = np.array(stats["percentiles"]["p01"], dtype=np.float32).reshape(1, -1, 1, 1)
    p99 = np.array(stats["percentiles"]["p99"], dtype=np.float32).reshape(1, -1, 1, 1)

    test_manifest = Path(cfg["test"])
    rel_paths = {Path(L.strip()).stem: L.strip()
                 for L in test_manifest.read_text().splitlines() if L.strip()}

    print(f"[load] {WEIGHTS}")
    yolo = YOLO(WEIGHTS)
    yolo.model.cuda().eval()
    model_nn = yolo.model

    # Auto-pick the deepest P3-stride layer
    target_layer_idx = find_p3_layer(model_nn)
    target_layer = model_nn.model[target_layer_idx]

    # YOLO's model.forward returns a tuple in eval mode; pytorch-grad-cam's
    # base CAM needs a single tensor to (a) auto-detect categories via argmax
    # and (b) invoke per-sample targets. EigenCAM doesn't actually USE either
    # for the heatmap (it does SVD on hooked activations), but we still need
    # to make the model output and target callable shapes parse cleanly.
    class _SingleTensorWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            if isinstance(out, (tuple, list)):
                # YOLOv11 returns (preds, feat_list); take preds tensor
                out = out[0]
                if isinstance(out, (tuple, list)):
                    out = out[0]
            return out

    class _DummyTarget:
        """Bypass per-sample target requirement — EigenCAM ignores its value."""
        def __call__(self, model_output):
            return model_output.sum() if isinstance(model_output, torch.Tensor) else torch.tensor(0.0)

    wrapped = _SingleTensorWrapper(model_nn)
    cam = EigenCAM(model=wrapped, target_layers=[target_layer])
    dummy_targets = [_DummyTarget()]

    fig, axes = plt.subplots(N_TILES, 3, figsize=(12, 3.7 * N_TILES), dpi=140)
    if N_TILES == 1:
        axes = np.array([axes])

    for row_idx, tile_name in enumerate(selected_tiles):
        rel = rel_paths[tile_name]
        npy_path = dataset_root / rel
        arr = np.load(npy_path)
        rgb = stretch_rgb(arr[..., [0, 1, 2]])
        H, W = arr.shape[:2]

        # Inference (for box predictions in col 3)
        x = preprocess_tensor(arr, p01, p99).cuda()
        with torch.no_grad():
            preds = yolo.predict(x, imgsz=IMGSZ, conf=CONF_THR, iou=0.6,
                                  verbose=False, device=0)[0]

        # EigenCAM heatmap: feed letterboxed (1024) input to match inference path
        x_up = F.interpolate(x, size=(IMGSZ, IMGSZ), mode="bilinear",
                              align_corners=False)
        # EigenCAM expects (B, H, W) input image but only uses tensor for forward;
        # targets=None lets it use the principal component of activations
        heat_imgsz = cam(input_tensor=x_up, targets=dummy_targets)[0]  # (IMGSZ, IMGSZ) in [0,1]
        # Downsample to native H×W for overlay on RGB
        heat_norm = zoom(heat_imgsz, (H / IMGSZ, W / IMGSZ), order=1)
        heat_norm = (heat_norm - heat_norm.min()) / (heat_norm.max() - heat_norm.min() + 1e-9)

        # GT + Pred for col 3 (preds.xyxy is in input-tensor-pixel space = imgsz scale)
        lp = Path(str(npy_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        gt = load_labels(lp)
        if preds.boxes is not None and len(preds.boxes) > 0:
            pred_xyxy = preds.boxes.xyxy.cpu().numpy()
            pred_cls = preds.boxes.cls.cpu().numpy().astype(int)
            pred_conf = preds.boxes.conf.cpu().numpy()
            # rescale from imgsz back to native HxW
            scale_x = W / IMGSZ
            scale_y = H / IMGSZ
            pred_xyxy_native = pred_xyxy.copy()
            pred_xyxy_native[:, [0, 2]] *= scale_x
            pred_xyxy_native[:, [1, 3]] *= scale_y
        else:
            pred_xyxy_native = np.empty((0, 4))
            pred_cls = np.array([], dtype=int)
            pred_conf = np.array([])

        # col 1: RGB
        ax = axes[row_idx, 0]
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{tile_name}\n(GT={int(df.iloc[row_idx]['n_gt'])}, "
                      f"TP={int(df.iloc[row_idx]['n_tp'])})", fontsize=8)
        if row_idx == 0:
            ax.set_title("(a) RGB (B4/B3/B2)", fontsize=11)

        # col 2: RGB + EigenCAM overlay
        ax = axes[row_idx, 1]
        ax.imshow(rgb)
        ax.imshow(heat_norm, alpha=0.55, cmap="jet")
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0:
            ax.set_title("(b) EigenCAM heatmap\n(P3-stride BiFPN feature)", fontsize=11)

        # col 3: RGB + boxes
        ax = axes[row_idx, 2]
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
        for cls, xn, yn, wn, hn in gt:
            if cls != 0:  # only Marine_Debris
                continue
            x1 = (xn - wn / 2) * W
            y1 = (yn - hn / 2) * H
            rect = mpatches.Rectangle((x1, y1), wn * W, hn * H, fill=False,
                                       edgecolor="lime", linewidth=1.8, linestyle="--")
            ax.add_patch(rect)
        for (x1, y1, x2, y2), c, cf in zip(pred_xyxy_native, pred_cls, pred_conf):
            if c != 0:  # only Marine_Debris
                continue
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor="red", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x1, max(y1 - 2, 0), f"{cf:.2f}",
                    color="red", fontsize=6,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.3))
        if row_idx == 0:
            ax.set_title("(c) GT (green dashed) vs Pred (red)", fontsize=11)

    handles_legend = [
        mpatches.Patch(facecolor="none", edgecolor="lime", linestyle="--",
                       label="GT Marine_Debris"),
        mpatches.Patch(facecolor="none", edgecolor="red", linestyle="-",
                       label="Predicted Marine_Debris (τ ≥ 0.25)"),
    ]
    fig.legend(handles=handles_legend, loc="lower center", ncol=2, frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Activation heatmap on representative test tiles "
                  "(BiFPN @ imgsz=1024, seed 0; Marine_Debris only)",
                  fontsize=12, fontweight="bold", y=0.997)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.015, 1, 0.985])
    plt.savefig(OUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"[save] {OUT_FIG}")


if __name__ == "__main__":
    main()
