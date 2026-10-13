"""
Threshold sweep + debris patch diagnostic for MARIDA v2.

NOT a re-evaluation — this asks a different question:
  "Does the model produce ANY debris predictions, even at conf=0.001?
   If yes, at what confidence? Do they overlap with GT boxes?"

Key outputs:
  1. Per-GT-instance table: max IoU with any prediction, conf of best match
  2. Threshold sweep table: debris TP/total at [0.001..0.5]
  3. PNG visualizations of all 9 debris test tiles (GT=red, preds=blue)

Interpretation:
  - If max_iou > 0.5 but conf < 0.25 → threshold tuning or focal loss tweak
  - If max_iou < 0.1 for ALL GT instances → model never activates on debris
  - If predictions cluster near GT but miss center → stride/anchor problem

Usage:
  python scripts/v2_sweep_thresholds.py \\
      --weights runs_v2/baseline_6ch_s0/weights/best.pt \\
      --tag baseline_s0

  python scripts/v2_sweep_thresholds.py \\
      --weights runs_v2/p2head_s0/weights/best.pt --tag p2head_s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATA_ROOT    = ROOT / "data" / "processed" / "marida_v2"
STATS_PATH   = DATA_ROOT / "channel_stats_6ch.json"
DEBRIS_CLS   = 0
IMGSZ        = 512
CLASS_NAMES  = ["Marine_Debris", "Algae_Organic", "Ship"]

# Threshold sweep points
CONF_SWEEP   = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
# IoU thresholds for "match" determination
IOU_LOOSE    = 0.10   # anything overlapping
IOU_STRICT   = 0.50   # COCO-standard TP


# ---------------------------------------------------------------------------
# Preprocessing (replicates MaskRescueNPYDataset._load_array + letterbox)
# ---------------------------------------------------------------------------

def _load_channel_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (low, high) arrays of shape (1,1,6) for clipping+scaling."""
    payload = json.loads(stats_path.read_text())
    p01 = np.array(payload["percentiles"]["p01"], dtype=np.float32).reshape(1, 1, -1)
    p99 = np.array(payload["percentiles"]["p99"], dtype=np.float32).reshape(1, 1, -1)
    return p01, p99


def _preprocess_npy(npy_path: Path, low: np.ndarray, high: np.ndarray,
                    channels: int = 6, imgsz: int = IMGSZ) -> np.ndarray:
    """Load npy → normalize → letterbox → float32 [0,255] (H,W,C)."""
    arr = np.load(npy_path, allow_pickle=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D array, got {arr.shape}")
    # Transpose (C,H,W) → (H,W,C) if needed
    if arr.shape[0] in (6, 7) and arr.shape[0] <= arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)
    arr = arr[..., :channels].astype(np.float32)

    # Channel stats normalization
    span = np.maximum(high - low, 1e-6)
    arr = np.clip(arr, low, high)
    arr = (arr - low) / span * 255.0

    # Fix non-finite values
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 255.0)

    # Letterbox to imgsz × imgsz (square, no distortion)
    h0, w0 = arr.shape[:2]
    scale = imgsz / max(h0, w0)
    new_h, new_w = int(round(h0 * scale)), int(round(w0 * scale))
    if (h0, w0) != (new_h, new_w):
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Pad to imgsz × imgsz
    pad_h = imgsz - new_h
    pad_w = imgsz - new_w
    top, left = pad_h // 2, pad_w // 2
    arr = np.pad(arr,
                 ((top, pad_h - top), (left, pad_w - left), (0, 0)),
                 constant_values=114.0)
    return arr.astype(np.float32), (h0, w0), (top, left, new_h, new_w)


def _arr_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H,W,C) float32 → (1,C,H,W) float32 CUDA tensor, scaled to [0,1]."""
    t = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
    return t.unsqueeze(0).cuda()


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def _load_gt_boxes(label_path: Path) -> np.ndarray:
    """Return (N,4) normalized cx,cy,w,h for class DEBRIS_CLS."""
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if int(parts[0]) == DEBRIS_CLS:
                boxes.append([float(x) for x in parts[1:5]])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def _xywh_norm_to_xyxy_pixel(boxes_xywh: np.ndarray, imgsz: int = IMGSZ) -> np.ndarray:
    """Convert normalized cx,cy,w,h → pixel x1,y1,x2,y2."""
    if len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    b = boxes_xywh.copy()
    out = np.zeros_like(b)
    out[:, 0] = (b[:, 0] - b[:, 2] / 2) * imgsz
    out[:, 1] = (b[:, 1] - b[:, 3] / 2) * imgsz
    out[:, 2] = (b[:, 0] + b[:, 2] / 2) * imgsz
    out[:, 3] = (b[:, 1] + b[:, 3] / 2) * imgsz
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU for all pairs. a: (M,4), b: (N,4) in xyxy pixel format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    iou = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, ba in enumerate(a):
        for j, bb in enumerate(b):
            x1 = max(ba[0], bb[0]); y1 = max(ba[1], bb[1])
            x2 = min(ba[2], bb[2]); y2 = min(ba[3], bb[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = (ba[2] - ba[0]) * (ba[3] - ba[1])
            area_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
            union = area_a + area_b - inter
            iou[i, j] = inter / (union + 1e-7)
    return iou


# ---------------------------------------------------------------------------
# Raw model predict (bypass YOLO's full pipeline)
# ---------------------------------------------------------------------------

def _raw_predict(model_nn, tensor: torch.Tensor, conf_thr: float = 0.001,
                 iou_thr: float = 0.45, nc: int = 3):
    """
    Run model forward and decode predictions.
    Returns np.ndarray of shape (K, 6): [x1, y1, x2, y2, conf, cls]
    """
    from ultralytics.utils.nms import non_max_suppression
    model_nn.eval()
    with torch.no_grad():
        raw = model_nn(tensor)

    # raw is a list of tensors in training mode, or (preds, protos) in eval
    if isinstance(raw, (list, tuple)):
        preds = raw[0] if isinstance(raw[0], torch.Tensor) and raw[0].ndim == 3 else raw
    else:
        preds = raw

    if isinstance(preds, (list, tuple)):
        # Some YOLO variants return (decoded, raw_feats) — take decoded
        preds = preds[0]

    results = non_max_suppression(
        preds, conf_thres=conf_thr, iou_thres=iou_thr, nc=nc, max_det=300
    )
    if results and results[0] is not None and len(results[0]):
        return results[0].cpu().numpy()  # (K, 6): x1y1x2y2 conf cls
    return np.zeros((0, 6), dtype=np.float32)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_tile_png(arr_hwc: np.ndarray, gt_xyxy: np.ndarray,
                   preds_xyxy: np.ndarray, out_path: Path,
                   title: str = "") -> None:
    """
    Save RGB-composite (channels 0,1,2 = B4,B3,B2) with GT (red) and preds (blue).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [viz] matplotlib not available — skipping PNG save")
        return

    # Build RGB image from first 3 channels (already 0-255)
    rgb = arr_hwc[:, :, :3].astype(np.float32)
    # Stretch each channel independently for visibility
    for c in range(3):
        ch = rgb[:, :, c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        rgb[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-6) * 255, 0, 255)
    rgb = rgb.astype(np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=9)

    # GT boxes — red
    for box in gt_xyxy:
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Predictions — blue, opacity by confidence
    for pred in preds_xyxy:
        x1, y1, x2, y2, conf, cls = pred
        if int(cls) != DEBRIS_CLS:
            continue
        alpha = min(1.0, 0.3 + conf * 2)
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1.0, edgecolor='blue',
                                   facecolor='none', alpha=alpha, linestyle='--')
        ax.add_patch(rect)
        ax.text(x1, y1 - 2, f"{conf:.3f}", color='cyan', fontsize=6)

    handles = [
        mpatches.Patch(edgecolor='red', facecolor='none', label='GT debris'),
        mpatches.Patch(edgecolor='blue', facecolor='none', linestyle='--',
                       label='Pred debris (conf)'),
    ]
    ax.legend(handles=handles, fontsize=7, loc='upper right')
    ax.axis('off')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic(weights: str, tag: str, channels: int = 6,
                   imgsz: int = IMGSZ, save_viz: bool = True,
                   split: str = "test") -> dict:
    """
    Full diagnostic pass:
      1. Load model
      2. For each debris tile in `split`: preprocess, predict at conf=0.001, match to GT
      3. Print per-instance table and threshold sweep summary
      4. Save PNG visualizations
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    label_dir = DATA_ROOT / split / "labels"
    image_dir = DATA_ROOT / split / "images"

    print(f"\n{'='*65}")
    print(f"Threshold sweep diagnostic: {tag}  [split={split}]")
    print(f"  Weights: {weights}")
    print(f"{'='*65}")

    print(f"  imgsz: {imgsz}")

    # Load channel stats
    low, high = _load_channel_stats(STATS_PATH)
    low_6 = low[:, :, :channels]
    high_6 = high[:, :, :channels]

    # Load model
    yolo = YOLO(weights)
    nn_model = yolo.model.cuda().eval()
    nc = nn_model.yaml.get("nc", 3)

    # Find debris test tiles
    debris_tiles = sorted([
        lf for lf in label_dir.glob("*.txt")
        if any(int(l.split()[0]) == DEBRIS_CLS
               for l in lf.read_text().splitlines() if l.strip())
    ])
    print(f"\nFound {len(debris_tiles)} debris {split} tiles\n")

    # --- Per-instance analysis ---
    out_dir = ROOT / "runs_v2" / f"sweep_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_gt_count = 0
    # For each threshold: [TP_loose, TP_strict, total_preds_debris]
    thresh_counts = {c: {"tp_loose": 0, "tp_strict": 0, "n_preds": 0}
                     for c in CONF_SWEEP}

    instance_rows = []  # (tile_name, gt_idx, gt_box, max_iou, best_conf)

    print(f"{'Tile':40s}  {'GT#':4s}  {'GT_wh(px)':12s}  {'MaxIoU':8s}  {'BestConf':8s}  {'Status'}")
    print("-" * 90)

    for label_path in debris_tiles:
        stem = label_path.stem
        npy_path = image_dir / f"{stem}.npy"
        if not npy_path.exists():
            print(f"  [WARN] No image for {stem}, skipping")
            continue

        # Preprocess
        arr, (h0, w0), (pad_top, pad_left, new_h, new_w) = _preprocess_npy(
            npy_path, low_6, high_6, channels=channels, imgsz=imgsz
        )
        tensor = _arr_to_tensor(arr)

        # Predict at lowest threshold
        preds = _raw_predict(nn_model, tensor, conf_thr=0.001, iou_thr=0.45, nc=nc)
        # preds: (K, 6) — x1,y1,x2,y2 in pixel coords of the 512×512 input

        # GT boxes
        gt_xywh = _load_gt_boxes(label_path)
        gt_xyxy = _xywh_norm_to_xyxy_pixel(gt_xywh, imgsz)
        all_gt_count += len(gt_xyxy)

        # Match each GT box to best prediction
        # Only debris predictions for matching
        debris_preds = preds[preds[:, 5].astype(int) == DEBRIS_CLS] if len(preds) else preds

        if len(gt_xyxy) > 0 and len(debris_preds) > 0:
            iou_mat = _iou_matrix(gt_xyxy, debris_preds[:, :4])
        else:
            iou_mat = np.zeros((len(gt_xyxy), len(debris_preds)))

        for gi, gt_box in enumerate(gt_xyxy):
            gt_w_px = (gt_box[2] - gt_box[0])
            gt_h_px = (gt_box[3] - gt_box[1])

            if len(debris_preds) > 0:
                best_j = np.argmax(iou_mat[gi])
                max_iou = float(iou_mat[gi, best_j])
                best_conf = float(debris_preds[best_j, 4])
            else:
                max_iou = 0.0
                best_conf = 0.0

            status = (
                "TP@0.5" if max_iou >= IOU_STRICT else
                "weak@0.1" if max_iou >= IOU_LOOSE else
                "MISS"
            )
            print(f"{stem:40s}  {gi:4d}  {gt_w_px:5.1f}x{gt_h_px:<5.1f}  "
                  f"{max_iou:8.4f}  {best_conf:8.4f}  {status}")
            instance_rows.append({
                "tile": stem, "gt_idx": gi,
                "gt_w_px": float(gt_w_px), "gt_h_px": float(gt_h_px),
                "max_iou": max_iou, "best_conf": best_conf, "status": status
            })

        # Threshold sweep counts
        for c_thr in CONF_SWEEP:
            debris_above = debris_preds[debris_preds[:, 4] >= c_thr] if len(debris_preds) else debris_preds
            thresh_counts[c_thr]["n_preds"] += len(debris_above)
            if len(gt_xyxy) > 0 and len(debris_above) > 0:
                iou_c = _iou_matrix(gt_xyxy, debris_above[:, :4])
                matched_loose = (iou_c.max(axis=1) >= IOU_LOOSE).sum()
                matched_strict = (iou_c.max(axis=1) >= IOU_STRICT).sum()
            else:
                matched_loose = 0
                matched_strict = 0
            thresh_counts[c_thr]["tp_loose"] += int(matched_loose)
            thresh_counts[c_thr]["tp_strict"] += int(matched_strict)

        # Visualization
        if save_viz:
            out_png = out_dir / f"{stem}.png"
            _save_tile_png(
                arr_hwc=arr,
                gt_xyxy=gt_xyxy,
                preds_xyxy=preds,
                out_path=out_png,
                title=f"{stem}  |  {len(gt_xyxy)} GT debris  |  {len(debris_preds)} debris preds"
            )

    # --- Scene-level aggregation ---
    from collections import defaultdict
    scene_stats = defaultdict(lambda: {"n_gt": 0, "n_tp_strict": 0, "n_tp_loose": 0,
                                       "confs": [], "ious": []})
    for row in instance_rows:
        scene = row["tile"].rsplit("_", 1)[0]
        scene_stats[scene]["n_gt"] += 1
        if row["status"] == "TP@0.5":
            scene_stats[scene]["n_tp_strict"] += 1
        if row["status"] in ("TP@0.5", "weak@0.1"):
            scene_stats[scene]["n_tp_loose"] += 1
        scene_stats[scene]["confs"].append(row["best_conf"])
        scene_stats[scene]["ious"].append(row["max_iou"])

    print(f"\n{'='*65}")
    print(f"Scene-Level Breakdown  ({len(scene_stats)} scenes)")
    print(f"{'='*65}")
    print(f"{'Scene':40s}  {'GT':>3s}  {'TP@0.5':>6s}  {'TP@0.1':>6s}  {'MeanConf':>8s}  {'MeanIoU':>8s}")
    print("-" * 80)
    for scene in sorted(scene_stats):
        s = scene_stats[scene]
        mc = float(np.mean(s["confs"])) if s["confs"] else 0.0
        mi = float(np.mean(s["ious"])) if s["ious"] else 0.0
        print(f"{scene:40s}  {s['n_gt']:3d}  {s['n_tp_strict']:6d}  {s['n_tp_loose']:6d}  {mc:8.4f}  {mi:8.4f}")

    # --- Threshold sweep table ---
    print(f"\n{'='*65}")
    print(f"Threshold Sweep  ({all_gt_count} GT debris instances)")
    print(f"{'='*65}")
    print(f"{'conf_thr':>10s}  {'n_preds':>8s}  {'TP(iou>0.1)':>12s}  "
          f"{'TP(iou>0.5)':>12s}  {'R@0.5':>7s}")
    print("-" * 65)
    for c_thr in CONF_SWEEP:
        tc = thresh_counts[c_thr]
        r50 = tc["tp_strict"] / max(all_gt_count, 1)
        print(f"{c_thr:10.3f}  {tc['n_preds']:8d}  "
              f"{tc['tp_loose']:12d}  {tc['tp_strict']:12d}  {r50:7.4f}")

    # --- Summary verdict ---
    best_tp_strict = max(v["tp_strict"] for v in thresh_counts.values())
    best_tp_loose  = max(v["tp_loose"]  for v in thresh_counts.values())

    print(f"\n{'='*65}")
    print(f"VERDICT for {tag}")
    print(f"{'='*65}")
    if best_tp_strict > 0:
        print(f"  >> IoU>0.5 TPs found at some threshold: max={best_tp_strict}/{all_gt_count}")
        print("     Hypothesis: threshold / confidence calibration issue")
        print("     Next: focal loss / class weight tuning OR score calibration")
    elif best_tp_loose > 0:
        print(f"  >> Predictions exist NEAR debris boxes (IoU 0.1-0.5): max={best_tp_loose}")
        print("     Model partially localises debris but can't tighten the box")
        print("     Next: larger input resolution (imgsz=1024) to improve localization")
    else:
        print("  >> NO predictions anywhere near GT debris boxes (iou < 0.1)")
        print("     Model never activates on debris pixels at any threshold")
        print("     Next: re-examine feature maps / consider segmentation approach")

    # Save JSON
    scene_stats_json = {
        scene: {
            "n_gt": s["n_gt"],
            "n_tp_strict": s["n_tp_strict"],
            "n_tp_loose": s["n_tp_loose"],
            "mean_conf": float(np.mean(s["confs"])) if s["confs"] else 0.0,
            "mean_iou": float(np.mean(s["ious"])) if s["ious"] else 0.0,
        }
        for scene, s in sorted(scene_stats.items())
    }
    results = {
        "tag": tag, "weights": weights, "split": split,
        "n_debris_tiles": len(debris_tiles),
        "n_gt_instances": all_gt_count,
        "instance_rows": instance_rows,
        "scene_stats": scene_stats_json,
        "threshold_sweep": {
            str(c): thresh_counts[c] for c in CONF_SWEEP
        },
        "verdict": {
            "best_tp_strict": best_tp_strict,
            "best_tp_loose":  best_tp_loose,
        }
    }
    out_json = out_dir / "sweep_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n  JSON: {out_json}")
    if save_viz:
        print(f"  PNGs: {out_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="MARIDA v2 threshold sweep + debris patch diagnostic")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to best.pt weights")
    parser.add_argument("--tag", type=str, default=None,
                        help="Label for this run (defaults to parent dir name)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip PNG visualization (faster)")
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Inference image size (must match training imgsz)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Data split to diagnose (default: test)")
    args = parser.parse_args()

    tag = args.tag or Path(args.weights).parent.parent.name
    KMP_fix = "KMP_DUPLICATE_LIB_OK"
    import os
    os.environ[KMP_fix] = "TRUE"

    run_diagnostic(
        weights=args.weights,
        tag=tag,
        channels=args.channels,
        imgsz=args.imgsz,
        save_viz=not args.no_viz,
        split=args.split,
    )


if __name__ == "__main__":
    main()
