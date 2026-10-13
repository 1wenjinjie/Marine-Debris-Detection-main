"""
Post-hoc Platt scaling calibration for MARIDA v2 debris detector.

Fits sigmoid(a * logit(conf) + b) on Marine_Debris val-set detections,
applies it to test predictions, and re-runs the debris threshold sweep.

The calibration is post-hoc and bounded: no retraining, no new model.
Success criterion: TP@0.05 (test) improves over the uncalibrated sweep.

Usage:
  # Single model
  python scripts/v2_calibrate.py \\
      --weights runs_v2/focal_g2.0_i1024_s0/weights/best.pt \\
      --tag focal_g2.0_i1024_s0 --imgsz 1024

  # All 3 seeds (pattern: focal_g2.0_i1024_s{seed})
  python scripts/v2_calibrate.py --seeds 0 1 2 --gamma 2.0 --imgsz 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import helpers from the sweep script (no duplication)
from v2_sweep_thresholds import (
    _arr_to_tensor,
    _iou_matrix,
    _load_channel_stats,
    _load_gt_boxes,
    _preprocess_npy,
    _raw_predict,
    _xywh_norm_to_xyxy_pixel,
)

DATA_ROOT  = ROOT / "data" / "processed" / "marida_v2"
STATS_PATH = DATA_ROOT / "channel_stats_6ch.json"
DEBRIS_CLS = 0
CONF_SWEEP = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
IOU_STRICT = 0.50


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

def _logit_safe(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 1e-7, 1.0 - 1e-7)
    return np.log(c / (1.0 - c))


def fit_platt(confs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Fit sigmoid(a*logit(c) + b) via NLL minimization.
    Returns (a, b) — identity is (1.0, 0.0).
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    print(f"  Platt fit: {n_pos} TP, {n_neg} FP (total {len(labels)} debris predictions)")

    if n_pos == 0 or n_neg == 0:
        print("  [WARN] Only one class present in calibration data — returning identity")
        return 1.0, 0.0

    x = _logit_safe(confs)
    y = labels.astype(np.float64)

    def nll(params: np.ndarray) -> float:
        a, b = params
        p = expit(a * x + b)
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    result = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10000})
    a, b = float(result.x[0]), float(result.x[1])
    print(f"  Platt params: a={a:.4f}, b={b:.4f}  (NLL={result.fun:.4f})")
    return a, b


def apply_platt(confs: np.ndarray, a: float, b: float) -> np.ndarray:
    from scipy.special import expit
    return expit(a * _logit_safe(confs) + b).astype(np.float32)


# ---------------------------------------------------------------------------
# Val-set collection
# ---------------------------------------------------------------------------

def collect_val_detections(
    nn_model, low: np.ndarray, high: np.ndarray,
    channels: int, imgsz: int, nc: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run prediction on all val tiles.
    Returns (confs, labels) for Marine_Debris predictions only.
    labels[i] = 1 if IoU > 0.5 with any GT debris box, else 0.
    """
    val_dir   = DATA_ROOT / "val"
    label_dir = val_dir / "labels"
    image_dir = val_dir / "images"

    all_confs:  list[float] = []
    all_labels: list[int]   = []

    npy_paths = sorted(image_dir.glob("*.npy"))
    print(f"  Collecting val detections from {len(npy_paths)} tiles ...")

    for npy_path in npy_paths:
        stem = npy_path.stem
        label_path = label_dir / f"{stem}.txt"

        arr, _, _ = _preprocess_npy(npy_path, low, high, channels=channels, imgsz=imgsz)
        tensor = _arr_to_tensor(arr)
        preds = _raw_predict(nn_model, tensor, conf_thr=0.001, nc=nc)

        debris_preds = preds[preds[:, 5].astype(int) == DEBRIS_CLS] if len(preds) else preds
        if len(debris_preds) == 0:
            continue

        gt_xywh = _load_gt_boxes(label_path)
        gt_xyxy = _xywh_norm_to_xyxy_pixel(gt_xywh, imgsz)

        if len(gt_xyxy) > 0:
            iou_mat = _iou_matrix(gt_xyxy, debris_preds[:, :4])
            for j in range(len(debris_preds)):
                max_iou = float(iou_mat[:, j].max())
                all_confs.append(float(debris_preds[j, 4]))
                all_labels.append(1 if max_iou >= IOU_STRICT else 0)
        else:
            for j in range(len(debris_preds)):
                all_confs.append(float(debris_preds[j, 4]))
                all_labels.append(0)

    return np.array(all_confs, dtype=np.float32), np.array(all_labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Calibrated test sweep
# ---------------------------------------------------------------------------

def calibrated_sweep(
    nn_model, a: float, b: float,
    low: np.ndarray, high: np.ndarray,
    channels: int, imgsz: int, nc: int,
    tag: str,
) -> tuple[dict, dict]:
    """
    Run test sweep with original and calibrated confs side by side.
    Returns (uncal_counts, cal_counts) — same structure as v2_sweep_thresholds.
    """
    test_dir  = DATA_ROOT / "test"
    label_dir = test_dir / "labels"
    image_dir = test_dir / "images"

    debris_tiles = sorted([
        lf for lf in label_dir.glob("*.txt")
        if any(int(l.split()[0]) == DEBRIS_CLS
               for l in lf.read_text().splitlines() if l.strip())
    ])

    uncal = {c: {"tp_strict": 0, "n_preds": 0} for c in CONF_SWEEP}
    cal   = {c: {"tp_strict": 0, "n_preds": 0} for c in CONF_SWEEP}
    n_gt  = 0

    instance_rows = []

    for label_path in debris_tiles:
        stem = label_path.stem
        npy_path = image_dir / f"{stem}.npy"
        if not npy_path.exists():
            continue

        arr, _, _ = _preprocess_npy(npy_path, low, high, channels=channels, imgsz=imgsz)
        tensor = _arr_to_tensor(arr)
        preds = _raw_predict(nn_model, tensor, conf_thr=0.001, nc=nc)

        gt_xywh = _load_gt_boxes(label_path)
        gt_xyxy = _xywh_norm_to_xyxy_pixel(gt_xywh, imgsz)
        n_gt += len(gt_xyxy)

        debris_preds = preds[preds[:, 5].astype(int) == DEBRIS_CLS] if len(preds) else preds

        if len(debris_preds) == 0:
            for gi in range(len(gt_xyxy)):
                instance_rows.append({"tile": stem, "gt_idx": gi,
                                      "orig_conf": 0.0, "cal_conf": 0.0,
                                      "max_iou": 0.0})
            continue

        # Calibrate confs
        orig_confs = debris_preds[:, 4].copy()
        cal_confs  = apply_platt(orig_confs, a, b)

        # Per-GT-instance: find best match (original IoU stays the same)
        iou_mat = _iou_matrix(gt_xyxy, debris_preds[:, :4]) if len(gt_xyxy) > 0 else np.zeros((0, len(debris_preds)))
        for gi in range(len(gt_xyxy)):
            if len(debris_preds) > 0:
                best_j   = int(np.argmax(iou_mat[gi]))
                max_iou  = float(iou_mat[gi, best_j])
                o_conf   = float(orig_confs[best_j])
                c_conf   = float(cal_confs[best_j])
            else:
                max_iou = o_conf = c_conf = 0.0
            instance_rows.append({"tile": stem, "gt_idx": gi,
                                   "orig_conf": o_conf, "cal_conf": c_conf,
                                   "max_iou": max_iou})

        # Threshold sweep
        for c_thr in CONF_SWEEP:
            # Uncalibrated
            above = debris_preds[orig_confs >= c_thr]
            uncal[c_thr]["n_preds"] += len(above)
            if len(above) > 0 and len(gt_xyxy) > 0:
                iou_c = _iou_matrix(gt_xyxy, above[:, :4])
                uncal[c_thr]["tp_strict"] += int((iou_c.max(axis=1) >= IOU_STRICT).sum())

            # Calibrated
            above_cal = debris_preds[cal_confs >= c_thr]
            cal[c_thr]["n_preds"] += len(above_cal)
            if len(above_cal) > 0 and len(gt_xyxy) > 0:
                iou_cc = _iou_matrix(gt_xyxy, above_cal[:, :4])
                cal[c_thr]["tp_strict"] += int((iou_cc.max(axis=1) >= IOU_STRICT).sum())

    # Print per-instance table
    print(f"\n{'Tile':40s}  GT#  orig_conf  cal_conf   IoU     status(orig→cal)")
    print("-" * 90)
    for row in instance_rows:
        iou = row["max_iou"]
        def _status(c, iou):
            if iou < 0.5: return "MISS"
            return "TP" if c >= 0.05 else f"low({c:.3f})"
        os_ = _status(row["orig_conf"], iou)
        cs_ = _status(row["cal_conf"],  iou)
        flag = " <=" if os_ != cs_ else ""
        print(f"{row['tile']:40s}  {row['gt_idx']:3d}  "
              f"{row['orig_conf']:9.4f}  {row['cal_conf']:9.4f}  "
              f"{iou:6.4f}  {os_} → {cs_}{flag}")

    # Print sweep comparison
    print(f"\n{'='*65}")
    print(f"Calibrated Threshold Sweep — {tag}  ({n_gt} GT debris instances)")
    print(f"{'='*65}")
    print(f"{'conf':>8}  {'orig TP':>8}  {'cal TP':>8}  {'delta':>7}  {'orig R':>7}  {'cal R':>7}")
    print("-" * 65)
    for c_thr in CONF_SWEEP:
        u = uncal[c_thr]["tp_strict"]
        c = cal[c_thr]["tp_strict"]
        delta = c - u
        r_u = u / max(n_gt, 1)
        r_c = c / max(n_gt, 1)
        mark = " <<" if delta > 0 else (" >>" if delta < 0 else "")
        print(f"{c_thr:8.3f}  {u:8d}  {c:8d}  {delta:+7d}  {r_u:7.4f}  {r_c:7.4f}{mark}")

    return uncal, cal, instance_rows, n_gt


# ---------------------------------------------------------------------------
# Per-model entry point
# ---------------------------------------------------------------------------

def calibrate_one(weights: str, tag: str, channels: int = 6, imgsz: int = 1024) -> dict:
    from ultralytics import YOLO

    print(f"\n{'='*65}")
    print(f"Calibration: {tag}")
    print(f"  weights: {weights}")
    print(f"  imgsz:   {imgsz}")
    print(f"{'='*65}")

    low, high = _load_channel_stats(STATS_PATH)
    low  = low[:, :, :channels]
    high = high[:, :, :channels]

    yolo     = YOLO(weights)
    nn_model = yolo.model.cuda().eval()
    nc       = nn_model.yaml.get("nc", 3)

    # Step 1: collect val detections
    confs, labels = collect_val_detections(nn_model, low, high, channels, imgsz, nc)

    # Step 2: fit Platt scaling
    a, b = fit_platt(confs, labels)

    # Quick sanity: effect on val TP confs
    tp_confs = confs[labels == 1]
    if len(tp_confs) > 0:
        cal_tp = apply_platt(tp_confs, a, b)
        print(f"  Val TP conf:  orig {tp_confs.mean():.4f} ± {tp_confs.std():.4f}  →  "
              f"cal {cal_tp.mean():.4f} ± {cal_tp.std():.4f}")
        print(f"  Val TP conf @ orig thrs: "
              f">=0.05: {(tp_confs >= 0.05).sum()}/{len(tp_confs)}  "
              f"→ cal >=0.05: {(cal_tp >= 0.05).sum()}/{len(tp_confs)}")

    # Step 3: calibrated test sweep
    uncal, cal, instance_rows, n_gt = calibrated_sweep(
        nn_model, a, b, low, high, channels, imgsz, nc, tag
    )

    # Verdict
    orig_tp05 = uncal[0.05]["tp_strict"]
    cal_tp05  = cal[0.05]["tp_strict"]
    orig_tp025 = uncal[0.025]["tp_strict"]
    cal_tp025  = cal[0.025]["tp_strict"]

    print(f"\n{'='*65}")
    print(f"VERDICT — {tag}")
    print(f"{'='*65}")
    print(f"  conf=0.05:  {orig_tp05} → {cal_tp05}  ({'improvement' if cal_tp05 > orig_tp05 else 'no improvement' if cal_tp05 == orig_tp05 else 'regression'})")
    print(f"  conf=0.025: {orig_tp025} → {cal_tp025}  ({'improvement' if cal_tp025 > orig_tp025 else 'no improvement' if cal_tp025 == orig_tp025 else 'regression'})")

    result = {
        "tag":      tag,
        "weights":  weights,
        "imgsz":    imgsz,
        "platt_a":  a,
        "platt_b":  b,
        "n_val_tp": int((labels == 1).sum()),
        "n_val_fp": int((labels == 0).sum()),
        "threshold_sweep": {
            str(c): {
                "orig_tp": uncal[c]["tp_strict"],
                "cal_tp":  cal[c]["tp_strict"],
                "delta":   cal[c]["tp_strict"] - uncal[c]["tp_strict"],
            }
            for c in CONF_SWEEP
        },
        "instance_rows": instance_rows,
    }

    out_dir  = ROOT / "runs_v2" / f"calib_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Saved: {out_path}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARIDA v2 post-hoc Platt calibration")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--weights", type=str,
                     help="Direct path to best.pt (single model)")
    grp.add_argument("--seeds", type=int, nargs="+",
                     help="Run multiple seeds with auto-resolved pattern")
    parser.add_argument("--tag",    type=str, default=None,
                        help="Run tag (used with --weights)")
    parser.add_argument("--gamma",  type=float, default=2.0,
                        help="Focal gamma (for pattern with --seeds)")
    parser.add_argument("--alpha",  type=float, default=0.25,
                        help="Focal alpha (for pattern with --seeds)")
    parser.add_argument("--imgsz",  type=int,   default=1024)
    parser.add_argument("--channels", type=int, default=6)
    args = parser.parse_args()

    RUNS_DIR = ROOT / "runs_v2"

    if args.weights:
        tag = args.tag or Path(args.weights).parent.parent.name
        calibrate_one(args.weights, tag, args.channels, args.imgsz)
        return

    # Multi-seed
    alpha_tag = f"_a{args.alpha}" if args.alpha != 0.25 else ""
    results = []
    for seed in args.seeds:
        run_name = f"focal_g{args.gamma}{alpha_tag}_i{args.imgsz}_s{seed}"
        weights  = str(RUNS_DIR / run_name / "weights" / "best.pt")
        if not Path(weights).exists():
            print(f"[SKIP] Weights not found: {weights}")
            continue
        r = calibrate_one(weights, run_name, args.channels, args.imgsz)
        results.append(r)

    if len(results) < 2:
        return

    # Multi-seed summary
    print(f"\n{'='*65}")
    print(f"MULTI-SEED CALIBRATION SUMMARY  ({len(results)} seeds)")
    print(f"  gamma={args.gamma}, alpha={args.alpha}, imgsz={args.imgsz}")
    print(f"{'='*65}")
    print(f"{'conf':>8}  {'orig mean':>10}  {'cal mean':>10}  {'delta mean':>12}")
    print("-" * 50)
    for c_thr in CONF_SWEEP:
        key = str(c_thr)
        orig_vals = [r["threshold_sweep"][key]["orig_tp"] for r in results]
        cal_vals  = [r["threshold_sweep"][key]["cal_tp"]  for r in results]
        delta_mean = np.mean(cal_vals) - np.mean(orig_vals)
        print(f"{c_thr:8.3f}  "
              f"{np.mean(orig_vals):7.1f} ± {np.std(orig_vals):.1f}  "
              f"{np.mean(cal_vals):7.1f} ± {np.std(cal_vals):.1f}  "
              f"{delta_mean:+12.1f}")

    summary = {
        "experiment": "v2_calibration",
        "focal_gamma": args.gamma,
        "focal_alpha": args.alpha,
        "imgsz": args.imgsz,
        "seeds": args.seeds,
        "results": results,
        "summary": {
            str(c): {
                "orig_mean": float(np.mean([r["threshold_sweep"][str(c)]["orig_tp"] for r in results])),
                "orig_std":  float(np.std( [r["threshold_sweep"][str(c)]["orig_tp"] for r in results])),
                "cal_mean":  float(np.mean([r["threshold_sweep"][str(c)]["cal_tp"]  for r in results])),
                "cal_std":   float(np.std( [r["threshold_sweep"][str(c)]["cal_tp"]  for r in results])),
            }
            for c in CONF_SWEEP
        },
    }
    tag = f"focal_g{args.gamma}{alpha_tag}_i{args.imgsz}"
    out_path = RUNS_DIR / f"calib_{tag}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved: {out_path}")


if __name__ == "__main__":
    main()
