"""
Confidence calibration for Marine_Debris predictions via Platt scaling.

Pipeline:
  1. Load BiFPN @ imgsz=1024 (honest SOTA) checkpoint
  2. Run inference on val split with low conf threshold -> collect (logit, is_TP) pairs
     (TP defined via IoU >= 0.5 against GT Marine_Debris boxes)
  3. Fit Platt scaling: P(TP|s) = sigmoid(a * logit(s) + b)
  4. Apply calibration to test predictions, compute:
        - reliability diagram (binned predicted vs observed positive rate)
        - threshold sweep table: P / R / F1 at multiple thresholds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO

from scripts.v2_sweep_thresholds import (
    _load_channel_stats,
    _preprocess_npy,
    _arr_to_tensor,
    _load_gt_boxes,
    _xywh_norm_to_xyxy_pixel,
    _iou_matrix,
    _raw_predict,
)

MARINE_DEBRIS_CLS = 0
EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def collect_preds(
    model_nn,
    manifest: Path,
    dataset_root: Path,
    channel_stats: Path,
    channels: int,
    imgsz: int,
    nc: int,
    iou_match: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int]:
    low, high = _load_channel_stats(channel_stats)

    with open(manifest) as f:
        rel_paths = [l.strip() for l in f if l.strip()]
    print(f"[infer] {manifest.name}: {len(rel_paths)} tiles", flush=True)

    all_scores, all_is_tp = [], []
    total_gt = 0

    for ti, rel in enumerate(rel_paths):
        npy_path = dataset_root / rel
        if not npy_path.exists():
            continue
        arr, _, _ = _preprocess_npy(npy_path, low, high, channels=channels, imgsz=imgsz)
        tensor = _arr_to_tensor(arr)
        preds = _raw_predict(model_nn, tensor, conf_thr=0.001, iou_thr=0.45, nc=nc)

        label_path = dataset_root / rel.replace("/images/", "/labels/").replace(".npy", ".txt")
        gt_xywh = _load_gt_boxes(label_path)
        gt_xyxy = _xywh_norm_to_xyxy_pixel(gt_xywh, imgsz=imgsz)
        total_gt += len(gt_xyxy)

        if len(preds) == 0:
            continue
        mask = preds[:, 5].astype(int) == MARINE_DEBRIS_CLS
        if not mask.any():
            continue
        pred_xyxy = preds[mask, :4]
        pred_conf = preds[mask, 4]

        if len(gt_xyxy) == 0:
            for s in pred_conf:
                all_scores.append(float(s))
                all_is_tp.append(0)
            continue

        iou = _iou_matrix(pred_xyxy, gt_xyxy)
        order = np.argsort(-pred_conf)
        gt_matched = np.zeros(len(gt_xyxy), dtype=bool)
        for idx in order:
            best_j, best_iou = -1, 0.0
            for j in range(len(gt_xyxy)):
                if gt_matched[j]:
                    continue
                if iou[idx, j] > best_iou:
                    best_iou, best_j = float(iou[idx, j]), j
            if best_iou >= iou_match and best_j >= 0:
                gt_matched[best_j] = True
                all_is_tp.append(1)
            else:
                all_is_tp.append(0)
            all_scores.append(float(pred_conf[idx]))

        if (ti + 1) % 20 == 0:
            print(f"  [{ti+1}/{len(rel_paths)}] preds={len(all_scores)} tp={sum(all_is_tp)} gt={total_gt}",
                  flush=True)

    return np.array(all_scores), np.array(all_is_tp), total_gt


def fit_platt(scores: np.ndarray, is_tp: np.ndarray, l2: float = 1.0) -> tuple[float, float]:
    """P(TP|s) = sigmoid(a*logit(s) + b) via scipy L-BFGS-B with L2 regularization.

    Manual Newton diverged on near-separable cases (a -> 1e9 overflow); we keep things
    numerically safe with bounded variables and explicit L2 penalty.
    """
    from scipy.optimize import minimize
    x = logit(scores)
    y = is_tp.astype(np.float64)
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        return 1.0, 0.0

    def nll(theta):
        a, b = theta
        z = np.clip(a * x + b, -50, 50)
        log_p = -np.logaddexp(0, -z)         # log(sigmoid(z))
        log_1mp = -np.logaddexp(0, z)        # log(1 - sigmoid(z))
        nll_val = -(y * log_p + (1 - y) * log_1mp).sum()
        nll_val += 0.5 * l2 * (a * a + b * b)
        return nll_val

    def grad(theta):
        a, b = theta
        z = np.clip(a * x + b, -50, 50)
        p = sigmoid(z)
        diff = p - y
        ga = float((diff * x).sum() + l2 * a)
        gb = float(diff.sum() + l2 * b)
        return np.array([ga, gb])

    res = minimize(nll, x0=np.array([1.0, 0.0]), jac=grad,
                   method="L-BFGS-B", bounds=[(-100, 100), (-100, 100)],
                   options={"maxiter": 500, "ftol": 1e-9})
    return float(res.x[0]), float(res.x[1])


def reliability_diagram(scores_cal: np.ndarray, is_tp: np.ndarray, n_bins: int = 10) -> dict:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores_cal, bins) - 1, 0, n_bins - 1)
    centers, observed, counts = [], [], []
    for k in range(n_bins):
        mask = bin_idx == k
        if mask.sum() == 0:
            centers.append((bins[k] + bins[k + 1]) / 2)
            observed.append(0.0)
            counts.append(0)
        else:
            centers.append(float(scores_cal[mask].mean()))
            observed.append(float(is_tp[mask].mean()))
            counts.append(int(mask.sum()))
    total = sum(counts)
    ece = sum(n / total * abs(c - o) for c, o, n in zip(centers, observed, counts) if n > 0) if total > 0 else 0.0
    return {"bin_centers": centers, "observed": observed, "counts": counts, "ECE": float(ece)}


def threshold_sweep(scores: np.ndarray, is_tp: np.ndarray, total_gt: int,
                    thresholds: list[float]) -> list[dict]:
    rows = []
    for tau in thresholds:
        mask = scores >= tau
        tp = int(is_tp[mask].sum())
        fp = int(mask.sum() - tp)
        fn = total_gt - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append({"tau": tau, "tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--channels", type=int, default=7)
    ap.add_argument("--out-dir", default="docs/paper/calibration")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data) as f:
        cfg = yaml.safe_load(f)
    dataset_root = Path(cfg["path"])
    nc = int(cfg["nc"])
    val_manifest = Path(cfg["val"])
    test_manifest = Path(cfg["test"])
    channel_stats = dataset_root / f"channel_stats_{args.channels}ch.json"

    print(f"[load] model {args.weights}", flush=True)
    yolo = YOLO(args.weights)
    yolo.model.cuda().eval()
    model_nn = yolo.model

    print("[stage 1] collecting val preds for Platt fit ...", flush=True)
    val_s, val_t, val_gt = collect_preds(
        model_nn, val_manifest, dataset_root, channel_stats, args.channels, args.imgsz, nc)
    print(f"[stage 1] val: {len(val_s)} preds, {int(val_t.sum())} TP, {val_gt} GT", flush=True)

    if len(val_s) < 20 or val_t.sum() < 3:
        print("[error] too few val preds/TPs to fit Platt; abort", flush=True)
        return

    a, b = fit_platt(val_s, val_t)
    print(f"[platt] a = {a:.4f}, b = {b:.4f}", flush=True)

    val_cal = sigmoid(a * logit(val_s) + b)
    raw_rel = reliability_diagram(val_s, val_t)
    cal_rel = reliability_diagram(val_cal, val_t)
    print(f"[ECE] raw = {raw_rel['ECE']:.4f}, calibrated = {cal_rel['ECE']:.4f}", flush=True)

    print("[stage 2] inference on test ...", flush=True)
    test_s, test_t, test_gt = collect_preds(
        model_nn, test_manifest, dataset_root, channel_stats, args.channels, args.imgsz, nc)
    print(f"[stage 2] test: {len(test_s)} preds, {int(test_t.sum())} TP, {test_gt} GT", flush=True)

    test_cal = sigmoid(a * logit(test_s) + b) if len(test_s) > 0 else np.array([])

    taus = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75]
    raw_sweep = threshold_sweep(test_s, test_t, test_gt, taus)
    cal_sweep = threshold_sweep(test_cal, test_t, test_gt, taus)

    test_raw_rel = reliability_diagram(test_s, test_t)
    test_cal_rel = reliability_diagram(test_cal, test_t)
    print(f"[ECE-test] raw = {test_raw_rel['ECE']:.4f}, calibrated = {test_cal_rel['ECE']:.4f}", flush=True)

    fit_payload = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "platt": {"a": a, "b": b},
        "val": {"n_preds": int(len(val_s)), "n_tp": int(val_t.sum()), "n_gt": val_gt,
                "ECE_raw": raw_rel["ECE"], "ECE_calibrated": cal_rel["ECE"],
                "reliability_raw": raw_rel, "reliability_calibrated": cal_rel,
                "note": "in-sample: Platt fit on this set"},
        "test": {"n_preds": int(len(test_s)), "n_tp": int(test_t.sum()), "n_gt": test_gt,
                 "ECE_raw": test_raw_rel["ECE"], "ECE_calibrated": test_cal_rel["ECE"],
                 "reliability_raw": test_raw_rel, "reliability_calibrated": test_cal_rel,
                 "note": "held-out: val-fitted Platt applied unchanged"},
    }
    with open(out_dir / "platt_fit.json", "w") as f:
        json.dump(fit_payload, f, indent=2)
    print(f"[save] {out_dir / 'platt_fit.json'}", flush=True)

    with open(out_dir / "threshold_sweep_test.json", "w") as f:
        json.dump({"thresholds": taus, "raw": raw_sweep, "calibrated": cal_sweep, "test_gt": test_gt}, f, indent=2)
    print(f"[save] {out_dir / 'threshold_sweep_test.json'}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=140)
    rows = [
        ("val (in-sample, Platt fit set)", raw_rel, cal_rel),
        ("test (held-out)", test_raw_rel, test_cal_rel),
    ]
    for row_idx, (split_label, rraw, rcal) in enumerate(rows):
        for col_idx, (rel, kind) in enumerate([(rraw, "Raw"), (rcal, "Platt-calibrated")]):
            ax = axes[row_idx, col_idx]
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
            keep = [i for i, c in enumerate(rel["counts"]) if c > 0]
            centers = [rel["bin_centers"][i] for i in keep]
            observed = [rel["observed"][i] for i in keep]
            counts = [rel["counts"][i] for i in keep]
            if counts:
                max_c = max(counts)
                sizes = [max(20, c / max_c * 200) for c in counts]
                ax.scatter(centers, observed, s=sizes, alpha=0.7, edgecolor="black")
                for cx, cy, n in zip(centers, observed, counts):
                    ax.annotate(str(n), (cx, cy), fontsize=7, ha="center", va="bottom")
            ax.set_xlabel("Predicted confidence")
            ax.set_ylabel("Observed TP rate")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{split_label} — {kind} (ECE={rel['ECE']:.3f})")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = out_dir / "reliability_diagram.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[save] {fig_path}", flush=True)

    print("\n[summary] threshold sweep on test:", flush=True)
    print(f"  {'tau':>7}  {'raw P':>7} {'raw R':>7} {'raw F1':>7}  {'cal P':>7} {'cal R':>7} {'cal F1':>7}",
          flush=True)
    for r, c in zip(raw_sweep, cal_sweep):
        print(f"  {r['tau']:>7.3f}  {r['P']:>7.3f} {r['R']:>7.3f} {r['F1']:>7.3f}  "
              f"{c['P']:>7.3f} {c['R']:>7.3f} {c['F1']:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
