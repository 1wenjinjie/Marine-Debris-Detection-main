"""Per-tile inference on v1 honest test split (BiFPN @ imgsz=1024, seed 0).

Outputs CSV `docs/paper/v1_test_per_tile.csv` with one row per test tile:
    tile, lat, lon, date, n_gt, n_tp, n_fp, n_fn, conf_thr

TP/FP/FN are at instance level for Marine_Debris (cls=0), IoU >= 0.5,
default deployment confidence threshold tau=0.25 (see §4.5 of paper).

Reuses functions from scripts.v2_sweep_thresholds and scripts.v1_calibration_platt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import yaml
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
GEO_INDEX_PATH = ROOT / "data/processed/marida_v2/tile_geo_index.json"
OUT_CSV = ROOT / "docs/paper/v1_test_per_tile.csv"


def per_tile_eval(model_nn, manifest_path, dataset_root, channel_stats, channels,
                  imgsz, nc, conf_thr, geo, iou_match=0.5):
    low, high = _load_channel_stats(channel_stats)
    with open(manifest_path) as f:
        rel_paths = [l.strip() for l in f if l.strip()]
    print(f"[infer] {manifest_path.name}: {len(rel_paths)} tiles, tau={conf_thr}", flush=True)

    rows = []
    for ti, rel in enumerate(rel_paths):
        npy_path = dataset_root / rel
        if not npy_path.exists():
            continue
        tile_name = Path(rel).stem

        arr, _, _ = _preprocess_npy(npy_path, low, high, channels=channels, imgsz=imgsz)
        tensor = _arr_to_tensor(arr)
        preds = _raw_predict(model_nn, tensor, conf_thr=0.001, iou_thr=0.45, nc=nc)

        label_path = dataset_root / rel.replace("/images/", "/labels/").replace(".npy", ".txt")
        gt_xywh = _load_gt_boxes(label_path)
        gt_xyxy = _xywh_norm_to_xyxy_pixel(gt_xywh, imgsz=imgsz)
        n_gt = len(gt_xyxy)

        # filter MD preds + apply deployment threshold
        n_tp = n_fp = 0
        n_pred_after_tau = 0
        if len(preds) > 0:
            mask = (preds[:, 5].astype(int) == MARINE_DEBRIS_CLS) & (preds[:, 4] >= conf_thr)
            pred_xyxy = preds[mask, :4]
            pred_conf = preds[mask, 4]
            n_pred_after_tau = int(mask.sum())

            if len(pred_xyxy) > 0 and n_gt > 0:
                iou = _iou_matrix(pred_xyxy, gt_xyxy)
                order = np.argsort(-pred_conf)
                gt_matched = np.zeros(n_gt, dtype=bool)
                for idx in order:
                    best_j, best_iou = -1, 0.0
                    for j in range(n_gt):
                        if gt_matched[j]:
                            continue
                        if iou[idx, j] > best_iou:
                            best_iou, best_j = float(iou[idx, j]), j
                    if best_iou >= iou_match and best_j >= 0:
                        gt_matched[best_j] = True
                        n_tp += 1
                    else:
                        n_fp += 1
            elif len(pred_xyxy) > 0:
                n_fp = len(pred_xyxy)

        n_fn = n_gt - n_tp

        g = geo.get(tile_name, {})
        lat, lon = g.get("center_latlon", [np.nan, np.nan])
        date = g.get("date", "")

        rows.append({
            "tile": tile_name,
            "lat": float(lat), "lon": float(lon), "date": date,
            "n_gt": n_gt, "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn,
            "n_pred_after_tau": n_pred_after_tau,
            "conf_thr": conf_thr,
        })

        if (ti + 1) % 20 == 0:
            cumul = sum(r["n_tp"] for r in rows)
            cumgt = sum(r["n_gt"] for r in rows)
            print(f"  [{ti+1}/{len(rel_paths)}] cum TP={cumul} / GT={cumgt}", flush=True)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default="runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_bifpn_imgsz1024_clean_init_s0/weights/best.pt")
    ap.add_argument("--data",
                    default="configs/datasets/marida_v1_physidx_on_v2split_7ch.yaml")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--channels", type=int, default=7)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    with open(args.data) as f:
        cfg = yaml.safe_load(f)
    dataset_root = Path(cfg["path"])
    nc = int(cfg["nc"])
    test_manifest = Path(cfg["test"])
    channel_stats = dataset_root / f"channel_stats_{args.channels}ch.json"

    geo = json.loads(GEO_INDEX_PATH.read_text(encoding="utf-8"))

    print(f"[load] {args.weights}", flush=True)
    yolo = YOLO(args.weights)
    yolo.model.cuda().eval()
    model_nn = yolo.model

    df = per_tile_eval(model_nn, test_manifest, dataset_root, channel_stats,
                       args.channels, args.imgsz, nc, args.conf, geo)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[save] {OUT_CSV}")

    print("\n[summary]")
    print(f"  tiles:        {len(df)}")
    print(f"  total GT:     {int(df['n_gt'].sum())}")
    print(f"  total TP:     {int(df['n_tp'].sum())}")
    print(f"  total FP:     {int(df['n_fp'].sum())}")
    print(f"  total FN:     {int(df['n_fn'].sum())}")
    tp = int(df["n_tp"].sum())
    fp = int(df["n_fp"].sum())
    fn = int(df["n_fn"].sum())
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"  P / R / F1:   {p:.3f} / {r:.3f} / {f1:.3f}")


if __name__ == "__main__":
    main()
