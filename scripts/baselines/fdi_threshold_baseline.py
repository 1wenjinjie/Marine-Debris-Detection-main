"""
FDI / NDWI-inverse threshold baseline for Marine_Debris detection on MARIDA v2split test.

Method:
  1. read raw 11-band Sentinel-2 L2A tile (256x256)
  2. compute index map (FDI per Biermann 2020 or NDWI-inverse)
  3. for each threshold tau, binarize -> connected components -> bbox predictions
  4. match against GT Marine_Debris bboxes (IoU >= 0.5) -> instance Precision/Recall
  5. output PR curve + best-F1 table

MARIDA 11-band order (Kikaki 2022): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
                                    0   1   2   3   4   5   6   7   8    9    10
Wavelengths (nm)                  : 443 490 560 665 705 740 783 842 865  1610 2190

FDI (Biermann 2020):
    NIR_baseline = R_RE2 + (R_SWIR1 - R_RE2) * (lambda_NIR - lambda_RE2) / (lambda_SWIR1 - lambda_RE2)
    FDI = R_NIR - NIR_baseline
    with RE2=B6 (740), NIR=B8 (842), SWIR1=B11 (1610)

NDWI-inverse (project convention):
    NDWI = (B3 - B8) / (B3 + B8 + eps)
    NDWI_inv = (1 - NDWI) / 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = PROJECT_ROOT / "data" / "processed" / "mask_bbox_rescue_split_physidx"
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "mask_bbox_rescue_split_physidx_v2split"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "MARIDA" / "patches"

MARINE_DEBRIS_CLS = 0
LAMBDA_RE2, LAMBDA_NIR, LAMBDA_SWIR1 = 740.0, 842.0, 1610.0
B6_IDX, B8_IDX, B11_IDX, B3_IDX = 5, 7, 9, 2


def compute_fdi(arr11: np.ndarray) -> np.ndarray:
    re2 = arr11[B6_IDX].astype(np.float32)
    nir = arr11[B8_IDX].astype(np.float32)
    swir1 = arr11[B11_IDX].astype(np.float32)
    nir_baseline = re2 + (swir1 - re2) * (LAMBDA_NIR - LAMBDA_RE2) / (LAMBDA_SWIR1 - LAMBDA_RE2)
    return nir - nir_baseline


def compute_ndwi_inv(arr11: np.ndarray) -> np.ndarray:
    b3 = arr11[B3_IDX].astype(np.float32)
    b8 = arr11[B8_IDX].astype(np.float32)
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
    return (1.0 - ndwi) / 2.0


def stem_to_raw_tif(stem: str) -> Path:
    scene = stem.rsplit("_", 1)[0]
    return RAW_ROOT / scene / f"{stem}.tif"


def stem_to_label_txt(rel_path: str) -> Path:
    return SPLIT_ROOT / rel_path.replace("/images/", "/labels/").replace(".npy", ".txt")


def load_gt_debris_boxes(label_path: Path, img_size: int = 256) -> list[tuple[int, int, int, int]]:
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if cls != MARINE_DEBRIS_CLS:
                continue
            cx, cy, w, h = map(float, parts[1:])
            x1 = max(0, int(round((cx - w / 2) * img_size)))
            y1 = max(0, int(round((cy - h / 2) * img_size)))
            x2 = min(img_size - 1, int(round((cx + w / 2) * img_size)))
            y2 = min(img_size - 1, int(round((cy + h / 2) * img_size)))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def index_map_to_pred_boxes(
    idx_map: np.ndarray,
    threshold: float,
    min_pixels: int = 4,
    padding: int = 2,
) -> list[tuple[int, int, int, int, float]]:
    """Threshold -> connected components -> bbox predictions with mean score per component."""
    H, W = idx_map.shape
    binary = (idx_map >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []
    labeled, n = label(binary)
    boxes = []
    for comp_id in range(1, n + 1):
        ys, xs = np.where(labeled == comp_id)
        if len(ys) < min_pixels:
            continue
        score = float(idx_map[ys, xs].mean())
        x1 = max(0, int(xs.min()) - padding)
        y1 = max(0, int(ys.min()) - padding)
        x2 = min(W - 1, int(xs.max()) + padding)
        y2 = min(H - 1, int(ys.max()) + padding)
        boxes.append((x1, y1, x2, y2, score))
    return boxes


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def evaluate_threshold(
    per_tile_preds: list[list[tuple]],
    per_tile_gts: list[list[tuple]],
    iou_thresh: float = 0.5,
) -> tuple[int, int, int]:
    """Aggregate TP/FP/FN across all tiles for a single index threshold."""
    tp, fp, fn = 0, 0, 0
    for preds, gts in zip(per_tile_preds, per_tile_gts):
        if not preds and not gts:
            continue
        if not preds:
            fn += len(gts)
            continue
        if not gts:
            fp += len(preds)
            continue
        preds_sorted = sorted(preds, key=lambda p: -p[4])
        gt_matched = [False] * len(gts)
        for pred in preds_sorted:
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if gt_matched[j]:
                    continue
                iou = bbox_iou(pred, gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                gt_matched[best_j] = True
                tp += 1
            else:
                fp += 1
        fn += sum(1 for m in gt_matched if not m)
    return tp, fp, fn


def sweep(
    index_name: str,
    per_tile_index_maps: list[np.ndarray],
    per_tile_gts: list[list[tuple]],
    thresholds: np.ndarray,
    min_pixels: int = 4,
) -> dict:
    results = []
    for tau in thresholds:
        per_tile_preds = [
            index_map_to_pred_boxes(im, float(tau), min_pixels=min_pixels) for im in per_tile_index_maps
        ]
        tp, fp, fn = evaluate_threshold(per_tile_preds, per_tile_gts)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results.append({"tau": float(tau), "tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1})
    best = max(results, key=lambda d: d["F1"])
    return {
        "index": index_name,
        "n_thresholds": len(thresholds),
        "results": results,
        "best": best,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--out-dir", default="docs/paper/fdi_baseline")
    ap.add_argument("--n-thresh", type=int, default=40)
    ap.add_argument("--min-pixels", type=int, default=4)
    args = ap.parse_args()

    manifest = SPLIT_MANIFEST / f"{args.split}.txt"
    with open(manifest) as f:
        rel_paths = [l.strip() for l in f if l.strip()]
    print(f"[load] {args.split} split: {len(rel_paths)} tiles")

    per_tile_arr, per_tile_gts, missing = [], [], []
    for rel in rel_paths:
        stem = os.path.splitext(os.path.basename(rel))[0]
        tif = stem_to_raw_tif(stem)
        if not tif.exists():
            missing.append(stem)
            continue
        with rasterio.open(tif) as src:
            arr11 = src.read()  # (11, H, W)
        per_tile_arr.append(arr11)
        per_tile_gts.append(load_gt_debris_boxes(stem_to_label_txt(rel)))

    if missing:
        print(f"[warn] {len(missing)} raw tifs not found, e.g. {missing[:3]}")

    total_gt = sum(len(g) for g in per_tile_gts)
    print(f"[gt] total Marine_Debris GT instances: {total_gt}")

    fdi_maps = [compute_fdi(a) for a in per_tile_arr]
    ndwi_maps = [compute_ndwi_inv(a) for a in per_tile_arr]

    fdi_all = np.concatenate([m.flatten() for m in fdi_maps])
    ndwi_all = np.concatenate([m.flatten() for m in ndwi_maps])
    print(f"[stats] FDI percentiles 50/90/95/99: "
          f"{np.percentile(fdi_all, [50, 90, 95, 99]).round(4).tolist()}")
    print(f"[stats] NDWI-inv percentiles 50/90/95/99: "
          f"{np.percentile(ndwi_all, [50, 90, 95, 99]).round(4).tolist()}")

    fdi_thresh = np.percentile(fdi_all, np.linspace(50, 99.5, args.n_thresh))
    ndwi_thresh = np.percentile(ndwi_all, np.linspace(50, 99.5, args.n_thresh))

    print("[sweep] FDI ...")
    fdi_res = sweep("FDI", fdi_maps, per_tile_gts, fdi_thresh, min_pixels=args.min_pixels)
    print(f"[sweep] FDI best F1 = {fdi_res['best']['F1']:.4f} "
          f"@ tau={fdi_res['best']['tau']:.4f}, "
          f"P={fdi_res['best']['P']:.3f}, R={fdi_res['best']['R']:.3f}")

    print("[sweep] NDWI-inv ...")
    ndwi_res = sweep("NDWI_inv", ndwi_maps, per_tile_gts, ndwi_thresh, min_pixels=args.min_pixels)
    print(f"[sweep] NDWI-inv best F1 = {ndwi_res['best']['F1']:.4f} "
          f"@ tau={ndwi_res['best']['tau']:.4f}, "
          f"P={ndwi_res['best']['P']:.3f}, R={ndwi_res['best']['R']:.3f}")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"fdi_ndwi_sweep_{args.split}.json"
    with open(out_json, "w") as f:
        json.dump({
            "split": args.split,
            "n_tiles": len(per_tile_arr),
            "total_gt": total_gt,
            "missing_raw": missing,
            "min_pixels": args.min_pixels,
            "FDI": fdi_res,
            "NDWI_inv": ndwi_res,
        }, f, indent=2)
    print(f"[save] {out_json}")


if __name__ == "__main__":
    main()
