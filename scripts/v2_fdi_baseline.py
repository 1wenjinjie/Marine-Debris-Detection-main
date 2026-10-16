"""FDI (Floating Debris Index) threshold baseline on the exact 9-fold scene CV.

Read-only with respect to existing results; writes only into --out.

Protocol (pre-specified, see summary.json "protocol"):
  * folds: cv_pipeline.FOLDS (the assignment actually used by the 54 YOLO runs);
    test = fold k, val = fold (k+1)%9, train = the rest.
  * FDI per pixel from the raw 11-band MARIDA patch
    (band order B1,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12):
    FDI = B8 - (B6 + (B11 - B6) * (832.8 - 664.6) / (1613.7 - 664.6) * 10)
  * detector: pixels with FDI > t, 4-connected components, area >= 4 px
    (same MIN4 rule as the labels), bbox padded by 2 px and clipped (same as
    the label builder), score = max FDI inside the component, top MAX_DET=300
    per patch.
  * t is chosen per fold among label-free quantiles of the train+val FDI pixels,
    maximising pooled IoU>=0.5 AP on the train+val scenes; ties -> higher t.
    Test labels are never used for the choice.
  * GT boxes and AP/ledger code are the ones used for the YOLO OOF analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "mpb"))

import audit_toolkit as at  # noqa: E402
import cv_pipeline as cv  # noqa: E402
import v2_relaxed_match as rm  # noqa: E402

SEEDS = tuple(cv.FULL_SEEDS)
FOLDS = tuple(range(9))
TOL = 1e-4
IMGSZ = 512
MIN_AREA = 4
PAD = 2
MAX_DET = 300
STRUCTURE = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
WL_RED, WL_NIR, WL_SWIR1 = 664.6, 832.8, 1613.7
B6, B8, B11 = 5, 7, 9  # 0-based indices in the 11-band MARIDA stack
QUANTILES = (0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999)
PIXEL_STRIDE = 8
EVAL_MODES = (("iou_0.50", "iou", 0.50), ("iou_0.10", "iou", 0.10),
              ("center_2px", "center", 2.0))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guard_output_dir(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"[error] output directory exists and is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def compute_fdi(tif: Path) -> np.ndarray:
    with rasterio.open(tif) as src:
        if src.count != 11:
            raise SystemExit(f"[error] {tif}: expected 11 bands, found {src.count}")
        x = src.read().astype(np.float64)
    slope = (WL_NIR - WL_RED) / (WL_SWIR1 - WL_RED) * 10.0
    fdi = x[B8] - (x[B6] + (x[B11] - x[B6]) * slope)
    return fdi.astype(np.float32)


def components_to_boxes(mask: np.ndarray, score_map: np.ndarray | None):
    """4-connected components with area >= MIN_AREA -> (N,4) xyxy on the IMGSZ grid
    using the label builder geometry (pad 2 px, clip, inclusive pixel extent),
    plus per-component max score (or None)."""
    height, width = mask.shape
    labeled, n = ndimage.label(mask, structure=STRUCTURE)
    if n == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    index = np.arange(1, n + 1)
    areas = np.asarray(ndimage.sum(mask, labeled, index))
    scores = (np.asarray(ndimage.maximum(score_map, labeled, index), np.float32)
              if score_map is not None else np.zeros(n, np.float32))
    sx, sy = IMGSZ / width, IMGSZ / height
    objects = ndimage.find_objects(labeled)
    boxes, keep_scores = [], []
    for comp in np.flatnonzero(areas >= MIN_AREA):
        slc = objects[comp]
        y0, y1 = slc[0].start, slc[0].stop - 1
        x0, x1 = slc[1].start, slc[1].stop - 1
        x0, y0 = max(0, x0 - PAD), max(0, y0 - PAD)
        x1, y1 = min(width - 1, x1 + PAD), min(height - 1, y1 + PAD)
        boxes.append([x0 * sx, y0 * sy, (x1 + 1) * sx, (y1 + 1) * sy])
        keep_scores.append(scores[comp])
    return (np.asarray(boxes, np.float32).reshape(-1, 4),
            np.asarray(keep_scores, np.float32))


def fdi_predictions(fdi: np.ndarray, threshold: float) -> np.ndarray:
    finite = np.isfinite(fdi)
    mask = finite & (fdi > threshold)
    score_map = np.where(finite, fdi, -np.inf).astype(np.float32)
    boxes, scores = components_to_boxes(mask, score_map)
    if not len(boxes):
        return np.zeros((0, 6), np.float32)
    order = np.argsort(-scores, kind="stable")[:MAX_DET]
    preds = np.zeros((len(order), 6), np.float32)
    preds[:, :4] = boxes[order]
    preds[:, 4] = scores[order]
    return preds  # class column = 0 (debris)


def ledger_from_entries(entries, matcher):
    led = {}
    for e in entries:
        preds, gts = e["predictions"], e["ground_truth"]
        tp = matcher(preds, gts)
        s0, t0, n0 = led.get(e["scene"], (np.array([]), np.array([]), 0))
        led[e["scene"]] = (np.concatenate([s0, preds[:, 4]]),
                           np.concatenate([t0, tp]), n0 + len(gts))
    return led


def fold_mean(ledger, fold_scenes):
    vals = [at.pooled_ap(ledger, fold_scenes[k]) for k in FOLDS
            if sum(ledger[s][2] for s in fold_scenes[k]) > 0]
    return float(np.mean(vals)) if vals else float("nan")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw/MARIDA/patches")
    p.add_argument("--out", default="runs_v2/_tmp_eval/fdi_baseline")
    p.add_argument("--ref-json", default="runs_cv/cv_stat_results.json")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir, out = Path(args.raw_dir), Path(args.out)
    guard_output_dir(out)

    print("[1/7] loading validated YOLO OOF (GT source and comparison) ...", flush=True)
    entries_by_run, fold_details = cv.load_validated_oof(SEEDS, smoke=False)
    fold_scenes = {k: sorted(cv.FOLDS[k]) for k in FOLDS}
    for k in FOLDS:
        if fold_scenes[k] != sorted(fold_details[k]["splits"]["test"]["scenes"]):
            raise SystemExit(f"[error] fold {k}: cv.FOLDS differs from OOF fold inventory")
    gt_by_stem, scene_by_stem = {}, {}
    for k in FOLDS:
        for e in entries_by_run[(k, "c6", SEEDS[0])]:
            gt_by_stem[e["stem"]] = np.asarray(e["ground_truth"], np.float32).reshape(-1, 4)
            scene_by_stem[e["stem"]] = e["scene"]
    if len(gt_by_stem) != 1381 or sum(len(v) for v in gt_by_stem.values()) != 212:
        raise SystemExit("[error] expected 1381 patches / 212 GT boxes in the OOF inventory")

    print("[2/7] computing FDI and re-deriving GT from *_cl.tif (geometry check) ...",
          flush=True)
    fdi = {}
    n_gt_checked = 0
    for i, stem in enumerate(sorted(gt_by_stem), start=1):
        scene = scene_by_stem[stem]
        tif = raw_dir / scene / f"{stem}.tif"
        cl = raw_dir / scene / f"{stem}_cl.tif"
        if not tif.is_file() or not cl.is_file():
            raise SystemExit(f"[error] missing raw files for {stem} under {raw_dir / scene}")
        fdi[stem] = compute_fdi(tif)
        with rasterio.open(cl) as src:
            class_mask = src.read(1)
        mine, _ = components_to_boxes(class_mask == 1, None)
        ref = gt_by_stem[stem]
        a = mine[np.lexsort(mine.T[::-1])] if len(mine) else mine
        b = ref[np.lexsort(ref.T[::-1])] if len(ref) else ref
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-3):
            raise SystemExit(f"[error] GT geometry mismatch for {stem}: re-derived {a.tolist()} "
                             f"vs OOF {b.tolist()}. No output written.")
        n_gt_checked += len(ref)
        if i % 200 == 0 or i == len(gt_by_stem):
            print(f"      {i}/{len(gt_by_stem)} patches", flush=True)
    print(f"      GT geometry identical for all patches ({n_gt_checked} boxes)")
    n_nonfinite = int(sum((~np.isfinite(v)).sum() for v in fdi.values()))

    stems_by_scene = {}
    for stem, scene in scene_by_stem.items():
        stems_by_scene.setdefault(scene, []).append(stem)
    iou50 = rm.make_matcher("iou", 0.50, 1.0)

    def entries_for(scenes, threshold):
        return [{"stem": st, "scene": sc, "ground_truth": gt_by_stem[st],
                 "predictions": fdi_predictions(fdi[st], threshold)}
                for sc in scenes for st in sorted(stems_by_scene[sc])]

    print("[3/7] per-fold threshold selection on train+val (test labels unused) ...",
          flush=True)
    selection_rows, fold_choice, oof_entries = [], {}, []
    for k in FOLDS:
        train_sc, val_sc, test_sc = cv.fold_scene_sets(k)
        fit_scenes = sorted(train_sc | val_sc)
        pixels = np.concatenate([fdi[st].ravel()[::PIXEL_STRIDE]
                                 for sc in fit_scenes for st in stems_by_scene[sc]])
        pixels = pixels[np.isfinite(pixels)]
        best = None
        for q in QUANTILES:
            t = float(np.quantile(pixels, q))
            led = ledger_from_entries(entries_for(fit_scenes, t), iou50)
            ap = at.pooled_ap(led, fit_scenes)
            n_pred = int(sum(len(v[0]) for v in led.values()))
            selection_rows.append({"fold": k, "quantile": q, "threshold": t,
                                   "fit_AP50": ap, "fit_n_predictions": n_pred})
            if best is None or ap > best[2] or (ap == best[2] and t > best[1]):
                best = (q, t, ap)
        fold_choice[k] = {"quantile": best[0], "threshold": best[1], "fit_AP50": best[2]}
        oof_entries.extend(entries_for(sorted(test_sc), best[1]))
        print(f"      fold {k}: q={best[0]} t={best[1]:.6f} fit AP50={best[2]:.4f}",
              flush=True)

    print("[4/7] OOF evaluation of the FDI baseline ...", flush=True)
    if sorted({e["stem"] for e in oof_entries}) != sorted(gt_by_stem):
        raise SystemExit("[error] FDI OOF does not cover every patch exactly once")
    fdi_ledgers = {name: ledger_from_entries(oof_entries, rm.make_matcher(kind, thr, 0.5))
                   for name, kind, thr in EVAL_MODES}
    ref_iou = cv.entries_to_scene_ledger(oof_entries)
    for sc, (s, t, n) in ref_iou.items():
        s2, t2, n2 = fdi_ledgers["iou_0.50"][sc]
        if not (np.array_equal(s, s2) and np.array_equal(t, t2) and n == n2):
            raise SystemExit(f"[error] FDI IoU0.5 ledger differs from cv_pipeline for {sc}")
    scenes = sorted(ref_iou)
    fdi_results = {}
    for name, _, _ in EVAL_MODES:
        led = fdi_ledgers[name]
        fdi_results[name] = {
            "pooled_AP": at.pooled_ap(led, scenes),
            "fold_mean_AP": fold_mean(led, fold_scenes),
            "max_recall": rm.max_recall(led),
            "n_predictions": int(sum(len(v[0]) for v in led.values())),
        }
        print(f"      {name}: pooled AP {fdi_results[name]['pooled_AP']:.4f}, fold-mean AP "
              f"{fdi_results[name]['fold_mean_AP']:.4f}, max recall "
              f"{fdi_results[name]['max_recall']:.4f}", flush=True)

    print("[5/7] YOLO reference ledgers (must reproduce cv_stat_results) ...", flush=True)
    expected = rm.load_reference_merged_ap(Path(args.ref_json))
    yolo = {}
    for ch in ("c6", "c7"):
        yolo[ch] = []
        for seed in SEEDS:
            led = cv.entries_to_scene_ledger(
                [e for k in FOLDS for e in entries_by_run[(k, ch, seed)]])
            ap = at.pooled_ap(led, scenes)
            if not np.isclose(ap, expected[ch][seed], atol=TOL, rtol=0.0):
                raise SystemExit(f"[error] YOLO {ch} seed {seed} AP {ap} != {expected[ch][seed]}")
            yolo[ch].append(led)
    print("      reproduced")

    print(f"[6/7] paired scene bootstrap YOLO - FDI (IoU0.5, B={args.n_boot}) ...", flush=True)
    contrasts = {}
    fdi_rep = [fdi_ledgers["iou_0.50"]] * len(SEEDS)
    for ch in ("c6", "c7"):
        point, (lo, hi), draws = at.scene_bootstrap(yolo[ch], fdi_rep, B=args.n_boot,
                                                    seed=args.seed)
        yolo_fold_mean = float(np.mean([fold_mean(l, fold_scenes) for l in yolo[ch]]))
        contrasts[f"{ch}_minus_fdi"] = {
            "statistic": "mean over 3 YOLO seeds of pooled AP50(YOLO) - pooled AP50(FDI)",
            "point": float(point), "ci95": [lo, hi], "n_draws_used": int(draws.size),
            "yolo_pooled_AP_mean": float(np.mean([at.pooled_ap(l, scenes) for l in yolo[ch]])),
            "yolo_fold_mean_AP_mean": yolo_fold_mean,
            "fold_mean_difference": yolo_fold_mean - fdi_results["iou_0.50"]["fold_mean_AP"],
        }
        print(f"      {ch} - FDI = {point:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)

    print("[7/7] writing outputs ...", flush=True)
    assign_rows = []
    for k in FOLDS:
        for sc in fold_scenes[k]:
            assign_rows.append({"fold": k, "role": "test", "val_fold": (k + 1) % 9,
                                "scene": sc, "n_patches": len(stems_by_scene[sc]),
                                "n_debris_min4": int(sum(len(gt_by_stem[st])
                                                         for st in stems_by_scene[sc]))})
    write_csv(out / "cv9_fold_assignment_actual.csv",
              ["fold", "role", "val_fold", "scene", "n_patches", "n_debris_min4"], assign_rows)
    write_csv(out / "threshold_selection.csv",
              ["fold", "quantile", "threshold", "fit_AP50", "fit_n_predictions"],
              selection_rows)
    fold_rows = []
    for k in FOLDS:
        row = {"fold": k, **fold_choice[k]}
        for name, _, _ in EVAL_MODES:
            row[f"test_AP_{name}"] = at.pooled_ap(fdi_ledgers[name], fold_scenes[k])
        row["n_gt"] = int(sum(fdi_ledgers["iou_0.50"][s][2] for s in fold_scenes[k]))
        fold_rows.append(row)
    write_csv(out / "fdi_fold_results.csv", list(fold_rows[0]), fold_rows)
    oof_dump = [{"stem": e["stem"], "scene": e["scene"],
                 "predictions_xyxy_conf_cls": e["predictions"].tolist(),
                 "ground_truth_xyxy": e["ground_truth"].tolist()} for e in oof_entries]
    (out / "fdi_oof.json").write_text(json.dumps(oof_dump), encoding="utf-8")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/v2_fdi_baseline.py",
        "script_sha256": sha256_file(Path(__file__)),
        "protocol": {
            "folds": "cv_pipeline.FOLDS (grouped_scene_folds seed 20261012); test=k, val=(k+1)%9",
            "fdi_formula": "B8 - (B6 + (B11 - B6) * (832.8-664.6)/(1613.7-664.6) * 10)",
            "band_indices_0based": {"B6": B6, "B8": B8, "B11": B11},
            "detector": f"FDI>t, 4-connected, area>={MIN_AREA}px, bbox pad {PAD}px clipped, "
                        f"score=max FDI in component, top {MAX_DET} per patch",
            "threshold_grid": f"quantiles {list(QUANTILES)} of train+val FDI pixels "
                              f"(every {PIXEL_STRIDE}th pixel, label-free)",
            "selection": "max pooled IoU>=0.5 AP on train+val scenes; ties -> higher t",
            "matching": "same greedy matcher / 101-point ledger AP as the YOLO OOF analysis; "
                        "centre-distance matching has no box-size constraint",
            "note": "FDI responds to all floating matter (Sargassum, ships, foam); no class "
                    "discrimination, so it is a naive physics baseline",
        },
        "checks": {
            "gt_geometry_rederived_from_cl_tif_identical": True,
            "n_gt_checked": n_gt_checked,
            "fdi_iou50_ledger_identical_to_cv_pipeline": True,
            "yolo_reference_AP_reproduced": True,
            "n_nonfinite_fdi_pixels": n_nonfinite,
        },
        "fold_choice": {str(k): v for k, v in fold_choice.items()},
        "fdi_results": fdi_results,
        "yolo_minus_fdi_iou50": contrasts,
        "bootstrap": {"replicates": args.n_boot, "seed": args.seed,
                      "kind": "audit_toolkit.scene_bootstrap over 63 scenes; FDI ledger "
                              "is deterministic and paired with each YOLO seed"},
        "outputs": ["summary.json", "fdi_fold_results.csv", "threshold_selection.csv",
                    "cv9_fold_assignment_actual.csv", "fdi_oof.json"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
