"""POST-HOC sensitivity analysis for the FDI baseline (added after inspecting
runs_v2/_tmp_eval/fdi_baseline, where every fold selected the lowest grid
quantile q=0.90). Read-only; writes only into --out.

Variants (threshold still chosen on train+val scenes only, test labels unused):
  ext_grid   : FDI as in v2_fdi_baseline.py, quantile grid extended down to 0.50
  water_ctx  : same, but FDI kept only where the 15x15 local mean of
               NDWI = (B3 - B8) / (B3 + B8) is > 0 (water context; the local mean
               keeps small debris pixels whose own NDWI is lowered by debris NIR);
               quantiles computed on the retained pixels
All detector, GT, matching and AP code is imported from v2_fdi_baseline.py.
Sanity check: ext_grid fit AP at the quantiles shared with the original grid must
equal runs_v2/_tmp_eval/fdi_baseline/threshold_selection.csv.
"""
from __future__ import annotations

import argparse
import csv
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
import v2_fdi_baseline as fb  # noqa: E402
import v2_relaxed_match as rm  # noqa: E402

QUANTILES_EXT = (0.50, 0.60, 0.70, 0.80, 0.85) + tuple(fb.QUANTILES)
B3 = 2
WATER_WINDOW = 15
VARIANTS = ("ext_grid", "water_ctx")


def read_bands(tif: Path):
    with rasterio.open(tif) as src:
        if src.count != 11:
            raise SystemExit(f"[error] {tif}: expected 11 bands, found {src.count}")
        x = src.read().astype(np.float64)
    slope = (fb.WL_NIR - fb.WL_RED) / (fb.WL_SWIR1 - fb.WL_RED) * 10.0
    fdi = x[fb.B8] - (x[fb.B6] + (x[fb.B11] - x[fb.B6]) * slope)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (x[B3] - x[fb.B8]) / (x[B3] + x[fb.B8])
    valid = np.isfinite(ndwi)
    num = ndimage.uniform_filter(np.where(valid, ndwi, 0.0), size=WATER_WINDOW, mode="nearest")
    den = ndimage.uniform_filter(valid.astype(np.float64), size=WATER_WINDOW, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        local = num / den
    water = np.isfinite(local) & (local > 0)
    fdi_all = fdi.astype(np.float32)
    fdi_water = np.where(water, fdi, np.nan).astype(np.float32)
    return fdi_all, fdi_water, float(water.mean())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw/MARIDA/patches")
    p.add_argument("--baseline-dir", default="runs_v2/_tmp_eval/fdi_baseline")
    p.add_argument("--out", default="runs_v2/_tmp_eval/fdi_sensitivity")
    p.add_argument("--ref-json", default="runs_cv/cv_stat_results.json")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir, out, base_dir = Path(args.raw_dir), Path(args.out), Path(args.baseline_dir)
    base_sel = base_dir / "threshold_selection.csv"
    if not base_sel.is_file():
        raise SystemExit(f"[error] baseline output not found: {base_sel}")
    fb.guard_output_dir(out)

    print("[1/6] loading validated YOLO OOF ...", flush=True)
    entries_by_run, fold_details = cv.load_validated_oof(fb.SEEDS, smoke=False)
    fold_scenes = {k: sorted(cv.FOLDS[k]) for k in fb.FOLDS}
    for k in fb.FOLDS:
        if fold_scenes[k] != sorted(fold_details[k]["splits"]["test"]["scenes"]):
            raise SystemExit(f"[error] fold {k}: cv.FOLDS differs from OOF fold inventory")
    gt_by_stem, scene_by_stem = {}, {}
    for k in fb.FOLDS:
        for e in entries_by_run[(k, "c6", fb.SEEDS[0])]:
            gt_by_stem[e["stem"]] = np.asarray(e["ground_truth"], np.float32).reshape(-1, 4)
            scene_by_stem[e["stem"]] = e["scene"]
    if len(gt_by_stem) != 1381 or sum(len(v) for v in gt_by_stem.values()) != 212:
        raise SystemExit("[error] expected 1381 patches / 212 GT boxes")
    stems_by_scene = {}
    for stem, scene in scene_by_stem.items():
        stems_by_scene.setdefault(scene, []).append(stem)

    print("[2/6] reading bands (FDI + local-NDWI water context) ...", flush=True)
    maps = {"ext_grid": {}, "water_ctx": {}}
    water_frac = []
    for i, stem in enumerate(sorted(gt_by_stem), start=1):
        tif = raw_dir / scene_by_stem[stem] / f"{stem}.tif"
        if not tif.is_file():
            raise SystemExit(f"[error] missing {tif}")
        maps["ext_grid"][stem], maps["water_ctx"][stem], wf = read_bands(tif)
        water_frac.append(wf)
        if i % 300 == 0 or i == len(gt_by_stem):
            print(f"      {i}/{len(gt_by_stem)} patches", flush=True)
    gt_pixels_in_water = 0
    gt_pixels_total = 0
    for stem, boxes in gt_by_stem.items():
        for x0, y0, x1, y1 in (boxes / 2.0).astype(int):
            region = maps["water_ctx"][stem][y0:y1, x0:x1]
            gt_pixels_total += region.size
            gt_pixels_in_water += int(np.isfinite(region).sum())

    iou50 = rm.make_matcher("iou", 0.50, 1.0)

    def entries_for(variant, scenes, threshold):
        return [{"stem": st, "scene": sc, "ground_truth": gt_by_stem[st],
                 "predictions": fb.fdi_predictions(maps[variant][st], threshold)}
                for sc in scenes for st in sorted(stems_by_scene[sc])]

    print("[3/6] per-fold threshold selection on train+val ...", flush=True)
    selection_rows, fold_rows, results, oof = [], [], {}, {}
    for variant in VARIANTS:
        oof[variant] = []
        for k in fb.FOLDS:
            train_sc, val_sc, test_sc = cv.fold_scene_sets(k)
            fit_scenes = sorted(train_sc | val_sc)
            pixels = np.concatenate([maps[variant][st].ravel()[::fb.PIXEL_STRIDE]
                                     for sc in fit_scenes for st in stems_by_scene[sc]])
            pixels = pixels[np.isfinite(pixels)]
            best = None
            for q in QUANTILES_EXT:
                t = float(np.quantile(pixels, q))
                led = fb.ledger_from_entries(entries_for(variant, fit_scenes, t), iou50)
                ap = at.pooled_ap(led, fit_scenes)
                selection_rows.append({"variant": variant, "fold": k, "quantile": q,
                                       "threshold": t, "fit_AP50": ap,
                                       "fit_n_predictions": int(sum(len(v[0]) for v in led.values()))})
                if best is None or ap > best[2] or (ap == best[2] and t > best[1]):
                    best = (q, t, ap)
            test_entries = entries_for(variant, sorted(test_sc), best[1])
            oof[variant].extend(test_entries)
            fold_rows.append({"variant": variant, "fold": k, "quantile": best[0],
                              "threshold": best[1], "fit_AP50": best[2],
                              "test_AP50": at.pooled_ap(
                                  fb.ledger_from_entries(test_entries, iou50), sorted(test_sc)),
                              "at_grid_edge": best[0] in (QUANTILES_EXT[0], QUANTILES_EXT[-1])})
            print(f"      {variant} fold {k}: q={best[0]} t={best[1]:.6f} "
                  f"fit AP50={best[2]:.4f}", flush=True)

    print("[4/6] sanity check vs original baseline selection ...", flush=True)
    with base_sel.open(encoding="utf-8", newline="") as handle:
        base = {(int(r["fold"]), float(r["quantile"])): float(r["fit_AP50"])
                for r in csv.DictReader(handle)}
    n_cmp = 0
    for r in selection_rows:
        key = (r["fold"], r["quantile"])
        if r["variant"] == "ext_grid" and key in base:
            if not np.isclose(r["fit_AP50"], base[key], atol=1e-9, rtol=0.0):
                raise SystemExit(f"[error] ext_grid fit AP {r['fit_AP50']} != baseline "
                                 f"{base[key]} at fold {key[0]} q={key[1]}")
            n_cmp += 1
    if n_cmp != len(base):
        raise SystemExit(f"[error] compared {n_cmp} of {len(base)} baseline rows")
    print(f"      {n_cmp} shared (fold, quantile) rows identical")

    print("[5/6] OOF evaluation + YOLO comparison ...", flush=True)
    expected = rm.load_reference_merged_ap(Path(args.ref_json))
    yolo = {}
    for ch in ("c6", "c7"):
        yolo[ch] = []
        for seed in fb.SEEDS:
            led = cv.entries_to_scene_ledger(
                [e for k in fb.FOLDS for e in entries_by_run[(k, ch, seed)]])
            if not np.isclose(at.pooled_ap(led, sorted(led)), expected[ch][seed],
                              atol=fb.TOL, rtol=0.0):
                raise SystemExit(f"[error] YOLO {ch} seed {seed} reference AP not reproduced")
            yolo[ch].append(led)
    for variant in VARIANTS:
        if sorted({e["stem"] for e in oof[variant]}) != sorted(gt_by_stem):
            raise SystemExit(f"[error] {variant}: OOF does not cover every patch once")
        res = {}
        for name, kind, thr in fb.EVAL_MODES:
            led = fb.ledger_from_entries(oof[variant], rm.make_matcher(kind, thr, 0.5))
            scenes = sorted(led)
            res[name] = {"pooled_AP": at.pooled_ap(led, scenes),
                         "fold_mean_AP": fb.fold_mean(led, fold_scenes),
                         "max_recall": rm.max_recall(led),
                         "n_predictions": int(sum(len(v[0]) for v in led.values()))}
            if name == "iou_0.50":
                led50 = led
        contrasts = {}
        for ch in ("c6", "c7"):
            point, (lo, hi), draws = at.scene_bootstrap(
                yolo[ch], [led50] * len(fb.SEEDS), B=args.n_boot, seed=args.seed)
            contrasts[f"{ch}_minus_fdi"] = {"point": float(point), "ci95": [lo, hi],
                                            "n_draws_used": int(draws.size)}
        results[variant] = {"metrics": res, "yolo_minus_fdi_iou50": contrasts}
        print(f"      {variant}: AP50 pooled {res['iou_0.50']['pooled_AP']:.4f} "
              f"fold-mean {res['iou_0.50']['fold_mean_AP']:.4f} recall "
              f"{res['iou_0.50']['max_recall']:.4f} | C6-FDI "
              f"{contrasts['c6_minus_fdi']['point']:+.4f} {contrasts['c6_minus_fdi']['ci95']} "
              f"| C7-FDI {contrasts['c7_minus_fdi']['point']:+.4f} "
              f"{contrasts['c7_minus_fdi']['ci95']}", flush=True)

    print("[6/6] writing outputs ...", flush=True)
    fb.write_csv(out / "threshold_selection.csv", list(selection_rows[0]), selection_rows)
    fb.write_csv(out / "fold_results.csv", list(fold_rows[0]), fold_rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/v2_fdi_sensitivity.py",
        "script_sha256": fb.sha256_file(Path(__file__)),
        "status": "POST HOC sensitivity analysis, added after the primary FDI baseline "
                  "selected the lowest grid quantile in every fold",
        "variants": {
            "ext_grid": f"FDI, quantile grid {list(QUANTILES_EXT)}",
            "water_ctx": f"FDI restricted to pixels whose {WATER_WINDOW}x{WATER_WINDOW} local "
                         "mean NDWI (B3-B8)/(B3+B8) > 0; same grid on retained pixels",
        },
        "selection": "per fold, max pooled IoU>=0.5 AP on train+val scenes; ties -> higher t",
        "checks": {"ext_grid_matches_baseline_rows": n_cmp,
                   "yolo_reference_AP_reproduced": True},
        "water_context": {"mean_fraction_of_patch_retained": float(np.mean(water_frac)),
                          "fraction_of_gt_box_pixels_retained":
                              gt_pixels_in_water / gt_pixels_total if gt_pixels_total else None},
        "folds_selected_at_grid_edge": {
            v: [r["fold"] for r in fold_rows if r["variant"] == v and r["at_grid_edge"]]
            for v in VARIANTS},
        "results": results,
        "bootstrap": {"replicates": args.n_boot, "seed": args.seed,
                      "kind": "audit_toolkit.scene_bootstrap, 63 scenes"},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
