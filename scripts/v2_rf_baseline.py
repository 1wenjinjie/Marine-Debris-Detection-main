"""Pixel-level Random Forest baseline (MARIDA-style) on the exact 9-fold scene CV.

Read-only with respect to existing results; writes only into --out.

Protocol (fixed before running; recorded in protocol.json):
  * folds: cv_pipeline.FOLDS; test = fold k, val = fold (k+1)%9, train = the rest
    (identical to the YOLO CV runs and the FDI baseline).
  * classifier: MARIDA official RF (marine-debris.github.io,
    semantic_segmentation/random_forest/random_forest.py): StandardScaler +
    RandomForestClassifier(n_estimators=125, criterion='gini', max_depth=20,
    min_samples_leaf=1, min_impurity_decrease=0, class_weight='balanced_subsample').
    random_state = run seed (official code uses 5); oob_score is disabled (it does not
    change the fitted trees, only adds an OOB estimate).
  * training pixels: every annotated pixel (_cl.tif > 0) of the TRAIN scenes only;
    sample weight = 1 / confidence level (_conf.tif: 1 High, 2 Moderate, 3 Low), as in
    the official train_eval.py; classes Mixed Water(15), Wakes(14), Cloud Shadows(13)
    and Waves(12) are merged into Marine Water(7) (official agg_to_water default).
  * features: 11 bands (NaN filled with the official bands_mean) + 8 spectral indices
    computed exactly as engineering_patches.py (NDVI, FAI, FDI, SI, NDWI, NRD, NDMI,
    BSI; NaN/inf -> 0). The 6 GLCM texture features of the official RF are NOT used.
  * detector: per-pixel P(Marine Debris); pixels with P > t, 4-connected components,
    area >= 4 px, bbox padded by 2 px and clipped, score = max P in component,
    top MAX_DET=300 per patch (same geometry as labels / FDI baseline).
  * t is chosen per fold and seed on the VAL scenes (max pooled IoU>=0.5 AP; ties ->
    higher t) from a fixed grid; test labels are never used for any choice.
    Grid v2: 0.95-0.99 were added after a fold-0 trial selected the old upper bound
    0.9 for all three seeds on VAL (test AP was not inspected).
  * GT boxes and AP/ledger code are the ones used for the YOLO OOF analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "mpb"))

import audit_toolkit as at  # noqa: E402
import cv_pipeline as cv  # noqa: E402
import v2_relaxed_match as rm  # noqa: E402
from v2_fdi_baseline import (components_to_boxes, fold_mean, ledger_from_entries,  # noqa: E402
                             sha256_file, write_csv, EVAL_MODES, MAX_DET, TOL)

SEEDS = tuple(cv.FULL_SEEDS)
FOLDS = tuple(range(9))
DEBRIS = 1
AGG_TO_WATER = {12: 7, 13: 7, 14: 7, 15: 7}
BANDS_MEAN = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606,
                       0.03457443, 0.03875053, 0.03436435, 0.0392113, 0.02358126,
                       0.01588816], np.float32)
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99)
RF_PARAMS = dict(n_estimators=125, criterion="gini", max_depth=20, min_samples_leaf=1,
                 min_impurity_decrease=0.0, class_weight="balanced_subsample",
                 oob_score=False)
FEATURES = ["nm440", "nm490", "nm560", "nm665", "nm705", "nm740", "nm783", "nm842",
            "nm865", "nm1600", "nm2200", "NDVI", "FAI", "FDI", "SI", "NDWI", "NRD",
            "NDMI", "BSI"]
CHUNK_PATCHES = 24


def features(tif: Path) -> np.ndarray:
    """(H*W, 19) float32 feature matrix, MARIDA engineering_patches.py formulas.
    Band numbers below are 1-based as in the official code."""
    with rasterio.open(tif) as src:
        if src.count != 11:
            raise SystemExit(f"[error] {tif}: expected 11 bands, found {src.count}")
        x = src.read().astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = {i + 1: x[i] for i in range(11)}
        ndvi = (b[8] - b[4]) / (b[8] + b[4])
        fai = b[8] - (b[4] + (b[10] - b[4]) * (833.0 - 665.0) / (1614.0 - 665.0))
        fdi = b[8] - (b[6] + 10 * (b[10] - b[6]) * (833.0 - 740.0) / (1614.0 - 740.0))
        si = ((1 - b[2]) * (1 - b[3]) * (1 - b[4])) ** (1 / 3)
        ndwi = (b[3] - b[8]) / (b[3] + b[8])
        nrd = b[8] - b[4]
        ndmi = (b[8] - b[10]) / (b[8] + b[10])
        bsi = ((b[10] + b[4]) - (b[8] + b[2])) / ((b[10] + b[4]) + (b[8] + b[2]))
    idx = np.stack([ndvi, fai, fdi, si, ndwi, nrd, ndmi, bsi]).astype(np.float32)
    idx = np.nan_to_num(idx, nan=0.0, posinf=0.0, neginf=0.0)
    nan = ~np.isfinite(x)
    if nan.any():
        x = np.where(nan, BANDS_MEAN[:, None, None], x)
    feat = np.concatenate([x, idx], axis=0)
    return feat.reshape(feat.shape[0], -1).T.copy()


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw/MARIDA/patches")
    p.add_argument("--out", default="runs_v2/_tmp_eval/rf_baseline")
    p.add_argument("--ref-json", default="runs_cv/cv_stat_results.json")
    p.add_argument("--folds", default="all", help="'all' or comma list (trial runs)")
    p.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    raw_dir, out = Path(args.raw_dir), Path(args.out)
    run_folds = FOLDS if args.folds == "all" else tuple(int(v) for v in args.folds.split(","))
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    protocol = {
        "script_sha256": sha256_file(Path(__file__)),
        "fdi_helper_sha256": sha256_file(ROOT / "scripts" / "v2_fdi_baseline.py"),
        "sklearn_version": sklearn.__version__,
        "rf_params": RF_PARAMS, "rf_random_state": "run seed", "seeds": list(SEEDS),
        "features": FEATURES, "agg_to_water": {str(k): v for k, v in AGG_TO_WATER.items()},
        "sample_weight": "1/conf (_conf.tif 1/2/3)", "thresholds": list(THRESHOLDS),
        "max_det": MAX_DET, "train_pixels": "all _cl>0 pixels of train scenes",
        "threshold_selection": "max pooled IoU>=0.5 AP on val scenes; ties -> higher t",
    }
    ppath = out / "protocol.json"
    if ppath.exists():
        old = json.loads(ppath.read_text(encoding="utf-8"))
        if old != protocol:
            raise SystemExit(f"[error] {ppath} differs from the current protocol; use a new --out")
    else:
        ppath.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"[info] sklearn {sklearn.__version__}, n_jobs={args.n_jobs}, folds={run_folds}",
          flush=True)

    print("[1/6] loading validated YOLO OOF (GT source and comparison) ...", flush=True)
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
    stems_by_scene = {}
    for stem, scene in scene_by_stem.items():
        stems_by_scene.setdefault(scene, []).append(stem)

    print("[2/6] annotated-pixel table + GT geometry check ...", flush=True)
    tab_X, tab_y, tab_w, tab_scene = [], [], [], []
    n_gt_checked = 0
    for i, stem in enumerate(sorted(gt_by_stem), start=1):
        scene = scene_by_stem[stem]
        tif = raw_dir / scene / f"{stem}.tif"
        cl_path = raw_dir / scene / f"{stem}_cl.tif"
        conf_path = raw_dir / scene / f"{stem}_conf.tif"
        for pth in (tif, cl_path, conf_path):
            if not pth.is_file():
                raise SystemExit(f"[error] missing raw file {pth}")
        cl = read_band(cl_path).astype(np.int64)
        mine, _ = components_to_boxes(cl == DEBRIS, None)
        ref = gt_by_stem[stem]
        a = mine[np.lexsort(mine.T[::-1])] if len(mine) else mine
        b = ref[np.lexsort(ref.T[::-1])] if len(ref) else ref
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-3):
            raise SystemExit(f"[error] GT geometry mismatch for {stem}")
        n_gt_checked += len(ref)
        lab = cl.ravel() > 0
        if lab.any():
            conf = read_band(conf_path).ravel()[lab].astype(np.int64)
            if conf.min() < 1 or conf.max() > 3:
                raise SystemExit(f"[error] {stem}: confidence outside 1..3 on labelled pixels")
            y = cl.ravel()[lab]
            y = np.array([AGG_TO_WATER.get(int(v), int(v)) for v in y], np.int64)
            tab_X.append(features(tif)[lab])
            tab_y.append(y)
            tab_w.append(1.0 / conf)
            tab_scene.append(np.full(int(lab.sum()), scene))
        if i % 200 == 0 or i == len(gt_by_stem):
            print(f"      {i}/{len(gt_by_stem)} patches", flush=True)
    tab_X = np.concatenate(tab_X)
    tab_y = np.concatenate(tab_y)
    tab_w = np.concatenate(tab_w)
    tab_scene = np.concatenate(tab_scene)
    print(f"      {len(tab_y)} annotated pixels, classes {sorted(set(tab_y.tolist()))}, "
          f"debris px {(tab_y == DEBRIS).sum()}", flush=True)

    iou50 = rm.make_matcher("iou", 0.50, 1.0)

    def prob_maps(model, stems):
        cls = list(model.classes_)
        j = cls.index(DEBRIS)
        res = {}
        for c0 in range(0, len(stems), CHUNK_PATCHES):
            chunk = stems[c0:c0 + CHUNK_PATCHES]
            feats = [features(raw_dir / scene_by_stem[st] / f"{st}.tif") for st in chunk]
            sizes = [len(f) for f in feats]
            proba = model.predict_proba(np.concatenate(feats))[:, j].astype(np.float32)
            off = 0
            for st, n in zip(chunk, sizes):
                side = int(round(np.sqrt(n)))
                res[st] = proba[off:off + n].reshape(side, side)
                off += n
        return res

    def preds_at(pm, t):
        mask = pm > t
        boxes, scores = components_to_boxes(mask, pm)
        if not len(boxes):
            return np.zeros((0, 6), np.float32)
        order = np.argsort(-scores, kind="stable")[:MAX_DET]
        p = np.zeros((len(order), 6), np.float32)
        p[:, :4] = boxes[order]
        p[:, 4] = scores[order]
        return p

    print("[3/6] per fold x seed: fit on train, choose t on val, predict test ...", flush=True)
    for k in run_folds:
        train_sc, val_sc, test_sc = cv.fold_scene_sets(k)
        sel = np.isin(tab_scene, sorted(train_sc))
        for seed in SEEDS:
            rpath = out / "runs" / f"fold{k}_s{seed}.json"
            if rpath.exists():
                print(f"      fold {k} seed {seed}: exists, skipped", flush=True)
                continue
            t0 = time.time()
            model = Pipeline(steps=[("scaler", StandardScaler()), ("rf", RandomForestClassifier(
                random_state=seed, n_jobs=args.n_jobs, **RF_PARAMS))])
            model.fit(tab_X[sel], tab_y[sel], rf__sample_weight=tab_w[sel])
            t_fit = time.time() - t0
            val_stems = [st for sc in sorted(val_sc) for st in sorted(stems_by_scene[sc])]
            test_stems = [st for sc in sorted(test_sc) for st in sorted(stems_by_scene[sc])]
            pm_val = prob_maps(model, val_stems)
            grid = []
            best = None
            for t in THRESHOLDS:
                ents = [{"stem": st, "scene": scene_by_stem[st], "ground_truth": gt_by_stem[st],
                         "predictions": preds_at(pm_val[st], t)} for st in val_stems]
                ap = at.pooled_ap(ledger_from_entries(ents, iou50), sorted(val_sc))
                grid.append({"threshold": t, "val_AP50": ap,
                             "val_n_predictions": int(sum(len(e["predictions"]) for e in ents))})
                key = -1.0 if np.isnan(ap) else ap
                if best is None or key > best[1] or (key == best[1] and t > best[0]):
                    best = (t, key)
            del pm_val
            pm_test = prob_maps(model, test_stems)
            test_pred = {st: preds_at(pm_test[st], best[0]) for st in test_stems}
            # pixel-level sanity on annotated test pixels (argmax)
            tsel = np.isin(tab_scene, sorted(test_sc))
            yhat = model.predict(tab_X[tsel])
            ytrue = tab_y[tsel]
            tp = int(((yhat == DEBRIS) & (ytrue == DEBRIS)).sum())
            fp = int(((yhat == DEBRIS) & (ytrue != DEBRIS)).sum())
            fn = int(((yhat != DEBRIS) & (ytrue == DEBRIS)).sum())
            row = {"fold": k, "seed": seed, "chosen_threshold": best[0],
                   "chosen_val_AP50": None if best[1] < 0 else best[1], "val_grid": grid,
                   "n_train_pixels": int(sel.sum()),
                   "n_train_debris_pixels": int((tab_y[sel] == DEBRIS).sum()),
                   "test_pixel_debris_tp_fp_fn": [tp, fp, fn],
                   "fit_seconds": round(t_fit, 1), "total_seconds": round(time.time() - t0, 1),
                   "test_predictions": {st: p.tolist() for st, p in test_pred.items()}}
            rpath.write_text(json.dumps(row), encoding="utf-8")
            print(f"      fold {k} seed {seed}: t={best[0]} val AP50={row['chosen_val_AP50']} "
                  f"pixel MD tp/fp/fn={tp}/{fp}/{fn} fit {t_fit:.0f}s total "
                  f"{row['total_seconds']:.0f}s", flush=True)
            del model, pm_test

    missing = [(k, s) for k in FOLDS for s in SEEDS
               if not (out / "runs" / f"fold{k}_s{s}.json").exists()]
    if missing:
        print(f"[stop] {len(missing)} fold/seed runs missing; summary not written "
              f"(trial mode).", flush=True)
        return 0

    print("[4/6] OOF evaluation ...", flush=True)
    runs = {(k, s): json.loads((out / "runs" / f"fold{k}_s{s}.json").read_text(encoding="utf-8"))
            for k in FOLDS for s in SEEDS}
    rf_leds = {name: [] for name, _, _ in EVAL_MODES}
    rf_results = {name: [] for name, _, _ in EVAL_MODES}
    oof_by_seed = {}
    for s in SEEDS:
        ents = []
        for k in FOLDS:
            for st, p in runs[(k, s)]["test_predictions"].items():
                ents.append({"stem": st, "scene": scene_by_stem[st],
                             "ground_truth": gt_by_stem[st],
                             "predictions": np.asarray(p, np.float32).reshape(-1, 6)})
        if sorted(e["stem"] for e in ents) != sorted(gt_by_stem):
            raise SystemExit(f"[error] RF seed {s} OOF does not cover every patch exactly once")
        oof_by_seed[s] = ents
        ref_iou = cv.entries_to_scene_ledger(ents)
        for name, kind, thr in EVAL_MODES:
            led = ledger_from_entries(ents, rm.make_matcher(kind, thr, 0.5))
            if name == "iou_0.50":
                for sc, (a_, b_, n_) in ref_iou.items():
                    a2, b2, n2 = led[sc]
                    if not (np.array_equal(a_, a2) and np.array_equal(b_, b2) and n_ == n2):
                        raise SystemExit(f"[error] RF ledger differs from cv_pipeline for {sc}")
            rf_leds[name].append(led)
            scenes = sorted(led)
            rf_results[name].append({
                "seed": s, "pooled_AP": at.pooled_ap(led, scenes),
                "fold_mean_AP": fold_mean(led, fold_scenes),
                "max_recall": rm.max_recall(led),
                "n_predictions": int(sum(len(v[0]) for v in led.values()))})
    scenes = sorted(rf_leds["iou_0.50"][0])
    for name in rf_results:
        aps = [r["pooled_AP"] for r in rf_results[name]]
        print(f"      {name}: pooled AP per seed {[round(a, 4) for a in aps]} "
              f"mean {np.mean(aps):.4f}", flush=True)
    px = np.array([runs[(k, s)]["test_pixel_debris_tp_fp_fn"] for k in FOLDS for s in SEEDS])
    px_by_seed = {}
    for s in SEEDS:
        tp, fp, fn = np.sum([runs[(k, s)]["test_pixel_debris_tp_fp_fn"] for k in FOLDS], axis=0)
        px_by_seed[str(s)] = {"tp": int(tp), "fp": int(fp), "fn": int(fn),
                              "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                              "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                              "iou": tp / max(tp + fp + fn, 1)}

    print("[5/6] YOLO reference ledgers + paired scene bootstrap ...", flush=True)
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
    contrasts = {}
    for ch in ("c6", "c7"):
        point, (lo, hi), draws = at.scene_bootstrap(yolo[ch], rf_leds["iou_0.50"],
                                                    B=args.n_boot, seed=args.seed)
        contrasts[f"{ch}_minus_rf"] = {
            "statistic": "mean over 3 paired seeds of pooled AP50(YOLO) - pooled AP50(RF)",
            "point": float(point), "ci95": [lo, hi], "n_draws_used": int(draws.size),
            "yolo_pooled_AP_mean": float(np.mean([at.pooled_ap(l, scenes) for l in yolo[ch]])),
        }
        print(f"      {ch} - RF = {point:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)

    print("[6/6] writing outputs ...", flush=True)
    fold_rows = []
    for k in FOLDS:
        for s in SEEDS:
            r = runs[(k, s)]
            led = rf_leds["iou_0.50"][SEEDS.index(s)]
            fold_rows.append({"fold": k, "seed": s, "chosen_threshold": r["chosen_threshold"],
                              "chosen_val_AP50": r["chosen_val_AP50"],
                              "test_AP50": at.pooled_ap(led, fold_scenes[k]),
                              "n_gt": int(sum(led[sc][2] for sc in fold_scenes[k])),
                              "n_train_pixels": r["n_train_pixels"],
                              "fit_seconds": r["fit_seconds"]})
    write_csv(out / "rf_fold_results.csv", list(fold_rows[0]), fold_rows)
    grid_rows = [{"fold": k, "seed": s, **g} for k in FOLDS for s in SEEDS
                 for g in runs[(k, s)]["val_grid"]]
    write_csv(out / "threshold_selection.csv", list(grid_rows[0]), grid_rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/v2_rf_baseline.py",
        "protocol": protocol,
        "checks": {"gt_geometry_rederived_from_cl_tif_identical": True,
                   "n_gt_checked": n_gt_checked,
                   "rf_iou50_ledger_identical_to_cv_pipeline": True,
                   "yolo_reference_AP_reproduced": True,
                   "n_annotated_pixels": int(len(tab_y))},
        "rf_results": rf_results,
        "rf_mean": {name: {"pooled_AP": float(np.mean([r["pooled_AP"] for r in v])),
                           "fold_mean_AP": float(np.mean([r["fold_mean_AP"] for r in v]))}
                    for name, v in rf_results.items()},
        "pixel_level_debris_on_annotated_test_pixels_argmax": px_by_seed,
        "yolo_minus_rf_iou50": contrasts,
        "bootstrap": {"replicates": args.n_boot, "seed": args.seed,
                      "kind": "audit_toolkit.scene_bootstrap over 63 scenes, YOLO seed i "
                              "paired with RF seed i"},
        "notes": ["GLCM texture features of the official MARIDA RF are omitted",
                  "threshold chosen on val scenes only; RF never sees val/test pixels in fit"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
