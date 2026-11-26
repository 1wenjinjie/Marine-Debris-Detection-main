"""Phase 3A: Post-training CV diagnostics on D4 runs (read-only on historical outputs).

Writes ONLY into runs_v2/_tmp_eval_d4/:
  1. cv_fold_ap/      - Per-fold AP (Criterion B), epoch audit, stratified fold bootstrap
  2. relaxed_match/   - Relaxed localization (IoU 0.5/0.3/0.1, center 1/2/3 px) raw + val-calibrated
  3. error_analysis/  - Patch/pixel FP attribution + size/confidence stratified recall (C6, C7, FDI)
  4. val_calibration/ - Formal strict val-only cross-fold FP-rate calibration + 10,000 scene bootstrap
                        for BOTH 9-fold Channel CV (with RF contrasts) and 9-fold 2x2 Factorial Audit
  5. hash_inventory/  - Complete SHA-256 inventory of all 189 D4 checkpoints, OOFs, and Phase 3 outputs
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "mpb"))

import audit_toolkit as at  # noqa: E402
import cv_pipeline as cv  # noqa: E402
import run_experiments as rex  # noqa: E402
import v2_cv_factorial as fact  # noqa: E402
import v2_cv_fold_ap as fold_ap_mod  # noqa: E402
import v2_error_analysis as err_mod  # noqa: E402
import v2_relaxed_match as rm_mod  # noqa: E402

RUNS_CV_D4 = ROOT / "runs_cv_d4"
OOF_CV_D4 = RUNS_CV_D4 / "oof"
REF_CV_D4 = RUNS_CV_D4 / "cv_stat_results.json"

RUNS_FACT_D4 = ROOT / "runs_cv_factorial_d4"
OOF_FACT_D4 = RUNS_FACT_D4 / "oof"

RUNS_MPB_D4 = ROOT / "runs_mpb_d4"
OUT_ROOT = ROOT / "runs_v2" / "_tmp_eval_d4"

SEEDS = (0, 1, 2)
FOLDS = tuple(range(9))
CHANNELS = ("c6", "c7")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def build_val_fp_calibrator(val_entries: list[dict]):
    """Strictly val-only monotonic score calibrator based on validation FP/patch rate.

    Maps each raw confidence s in [0, 1] to 1 / (1 + estimated_val_fp_per_patch(s)),
    using piecewise-linear empirical survival interpolation between sorted val FP scores.
    Strictly increasing in s everywhere, preserving intra-fold AP identically while
    aligning all fold models onto one shared operational false-alarm-rate scale.
    """
    n_val = max(len(val_entries), 1)
    fp_scores = []
    for e in val_entries:
        preds = np.asarray(e["predictions"], dtype=float).reshape(-1, 6)
        preds = preds[preds[:, 5] == 0.0]
        if not len(preds):
            continue
        gts = np.asarray(e["ground_truth"], dtype=float).reshape(-1, 4)
        tp = cv.match_patch_tp(preds, gts)
        fp = preds[tp == 0.0, 4]
        if len(fp):
            fp_scores.append(fp)
    u = np.sort(np.concatenate(fp_scores)) if fp_scores else np.zeros(0, dtype=float)
    m = len(u)

    def transform(scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if not len(s):
            return s.copy()
        if m == 0:
            fp_above = 1.0 - np.clip(s, 0.0, 1.0)
            return 1.0 / (1.0 + fp_above / n_val)
        idx = np.searchsorted(u, s, side="right")
        frac = np.empty_like(s, dtype=float)
        mid = (idx > 0) & (idx < m)
        if np.any(mid):
            lo = u[idx[mid] - 1]
            hi = u[idx[mid]]
            frac[mid] = np.clip((s[mid] - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)
        high = idx == m
        if np.any(high):
            lo = u[m - 1]
            frac[high] = np.clip((s[high] - lo) / np.maximum(1.0 - lo, 1e-12), 0.0, 1.0)
        low = idx == 0
        if np.any(low):
            hi = u[0]
            frac[low] = np.clip(s[low] / np.maximum(hi, 1e-12), 0.0, 1.0)
        fp_above = (m - idx) + (1.0 - frac)
        return 1.0 / (1.0 + fp_above / n_val)

    return transform, {"n_val_patches": n_val, "n_val_fp": m}


def calibrate_entries(entries: list[dict], transform) -> list[dict]:
    out = []
    for e in entries:
        preds = np.asarray(e["predictions"], dtype=float).reshape(-1, 6).copy()
        if len(preds):
            mask = preds[:, 5] == 0.0
            if np.any(mask):
                preds[mask, 4] = transform(preds[mask, 4])
        out.append({**e, "predictions": preds})
    return out


def patch_cv_modules() -> dict:
    ref = json.loads(REF_CV_D4.read_text(encoding="utf-8"))
    doc = {
        ch: tuple(float(v) for v in ref["channels"][ch]["per_seed_AP"])
        for ch in CHANNELS
    }
    cv.RUNS_CV = RUNS_CV_D4
    cv.OOF_DIR = OOF_CV_D4
    fold_ap_mod.DOCUMENTED_MERGED_AP = doc
    rm_mod.DOCUMENTED_MERGED_AP = doc
    return ref


def run_cv_fold_ap() -> dict:
    print("\n" + "=" * 72, flush=True)
    print("[Phase 3A-1] Running v2_cv_fold_ap on runs_cv_d4 ...", flush=True)
    print("=" * 72, flush=True)
    out = OUT_ROOT / "cv_fold_ap"
    if out.exists():
        shutil.rmtree(out)
    old_argv = sys.argv
    try:
        sys.argv = [
            "v2_cv_fold_ap.py",
            "--oof-dir", str(OOF_CV_D4),
            "--runs-dir", str(RUNS_CV_D4),
            "--fold-csv", str(ROOT / "runs_mpb/stat_revision/cv9_fold_assignment.csv"),
            "--out", str(out),
            "--ref-json", str(REF_CV_D4),
            "--plan", str(ROOT / "runs_v2/_tmp_eval/plan_cv_relaxed.md"),
            "--n-boot", "10000",
            "--seed", "0",
        ]
        fold_ap_mod.main()
    finally:
        sys.argv = old_argv
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def run_relaxed_match() -> dict:
    print("\n" + "=" * 72, flush=True)
    print("[Phase 3A-2] Running v2_relaxed_match on runs_cv_d4 ...", flush=True)
    print("=" * 72, flush=True)
    out = OUT_ROOT / "relaxed_match"
    if out.exists():
        shutil.rmtree(out)
    old_argv = sys.argv
    try:
        sys.argv = [
            "v2_relaxed_match.py",
            "--oof-dir", str(OOF_CV_D4),
            "--fold-csv", str(ROOT / "runs_mpb/stat_revision/cv9_fold_assignment.csv"),
            "--out", str(out),
            "--ref-json", str(REF_CV_D4),
            "--plan", str(ROOT / "runs_v2/_tmp_eval/plan_cv_relaxed.md"),
            "--n-boot", "10000",
            "--seed", "0",
        ]
        rm_mod.main()
    finally:
        sys.argv = old_argv
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def run_error_analysis() -> dict:
    print("\n" + "=" * 72, flush=True)
    print("[Phase 3A-3] Running v2_error_analysis on runs_cv_d4 ...", flush=True)
    print("=" * 72, flush=True)
    out = OUT_ROOT / "error_analysis"
    if out.exists():
        shutil.rmtree(out)
    old_argv = sys.argv
    try:
        sys.argv = [
            "v2_error_analysis.py",
            "--raw-dir", str(ROOT / "data/raw/MARIDA/patches"),
            "--fdi-dir", str(ROOT / "runs_v2/_tmp_eval/fdi_baseline"),
            "--ref-json", str(REF_CV_D4),
            "--out", str(out),
        ]
        err_mod.main()
    finally:
        sys.argv = old_argv
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def load_rf_scene_ledgers(gt_by_stem: dict[str, np.ndarray], scene_by_stem: dict[str, str]) -> dict[int, dict]:
    rf_runs = ROOT / "runs_v2" / "_tmp_eval" / "rf_baseline_v2" / "runs"
    ledgers = {}
    for s in SEEDS:
        entries = []
        for k in FOLDS:
            p = rf_runs / f"fold{k}_s{s}.json"
            d = json.loads(p.read_text(encoding="utf-8"))
            for stem, boxes in d["test_predictions"].items():
                arr = np.asarray(boxes, dtype=float).reshape(-1, 6) if boxes else np.zeros((0, 6), dtype=float)
                entries.append({
                    "stem": stem,
                    "scene": scene_by_stem[stem],
                    "predictions": arr,
                    "ground_truth": gt_by_stem[stem],
                })
        ledgers[s] = cv.entries_to_scene_ledger(entries)
    return ledgers


def run_val_calibration(n_boot: int = 10000) -> dict:
    print("\n" + "=" * 72, flush=True)
    print("[Phase 3A-4] Strict val-only cross-fold calibration (CV9 + Relaxed + Factorial) ...", flush=True)
    print("=" * 72, flush=True)
    out = OUT_ROOT / "val_calibration"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # --- A. 9-fold Channel CV ---
    entries_raw, fold_details = cv.load_validated_oof(SEEDS, smoke=False)
    fold_scenes = {k: sorted(fold_details[k]["splits"]["test"]["scenes"]) for k in FOLDS}
    gt_by_stem, scene_by_stem = {}, {}
    for k in FOLDS:
        for e in entries_raw[(k, "c6", 0)]:
            gt_by_stem[e["stem"]] = np.asarray(e["ground_truth"], float).reshape(-1, 4)
            scene_by_stem[e["stem"]] = e["scene"]
    rf_ledgers = load_rf_scene_ledgers(gt_by_stem, scene_by_stem)

    entries_cal = {}
    cal_meta = {}
    max_fold_ap_diff = 0.0
    for k in FOLDS:
        for ch in CHANNELS:
            for s in SEEDS:
                val_path = RUNS_CV_D4 / f"fold{k}_{ch}_s{s}" / "val_predictions.json"
                val_entries = rex._read_prediction_ledger(val_path)
                tf, meta = build_val_fp_calibrator(val_entries)
                cal_meta[f"fold{k}_{ch}_s{s}"] = meta
                cal_e = calibrate_entries(entries_raw[(k, ch, s)], tf)
                entries_cal[(k, ch, s)] = cal_e

                # Monotonicity verification: per-fold AP must match raw identically
                led_r = cv.entries_to_scene_ledger(entries_raw[(k, ch, s)])
                led_c = cv.entries_to_scene_ledger(cal_e)
                ap_r = at.pooled_ap(led_r, fold_scenes[k])
                ap_c = at.pooled_ap(led_c, fold_scenes[k])
                max_fold_ap_diff = max(max_fold_ap_diff, abs(ap_r - ap_c))
    print(f"  [check] CV9 max |fold_AP_cal - fold_AP_raw| = {max_fold_ap_diff:.2e}", flush=True)
    if max_fold_ap_diff > 1e-9:
        raise RuntimeError(f"Calibration broke intra-fold monotonicity: {max_fold_ap_diff}")

    # Build raw & calibrated merged scene ledgers for all 6 relaxed modes
    grid = rm_mod.detect_grid_scale(entries_raw)
    matchers = {name: rm_mod.make_matcher(mode, thr, grid["scale_to_original"]) for name, mode, thr in rm_mod.MODES}
    scenes_all = sorted(scene_by_stem[st] for st in set(scene_by_stem))
    scenes_all = sorted(set(scene_by_stem.values()))

    rng_cv = np.random.default_rng(20261015)
    resamples_cv = [list(rng_cv.choice(scenes_all, size=len(scenes_all), replace=True)) for _ in range(n_boot)]

    def boot_summary(fn_point_list):
        draws = np.array([fn_point_list(sc) for sc in resamples_cv], dtype=float)
        draws = draws[~np.isnan(draws)]
        lo, hi = np.quantile(draws, [0.025, 0.975])
        return [round(float(lo), 4), round(float(hi), 4)]

    # Primary IoU 0.50 CV9 raw vs cal vs RF
    led_cv_raw = {}
    led_cv_cal = {}
    for ch in CHANNELS:
        for s in SEEDS:
            led_cv_raw[(ch, s)] = cv.entries_to_scene_ledger([e for k in FOLDS for e in entries_raw[(k, ch, s)]])
            led_cv_cal[(ch, s)] = cv.entries_to_scene_ledger([e for k in FOLDS for e in entries_cal[(k, ch, s)]])

    cv_primary = {}
    for tag, led_dict in (("raw_pooled_oof", led_cv_raw), ("val_calibrated_pooled_oof", led_cv_cal)):
        res = {}
        for ch in CHANNELS:
            per_s = [at.pooled_ap(led_dict[(ch, s)], scenes_all) for s in SEEDS]
            ci = boot_summary(lambda sc, c=ch, d=led_dict: float(np.mean([at.pooled_ap(d[(c, s)], sc) for s in SEEDS])))
            res[ch] = {
                "per_seed_AP": [round(v, 4) for v in per_s],
                "mean_AP": round(float(np.mean(per_s)), 4),
                "std_AP": round(float(np.std(per_s, ddof=1)), 4),
                "ci95": ci,
            }
        # Contrasts: C7-C6, C6-RF, C7-RF
        contrasts = {}
        for cname, lhs_fn, rhs_fn in (
            ("c7_minus_c6", lambda s, sc, d=led_dict: at.pooled_ap(d[("c7", s)], sc), lambda s, sc, d=led_dict: at.pooled_ap(d[("c6", s)], sc)),
            ("c6_minus_rf", lambda s, sc, d=led_dict: at.pooled_ap(d[("c6", s)], sc), lambda s, sc: at.pooled_ap(rf_ledgers[s], sc)),
            ("c7_minus_rf", lambda s, sc, d=led_dict: at.pooled_ap(d[("c7", s)], sc), lambda s, sc: at.pooled_ap(rf_ledgers[s], sc)),
        ):
            diffs = [lhs_fn(s, scenes_all) - rhs_fn(s, scenes_all) for s in SEEDS]
            ci = boot_summary(lambda sc, lf=lhs_fn, rf=rhs_fn: float(np.mean([lf(s, sc) - rf(s, sc) for s in SEEDS])))
            contrasts[cname] = {
                "point": round(float(np.mean(diffs)), 4),
                "per_seed": [round(float(v), 4) for v in diffs],
                "ci95": ci,
                "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            }
        res["contrasts"] = contrasts
        cv_primary[tag] = res

    # Relaxed modes with val-calibrated scores
    print("  computing val-calibrated AP across all 6 localization criteria ...", flush=True)
    fold_led_cal_modes = rm_mod.build_fold_ledgers(entries_cal, matchers)
    relaxed_cal = {}
    for name, _, _ in rm_mod.MODES:
        merged_m = {(ch, s): rm_mod.merge_fold_ledgers(fold_led_cal_modes[name], ch, s)
                    for ch in CHANNELS for s in SEEDS}
        mode_entry = {}
        for ch in CHANNELS:
            per_s = [at.pooled_ap(merged_m[(ch, s)], scenes_all) for s in SEEDS]
            ci = boot_summary(lambda sc, c=ch, m=merged_m: float(np.mean([at.pooled_ap(m[(c, s)], sc) for s in SEEDS])))
            mode_entry[ch] = {
                "per_seed_AP": [round(v, 4) for v in per_s],
                "mean_AP": round(float(np.mean(per_s)), 4),
                "std_AP": round(float(np.std(per_s, ddof=1)), 4),
                "ci95": ci,
            }
        diffs = [at.pooled_ap(merged_m[("c7", s)], scenes_all) - at.pooled_ap(merged_m[("c6", s)], scenes_all) for s in SEEDS]
        ci_d = boot_summary(lambda sc, m=merged_m: float(np.mean([at.pooled_ap(m[("c7", s)], sc) - at.pooled_ap(m[("c6", s)], sc) for s in SEEDS])))
        mode_entry["c7_minus_c6"] = {"point": round(float(np.mean(diffs)), 4), "ci95": ci_d}
        relaxed_cal[name] = mode_entry

    # --- B. 9-fold 2x2 Factorial Audit (val-only calibrated) ---
    print("  computing val-only calibrated 9-fold 2x2 factorial audit ...", flush=True)
    fact.load_runtime()
    designs = fact.all_designs()
    fact_entries_raw = fact.load_oof(designs, OOF_FACT_D4, FOLDS, SEEDS)
    fact_entries_cal = {}
    max_fact_diff = 0.0
    for k in FOLDS:
        k_scenes = sorted({e["scene"] for e in fact_entries_raw[(k, "clean_ndwi", 0)]})
        for cell in fact.CELLS:
            for s in SEEDS:
                val_path = RUNS_FACT_D4 / f"fold{k}_{cell}_s{s}" / "val_predictions.json"
                val_entries = rex._read_prediction_ledger(val_path)
                tf, _ = build_val_fp_calibrator(val_entries)
                cal_e = calibrate_entries(fact_entries_raw[(k, cell, s)], tf)
                fact_entries_cal[(k, cell, s)] = cal_e
                lr = cv.entries_to_scene_ledger(fact_entries_raw[(k, cell, s)])
                lc = cv.entries_to_scene_ledger(cal_e)
                max_fact_diff = max(max_fact_diff, abs(at.pooled_ap(lr, k_scenes) - at.pooled_ap(lc, k_scenes)))
    print(f"  [check] Factorial max |fold_AP_cal - fold_AP_raw| = {max_fact_diff:.2e}", flush=True)
    if max_fact_diff > 1e-9:
        raise RuntimeError(f"Factorial calibration broke monotonicity: {max_fact_diff}")

    led_fact_raw = {
        cell: [cv.entries_to_scene_ledger([e for k in FOLDS for e in fact_entries_raw[(k, cell, s)]]) for s in SEEDS]
        for cell in fact.CELLS
    }
    led_fact_cal = {
        cell: [cv.entries_to_scene_ledger([e for k in FOLDS for e in fact_entries_cal[(k, cell, s)]]) for s in SEEDS]
        for cell in fact.CELLS
    }
    exposed_scenes = sorted({sc for d in designs for sc in d["exposed_scenes"]})
    fact_raw_sum = fact.summarize_panel(led_fact_raw, exposed_scenes, n_boot=n_boot, seed=20261015)
    fact_cal_sum = fact.summarize_panel(led_fact_cal, exposed_scenes, n_boot=n_boot, seed=20261015)

    payload = {
        "generated_utc": utc_now(),
        "method": "Piecewise-linear monotonic validation FP-rate score calibration: s_cal = 1 / (1 + val_fp_per_patch(s)), fitted strictly on each fold's validation split only.",
        "checks": {
            "cv9_max_intra_fold_ap_diff": max_fold_ap_diff,
            "factorial_max_intra_fold_ap_diff": max_fact_diff,
            "test_used_for_calibration": False,
        },
        "bootstrap_replicates": n_boot,
        "cv9_channel_primary": cv_primary,
        "cv9_relaxed_val_calibrated": relaxed_cal,
        "factorial_raw_pooled_oof": fact_raw_sum,
        "factorial_val_calibrated_pooled_oof": fact_cal_sum,
    }
    write_json(out / "summary.json", payload)

    # Also write compact CSV table
    rows = []
    for mode, md in relaxed_cal.items():
        for ch in CHANNELS:
            rows.append({
                "mode": mode, "channel": ch,
                "cal_mean_AP": md[ch]["mean_AP"], "cal_std_AP": md[ch]["std_AP"],
                "ci95_lo": md[ch]["ci95"][0], "ci95_hi": md[ch]["ci95"][1],
            })
    write_csv(out / "cv9_relaxed_calibrated.csv",
              ["mode", "channel", "cal_mean_AP", "cal_std_AP", "ci95_lo", "ci95_hi"], rows)
    print(f"[done] val_calibration -> {out / 'summary.json'}", flush=True)
    return payload


def run_hash_inventory() -> dict:
    print("\n" + "=" * 72, flush=True)
    print("[Phase 3A-5] Generating SHA-256 hash inventory for all 189 D4 runs ...", flush=True)
    print("=" * 72, flush=True)
    out = OUT_ROOT / "hash_inventory"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    # Core modified code & scripts
    code_files = [
        "ultralytics_plugins/mask_rescue_dataset.py",
        "tests/unit/test_random_d4.py",
        "scripts/mpb/run_cv_d4.py",
        "scripts/mpb/run_phase2_d4.py",
        "scripts/mpb/run_phase3_cv_d4.py",
        "scripts/mpb/run_phase3_external_d4.py",
    ]
    for rel in code_files:
        p = ROOT / rel
        if p.is_file():
            rows.append({"kind": "code", "path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})

    # 54 CV9 checkpoints + 54 OOFs + summary
    cv_ckpts = sorted(RUNS_CV_D4.glob("fold*_s*/weights/best.pt"))
    cv_oofs = sorted(OOF_CV_D4.glob("fold*_s*_oof.json"))
    fact_ckpts = sorted(RUNS_FACT_D4.glob("fold*_s*/weights/best.pt"))
    fact_oofs = sorted(OOF_FACT_D4.glob("fold*_s*_oof.json"))
    mpb_ckpts = sorted((RUNS_MPB_D4 / "full").glob("*_s*/weights/best.pt"))
    mpb_ledgers = sorted((RUNS_MPB_D4 / "full").glob("*_s*/test_predictions_locked.json"))

    for kind, files in (
        ("cv9_checkpoint", cv_ckpts),
        ("cv9_oof", cv_oofs),
        ("factorial_checkpoint", fact_ckpts),
        ("factorial_oof", fact_oofs),
        ("mpb10_checkpoint", mpb_ckpts),
        ("mpb10_test_ledger", mpb_ledgers),
    ):
        for p in files:
            rows.append({
                "kind": kind,
                "path": p.relative_to(ROOT).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })

    # Top-level summaries & Phase 3A outputs
    for p in [
        REF_CV_D4,
        RUNS_FACT_D4 / "analysis" / "summary.json",
        RUNS_MPB_D4 / "summary_all.json",
        *sorted((OUT_ROOT / "cv_fold_ap").glob("*.json")),
        *sorted((OUT_ROOT / "relaxed_match").glob("*.json")),
        *sorted((OUT_ROOT / "error_analysis").glob("*.json")),
        *sorted((OUT_ROOT / "val_calibration").glob("*.json")),
    ]:
        if p.is_file():
            rows.append({
                "kind": "summary",
                "path": p.relative_to(ROOT).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })

    write_csv(out / "hashes_d4.csv", ["kind", "path", "bytes", "sha256"], rows)
    counts = {
        "cv9_checkpoints": len(cv_ckpts),
        "cv9_oofs": len(cv_oofs),
        "factorial_checkpoints": len(fact_ckpts),
        "factorial_oofs": len(fact_oofs),
        "mpb10_checkpoints": len(mpb_ckpts),
        "mpb10_test_ledgers": len(mpb_ledgers),
        "total_checkpoints": len(cv_ckpts) + len(fact_ckpts) + len(mpb_ckpts),
        "total_hashed_files": len(rows),
    }
    write_json(out / "manifest.json", {"generated_utc": utc_now(), "counts": counts})
    print(f"[done] hash_inventory: {counts}", flush=True)
    return counts


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    patch_cv_modules()
    skip_done = "--skip-done" in sys.argv
    if skip_done and (OUT_ROOT / "cv_fold_ap" / "summary.json").is_file():
        fold_sum = json.loads((OUT_ROOT / "cv_fold_ap" / "summary.json").read_text(encoding="utf-8"))
    else:
        fold_sum = run_cv_fold_ap()
    if skip_done and (OUT_ROOT / "relaxed_match" / "summary.json").is_file():
        rel_sum = json.loads((OUT_ROOT / "relaxed_match" / "summary.json").read_text(encoding="utf-8"))
    else:
        rel_sum = run_relaxed_match()
    if skip_done and (OUT_ROOT / "error_analysis" / "summary.json").is_file():
        err_sum = json.loads((OUT_ROOT / "error_analysis" / "summary.json").read_text(encoding="utf-8"))
    else:
        err_sum = run_error_analysis()
    cal_sum = run_val_calibration(n_boot=10000)
    hash_cnt = run_hash_inventory()

    print("\n" + "=" * 72, flush=True)
    print("PHASE 3A COMPLETE SUMMARY", flush=True)
    print("=" * 72, flush=True)
    for tag in ("raw_pooled_oof", "val_calibrated_pooled_oof"):
        d = cal_sum["cv9_channel_primary"][tag]
        print(f"CV9 {tag}: C6={d['c6']['mean_AP']:.4f} {d['c6']['ci95']} (sd={d['c6']['std_AP']:.4f}), "
              f"C7={d['c7']['mean_AP']:.4f} {d['c7']['ci95']} | "
              f"C7-C6={d['contrasts']['c7_minus_c6']['point']:+.4f} {d['contrasts']['c7_minus_c6']['ci95']} | "
              f"C6-RF={d['contrasts']['c6_minus_rf']['point']:+.4f} {d['contrasts']['c6_minus_rf']['ci95']}", flush=True)
    print("Relaxed Match (Raw vs Val-Cal vs Fold-Mean, C6):", flush=True)
    for mode, _, _ in rm_mod.MODES:
        pm = rel_sum["per_mode"][mode]
        cal_c6 = cal_sum["cv9_relaxed_val_calibrated"][mode]["c6"]
        print(f"  {mode:11s}: raw={pm['pooled_AP_mean_sd']['c6']['mean']:.4f}, val_cal={cal_c6['mean_AP']:.4f} {cal_c6['ci95']}, "
              f"fold_mean={pm['fold_mean_AP_over_seeds']['c6']:.4f}, maxR={pm['max_recall_at_conf_0.001']['mean_over_seeds']['c6']:.4f}", flush=True)
    fc = cal_sum["factorial_val_calibrated_pooled_oof"]
    print("Factorial Val-Calibrated:", flush=True)
    for cell in fact.CELLS:
        print(f"  {cell:14s}: {fc['cells'][cell]['mean_AP']:.4f} {fc['cells'][cell]['ci95']}", flush=True)
    for cname in ("oracle_prior_main_effect", "scene_exposure_main_effect", "interaction"):
        c = fc["contrasts"][cname]
        print(f"  {cname:28s}: {c['point']:+.4f} {c['ci95']}", flush=True)
    print("Hashes:", hash_cnt, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
