"""Per-fold AP (criterion B) and provenance audit for the 9-fold scene CV.

Read-only re-analysis of the 54 finished OOF predictions. No retraining, no
modification of any existing file. Writes only into --out.

Reuses the existing ledger AP / bootstrap implementation:
  audit_toolkit.pooled_ap, ap_from_ledger, scene_bootstrap, ci_vs_margin
  cv_pipeline.entries_to_scene_ledger, load_validated_oof, box_iou_one_vs_many
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "mpb"))

import audit_toolkit as at  # noqa: E402
import cv_pipeline as cv  # noqa: E402

SEEDS = tuple(cv.FULL_SEEDS)
CHANNELS = ("c6", "c7")
FOLDS = tuple(range(9))
N_FORMAL_RUNS = 54
TOL = 1e-4
ORIGINAL_GRID = 256.0
INPUT_GRID = 512.0

# per-seed merged (criterion A) AP documented in runs_cv/cv_stat_results.json
DOCUMENTED_MERGED_AP = {
    "c6": (0.0661, 0.0498, 0.1002),
    "c7": (0.0956, 0.0269, 0.0315),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def guard_output_dir(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"[error] output directory exists and is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_reference_merged_ap(ref_json: Path) -> dict:
    payload = json.loads(ref_json.read_text(encoding="utf-8"))
    expected = {ch: [float(v) for v in payload["channels"][ch]["per_seed_AP"]]
                for ch in CHANNELS}
    if [len(v) for v in expected.values()] != [len(SEEDS)] * 2:
        raise SystemExit(f"[error] unexpected per_seed_AP length in {ref_json}")
    for ch, values in expected.items():
        for seed, value, documented in zip(SEEDS, values, DOCUMENTED_MERGED_AP[ch]):
            if abs(value - documented) > TOL:
                raise SystemExit(
                    f"[error] {ref_json}: {ch} seed {seed} AP {value!r} does not match "
                    f"the documented value {documented!r}")
    return expected


def read_fold_csv(path: Path) -> dict:
    """fold -> sorted test scenes, from the stat_revision assignment sheet."""
    mapping: dict[int, set] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            role = (row.get("role") or "test").strip().lower()
            if role != "test":
                continue
            mapping.setdefault(int(row["fold"]), set()).add(row["scene"].strip())
    return {k: sorted(v) for k, v in sorted(mapping.items())}


def detect_grid_scale(entries_by_run) -> dict:
    """OOF boxes live on the 512 input grid (256 px patches, imgsz=512, no padding)."""
    max_coord = 0.0
    for key in sorted(entries_by_run):
        for entry in entries_by_run[key]:
            for array in (entry["predictions"], entry["ground_truth"]):
                if len(array):
                    max_coord = max(max_coord, float(np.asarray(array)[:, :4].max()))
    grid = INPUT_GRID if max_coord > ORIGINAL_GRID + 0.5 else ORIGINAL_GRID
    converted = bool(grid != ORIGINAL_GRID)
    return {
        "max_observed_coordinate": max_coord,
        "oof_box_grid": int(grid),
        "original_grid": int(ORIGINAL_GRID),
        "scale_to_original": float(ORIGINAL_GRID / grid),
        "converted_to_original_grid": converted,
        "note": (f"OOF boxes are on the {int(grid)} input grid; coordinates are multiplied "
                 f"by {ORIGINAL_GRID / grid:g} for centre-distance matching"
                 if converted else "OOF boxes are already on the original 256 grid"),
    }


def audit_results_csv(path: Path) -> dict:
    if not path.is_file():
        return {"present": False}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{k.strip(): v for k, v in r.items()} for r in reader]
    epochs, unparsable = [], 0
    for row in rows:
        try:
            epochs.append(int(float(row["epoch"])))
        except (TypeError, ValueError, KeyError):
            unparsable += 1
    strictly_increasing = all(b > a for a, b in zip(epochs, epochs[1:]))
    return {
        "present": True,
        "n_rows": len(rows),
        "n_unique_epochs": len(set(epochs)),
        "min_epoch": min(epochs) if epochs else None,
        "max_epoch": max(epochs) if epochs else None,
        "has_duplicate_epoch": len(epochs) != len(set(epochs)),
        "strictly_increasing": strictly_increasing,
        "non_monotonic_or_duplicate": bool(not strictly_increasing),
        "n_unparsable_epoch": unparsable,
    }


# --------------------------------------------------------------------------
# criterion B
# --------------------------------------------------------------------------
def fold_ap_table(fold_ledgers, fold_scenes):
    """fold -> (ap, n_gt, n_scenes, n_scenes_with_gt) for one (channel, seed)."""
    table = {}
    for k in FOLDS:
        scenes = fold_scenes[k]
        ledger = fold_ledgers[k]
        n_gt = int(sum(ledger[scene][2] for scene in scenes))
        n_with_gt = int(sum(1 for scene in scenes if ledger[scene][2] > 0))
        table[k] = {
            "ap": at.pooled_ap(ledger, scenes),
            "n_gt": n_gt,
            "n_scenes": len(scenes),
            "n_scenes_with_gt": n_with_gt,
        }
    return table


def fold_average(table, weighted: bool):
    usable = [k for k in FOLDS if table[k]["n_gt"] > 0]
    if not usable:
        return float("nan"), []
    if weighted:
        weights = [table[k]["n_gt"] for k in usable]
        value = float(np.average([table[k]["ap"] for k in usable], weights=weights))
    else:
        value = float(np.mean([table[k]["ap"] for k in usable]))
    return value, usable


def stratified_fold_bootstrap(ledgers_c7, ledgers_c6, fold_scenes, n_boot: int, seed: int):
    """Resample scenes inside each fold (fold sizes kept), then average over folds.

    Per replicate each fold is resampled once (picks_by_fold); the same picks are
    reused by every seed and by C6/C7, so the seeds stay paired on one resample.
    """
    rng = np.random.default_rng(seed)

    def replicate():
        picks_by_fold = {
            k: [fold_scenes[k][i]
                for i in rng.integers(0, len(fold_scenes[k]), len(fold_scenes[k]))]
            for k in FOLDS}
        usable = [k for k in FOLDS
                  if min(sum(led[s][2] for s in picks_by_fold[k])
                         for led in list(ledgers_c7) + list(ledgers_c6)) > 0]
        per_seed = []
        for led7, led6 in zip(ledgers_c7, ledgers_c6):
            fold_values_7 = [at.pooled_ap(led7, picks_by_fold[k]) for k in usable]
            fold_values_6 = [at.pooled_ap(led6, picks_by_fold[k]) for k in usable]
            if fold_values_7:
                per_seed.append(np.mean(fold_values_7) - np.mean(fold_values_6))
        return float(np.mean(per_seed)) if per_seed else float("nan")

    draws = np.array([replicate() for _ in range(n_boot)])
    draws = draws[~np.isnan(draws)]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi), draws


# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-dir", default="runs_cv/oof")
    parser.add_argument("--runs-dir", default="runs_cv")
    parser.add_argument("--fold-csv", default="runs_mpb/stat_revision/cv9_fold_assignment.csv")
    parser.add_argument("--out", default="runs_v2/_tmp_eval/cv_fold_ap")
    parser.add_argument("--ref-json", default="runs_cv/cv_stat_results.json")
    parser.add_argument("--plan", default="runs_v2/_tmp_eval/plan_cv_relaxed.md")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    oof_dir, runs_dir = Path(args.oof_dir), Path(args.runs_dir)
    out = Path(args.out)
    plan = Path(args.plan)
    if not plan.is_file():
        raise SystemExit(f"[error] plan file not found: {plan}")
    plan_sha = sha256_file(plan)
    guard_output_dir(out)

    print(f"[1/8] loading 54 formal OOF files from {oof_dir} ...")
    entries_by_run, fold_details = cv.load_validated_oof(SEEDS, smoke=False)
    formal_files = sorted(p for p in oof_dir.glob("*_oof.json") if "smoke" not in p.name)
    smoke_files = sorted(p for p in oof_dir.glob("*_oof.json") if "smoke" in p.name)
    if len(formal_files) != N_FORMAL_RUNS:
        raise SystemExit(f"[error] expected {N_FORMAL_RUNS} formal OOF files, "
                         f"found {len(formal_files)} in {oof_dir}")
    print(f"      {len(formal_files)} formal OOF files loaded; "
          f"{len(smoke_files)} smoke file(s) excluded: "
          f"{', '.join(p.name for p in smoke_files) or 'none'}")

    print("[2/8] determining OOF box coordinate grid ...")
    grid = detect_grid_scale(entries_by_run)
    print(f"      max coordinate {grid['max_observed_coordinate']:.1f} -> "
          f"{grid['oof_box_grid']} input grid (scale to original "
          f"{grid['scale_to_original']:g})")

    print("[3/8] criterion A sanity check against "
          f"{Path(args.ref_json).name} (tolerance {TOL}) ...")
    expected = load_reference_merged_ap(Path(args.ref_json))
    merged_ledgers = {}
    recomputed = {}
    for channel in CHANNELS:
        recomputed[channel] = []
        for seed in SEEDS:
            merged = []
            for k in FOLDS:
                merged.extend(entries_by_run[(k, channel, seed)])
            ledger = cv.entries_to_scene_ledger(merged)
            merged_ledgers[(channel, seed)] = ledger
            ap = at.pooled_ap(ledger, sorted(ledger))
            recomputed[channel].append(ap)
            print(f"      {channel} seed {seed}: {ap:.6f} "
                  f"(reference {expected[channel][seed]:.6f})")
        for seed, got, want in zip(SEEDS, recomputed[channel], expected[channel]):
            if not np.isclose(got, want, atol=TOL, rtol=0.0):
                raise SystemExit(
                    f"[error] criterion A mismatch {channel} seed {seed}: "
                    f"{got!r} != {want!r} (tolerance {TOL}). No output written.")
    print("      criterion A reproduced within tolerance")

    print("[4/8] building per-fold ledgers (criterion B) ...")
    fold_scenes = {k: sorted(fold_details[k]["splits"]["test"]["scenes"]) for k in FOLDS}
    fold_ledgers = {}
    for channel in CHANNELS:
        for seed in SEEDS:
            for k in FOLDS:
                fold_ledgers[(k, channel, seed)] = cv.entries_to_scene_ledger(
                    entries_by_run[(k, channel, seed)])
    print(f"      {len(fold_ledgers)} fold ledgers built")

    print("[5/8] criterion B fold AP ...")
    fold_rows = []
    per_seed_simple, per_seed_weighted = {}, {}
    for channel in CHANNELS:
        per_seed_simple[channel], per_seed_weighted[channel] = [], []
        for seed in SEEDS:
            table = fold_ap_table({k: fold_ledgers[(k, channel, seed)] for k in FOLDS},
                                  fold_scenes)
            for k in FOLDS:
                fold_rows.append({
                    "fold": k, "channel": channel, "seed": seed,
                    "n_scenes_in_fold": table[k]["n_scenes"],
                    "n_scenes_with_gt": table[k]["n_scenes_with_gt"],
                    "n_gt_in_fold": table[k]["n_gt"],
                    "ap": table[k]["ap"],
                })
            simple, usable = fold_average(table, weighted=False)
            weighted, _ = fold_average(table, weighted=True)
            per_seed_simple[channel].append(simple)
            per_seed_weighted[channel].append(weighted)
            print(f"      {channel} seed {seed}: fold mean {simple:.6f} "
                  f"({len(usable)} folds with GT), GT-weighted {weighted:.6f}")

    print(f"[6/8] C7-C6 fold-average contrast, stratified scene bootstrap "
          f"B={args.n_boot} seed={args.seed} ...")
    point_foldavg = float(np.mean([w - s for s, w in
                                   zip(per_seed_simple["c6"], per_seed_simple["c7"])]))
    lo, hi, draws = stratified_fold_bootstrap(
        [merged_ledgers[("c7", seed)] for seed in SEEDS],
        [merged_ledgers[("c6", seed)] for seed in SEEDS],
        fold_scenes, args.n_boot, args.seed)
    print(f"      fold-average C7-C6 = {point_foldavg:+.6f}, 95% CI [{lo:+.6f}, {hi:+.6f}]")

    pooled_point, (p_lo, p_hi), _ = at.scene_bootstrap(
        [merged_ledgers[("c7", s)] for s in SEEDS],
        [merged_ledgers[("c6", s)] for s in SEEDS],
        B=args.n_boot, seed=args.seed)
    print(f"      pooled (criterion A) C7-C6 = {pooled_point:+.6f}, "
          f"95% CI [{p_lo:+.6f}, {p_hi:+.6f}]")

    print("[7/8] SHA-256 of 54 checkpoints and 54 OOF files ...")
    hash_rows = []
    for channel in CHANNELS:
        for seed in SEEDS:
            for k in FOLDS:
                run = f"fold{k}_{channel}_s{seed}"
                for kind, path in (("checkpoint", runs_dir / run / "weights" / "best.pt"),
                                   ("oof_json", oof_dir / f"{run}_oof.json")):
                    if not path.is_file():
                        hash_rows.append({"fold": k, "channel": channel, "seed": seed,
                                          "kind": kind, "path": str(path),
                                          "sha256": "MISSING", "size_bytes": "",
                                          "mtime_utc": ""})
                        print(f"      MISSING {path}")
                        continue
                    stat = path.stat()
                    hash_rows.append({
                        "fold": k, "channel": channel, "seed": seed, "kind": kind,
                        "path": path.as_posix(), "sha256": sha256_file(path),
                        "size_bytes": stat.st_size,
                        "mtime_utc": datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                    })
    print(f"      {len(hash_rows)} hash records")

    print("[8/8] auditing results.csv epoch columns ...")
    epoch_rows = []
    for channel in CHANNELS:
        for seed in SEEDS:
            for k in FOLDS:
                run = f"fold{k}_{channel}_s{seed}"
                info = audit_results_csv(runs_dir / run / "results.csv")
                epoch_rows.append({"fold": k, "channel": channel, "seed": seed,
                                   "run": run, "path": (runs_dir / run / "results.csv").as_posix(),
                                   **{key: ("" if value is None else value)
                                      for key, value in info.items()}})
    flagged = [r["run"] for r in epoch_rows
               if r.get("non_monotonic_or_duplicate") in (True, "True")]
    print(f"      {len(epoch_rows)} runs audited; "
          f"{len(flagged)} with duplicate/non-monotonic epochs"
          + (f": {', '.join(flagged)}" if flagged else ""))

    n_gt_per_fold = {k: int(sum(fold_ledgers[(k, "c6", SEEDS[0])][scene][2]
                                for scene in fold_scenes[k])) for k in FOLDS}
    folds_with_gt = [k for k in FOLDS if n_gt_per_fold[k] > 0]

    fold_csv = read_fold_csv(Path(args.fold_csv))
    csv_check = {
        "fold_csv": Path(args.fold_csv).as_posix(),
        "n_folds_in_csv": len(fold_csv),
        "identical_to_oof_folds": all(
            fold_csv.get(k) == fold_scenes[k] for k in FOLDS) and len(fold_csv) == len(FOLDS),
        "per_fold": {
            str(k): {"n_scenes_csv": len(fold_csv.get(k, [])),
                     "n_scenes_in_oof": len(fold_scenes[k]),
                     "identical": fold_csv.get(k) == fold_scenes[k]}
            for k in FOLDS
        },
        "handling": ("fold -> scene mapping is taken from the OOF file contents "
                     "(authoritative: each OOF holds only its own test scenes and is "
                     "validated by cv_pipeline.load_validated_oof); the CSV is a "
                     "cross-check only"),
    }

    write_csv(out / "fold_ap.csv",
              ["fold", "channel", "seed", "n_scenes_in_fold", "n_scenes_with_gt",
               "n_gt_in_fold", "ap"], fold_rows)
    write_csv(out / "checkpoint_hashes.csv",
              ["fold", "channel", "seed", "kind", "path", "sha256", "size_bytes",
               "mtime_utc"], hash_rows)
    write_csv(out / "epoch_audit.csv",
              ["fold", "channel", "seed", "run", "path", "present", "n_rows",
               "n_unique_epochs", "min_epoch", "max_epoch", "has_duplicate_epoch",
               "strictly_increasing", "non_monotonic_or_duplicate",
               "n_unparsable_epoch"], epoch_rows)

    summary = {
        "generated_utc": utc_now(),
        "script": "scripts/v2_cv_fold_ap.py",
        "plan_file": plan.as_posix(),
        "plan_sha256": plan_sha,
        "inputs": {
            "oof_dir": oof_dir.as_posix(),
            "runs_dir": runs_dir.as_posix(),
            "fold_csv": Path(args.fold_csv).as_posix(),
            "ref_json": Path(args.ref_json).as_posix(),
            "n_formal_oof": len(formal_files),
            "excluded_smoke_oof": [p.name for p in smoke_files],
            "n_scenes_per_channel_seed": 63,
            "n_patches_per_channel_seed": 1381,
            "n_gt_per_channel_seed": 212,
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
        },
        "coordinate_grid": grid,
        "sanity_check_criterion_a": {
            "tolerance": TOL,
            "expected_per_seed_AP": expected,
            "recomputed_per_seed_AP": recomputed,
            "passed": True,
        },
        "fold_csv_check": csv_check,
        "fold_ap": {
            "per_seed_fold_mean": per_seed_simple,
            "per_seed_fold_gt_weighted_mean": per_seed_weighted,
            "mean_over_seeds_fold_mean": {ch: float(np.mean(v))
                                          for ch, v in per_seed_simple.items()},
            "mean_over_seeds_fold_gt_weighted_mean": {ch: float(np.mean(v))
                                                      for ch, v in per_seed_weighted.items()},
            "n_folds_with_gt": len(folds_with_gt),
            "n_gt_per_fold": {str(k): n_gt_per_fold[k] for k in FOLDS},
        },
        "contrast_c7_minus_c6_fold_average": {
            "statistic": "mean over 3 paired seeds of (fold-mean AP(C7) - fold-mean AP(C6)); "
                         "fold mean is the simple average over folds with GT>0",
            "point": point_foldavg,
            "ci95": [lo, hi],
            "bootstrap_replicates": args.n_boot,
            "bootstrap_seed": args.seed,
            "bootstrap_kind": "stratified: scenes resampled with replacement inside each fold, "
                              "fold sizes preserved; folds whose resample has GT=0 are skipped",
            "bootstrap_note": ("bootstrap: scenes resampled once per replicate (within fold "
                               "for stratified), shared across seeds and channels; folds "
                               "whose resample has zero GT are skipped consistently"),
            "n_draws_used": int(draws.size),
            f"ci_vs_margin_{cv.MARGIN}": at.ci_vs_margin(lo, hi, cv.MARGIN),
            "margin": cv.MARGIN,
        },
        "contrast_c7_minus_c6_pooled_reference": {
            "statistic": "at.scene_bootstrap on the globally merged 63-scene ledgers",
            "point": float(pooled_point),
            "ci95": [float(p_lo), float(p_hi)],
            "bootstrap_replicates": args.n_boot,
            "bootstrap_seed": args.seed,
            f"ci_vs_margin_{cv.MARGIN}": at.ci_vs_margin(p_lo, p_hi, cv.MARGIN),
        },
        "notes": {
            "bootstrap": ("bootstrap: scenes resampled once per replicate (within fold for "
                          "stratified), shared across seeds and channels; folds whose "
                          "resample has zero GT are skipped consistently"),
        },
        "epoch_audit": {
            "n_runs_audited": len(epoch_rows),
            "n_runs_missing_results_csv": sum(1 for r in epoch_rows if not r["present"]),
            "runs_with_duplicate_or_non_monotonic_epoch": flagged,
            "files_modified": False,
        },
        "outputs": ["fold_ap.csv", "summary.json", "checkpoint_hashes.csv", "epoch_audit.csv"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"[done] wrote {out}/fold_ap.csv, summary.json, checkpoint_hashes.csv, epoch_audit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
