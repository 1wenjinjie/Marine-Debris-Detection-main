"""Phase 2 high-throughput parallel D4 rerun:
  - Stage 2A: 9-fold 2x2 factorial audit (108 runs) -> runs_cv_factorial_d4/
  - Stage 2B: 10-scene locked channels (9 runs) + MIN sensitivity (18 runs) = 27 runs -> runs_mpb_d4/

Preserves all historical runs_cv_factorial/ and runs_mpb/ untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "mpb"))

from cv_factorial_scheduler import run_locked, schedule  # noqa: E402

FACT_RUNS = ROOT / "runs_cv_factorial_d4"
FACT_OOF = FACT_RUNS / "oof"
FACT_LOCKS = FACT_RUNS / ".locks"
FACT_ANALYSIS = FACT_RUNS / "analysis"

MPB_RUNS = ROOT / "runs_mpb_d4"
MPB_FULL = MPB_RUNS / "full"
MPB_LOCKS = MPB_RUNS / ".locks"

SEEDS = (0, 1, 2)
FACT_CELLS = ("clean_ndwi", "clean_oracle", "shared_ndwi", "shared_oracle")
MPB_CELLS = (
    "full_scene_6ch",
    "full_scene_ndwi7",
    "full_scene_shuffled7",
    "min_train1",
    "min_train2",
    "min_train8",
    "min_matched1",
    "min_matched2",
    "min_matched8",
)


def _worker_init(device: str, threads_per_worker: int) -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)

    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(threads_per_worker)

    global _DEVICE, _rex, _cvp, _fact
    _DEVICE = device
    import cv_pipeline as cvp
    import run_experiments as rex
    import v2_cv_factorial as fact

    _rex = rex
    _cvp = cvp
    _fact = fact


def _atomic_write_ledger(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    _rex._write_prediction_ledger(tmp, entries)
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _worker_fact(task: tuple[int, str, int]) -> str:
    k, cell, seed = task
    run_name = f"fold{k}_{cell}_s{seed}"
    ddir = _fact.FACT_ROOT / f"fold{k}_{cell}"
    cv_yaml = ddir / "cv.yaml"
    run_dir = FACT_RUNS / run_name
    val_path = run_dir / "val_predictions.json"
    oof_path = FACT_OOF / f"{run_name}_oof.json"
    lock_path = FACT_LOCKS / f"{run_name}.lock"

    def action() -> None:
        if run_dir.exists() and not (oof_path.is_file() and val_path.is_file() and (run_dir / "weights" / "best.pt").is_file()):
            shutil.rmtree(run_dir, ignore_errors=True)
        if oof_path.exists():
            oof_path.unlink(missing_ok=True)

        t0 = time.time()
        init = _rex._ensure_initialization()[7]
        overrides = {
            **_rex.LOCKED_HYPERPARAMS,
            "model": str(init),
            "data": str(cv_yaml),
            "project": str(FACT_RUNS),
            "name": run_name,
            "seed": seed,
            "device": _DEVICE,
            "exist_ok": True,
        }
        print(f"[2A start] {run_name} (pid={os.getpid()})", flush=True)
        trainer = _rex.MPBTrainer(overrides=overrides)
        trainer.train()

        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(best)

        imgsz = int(_rex.LOCKED_HYPERPARAMS["imgsz"])
        val_entries = _rex.collect_predictions(best, cv_yaml, "val", imgsz=imgsz, device=_DEVICE)
        _atomic_write_ledger(val_path, val_entries)

        test_entries = _rex.collect_predictions(best, cv_yaml, "test", imgsz=imgsz, device=_DEVICE, allow_test=True)
        _atomic_write_ledger(oof_path, test_entries)
        print(f"[2A done]  {run_name} in {(time.time() - t0) / 60:.1f} min (entries={len(test_entries)})", flush=True)

    return run_locked(lock_path, run_dir, oof_path, action, skip_completed=True)


def _worker_mpb(task: tuple[str, int]) -> str:
    cell_name, seed = task
    cell = _rex.CELLS[cell_name]
    data_config = _rex._config_path(cell)
    run_name = f"{cell.name}_s{seed}"
    run_dir = MPB_FULL / run_name
    val_ledger = run_dir / "val_predictions.json"
    val_op = run_dir / "val_operating_points.json"
    test_ledger = run_dir / "test_predictions_locked.json"
    eval_locked = run_dir / "eval_test_locked.json"
    lock_path = MPB_LOCKS / f"{run_name}.lock"

    def action() -> None:
        complete = all(p.is_file() for p in (run_dir / "weights" / "best.pt", val_ledger, val_op, test_ledger, eval_locked))
        if run_dir.exists() and not complete:
            shutil.rmtree(run_dir, ignore_errors=True)

        t0 = time.time()
        init = _rex._ensure_initialization()[cell.channels]
        overrides = {
            **_rex.LOCKED_HYPERPARAMS,
            "model": str(init),
            "data": str(data_config),
            "project": str(MPB_FULL),
            "name": run_name,
            "seed": seed,
            "device": _DEVICE,
            "exist_ok": True,
        }
        print(f"[2B start] {run_name} (pid={os.getpid()})", flush=True)
        trainer = _rex.MPBTrainer(overrides=overrides)
        trainer.train()

        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(best)

        imgsz = int(_rex.LOCKED_HYPERPARAMS["imgsz"])
        val_entries = _rex.collect_predictions(best, data_config, "val", imgsz=imgsz, device=_DEVICE)
        _atomic_write_ledger(val_ledger, val_entries)
        operating = _rex.validation_operating_points(val_entries)
        _atomic_write_json(val_op, operating)

        balanced_thr = float(operating["selected"]["balanced"]["threshold"])
        test_entries = _rex.collect_predictions(best, data_config, "test", imgsz=imgsz, device=_DEVICE, allow_test=True)
        _atomic_write_ledger(test_ledger, test_entries)

        locked_metrics = {
            "cell": cell.name,
            "seed": seed,
            "channels": cell.channels,
            "augmentation": "RandomD4",
            "val_locked_balanced_threshold": balanced_thr,
            "val_balanced_f1": float(operating["selected"]["balanced"]["f1"]),
            "val_high_recall_feasible": operating["selected"]["high_recall_r_ge_0.80"] is not None,
            "test_ap50_ledger": float(_rex.ap50(test_entries)),
            "test_at_val_balanced": _rex.operating_metrics(test_entries, balanced_thr),
            "test_scene_bootstrap": _rex.bootstrap_scene_ci(test_entries, balanced_thr, iterations=10000, seed=20260818),
        }
        _atomic_write_json(eval_locked, locked_metrics)
        print(
            f"[2B done]  {run_name} in {(time.time() - t0) / 60:.1f} min "
            f"(test AP50={locked_metrics['test_ap50_ledger']:.4f}, thr={balanced_thr}, "
            f"test F1={locked_metrics['test_at_val_balanced']['f1']:.4f})",
            flush=True,
        )

    return run_locked(lock_path, run_dir, eval_locked, action, skip_completed=True)


def analyze_factorial(n_boot: int = 10000) -> dict:
    import v2_cv_factorial as fact

    fact.load_runtime()
    orig_runs, orig_oof, orig_out = fact.RUNS, fact.OOF_DIR, fact.ANALYSIS_OUT
    try:
        fact.RUNS = FACT_RUNS
        fact.OOF_DIR = FACT_OOF
        fact.ANALYSIS_OUT = FACT_ANALYSIS
        if FACT_ANALYSIS.exists():
            shutil.rmtree(FACT_ANALYSIS)
        fact.analyze(n_boot)
    finally:
        fact.RUNS, fact.OOF_DIR, fact.ANALYSIS_OUT = orig_runs, orig_oof, orig_out
    return json.loads((FACT_ANALYSIS / "summary.json").read_text(encoding="utf-8"))


def analyze_mpb() -> dict:
    import numpy as np
    import run_experiments as rex

    entries_by_run: dict[tuple[str, int], list[dict]] = {}
    thresholds: dict[tuple[str, int], float] = {}
    per_cell: dict[str, dict] = {}

    for cell_name in MPB_CELLS:
        aps, f1s, thrs = [], [], []
        seed_rows = {}
        for s in SEEDS:
            run_dir = MPB_FULL / f"{cell_name}_s{s}"
            ev = json.loads((run_dir / "eval_test_locked.json").read_text(encoding="utf-8"))
            ents = rex._read_prediction_ledger(run_dir / "test_predictions_locked.json")
            entries_by_run[(cell_name, s)] = ents
            thr = float(ev["val_locked_balanced_threshold"])
            thresholds[(cell_name, s)] = thr
            aps.append(float(ev["test_ap50_ledger"]))
            f1s.append(float(ev["test_at_val_balanced"]["f1"]))
            thrs.append(thr)
            seed_rows[str(s)] = {
                "ap50": round(float(ev["test_ap50_ledger"]), 4),
                "val_thr": thr,
                "test_f1": round(float(ev["test_at_val_balanced"]["f1"]), 4),
                "test_p": round(float(ev["test_at_val_balanced"]["precision"]), 4),
                "test_r": round(float(ev["test_at_val_balanced"]["recall"]), 4),
            }
        per_cell[cell_name] = {
            "ap50_mean": round(float(np.mean(aps)), 4),
            "ap50_std": round(float(np.std(aps, ddof=1)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "per_seed": seed_rows,
        }

    channel_contrasts = rex.paired_contrast_summary(
        entries_by_run,
        thresholds,
        ("full_scene_6ch", "full_scene_ndwi7", "full_scene_shuffled7"),
        {
            "ndwi7_minus_6ch": {"full_scene_ndwi7": 1.0, "full_scene_6ch": -1.0},
            "ndwi7_minus_shuffled7": {"full_scene_ndwi7": 1.0, "full_scene_shuffled7": -1.0},
            "shuffled7_minus_6ch": {"full_scene_shuffled7": 1.0, "full_scene_6ch": -1.0},
        },
        iterations=10000,
        seed=20260818,
    )
    min_fixed_contrasts = rex.paired_contrast_summary(
        entries_by_run,
        thresholds,
        ("full_scene_6ch", "min_train1", "min_train2", "min_train8"),
        {
            "min1_minus_min4": {"min_train1": 1.0, "full_scene_6ch": -1.0},
            "min2_minus_min4": {"min_train2": 1.0, "full_scene_6ch": -1.0},
            "min8_minus_min4": {"min_train8": 1.0, "full_scene_6ch": -1.0},
        },
        iterations=10000,
        seed=20260818,
    )

    summary = {
        "augmentation": "RandomD4",
        "per_cell": per_cell,
        "channel_contrasts": channel_contrasts,
        "min_fixed_contrasts": min_fixed_contrasts,
    }
    out = MPB_RUNS / "summary_all.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== STAGE 2B (10-SCENE + MIN SENSITIVITY) SUMMARY ===", flush=True)
    print(json.dumps({"per_cell": per_cell}, indent=2, ensure_ascii=False), flush=True)
    print(f"written: {out}", flush=True)
    return summary


def preflight() -> None:
    import run_experiments as rex
    import v2_cv_factorial as fact

    rex._ensure_ultralytics_font()
    rex._ensure_initialization()
    fact.load_runtime()
    if not fact.DESIGN_JSON.is_file():
        raise FileNotFoundError(fact.DESIGN_JSON)
    designs = json.loads(fact.DESIGN_JSON.read_text(encoding="utf-8"))
    for d in designs:
        for cell in FACT_CELLS:
            cv_yaml = fact.FACT_ROOT / f"fold{d['fold']}_{cell}" / "cv.yaml"
            if not cv_yaml.is_file():
                fact.prepare_cell(d, cell, smoke=False)
    for cell_name in MPB_CELLS:
        cfg = rex._config_path(rex.CELLS[cell_name])
        if not cfg.is_file():
            raise FileNotFoundError(cfg)
    print(f"[preflight] OK: 36 factorial fold-cell YAMLs + {len(MPB_CELLS)} MPB configs verified.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--device", default="0")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stage", choices=("all", "2a", "2b"), default="all")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    for p in (FACT_RUNS, FACT_OOF, FACT_LOCKS, MPB_RUNS, MPB_FULL, MPB_LOCKS):
        p.mkdir(parents=True, exist_ok=True)

    preflight()
    if args.check_only:
        return

    t_all = time.time()
    if args.stage in ("all", "2a"):
        if not args.analyze_only:
            fact_tasks = [(k, cell, s) for s in SEEDS for k in range(9) for cell in FACT_CELLS]
            t0 = time.time()
            print(f"\n[2A] launching {len(fact_tasks)} factorial runs (jobs={args.jobs}) ...", flush=True)
            counts = schedule(
                fact_tasks,
                _worker_fact,
                jobs=args.jobs,
                initializer=_worker_init,
                initargs=(args.device, args.threads),
            )
            print(f"[2A] finished in {(time.time() - t0) / 60:.1f} min: {dict(counts)}", flush=True)
        analyze_factorial(n_boot=10000)

    if args.stage in ("all", "2b"):
        if not args.analyze_only:
            mpb_tasks = [(cell_name, s) for s in SEEDS for cell_name in MPB_CELLS]
            t0 = time.time()
            print(f"\n[2B] launching {len(mpb_tasks)} 10-scene + MIN runs (jobs={args.jobs}) ...", flush=True)
            counts = schedule(
                mpb_tasks,
                _worker_mpb,
                jobs=args.jobs,
                initializer=_worker_init,
                initargs=(args.device, args.threads),
            )
            print(f"[2B] finished in {(time.time() - t0) / 60:.1f} min: {dict(counts)}", flush=True)
        analyze_mpb()

    print(f"\n[phase2_d4] ALL COMPLETE in {(time.time() - t_all) / 3600:.2f} hours.", flush=True)


if __name__ == "__main__":
    main()
