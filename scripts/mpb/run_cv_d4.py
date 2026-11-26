"""High-throughput parallel 9-fold scene CV with lossless D4 augmentation.

Writes all outputs to runs_cv_d4/ without modifying historical runs_cv/.
Supports --jobs N for multi-process GPU saturation on RTX 4090 and full resumability.
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

RUNS_CV_D4 = ROOT / "runs_cv_d4"
OOF_DIR = RUNS_CV_D4 / "oof"
LOCK_DIR = RUNS_CV_D4 / ".locks"
FULL_SEEDS = (0, 1, 2)
CHKEYS = ("c6", "c7")


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

    global _WORKER_DEVICE, _cvp, _rex
    _WORKER_DEVICE = device
    import cv_pipeline as cvp
    import run_experiments as rex

    _cvp = cvp
    _rex = rex


def _atomic_write_ledger(path: Path, entries: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    _rex._write_prediction_ledger(tmp, entries)
    tmp.replace(path)


def _worker_run(task: tuple[int, str, int]) -> str:
    k, chkey, seed = task
    channels = _cvp.CHDEF[chkey][0]
    ddir = _cvp.CV_ROOT / f"fold{k}_{chkey}"
    cv_yaml = ddir / "cv.yaml"
    run_name = f"fold{k}_{chkey}_s{seed}"
    run_dir = RUNS_CV_D4 / run_name
    val_path = run_dir / "val_predictions.json"
    oof_path = OOF_DIR / f"{run_name}_oof.json"
    lock_path = LOCK_DIR / f"{run_name}.lock"

    def action() -> None:
        if run_dir.exists() and not (oof_path.is_file() and val_path.is_file() and (run_dir / "weights" / "best.pt").is_file()):
            shutil.rmtree(run_dir, ignore_errors=True)
        if oof_path.exists():
            oof_path.unlink(missing_ok=True)

        t0 = time.time()
        init = _rex._ensure_initialization()[channels]
        overrides = {
            **_rex.LOCKED_HYPERPARAMS,
            "model": str(init),
            "data": str(cv_yaml),
            "project": str(RUNS_CV_D4),
            "name": run_name,
            "seed": seed,
            "device": _WORKER_DEVICE,
            "exist_ok": True,
        }
        print(f"[start] {run_name} (pid={os.getpid()})", flush=True)
        trainer = _rex.MPBTrainer(overrides=overrides)
        trainer.train()

        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(best)

        imgsz = int(_rex.LOCKED_HYPERPARAMS["imgsz"])
        val_entries = _rex.collect_predictions(best, cv_yaml, "val", imgsz=imgsz, device=_WORKER_DEVICE)
        _atomic_write_ledger(val_path, val_entries)

        test_entries = _rex.collect_predictions(best, cv_yaml, "test", imgsz=imgsz, device=_WORKER_DEVICE, allow_test=True)
        _atomic_write_ledger(oof_path, test_entries)
        print(f"[done]  {run_name} in {(time.time() - t0) / 60:.1f} min (entries={len(test_entries)})", flush=True)

    return run_locked(lock_path, run_dir, oof_path, action, skip_completed=True)


def analyze() -> dict:
    import numpy as np
    import audit_toolkit as at
    import cv_pipeline as cvp
    import run_experiments as rex

    orig_runs = cvp.RUNS_CV
    orig_oof = cvp.OOF_DIR
    try:
        cvp.RUNS_CV = RUNS_CV_D4
        cvp.OOF_DIR = OOF_DIR
        report = cvp.merge_and_analyze(FULL_SEEDS, smoke=False)
    finally:
        cvp.RUNS_CV = orig_runs
        cvp.OOF_DIR = orig_oof

    per_fold_c6 = [r["c6_AP"] for r in report["per_fold_per_seed"]]
    per_fold_c7 = [r["c7_AP"] for r in report["per_fold_per_seed"]]
    report["augmentation"] = "RandomD4 (lossless 90-deg rotations x horizontal flip, 8 symmetries)"
    report["fold_mean_AP"] = {
        "c6": round(float(np.mean(per_fold_c6)), 4),
        "c7": round(float(np.mean(per_fold_c7)), 4),
        "c7_minus_c6": round(float(np.mean(per_fold_c7) - np.mean(per_fold_c6)), 4),
    }

    old_stat_path = ROOT / "runs_cv" / "cv_stat_results.json"
    if old_stat_path.is_file():
        old = json.loads(old_stat_path.read_text(encoding="utf-8"))
        report["comparison_vs_old_noaug"] = {
            "old_c6_pooled_AP": old["channels"]["c6"]["mean_AP"],
            "new_c6_pooled_AP": report["channels"]["c6"]["mean_AP"],
            "delta_c6_pooled_AP": round(report["channels"]["c6"]["mean_AP"] - old["channels"]["c6"]["mean_AP"], 4),
            "old_c7_pooled_AP": old["channels"]["c7"]["mean_AP"],
            "new_c7_pooled_AP": report["channels"]["c7"]["mean_AP"],
            "delta_c7_pooled_AP": round(report["channels"]["c7"]["mean_AP"] - old["channels"]["c7"]["mean_AP"], 4),
        }

    out = RUNS_CV_D4 / "cv_stat_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== 9-FOLD CHANNEL CV (D4) SUMMARY ===", flush=True)
    print(json.dumps({
        "channels": report["channels"],
        "fold_mean_AP": report["fold_mean_AP"],
        "contrast_c7_minus_c6": {k: v for k, v in report["contrast_c7_minus_c6"].items() if k != "jackknife_all"},
        "comparison_vs_old_noaug": report.get("comparison_vs_old_noaug"),
    }, indent=2, ensure_ascii=False), flush=True)
    print(f"written: {out}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--device", default="0")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    RUNS_CV_D4.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        import cv_pipeline as cvp
        import run_experiments as rex

        rex._ensure_ultralytics_font()
        rex._ensure_initialization()
        for k in range(9):
            for chkey in CHKEYS:
                ddir = cvp.CV_ROOT / f"fold{k}_{chkey}"
                if not (ddir / "cv.yaml").is_file():
                    cvp.prepare_fold(k, chkey, smoke=False)

        tasks = [(k, chkey, seed) for seed in FULL_SEEDS for k in range(9) for chkey in CHKEYS]
        t0 = time.time()
        print(f"[cv_d4] launching {len(tasks)} runs with jobs={args.jobs} on device={args.device}", flush=True)
        counts = schedule(
            tasks,
            _worker_run,
            jobs=args.jobs,
            initializer=_worker_init,
            initargs=(args.device, args.threads),
        )
        print(f"[cv_d4] all tasks finished in {(time.time() - t0) / 60:.1f} min: {dict(counts)}", flush=True)

    analyze()


if __name__ == "__main__":
    main()
