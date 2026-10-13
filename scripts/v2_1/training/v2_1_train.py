"""
v2.1 Phase 3 production trainer — binary Marine_Debris instance segmentation.

Per SPEC §14 training matrix:
  main:      baseline_v2_1_s{0,1,2}      imgsz=512   3-seed
             baseline_v2_1_i1024_s{0,1,2} imgsz=1024  3-seed
  ablation:  many single-seed variants

This script handles the main runs. Ablations have their own thin
wrappers (see v2_1_ablation_*.py).

Usage:
  # one seed at imgsz=512
  python scripts/v2_1/training/v2_1_train.py --seed 0 --imgsz 512

  # multi-seed series at imgsz=1024
  python scripts/v2_1/training/v2_1_train.py --seed 0 1 2 --imgsz 1024

  # eval-only (load existing best.pt and re-run val)
  python scripts/v2_1/training/v2_1_train.py --seed 0 --imgsz 512 --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Triggers IMG_FORMATS / verify_image_label patches + module registration
import ultralytics_plugins_v2_1  # noqa: F401
from ultralytics_plugins_v2_1 import MaskRescueSegTrainer_v2_1


DATA_YAML = (
    ROOT / "configs" / "datasets" / "v2_1" /
    "marida_v2_1_binary_highmod.yaml"
)
RUNS_ROOT = ROOT / "runs_v2_1"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True, cwd=str(ROOT), timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _build_overrides(seed: int, imgsz: int, name: str, epochs: int,
                     workers: int) -> dict:
    """Per SPEC §20 — explicit zero on spectral-incompatible aug."""
    batch = 8 if imgsz <= 512 else 4
    return {
        "model":       "yolo11n-seg.pt",
        "data":        str(DATA_YAML),
        "project":     str(RUNS_ROOT),
        "name":        name,
        "exist_ok":    True,
        "task":        "segment",

        # Training
        "epochs":      epochs,
        "batch":       batch,
        "imgsz":       imgsz,
        "workers":     workers,
        "device":      "0",
        "seed":        seed,
        "patience":    50,
        "save_period": -1,
        "plots":       True,
        "val":         True,
        # AMP disabled: with 9-channel inflated input + small batch +
        # dilated 1-px masks, FP16 gradient overflow happens around
        # epoch 4-6 (val losses go NaN). FP32 is slower but stable
        # for this small-data regime.
        "amp":         False,
        # Slightly slower warmup helps avoid weight blowup with
        # inflated-conv first layer.
        "warmup_epochs": 5.0,

        # Aug: SPEC §20 — disable everything spectral-unsafe
        "mosaic":      0.0,
        "mixup":       0.0,
        "copy_paste":  0.0,
        "hsv_h":       0.0,
        "hsv_s":       0.0,
        "hsv_v":       0.0,
        "degrees":     180.0,
        "fliplr":      0.5,
        "flipud":      0.5,
        "scale":       0.5,
        "shear":       0.0,
        "perspective": 0.0,
    }


def _save_manifest(run_dir: Path, overrides: dict, seed: int, imgsz: int,
                    t_start: float, status: str = "running") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version":           "v2.1.2",
        "git_commit":        _git_commit(),
        "seed":              seed,
        "imgsz":             imgsz,
        "status":            status,
        "started_utc":       time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "elapsed_seconds":   time.time() - t_start,
        "overrides":         overrides,
        "spec_version":      "v2_1_DATASET_SPEC.md",
        "label_set":         "highmod",
        "task":              "binary instance segmentation",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))


def train_one_seed(seed: int, imgsz: int, epochs: int = 50,
                   eval_only: bool = False, workers: int = 0) -> dict:
    """Train one seed; return summary dict."""
    tag = f"baseline_v2_1{'_i1024' if imgsz > 512 else ''}_s{seed}"
    run_dir = RUNS_ROOT / tag
    t_start = time.time()

    overrides = _build_overrides(seed, imgsz, tag, epochs, workers)
    print(f"\n{'=' * 70}")
    print(f"v2.1 Phase 3 train  |  {tag}")
    print(f"  imgsz={imgsz}  seed={seed}  epochs={epochs}  batch={overrides['batch']}")
    print(f"  data: {DATA_YAML.name}")
    print(f"  out:  {run_dir}")
    print(f"{'=' * 70}\n")

    _save_manifest(run_dir, overrides, seed, imgsz, t_start, status="running")

    try:
        if eval_only:
            overrides["model"] = str(run_dir / "weights" / "best.pt")
            overrides["epochs"] = 0
            trainer = MaskRescueSegTrainer_v2_1(overrides=overrides)
            trainer.train()
        else:
            trainer = MaskRescueSegTrainer_v2_1(overrides=overrides)
            trainer.train()
    except Exception as e:
        _save_manifest(run_dir, overrides, seed, imgsz, t_start, status="error")
        raise

    _save_manifest(run_dir, overrides, seed, imgsz, t_start, status="done")

    # Try to surface key metrics if results.csv is present
    metrics = {}
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        lines = csv_path.read_text().strip().splitlines()
        if len(lines) >= 2:
            header = [c.strip() for c in lines[0].split(",")]
            vals = [c.strip() for c in lines[-1].split(",")]
            for h, v in zip(header, vals):
                try:
                    metrics[h] = float(v)
                except ValueError:
                    pass
    return {"tag": tag, "run_dir": str(run_dir),
             "elapsed_seconds": time.time() - t_start, "metrics": metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description="v2.1 binary MD seg trainer")
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--imgsz", type=int, default=512, choices=(512, 1024))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--workers", type=int, default=0,
                     help="0 to avoid Windows pagefile issues")
    args = ap.parse_args()

    all_summary = []
    for seed in args.seed:
        summary = train_one_seed(
            seed=seed, imgsz=args.imgsz, epochs=args.epochs,
            eval_only=args.eval_only, workers=args.workers,
        )
        all_summary.append(summary)
        print(f"\n[OK] {summary['tag']} done in "
              f"{summary['elapsed_seconds']:.0f}s")
        if summary["metrics"]:
            keys = ["metrics/mAP50(B)", "metrics/mAP50(M)",
                     "metrics/mAP50-95(B)", "metrics/mAP50-95(M)"]
            for k in keys:
                if k in summary["metrics"]:
                    print(f"  {k:25s} = {summary['metrics'][k]:.4f}")

    # multi-seed summary
    if len(all_summary) > 1:
        summary_path = RUNS_ROOT / (
            f"baseline_v2_1{'_i1024' if args.imgsz > 512 else ''}_summary.json"
        )
        summary_path.write_text(json.dumps(all_summary, indent=2, default=str))
        print(f"\nMulti-seed summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
