"""
MARIDA v2 baseline training: 6ch student model (YOLOv11n).

This is the reference baseline for all subsequent experiments.
Trains on marida_v2_6ch.yaml with standardized hyperparameters.

Usage:
  # Single seed
  python scripts/v2_baseline_train.py --seed 0

  # Multi-seed (3 seeds for stability)
  python scripts/v2_baseline_train.py --seed 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

DATA_6CH = str(ROOT / "configs" / "datasets" / "marida_v2_6ch.yaml")
MODEL_6CH = str(ROOT / "configs" / "models" / "yolov11n_v2_6ch.yaml")
RUNS_DIR = ROOT / "runs_v2"


def train_one_seed(seed: int) -> dict:
    """Train a 6ch baseline for one seed and return eval metrics."""
    run_name = f"baseline_6ch_s{seed}"
    project = str(RUNS_DIR)

    overrides = {
        **MASK_RESCUE_TRAIN_DEFAULTS,
        "model": MODEL_6CH,
        "data": DATA_6CH,
        "project": project,
        "name": run_name,
        "seed": seed,
        "exist_ok": True,
        "device": "0",
    }

    print(f"\n{'='*60}")
    print(f"Training: {run_name} (seed={seed})")
    print(f"{'='*60}")

    trainer = MaskRescueTrainer(overrides=overrides)
    trainer.train()

    # Evaluate on test set
    print(f"\nEvaluating {run_name} on test set...")
    from scripts.v2_evaluate import evaluate
    best_weights = str(RUNS_DIR / run_name / "weights" / "best.pt")
    metrics = evaluate(
        weights=best_weights,
        data_yaml=DATA_6CH,
        channels=6,
        split="test",
    )
    metrics["seed"] = seed
    metrics["run_name"] = run_name

    # Save per-seed results
    out_path = RUNS_DIR / run_name / "eval_test.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved: {out_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="MARIDA v2 baseline training")
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    all_results = []
    for seed in args.seed:
        result = train_one_seed(seed)
        all_results.append(result)

    # Summary
    if len(all_results) > 1:
        import numpy as np
        print(f"\n{'='*60}")
        print(f"MULTI-SEED SUMMARY ({len(all_results)} seeds)")
        print(f"{'='*60}")
        for metric in ["mAP50", "mAP50_95", "precision", "recall"]:
            vals = [r[metric] for r in all_results]
            print(f"  {metric:15s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
                  f"({', '.join(f'{v:.4f}' for v in vals)})")

        summary = {
            "experiment": "v2_baseline_6ch",
            "seeds": args.seed,
            "results": all_results,
            "summary": {
                metric: {"mean": float(np.mean([r[metric] for r in all_results])),
                         "std": float(np.std([r[metric] for r in all_results]))}
                for metric in ["mAP50", "mAP50_95", "precision", "recall"]
            },
        }
        summary_path = RUNS_DIR / "baseline_6ch_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
