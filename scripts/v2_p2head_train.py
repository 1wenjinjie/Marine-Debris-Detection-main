"""
P2-head training for MARIDA v2 — 6 spectral channels, 4-scale detection.

Motivation:
  Marine_Debris median size ~14px in 512px input.
    stride-8 (P3): ~1.75 feature cells — unreliable
    stride-4 (P2): ~3.5 feature cells — learnable

Architecture change:
  yolov11n_v2_6ch_p2.yaml adds a 4th detection head at stride 4 (P2/4).
  All other hyperparameters are identical to the baseline.

Hypothesis: Recall=0 is caused by insufficient feature map resolution
for small objects, not by class imbalance or spectral indistinguishability.

Usage:
  python scripts/v2_p2head_train.py --seed 0
  python scripts/v2_p2head_train.py --seed 0 1 2
  python scripts/v2_p2head_train.py --seed 0 --eval-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

DATA_6CH  = str(ROOT / "configs" / "datasets" / "marida_v2_6ch.yaml")
MODEL_P2  = str(ROOT / "configs" / "models" / "yolov11n_v2_6ch_p2.yaml")
RUNS_DIR  = ROOT / "runs_v2"

CLASS_NAMES = ["Marine_Debris", "Algae_Organic", "Ship"]


def train_one_seed(seed: int, eval_only: bool = False) -> dict:
    """Train the P2-head model for one seed and evaluate on the 6ch test set."""
    run_name = f"p2head_s{seed}"
    project   = str(RUNS_DIR)
    best_weights = str(RUNS_DIR / run_name / "weights" / "best.pt")

    if not eval_only:
        overrides = {
            **MASK_RESCUE_TRAIN_DEFAULTS,
            "model":    MODEL_P2,
            "data":     DATA_6CH,
            "project":  project,
            "name":     run_name,
            "seed":     seed,
            "exist_ok": True,
            "device":   "0",
        }

        print(f"\n{'='*60}")
        print(f"P2-Head Training: {run_name}  (seed={seed})")
        print(f"  Model: yolov11n_v2_6ch_p2.yaml  (4-scale: P2/P3/P4/P5)")
        print(f"  Hypothesis: stride-4 head resolves small-object recall=0")
        print(f"{'='*60}")

        trainer = MaskRescueTrainer(overrides=overrides)
        trainer.train()
    else:
        if not Path(best_weights).exists():
            raise FileNotFoundError(
                f"--eval-only but weights not found: {best_weights}"
            )
        print(f"\n[eval-only] Using: {best_weights}")

    print(f"\nEvaluating {run_name} on 6ch test set...")
    from scripts.v2_evaluate import evaluate
    metrics = evaluate(
        weights=best_weights,
        data_yaml=DATA_6CH,
        channels=6,
        split="test",
    )
    metrics["seed"]     = seed
    metrics["run_name"] = run_name
    metrics["model"]    = "yolov11n_v2_6ch_p2"

    out_path = RUNS_DIR / run_name / "eval_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    # ---- Print summary ----
    per    = metrics.get("per_class", {})
    debris = per.get("Marine_Debris", {})
    algae  = per.get("Algae_Organic", {})
    ship   = per.get("Ship", {})

    print(f"\n{'='*60}")
    print(f"  P2-Head Results — {run_name}")
    print(f"{'='*60}")
    print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print()
    print("  Per-class:")
    for name, cls_m in [("Marine_Debris", debris), ("Algae_Organic", algae), ("Ship", ship)]:
        print(f"    {name:20s}: AP50={cls_m.get('AP50', 0):.4f}  "
              f"P={cls_m.get('precision', 0):.4f}  "
              f"R={cls_m.get('recall', 0):.4f}")

    # Verdict
    debris_recall = debris.get("recall", 0.0)
    debris_ap50   = debris.get("AP50", 0.0)
    if debris_recall > 0.0 or debris_ap50 > 0.01:
        print(f"\n  >> SIGNAL: Marine_Debris recall={debris_recall:.4f} AP50={debris_ap50:.4f}")
        if debris_recall >= 0.154 and debris_ap50 > 0.05:
            print("  >> PAPER-GRADE: meets publication threshold!")
        else:
            print("  >> Partial signal — hypothesis A (scale) partially confirmed.")
    else:
        print(f"\n  >> NO SIGNAL: Marine_Debris recall={debris_recall:.4f}")
        print("     Scale alone insufficient — consider tiled inference or larger input.")

    print(f"\n  Saved: {out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="MARIDA v2 P2-head training")
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, re-evaluate existing weights")
    args = parser.parse_args()

    all_results = []
    for seed in args.seed:
        result = train_one_seed(seed, eval_only=args.eval_only)
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"P2-HEAD MULTI-SEED SUMMARY ({len(all_results)} seeds)")
        print(f"{'='*60}")

        for metric in ["mAP50", "mAP50_95", "precision", "recall"]:
            vals = [r[metric] for r in all_results]
            print(f"  {metric:15s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
                  f"({', '.join(f'{v:.4f}' for v in vals)})")

        print(f"\n  Marine_Debris:")
        for m in ["AP50", "recall", "precision"]:
            vals = [r["per_class"]["Marine_Debris"][m] for r in all_results]
            print(f"    {m:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

        summary = {
            "experiment": "v2_p2head",
            "model":      "yolov11n_v2_6ch_p2",
            "seeds":      args.seed,
            "results":    all_results,
            "summary": {
                metric: {
                    "mean": float(np.mean([r[metric] for r in all_results])),
                    "std":  float(np.std([r[metric] for r in all_results])),
                }
                for metric in ["mAP50", "mAP50_95", "precision", "recall"]
            },
        }
        summary_path = RUNS_DIR / "p2head_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
