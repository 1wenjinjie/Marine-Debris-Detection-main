"""
imgsz=1024 training for MARIDA v2 — clean scale hypothesis test.

Motivation (vs P2 head experiment):
  P2 head added stride-4 feature map but kept imgsz=512:
    14px debris → P2 cell coverage ~3.5 cells → model still underfits
  imgsz=1024 keeps the SAME 3-scale architecture but doubles resolution:
    14px debris → P3 stride-8 @ 1024px → ~3.5 cells (same as P2@512)
    14px debris → P2 stride-4 @ 1024px → ~7 cells   (clear)

  This is a CLEANER test of hypothesis A (scale) because:
    - Same architecture as baseline (3 detect heads, no structural changes)
    - Only variable changed is input/feature-map resolution
    - More publishable: "baseline + 2× resolution" is a standard ablation

Memory note:
  imgsz=1024, 6ch, ~2.7M params → ~6 GB GPU RAM @ batch=4 (RTX 3090 Ti: 24 GB)
  Default batch=4. If OOM, reduce to 2.

Usage:
  python scripts/v2_imgsz1024_train.py --seed 0
  python scripts/v2_imgsz1024_train.py --seed 0 1 2
  python scripts/v2_imgsz1024_train.py --seed 0 --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

DATA_6CH  = str(ROOT / "configs" / "datasets" / "marida_v2_6ch.yaml")
MODEL_6CH = str(ROOT / "configs" / "models" / "yolov11n_v2_6ch.yaml")   # SAME as baseline
RUNS_DIR  = ROOT / "runs_v2"

CLASS_NAMES = ["Marine_Debris", "Algae_Organic", "Ship"]

# Override defaults for 1024px training
IMGSZ1024_OVERRIDES = {
    **MASK_RESCUE_TRAIN_DEFAULTS,
    "imgsz": 1024,
    "batch": 4,    # halved from 8 to stay within GPU memory
}


def train_one_seed(seed: int, eval_only: bool = False, workers: int | None = None) -> dict:
    """Train imgsz=1024 baseline for one seed and evaluate on test set."""
    run_name     = f"imgsz1024_s{seed}"
    project      = str(RUNS_DIR)
    best_weights = str(RUNS_DIR / run_name / "weights" / "best.pt")

    if not eval_only:
        overrides = {
            **IMGSZ1024_OVERRIDES,
            "model":    MODEL_6CH,
            "data":     DATA_6CH,
            "project":  project,
            "name":     run_name,
            "seed":     seed,
            "exist_ok": True,
            "device":   "0",
        }
        if workers is not None:
            overrides["workers"] = workers

        print(f"\n{'='*65}")
        print(f"imgsz=1024 Training: {run_name}  (seed={seed})")
        print(f"  Architecture: yolov11n_v2_6ch.yaml  (same 3-scale as baseline)")
        print(f"  Key change: imgsz 512 → 1024, batch 8 → 4")
        print(f"  Hypothesis: 14px debris needs higher resolution, not more heads")
        print(f"{'='*65}")

        trainer = MaskRescueTrainer(overrides=overrides)
        trainer.train()
    else:
        if not Path(best_weights).exists():
            raise FileNotFoundError(
                f"--eval-only but weights not found: {best_weights}"
            )
        print(f"\n[eval-only] Using: {best_weights}")

    print(f"\nEvaluating {run_name} on 6ch test set (imgsz=1024)...")
    from scripts.v2_evaluate import evaluate
    metrics = evaluate(
        weights=best_weights,
        data_yaml=DATA_6CH,
        channels=6,
        split="test",
        imgsz=1024,    # evaluate at same resolution as training
        batch=4,
    )
    metrics["seed"]     = seed
    metrics["run_name"] = run_name
    metrics["model"]    = "yolov11n_v2_6ch"
    metrics["imgsz"]    = 1024

    out_path = RUNS_DIR / run_name / "eval_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    # ---- Print summary ----
    per    = metrics.get("per_class", {})
    debris = per.get("Marine_Debris", {})
    algae  = per.get("Algae_Organic", {})
    ship   = per.get("Ship", {})

    print(f"\n{'='*65}")
    print(f"  imgsz=1024 Results — {run_name}")
    print(f"{'='*65}")
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
    debris_ap50   = debris.get("AP50",   0.0)

    # Baseline reference (seed=0)
    baseline_map  = 0.193
    baseline_ship = 0.491

    print(f"\n  Comparison vs baseline (imgsz=512):")
    map_delta  = metrics["mAP50"] - baseline_map
    ship_delta = ship.get("AP50", 0) - baseline_ship
    print(f"    mAP50 delta:   {map_delta:+.4f}  "
          f"({'improved' if map_delta >= 0 else 'degraded'})")
    print(f"    Ship AP50 delta: {ship_delta:+.4f}")

    if debris_recall > 0.0 or debris_ap50 > 0.01:
        print(f"\n  >> SIGNAL: Marine_Debris recall={debris_recall:.4f} AP50={debris_ap50:.4f}")
        print("     Hypothesis A (scale) CONFIRMED — higher resolution helps")
        if debris_recall >= 0.154 and debris_ap50 > 0.05:
            print("  >> PAPER-GRADE: meets publication threshold!")
        else:
            print("  >> Next: try imgsz=1024 + oversample, or P2 head @ 1024px")
    else:
        print(f"\n  >> NO SIGNAL: Marine_Debris recall={debris_recall:.4f}")
        print("     Scale hypothesis NOT confirmed at 1024px either")
        print("     Consider: segmentation approach, or debris is not detectable")
        print("     by bbox detection in MARIDA L2A imagery at available resolution")

    print(f"\n  Saved: {out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="MARIDA v2 imgsz=1024 training")
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override DataLoader workers (set 0 to avoid Windows pagefile issues)")
    args = parser.parse_args()

    all_results = []
    for seed in args.seed:
        result = train_one_seed(seed, eval_only=args.eval_only, workers=args.workers)
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print(f"imgsz=1024 MULTI-SEED SUMMARY ({len(all_results)} seeds)")
        print(f"{'='*65}")

        for metric in ["mAP50", "mAP50_95", "precision", "recall"]:
            vals = [r[metric] for r in all_results]
            print(f"  {metric:15s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
                  f"({', '.join(f'{v:.4f}' for v in vals)})")

        print(f"\n  Marine_Debris:")
        for m in ["AP50", "recall", "precision"]:
            vals = [r["per_class"]["Marine_Debris"][m] for r in all_results]
            print(f"    {m:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

        summary = {
            "experiment": "v2_imgsz1024",
            "imgsz": 1024,
            "model": "yolov11n_v2_6ch",
            "seeds": args.seed,
            "results": all_results,
            "summary": {
                metric: {
                    "mean": float(np.mean([r[metric] for r in all_results])),
                    "std":  float(np.std([r[metric] for r in all_results])),
                }
                for metric in ["mAP50", "mAP50_95", "precision", "recall"]
            },
        }
        summary_path = RUNS_DIR / "imgsz1024_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
