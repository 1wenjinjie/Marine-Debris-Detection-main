"""
Focal loss training for MARIDA v2.

Replaces BCE classification loss with FocalBCE to address confidence
calibration: baseline TPs have conf < 0.006, far below any usable threshold.
Focal loss down-weights easy negatives, giving more gradient to hard Marine_Debris.

Default: imgsz=1024 (best result so far), gamma=1.5, alpha=0.25.

Usage:
  python scripts/v2_focal_loss_train.py --seed 0
  python scripts/v2_focal_loss_train.py --seed 0 --gamma 2.0
  python scripts/v2_focal_loss_train.py --seed 0 1 2
  python scripts/v2_focal_loss_train.py --seed 0 --eval-only
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
from ultralytics_plugins.focal_loss_trainer import FocalLossTrainer

DATA_6CH  = str(ROOT / "configs" / "datasets" / "marida_v2_6ch.yaml")
MODEL_6CH = str(ROOT / "configs" / "models" / "yolov11n_v2_6ch.yaml")
RUNS_DIR  = ROOT / "runs_v2"

CLASS_NAMES = ["Marine_Debris", "Algae_Organic", "Ship"]


def train_one_seed(seed: int, gamma: float, alpha: float,
                   imgsz: int, batch: int,
                   eval_only: bool = False) -> dict:
    alpha_tag = f"_a{alpha}" if alpha != 0.25 else ""
    run_name = f"focal_g{gamma}{alpha_tag}_i{imgsz}_s{seed}"
    project  = str(RUNS_DIR)
    best_weights = str(RUNS_DIR / run_name / "weights" / "best.pt")

    if not eval_only:
        overrides = {
            **MASK_RESCUE_TRAIN_DEFAULTS,
            "model":    MODEL_6CH,
            "data":     DATA_6CH,
            "project":  project,
            "name":     run_name,
            "seed":     seed,
            "exist_ok": True,
            "device":   "0",
            "imgsz":    imgsz,
            "batch":    batch,
        }

        print(f"\n{'='*65}")
        print(f"Focal Loss Training: {run_name}  (seed={seed})")
        print(f"  gamma={gamma}, alpha={alpha}, imgsz={imgsz}, batch={batch}")
        print(f"{'='*65}")

        trainer = FocalLossTrainer(
            focal_gamma=gamma,
            focal_alpha=alpha,
            overrides=overrides,
        )
        trainer.train()
    else:
        if not Path(best_weights).exists():
            raise FileNotFoundError(
                f"--eval-only but weights not found: {best_weights}"
            )
        print(f"\n[eval-only] Using: {best_weights}")

    print(f"\nEvaluating {run_name} on 6ch test set (imgsz={imgsz})...")
    from scripts.v2_evaluate import evaluate
    metrics = evaluate(
        weights=best_weights,
        data_yaml=DATA_6CH,
        channels=6,
        split="test",
        imgsz=imgsz,
        batch=batch,
    )
    metrics["seed"]     = seed
    metrics["run_name"] = run_name
    metrics["model"]    = "yolov11n_v2_6ch"
    metrics["imgsz"]    = imgsz
    metrics["focal_gamma"] = gamma
    metrics["focal_alpha"] = alpha

    out_path = RUNS_DIR / run_name / "eval_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    per    = metrics.get("per_class", {})
    debris = per.get("Marine_Debris", {})
    algae  = per.get("Algae_Organic", {})
    ship   = per.get("Ship", {})

    print(f"\n{'='*65}")
    print(f"  Focal Loss Results -- {run_name}")
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

    debris_recall = debris.get("recall", 0.0)
    debris_ap50   = debris.get("AP50",   0.0)

    baseline_map  = 0.193
    imgsz1024_map = 0.194

    ref_map = imgsz1024_map if imgsz == 1024 else baseline_map
    map_delta = metrics["mAP50"] - ref_map
    print(f"\n  vs reference (imgsz={imgsz}): mAP50 delta = {map_delta:+.4f}")

    if debris_recall > 0.0 or debris_ap50 > 0.01:
        print(f"\n  >> SIGNAL: Marine_Debris recall={debris_recall:.4f} AP50={debris_ap50:.4f}")
        if debris_recall >= 0.231:
            print("  >> Exceeds imgsz=1024 baseline (R=0.231) -- focal loss helps!")
        if debris_recall >= 0.3:
            print("  >> STRONG: R>=0.3, consider multi-seed validation")
    else:
        print(f"\n  >> NO SIGNAL: Marine_Debris recall={debris_recall:.4f}")
        print("     Focal loss did not help -- consider FDI/spectral features")

    print(f"\n  Saved: {out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="MARIDA v2 focal loss training")
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    all_results = []
    for seed in args.seed:
        result = train_one_seed(
            seed=seed,
            gamma=args.gamma,
            alpha=args.alpha,
            imgsz=args.imgsz,
            batch=args.batch,
            eval_only=args.eval_only,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print(f"Focal Loss MULTI-SEED SUMMARY ({len(all_results)} seeds)")
        print(f"  gamma={args.gamma}, alpha={args.alpha}, imgsz={args.imgsz}")
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
            "experiment": "v2_focal_loss",
            "focal_gamma": args.gamma,
            "focal_alpha": args.alpha,
            "imgsz": args.imgsz,
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
        tag = f"g{args.gamma}_i{args.imgsz}"
        summary_path = RUNS_DIR / f"focal_{tag}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
