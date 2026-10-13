"""
6ch vs 3ch spectral ablation for MARIDA v2.

Answers: does NIR (B8) + SWIR (B11, B12) help detection beyond RGB (B4, B3, B2)?

Runs two 3ch experiments (seed=0) and compares to existing 6ch results:
  1. 3ch baseline (imgsz=512, BCE)     — compare to baseline_6ch_s0
  2. 3ch focal g2.0 i1024              — compare to focal_g2.0_i1024_s0

Usage:
  python scripts/v2_3ch_ablation.py
  python scripts/v2_3ch_ablation.py --skip-baseline   # only focal
  python scripts/v2_3ch_ablation.py --eval-only        # just evaluate
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
from ultralytics_plugins.focal_loss_trainer import FocalLossTrainer

DATA_3CH  = str(ROOT / "configs" / "datasets" / "marida_v2_3ch.yaml")
DATA_6CH  = str(ROOT / "configs" / "datasets" / "marida_v2_6ch.yaml")
MODEL_3CH = str(ROOT / "configs" / "models" / "yolov11n_v2_3ch.yaml")
RUNS_DIR  = ROOT / "runs_v2"

CLASS_NAMES = ["Marine_Debris", "Algae_Organic", "Ship"]


def train_and_eval(run_name: str, model_yaml: str, data_yaml: str,
                   channels: int, imgsz: int, batch: int, seed: int,
                   focal_gamma: float = 0.0, focal_alpha: float = 0.25,
                   eval_only: bool = False) -> dict:
    project = str(RUNS_DIR)
    best_weights = str(RUNS_DIR / run_name / "weights" / "best.pt")

    if not eval_only:
        overrides = {
            **MASK_RESCUE_TRAIN_DEFAULTS,
            "model":    model_yaml,
            "data":     data_yaml,
            "project":  project,
            "name":     run_name,
            "seed":     seed,
            "exist_ok": True,
            "device":   "0",
            "imgsz":    imgsz,
            "batch":    batch,
            "workers":  0,
        }

        print(f"\n{'='*65}")
        print(f"Training: {run_name}  (ch={channels}, imgsz={imgsz}, seed={seed})")
        if focal_gamma > 0:
            print(f"  focal: gamma={focal_gamma}, alpha={focal_alpha}")
        print(f"{'='*65}")

        if focal_gamma > 0:
            trainer = FocalLossTrainer(
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                overrides=overrides,
            )
        else:
            trainer = MaskRescueTrainer(overrides=overrides)
        trainer.train()
    else:
        if not Path(best_weights).exists():
            raise FileNotFoundError(f"--eval-only but no weights: {best_weights}")
        print(f"\n[eval-only] {run_name}: {best_weights}")

    print(f"\nEvaluating {run_name} on test set (ch={channels}, imgsz={imgsz})...")
    from scripts.v2_evaluate import evaluate
    metrics = evaluate(
        weights=best_weights,
        data_yaml=data_yaml,
        channels=channels,
        split="test",
        imgsz=imgsz,
        batch=batch,
    )
    metrics["seed"]     = seed
    metrics["run_name"] = run_name
    metrics["channels"] = channels
    metrics["imgsz"]    = imgsz
    if focal_gamma > 0:
        metrics["focal_gamma"] = focal_gamma
        metrics["focal_alpha"] = focal_alpha

    out_path = RUNS_DIR / run_name / "eval_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved: {out_path}")
    return metrics


def print_comparison(label_3ch: str, m3: dict, label_6ch: str, m6: dict):
    print(f"\n{'='*70}")
    print(f"COMPARISON: {label_3ch} vs {label_6ch}")
    print(f"{'='*70}")
    print(f"{'Metric':25s}  {'3ch':>10s}  {'6ch':>10s}  {'delta':>10s}  {'verdict':>10s}")
    print("-" * 70)

    for metric_name, key_path in [
        ("mAP@0.5",           ("mAP50",)),
        ("mAP@0.5:0.95",     ("mAP50_95",)),
        ("Precision",         ("precision",)),
        ("Recall",            ("recall",)),
        ("Marine_Debris AP50", ("per_class", "Marine_Debris", "AP50")),
        ("Marine_Debris R",    ("per_class", "Marine_Debris", "recall")),
        ("Algae_Organic AP50", ("per_class", "Algae_Organic", "AP50")),
        ("Algae_Organic R",    ("per_class", "Algae_Organic", "recall")),
        ("Ship AP50",          ("per_class", "Ship", "AP50")),
        ("Ship R",             ("per_class", "Ship", "recall")),
    ]:
        v3 = m3
        v6 = m6
        for k in key_path:
            v3 = v3.get(k, 0) if isinstance(v3, dict) else 0
            v6 = v6.get(k, 0) if isinstance(v6, dict) else 0
        delta = v6 - v3
        verdict = "6ch wins" if delta > 0.005 else ("3ch wins" if delta < -0.005 else "~tie")
        print(f"{metric_name:25s}  {v3:10.4f}  {v6:10.4f}  {delta:+10.4f}  {verdict:>10s}")


def main():
    parser = argparse.ArgumentParser(description="MARIDA v2 6ch vs 3ch spectral ablation")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = {}

    # --- 3ch baseline (imgsz=512, BCE) ---
    if not args.skip_baseline:
        results["3ch_baseline"] = train_and_eval(
            run_name=f"ablation_3ch_baseline_s{args.seed}",
            model_yaml=MODEL_3CH, data_yaml=DATA_3CH,
            channels=3, imgsz=512, batch=8, seed=args.seed,
            eval_only=args.eval_only,
        )

    # --- 3ch focal g2.0 i1024 ---
    results["3ch_focal"] = train_and_eval(
        run_name=f"ablation_3ch_focal_g2.0_i1024_s{args.seed}",
        model_yaml=MODEL_3CH, data_yaml=DATA_3CH,
        channels=3, imgsz=1024, batch=4, seed=args.seed,
        focal_gamma=2.0, focal_alpha=0.25,
        eval_only=args.eval_only,
    )

    # --- Load existing 6ch results for comparison ---
    ref_6ch_baseline = RUNS_DIR / f"baseline_6ch_s{args.seed}" / "eval_test.json"
    ref_6ch_focal    = RUNS_DIR / f"focal_g2.0_i1024_s{args.seed}" / "eval_test.json"

    if "3ch_baseline" in results and ref_6ch_baseline.exists():
        m6 = json.loads(ref_6ch_baseline.read_text())
        print_comparison("3ch baseline", results["3ch_baseline"],
                         "6ch baseline", m6)

    if ref_6ch_focal.exists():
        m6 = json.loads(ref_6ch_focal.read_text())
        print_comparison("3ch focal g2.0 i1024", results["3ch_focal"],
                         "6ch focal g2.0 i1024", m6)

    # --- Also run sweep on 3ch focal for debris comparison ---
    print(f"\nRunning debris threshold sweep on 3ch focal model...")
    from v2_sweep_thresholds import run_diagnostic
    sweep = run_diagnostic(
        weights=str(RUNS_DIR / f"ablation_3ch_focal_g2.0_i1024_s{args.seed}" / "weights" / "best.pt"),
        tag=f"ablation_3ch_focal_g2.0_i1024_s{args.seed}",
        channels=3, imgsz=1024, save_viz=False,
    )

    # --- Save ablation summary ---
    summary = {
        "experiment": "v2_3ch_vs_6ch_ablation",
        "seed": args.seed,
        "results": results,
        "sweep_3ch_focal": sweep.get("threshold_sweep", {}),
    }
    out_path = RUNS_DIR / f"ablation_3ch_vs_6ch_s{args.seed}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  Ablation summary: {out_path}")


if __name__ == "__main__":
    main()
