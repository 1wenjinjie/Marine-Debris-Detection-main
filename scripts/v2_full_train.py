"""
MARIDA v2 Full model training: CBAM + BiFPN-lite + EIoU.

This is the final teacher-task model. Unlike the single-module gate runs,
the matrix marks Full as a required 3-seed experiment once the smoke test
passes. Start with seed=0 if you want an early sanity check.

Usage:
  python scripts/v2_full_train.py --seed 0
  python scripts/v2_full_train.py --seed 0 1 2
  python scripts/v2_full_train.py --seed 0 --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS
from ultralytics_plugins.bifpn_modules import install_bifpn_patch
from ultralytics_plugins.eiou_trainer import EIoUTrainer

install_bifpn_patch()

from scripts.v2_bifpn_train import _gate_verdict, _run_sweep, _save_eval, _weights_for_run

DATA_6CH = str(ROOT / "configs" / "datasets" / "marida_v2_6ch.yaml")
MODEL_FULL = str(ROOT / "configs" / "models" / "yolov11n_v2_6ch_full.yaml")
RUNS_DIR = ROOT / "runs_v2"


def _print_eval_summary(run_name: str, metrics: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Full Results - {run_name}")
    print(f"{'=' * 60}")
    print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print()
    print("  Per-class:")
    for name, cls_m in metrics.get("per_class", {}).items():
        print(
            f"    {name:20s}: AP50={cls_m.get('AP50', 0):.4f}  "
            f"P={cls_m.get('precision', 0):.4f}  "
            f"R={cls_m.get('recall', 0):.4f}"
        )


def train_one_seed(
    seed: int,
    eval_only: bool = False,
    run_sweep: bool = True,
    no_viz: bool = False,
    device: str = "0",
    workers: int | None = None,
) -> dict:
    run_name = f"full_v2_s{seed}"
    imgsz = int(MASK_RESCUE_TRAIN_DEFAULTS["imgsz"])

    if not eval_only:
        overrides = {
            **MASK_RESCUE_TRAIN_DEFAULTS,
            "model": MODEL_FULL,
            "data": DATA_6CH,
            "project": str(RUNS_DIR),
            "name": run_name,
            "seed": seed,
            "exist_ok": True,
            "device": device,
        }
        if workers is not None:
            overrides["workers"] = workers

        print(f"\n{'=' * 60}")
        print(f"Full Training: {run_name}  (seed={seed})")
        print("  Model: yolov11n_v2_6ch_full.yaml")
        print("  Setting: imgsz=512, BCE cls loss, EIoU box loss")
        print("  Components: CBAM + BiFPN-lite + EIoU")
        print(f"{'=' * 60}")

        trainer = EIoUTrainer(overrides=overrides)
        trainer.train()
    else:
        print(f"\n[eval-only] Using existing weights for {run_name}")

    weights = _weights_for_run(run_name)

    print(f"\nEvaluating {run_name} on 6ch test set...")
    from scripts.v2_evaluate import evaluate

    metrics = evaluate(
        weights=str(weights),
        data_yaml=DATA_6CH,
        channels=6,
        split="test",
        imgsz=imgsz,
        device=device,
    )
    metrics["seed"] = seed
    metrics["run_name"] = run_name
    metrics["model"] = "yolov11n_v2_6ch_full"
    metrics["components"] = ["CBAM", "BiFPN-lite", "EIoU"]
    metrics["box_loss"] = "EIoU"

    eval_path = _save_eval(metrics, run_name)
    _print_eval_summary(run_name, metrics)
    print(f"\n  Saved: {eval_path}")

    sweep = None
    if run_sweep:
        sweep = _run_sweep(weights, run_name, imgsz=imgsz, no_viz=no_viz)
        metrics["sweep_results"] = str(RUNS_DIR / f"sweep_{run_name}" / "sweep_results.json")

    gate = _gate_verdict(metrics, sweep)
    metrics["gate_verdict"] = gate
    gate_path = RUNS_DIR / run_name / "gate_verdict.json"
    gate_path.write_text(json.dumps(gate, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  Full Gate Reference - {run_name}")
    print(f"{'=' * 60}")
    print(f"  mAP50 delta vs baseline_s0: {gate['map50_delta_vs_baseline_s0']}")
    print(f"  TP@0.025 delta vs baseline_s0: {gate['debris_tp_conf_0.025_delta_vs_baseline_s0']}")
    print(f"  gate_pass/reference: {gate['gate_pass']}")
    print(f"  Saved: {gate_path}")

    _save_eval(metrics, run_name)
    return metrics


def _write_multi_seed_summary(seeds: list[int], results: list[dict]) -> Path:
    print(f"\n{'=' * 60}")
    print(f"FULL MULTI-SEED SUMMARY ({len(results)} seeds)")
    print(f"{'=' * 60}")

    for metric in ["mAP50", "mAP50_95", "precision", "recall"]:
        vals = [r[metric] for r in results]
        print(
            f"  {metric:15s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
            f"({', '.join(f'{v:.4f}' for v in vals)})"
        )

    print("\n  Marine_Debris:")
    for m in ["AP50", "recall", "precision"]:
        vals = [r["per_class"]["Marine_Debris"][m] for r in results]
        print(f"    {m:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    summary = {
        "experiment": "v2_full",
        "model": "yolov11n_v2_6ch_full",
        "components": ["CBAM", "BiFPN-lite", "EIoU"],
        "seeds": seeds,
        "results": results,
        "summary": {
            metric: {
                "mean": float(np.mean([r[metric] for r in results])),
                "std": float(np.std([r[metric] for r in results])),
            }
            for metric in ["mAP50", "mAP50_95", "precision", "recall"]
        },
    }
    summary_path = RUNS_DIR / "full_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary: {summary_path}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="MARIDA v2 Full model training")
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate existing weights")
    parser.add_argument("--no-sweep", action="store_true", help="Skip debris threshold sweep")
    parser.add_argument("--no-viz", action="store_true", help="Skip sweep PNG visualizations")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=None, help="Override DataLoader workers")
    args = parser.parse_args()

    all_results = []
    for seed in args.seed:
        result = train_one_seed(
            seed=seed,
            eval_only=args.eval_only,
            run_sweep=not args.no_sweep,
            no_viz=args.no_viz,
            device=args.device,
            workers=args.workers,
        )
        all_results.append(result)

    if len(all_results) > 1:
        _write_multi_seed_summary(args.seed, all_results)


if __name__ == "__main__":
    main()
