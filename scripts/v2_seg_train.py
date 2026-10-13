"""MARIDA v2-seg baseline trainer (instance segmentation, 6-channel)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS
from ultralytics_plugins.mask_rescue_seg_trainer import MaskRescueSegTrainer

DATA_6CH_SEG = str(ROOT / "configs" / "datasets" / "marida_v2_seg_6ch.yaml")
MODEL_6CH_SEG = str(ROOT / "configs" / "models" / "yolov11n_v2_6ch_seg.yaml")
RUNS_DIR = ROOT / "runs_v2_seg"


def train_one_seed(seed: int, epochs: int | None = None, imgsz: int | None = None,
                   batch: int | None = None, workers: int | None = None,
                   name_suffix: str = "") -> dict:
    run_name = f"baseline_seg_6ch{name_suffix}_s{seed}"
    project = str(RUNS_DIR)

    overrides = {
        **MASK_RESCUE_TRAIN_DEFAULTS,
        "model": MODEL_6CH_SEG,
        "data": DATA_6CH_SEG,
        "project": project,
        "name": run_name,
        "seed": seed,
        "exist_ok": True,
        "device": "0",
        "task": "segment",
    }
    if epochs is not None:
        overrides["epochs"] = epochs
    if imgsz is not None:
        overrides["imgsz"] = imgsz
    if batch is not None:
        overrides["batch"] = batch
    if workers is not None:
        overrides["workers"] = workers

    print(f"\n{'='*60}")
    print(f"SEG TRAIN: {run_name} (seed={seed})")
    print(f"  imgsz={overrides.get('imgsz')}, batch={overrides.get('batch')}, "
          f"workers={overrides.get('workers')}, epochs={overrides.get('epochs')}")
    print(f"{'='*60}")

    trainer = MaskRescueSegTrainer(overrides=overrides)
    trainer.train()

    print(f"\nEvaluating {run_name} on test set...")
    from scripts.v2_seg_evaluate import evaluate
    best_weights = str(RUNS_DIR / run_name / "weights" / "best.pt")
    metrics = evaluate(
        weights=best_weights,
        data_yaml=DATA_6CH_SEG,
        channels=6,
        split="test",
        imgsz=overrides.get("imgsz", 512),
        batch=overrides.get("batch", 8),
        workers=0,
    )
    metrics["seed"] = seed
    metrics["run_name"] = run_name

    out_path = RUNS_DIR / run_name / "eval_test.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved: {out_path}")
    return metrics


def main():
    p = argparse.ArgumentParser(description="MARIDA v2-seg baseline training")
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--name-suffix", type=str, default="")
    args = p.parse_args()

    results = []
    for s in args.seed:
        results.append(train_one_seed(
            s, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            workers=args.workers, name_suffix=args.name_suffix,
        ))

    if len(results) > 1:
        import numpy as np
        print(f"\n{'='*60}\nMULTI-SEED SUMMARY ({len(results)} seeds)\n{'='*60}")
        for metric in ["mask_mAP50", "mask_mAP50_95", "box_mAP50", "box_mAP50_95"]:
            vals = [r[metric] for r in results]
            print(f"  {metric:18s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
                  f"({', '.join(f'{v:.4f}' for v in vals)})")


if __name__ == "__main__":
    main()
