"""
Debris-aware oversampling training for MARIDA v2.

Addresses the background tile imbalance problem:
  - 75% of train tiles are background-only (730/973)
  - Only 52 tiles contain Marine_Debris (5.3%)

Strategy: WeightedRandomSampler with per-tile weights.
  - Debris-containing tiles:  weight = oversample_factor (default 8)
  - Background-only tiles:    weight = 1.0

This uses the existing WeightedRandomSampler infrastructure in
MaskRescueTrainer without modifying core classes.

Usage:
  python scripts/v2_oversample_train.py --seed 0
  python scripts/v2_oversample_train.py --seed 0 --oversample 5
  python scripts/v2_oversample_train.py --seed 0 1 2
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
from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer
from ultralytics.utils import colorstr, LOGGER
from ultralytics.utils.torch_utils import unwrap_model

DATA_6CH = str(ROOT / "configs" / "datasets" / "marida_v2_6ch.yaml")
MODEL_6CH = str(ROOT / "configs" / "models" / "yolov11n_v2_6ch.yaml")
RUNS_DIR = ROOT / "runs_v2"

DEBRIS_CLASS_ID = 0  # Marine_Debris is class 0


class DebrisOversampleDataset(MaskRescueNPYDataset):
    """MaskRescueNPYDataset that upweights debris-containing tiles."""

    def __init__(self, *args, debris_oversample_factor: float = 8.0, **kwargs):
        # Must be set before super().__init__() calls _align_sample_weights()
        self._debris_oversample_factor = debris_oversample_factor
        super().__init__(*args, **kwargs)

    def _align_sample_weights(self) -> None:
        """Compute per-tile weights: debris tiles get oversample_factor, background gets 1.0."""
        weights = []
        n_debris = 0
        n_background = 0
        for path_str in self.im_files:
            key = str(Path(path_str).resolve())
            hist = self.sample_class_hist.get(key, np.zeros(1))
            has_debris = (
                len(hist) > DEBRIS_CLASS_ID and hist[DEBRIS_CLASS_ID] > 0
            )
            w = self._debris_oversample_factor if has_debris else 1.0
            weights.append(w)
            if has_debris:
                n_debris += 1
            else:
                n_background += 1

        self.sample_weights = weights
        self._summarize_sample_weights(np.array(weights, dtype=np.float32))

        # Log effective sampling fractions
        total_weight = n_debris * self._debris_oversample_factor + n_background * 1.0
        debris_frac = (n_debris * self._debris_oversample_factor) / max(total_weight, 1)
        LOGGER.info(
            f"{self.prefix}DebrisOversampling: {n_debris} debris tiles x{self._debris_oversample_factor:.0f} "
            f"vs {n_background} background tiles x1.0 "
            f"=> effective debris tile fraction per epoch: {debris_frac*100:.1f}%"
        )


class DebrisOversampleTrainer(MaskRescueTrainer):
    """Trainer that uses DebrisOversampleDataset for training splits."""

    def __init__(self, *args, debris_oversample_factor: float = 8.0, **kwargs):
        self._debris_oversample_factor = debris_oversample_factor
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset_root = Path(self.data.get("path", ".")).resolve()
        channels = self.data.get("channels", 3)

        factor = self._debris_oversample_factor if mode == "train" else 1.0

        return DebrisOversampleDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect if mode == "train" else True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            dataset_root=dataset_root,
            channels=channels,
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            debris_oversample_factor=factor,
        )


def train_one_seed(seed: int, oversample_factor: float) -> dict:
    run_name = f"oversample_k{int(oversample_factor)}_s{seed}"
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
    print(f"Training: {run_name}  (seed={seed}, oversample_factor={oversample_factor})")
    print(f"Debris tiles: ~52/973 train tiles, upweighted x{int(oversample_factor)}")
    print(f"{'='*60}")

    trainer = DebrisOversampleTrainer(
        overrides=overrides,
        debris_oversample_factor=oversample_factor,
    )
    trainer.train()

    print(f"\nEvaluating {run_name} on 6ch test set...")
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
    metrics["oversample_factor"] = oversample_factor

    out_path = RUNS_DIR / run_name / "eval_test.json"
    out_path.write_text(json.dumps(metrics, indent=2))

    # ---- Print summary ----
    per = metrics.get("per_class", {})
    debris = per.get("Marine_Debris", {})
    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")
    print(f"  mAP@0.5:       {metrics['mAP50']:.4f}")
    print(f"  Marine_Debris: AP50={debris.get('AP50',0):.4f}  R={debris.get('recall',0):.4f}")
    algae = per.get("Algae_Organic", {})
    ship  = per.get("Ship", {})
    print(f"  Algae_Organic: AP50={algae.get('AP50',0):.4f}")
    print(f"  Ship:          AP50={ship.get('AP50',0):.4f}")
    print(f"  Saved: {out_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="MARIDA v2 debris oversampling training")
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--oversample", type=float, default=8.0,
        help="Debris tile oversample factor (default 8: debris tiles drawn 8x more often)"
    )
    args = parser.parse_args()

    all_results = []
    for seed in args.seed:
        result = train_one_seed(seed, args.oversample)
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"MULTI-SEED SUMMARY")
        print(f"{'='*60}")
        for metric in ["mAP50", "mAP50_95"]:
            vals = [r[metric] for r in all_results]
            print(f"  {metric:15s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
