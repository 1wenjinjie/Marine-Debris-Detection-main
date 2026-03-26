"""
Phase 0: LUPI Hypothesis Validation Experiments
================================================
Runs three targeted experiments to determine whether the Knowledge Heatmap
functions as legitimate Privileged Information (LUPI) or as test-time leakage.

Experiment Design
-----------------
P0-A  [6ch clean baseline]
      Train: 6ch model, no heatmap
      Test:  6ch, no heatmap
      Purpose: Establish honest baseline without any GT prior

P0-B  [LUPI candidate]
      Train: 7ch model, WITH heatmap (GT spatial prior as training signal)
      Test:  7ch model, heatmap channel ZEROED OUT
      Purpose: Test whether training-time GT prior improves inference
      LUPI hypothesis: P0-B > P0-A on Recall and/or F1

P0-C  [Historical comparison, information-leakage bound]
      Train: 7ch model, WITH heatmap
      Test:  7ch model, WITH heatmap (GT available at test time)
      Purpose: Upper bound; NOT a fair evaluation for deployment
      Note:  This is the setup that produced F1=0.6774 historically

All experiments use the same seed for a fair single-seed comparison.
Run with multiple seeds in Phase 1 to establish stability.

Usage
-----
  python scripts/phase0_lupi_experiments.py --device 0 --seed 0
  python scripts/phase0_lupi_experiments.py --device 0 --seed 0 --exp P0-A
  python scripts/phase0_lupi_experiments.py --device 0 --seed 0 --exp P0-B --eval-only --weights path/to/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data import build as yolo_build
from ultralytics.models.yolo.detect import val as detect_val

from ultralytics_plugins.mask_rescue_dataset_fixed import MaskRescueNPYDataset
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer
from scripts.training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS

SHARED_OVERRIDES = MASK_RESCUE_TRAIN_DEFAULTS

# Dataset and model configs
DATA_7CH = str(ROOT / "configs/datasets/marida_rescue_7ch.yaml")
DATA_6CH = str(ROOT / "configs/datasets/marida_rescue_split_6ch.yaml")
MODEL_7CH = str(ROOT / "configs/models/yolov11n_7ch.yaml")
MODEL_6CH = str(ROOT / "configs/models/yolov11n_6ch.yaml")


@contextmanager
def _heatmap_env(enabled: bool):
    """Temporarily set MASKRESCUE_ZERO_HEATMAP, restoring previous state on exit."""
    key = "MASKRESCUE_ZERO_HEATMAP"
    prev = os.environ.get(key)
    try:
        if enabled:
            os.environ[key] = "1"
        else:
            os.environ.pop(key, None)
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_experiment(
    exp_id: str,
    model_yaml: str,
    data_yaml: str,
    seed: int,
    device: str,
    project: str,
    zero_heatmap_at_eval: bool = False,
) -> Path:
    """Train one experiment and return the weights path."""
    run_name = f"{exp_id}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"  TRAINING: {run_name}")
    print(f"  model={Path(model_yaml).name}  data={Path(data_yaml).name}")
    print(f"  zero_heatmap_at_eval={zero_heatmap_at_eval}")
    print(f"{'='*60}\n")

    set_seed(seed)

    overrides = dict(SHARED_OVERRIDES)
    overrides.update({
        "data": data_yaml,
        "device": device,
        "project": project,
        "name": run_name,
        "seed": seed,
    })

    with _heatmap_env(zero_heatmap_at_eval):
        model = YOLO(model_yaml)
        model.train(trainer=MaskRescueTrainer, **overrides)

    weights = Path(project) / run_name / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Training finished but best.pt not found at {weights}")
    return weights


def _patch_val_dataset(data_yaml: str) -> None:
    """Monkey-patch Ultralytics val to use MaskRescueNPYDataset."""
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}
    channels = data_cfg.get("channels", 3)
    dataset_root = Path(data_cfg.get("path", Path(data_yaml).parent)).resolve()

    from ultralytics.utils import colorstr

    def _builder(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
        return MaskRescueNPYDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=cfg,
            rect=rect,
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            use_segments=cfg.task == "segment",
            use_keypoints=cfg.task == "pose",
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
            dataset_root=dataset_root,
            channels=channels,
        )

    yolo_build.build_yolo_dataset = _builder
    detect_val.build_yolo_dataset = _builder


def evaluate_experiment(
    exp_id: str,
    weights: Path,
    data_yaml: str,
    seed: int,
    device: str,
    project: str,
    zero_heatmap_at_eval: bool = False,
) -> dict:
    """Evaluate a trained model and return metrics."""
    run_name = f"{exp_id}_seed{seed}_eval"
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {run_name}")
    print(f"  weights={weights.name}  zero_heatmap={zero_heatmap_at_eval}")
    print(f"{'='*60}\n")

    _patch_val_dataset(data_yaml)

    with _heatmap_env(zero_heatmap_at_eval):
        model = YOLO(str(weights))
        results = model.val(
            data=data_yaml,
            imgsz=512,
            batch=8,
            device=device,
            project=project,
            name=run_name,
        )

    metrics = getattr(results, "results_dict", {})
    out_path = Path(project) / run_name / "phase0_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exp_id": exp_id,
        "seed": seed,
        "zero_heatmap_at_eval": zero_heatmap_at_eval,
        "weights": str(weights),
        "data": data_yaml,
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  → metrics saved to {out_path}")
    return payload


def print_summary(results: list[dict]) -> None:
    print("\n" + "="*70)
    print("  PHASE 0 SUMMARY")
    print("="*70)
    print(f"  {'Exp':<10} {'F1':>8} {'Precision':>12} {'Recall':>10} {'mAP50':>10}")
    print(f"  {'-'*52}")
    for r in results:
        m = r["metrics"]
        f1 = m.get("metrics/F1(B)", m.get("metrics/f1", "N/A"))
        prec = m.get("metrics/precision(B)", m.get("metrics/precision", "N/A"))
        rec = m.get("metrics/recall(B)", m.get("metrics/recall", "N/A"))
        map50 = m.get("metrics/mAP50(B)", m.get("metrics/mAP_0.5", "N/A"))

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        print(f"  {r['exp_id']:<10} {fmt(f1):>8} {fmt(prec):>12} {fmt(rec):>10} {fmt(map50):>10}")
    print("="*70)
    print()
    print("  Decision rule:")
    print("  ✅ P0-B Recall > P0-A Recall → LUPI framework valid, proceed to Phase 1")
    print("  ⚠️  P0-B ≈ P0-A              → heatmap has no residual value, revise narrative")
    print("  ❌ P0-B < P0-A              → heatmap hurts generalization, major redesign needed")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0: LUPI hypothesis validation")
    parser.add_argument("--device", default="0", help="Training device (0, cpu, etc.)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--project", default="runs_phase0", help="Output project directory")
    parser.add_argument(
        "--exp",
        choices=["P0-A", "P0-B", "P0-C", "all"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, only run evaluation (requires --weights)",
    )
    parser.add_argument("--weights", default=None, help="Weights path for --eval-only mode")
    args = parser.parse_args()

    collected_results: list[dict] = []
    run_exp = args.exp

    # ---- P0-A: 6ch clean baseline ----
    if run_exp in ("P0-A", "all"):
        if not args.eval_only:
            w = train_experiment(
                exp_id="P0-A",
                model_yaml=MODEL_6CH,
                data_yaml=DATA_6CH,
                seed=args.seed,
                device=args.device,
                project=args.project,
                zero_heatmap_at_eval=False,
            )
        else:
            w = Path(args.weights) if args.weights else None
            assert w and w.exists(), "Provide --weights for eval-only mode"

        r = evaluate_experiment(
            exp_id="P0-A",
            weights=w,
            data_yaml=DATA_6CH,
            seed=args.seed,
            device=args.device,
            project=args.project,
            zero_heatmap_at_eval=False,
        )
        collected_results.append(r)

    # ---- P0-B: LUPI (7ch train, zero heatmap at test) ----
    if run_exp in ("P0-B", "all"):
        if not args.eval_only:
            w = train_experiment(
                exp_id="P0-B",
                model_yaml=MODEL_7CH,
                data_yaml=DATA_7CH,
                seed=args.seed,
                device=args.device,
                project=args.project,
                zero_heatmap_at_eval=True,  # zero heatmap during val inside training loop too
            )
        else:
            w = Path(args.weights) if args.weights else None
            assert w and w.exists(), "Provide --weights for eval-only mode"

        r = evaluate_experiment(
            exp_id="P0-B",
            weights=w,
            data_yaml=DATA_7CH,
            seed=args.seed,
            device=args.device,
            project=args.project,
            zero_heatmap_at_eval=True,
        )
        collected_results.append(r)

    # ---- P0-C: 7ch with GT heatmap at test (historical/upper-bound) ----
    if run_exp in ("P0-C", "all"):
        if not args.eval_only:
            w = train_experiment(
                exp_id="P0-C",
                model_yaml=MODEL_7CH,
                data_yaml=DATA_7CH,
                seed=args.seed,
                device=args.device,
                project=args.project,
                zero_heatmap_at_eval=False,
            )
        else:
            w = Path(args.weights) if args.weights else None
            assert w and w.exists(), "Provide --weights for eval-only mode"

        r = evaluate_experiment(
            exp_id="P0-C",
            weights=w,
            data_yaml=DATA_7CH,
            seed=args.seed,
            device=args.device,
            project=args.project,
            zero_heatmap_at_eval=False,
        )
        collected_results.append(r)

    if collected_results:
        # Save combined summary
        summary_path = Path(args.project) / f"phase0_summary_seed{args.seed}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(collected_results, indent=2))
        print(f"\nFull results saved to {summary_path}")
        print_summary(collected_results)


if __name__ == "__main__":
    main()
