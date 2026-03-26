"""
Phase 1: Privileged Knowledge Distillation (PKD) Minimum Viable Verification
=============================================================================
Trains a 6ch student model with response-level KD from the 7ch P0-C teacher.

Experiment Design
-----------------
P1-RKD  [Response-level KD]
    Teacher: P0-C best.pt (7ch, mAP50=0.605)
    Student: 6ch YOLOv11n, trained with KD loss from teacher soft outputs
    Data:    7ch NPY (teacher gets all 7ch, student gets first 6ch)
    Compare: P0-A baseline (mAP50=0.312)

Decision Gate:
    mAP50 > 0.332 (P0-A + 0.02)  → KD signal exists, proceed to feature KD
    mAP50 ≈ P0-A                  → Response KD insufficient, try feature KD
    mAP50 < P0-A                  → KD harmful, investigate

Usage
-----
  python scripts/phase1_pkd_experiments.py --device 0 --seed 0
  python scripts/phase1_pkd_experiments.py --device 0 --seed 0 --eval-only --weights path/to/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from scripts.training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS
from ultralytics_plugins.pkd_trainer import PKDTrainer


# ---------- Paths ----------
TEACHER_WEIGHTS = ROOT / "runs_phase0" / "P0-C_seed0" / "weights" / "best.pt"
STUDENT_MODEL_YAML = ROOT / "configs" / "models" / "yolov11n_6ch.yaml"
DATA_7CH_YAML = str(ROOT / "configs" / "datasets" / "marida_rescue_7ch.yaml")
DATA_6CH_YAML = str(ROOT / "configs" / "datasets" / "marida_rescue_split_6ch.yaml")
PROJECT_DIR = ROOT / "runs_phase1"


def train_pkd(
    seed: int = 0,
    device: str = "0",
    kd_weight: float = 1.0,
    kd_temperature: float = 4.0,
    teacher_weights: str | None = None,
) -> Path:
    """Train 6ch student with response-level KD from 7ch teacher."""
    import numpy as np
    import torch
    import random as _random

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)

    teacher_path = teacher_weights or str(TEACHER_WEIGHTS)
    run_name = f"P1-RKD_seed{seed}"

    overrides = dict(MASK_RESCUE_TRAIN_DEFAULTS)
    overrides.update({
        "data": DATA_7CH_YAML,  # 7ch data: teacher gets heatmap, student gets 6ch slice
        "device": device,
        "project": str(PROJECT_DIR),
        "name": run_name,
        "seed": seed,
        "exist_ok": True,
        # PKD-specific
        "teacher_weights": teacher_path,
        "kd_weight": kd_weight,
        "kd_temperature": kd_temperature,
    })

    # Student model: 6ch architecture
    model = YOLO(str(STUDENT_MODEL_YAML))
    model.train(trainer=PKDTrainer, **overrides)

    best_pt = PROJECT_DIR / run_name / "weights" / "best.pt"
    print(f"\n[P1-RKD] Training complete. Best weights: {best_pt}")
    return best_pt


def evaluate_model(weights: str, data_yaml: str, device: str = "0", save_name: str = "eval") -> dict:
    """Evaluate a model using the MaskRescue NPY dataset with monkey-patched val."""
    from ultralytics.data import build as yolo_build
    from ultralytics.models.yolo.detect import val as detect_val
    from ultralytics_plugins.mask_rescue_dataset_fixed import MaskRescueNPYDataset
    from ultralytics.utils import colorstr
    from ultralytics.utils.torch_utils import unwrap_model

    # --- monkey-patch so val() uses our NPY dataset ---
    _orig_build = yolo_build.build_yolo_dataset

    def _patched_build(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
        from pathlib import Path as _Path
        dataset_root = _Path(data.get("path", "")).resolve()
        channels = data.get("channels", 3)
        return MaskRescueNPYDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=False,
            hyp=cfg,
            rect=rect,
            stride=stride,
            pad=0.5,
            prefix=colorstr(f"{mode}: "),
            dataset_root=dataset_root,
            channels=channels,
            task=cfg.task,
            data=data,
        )

    yolo_build.build_yolo_dataset = _patched_build
    detect_val.build_yolo_dataset = _patched_build

    try:
        model = YOLO(weights)
        results = model.val(data=data_yaml, device=device, batch=8, imgsz=512)
        metrics = {
            "metrics/precision(B)": float(results.results_dict.get("metrics/precision(B)", 0)),
            "metrics/recall(B)": float(results.results_dict.get("metrics/recall(B)", 0)),
            "metrics/mAP50(B)": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "metrics/mAP50-95(B)": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "fitness": float(results.results_dict.get("fitness", 0)),
        }
    finally:
        yolo_build.build_yolo_dataset = _orig_build
        detect_val.build_yolo_dataset = _orig_build

    return metrics


def run_experiment(args):
    """Run full P1-RKD experiment: train + evaluate."""
    seed = args.seed
    device = args.device
    run_name = f"P1-RKD_seed{seed}"

    if args.eval_only:
        if not args.weights:
            default_weights = PROJECT_DIR / run_name / "weights" / "best.pt"
            if default_weights.exists():
                args.weights = str(default_weights)
            else:
                raise ValueError(f"No weights found at {default_weights}. Provide --weights or train first.")
        weights = args.weights
    else:
        # Train
        weights = str(train_pkd(
            seed=seed,
            device=device,
            kd_weight=args.kd_weight,
            kd_temperature=args.kd_temperature,
            teacher_weights=args.teacher_weights,
        ))

    # Evaluate on 6ch test set (student is 6ch, fair evaluation)
    print(f"\n[P1-RKD] Evaluating {weights} on 6ch test set...")
    metrics = evaluate_model(weights, DATA_6CH_YAML, device=device, save_name=f"{run_name}_eval")

    # Save results
    eval_dir = PROJECT_DIR / f"{run_name}_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "exp_id": "P1-RKD",
        "seed": seed,
        "kd_weight": args.kd_weight,
        "kd_temperature": args.kd_temperature,
        "teacher_weights": args.teacher_weights or str(TEACHER_WEIGHTS),
        "student_weights": weights,
        "metrics": metrics,
    }
    result_path = eval_dir / "phase1_metrics.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[P1-RKD] Results saved to {result_path}")

    # Print comparison with Phase 0 baselines
    print_comparison(metrics)
    return result


def print_comparison(p1_metrics: dict):
    """Print P1-RKD results alongside Phase 0 baselines."""
    p0a_path = ROOT / "runs_phase0" / "P0-A_seed0_eval" / "phase0_metrics.json"
    p0c_path = ROOT / "runs_phase0" / "P0-C_seed0_eval" / "phase0_metrics.json"

    rows = []

    # Load P0-A
    if p0a_path.exists():
        with open(p0a_path) as f:
            p0a = json.load(f)["metrics"]
        rows.append(("P0-A (6ch baseline)", p0a))

    # P1-RKD
    rows.append(("P1-RKD (6ch + resp. KD)", p1_metrics))

    # Load P0-C
    if p0c_path.exists():
        with open(p0c_path) as f:
            p0c = json.load(f)["metrics"]
        rows.append(("P0-C (7ch oracle)", p0c))

    print("\n" + "=" * 85)
    print("Phase 1 PKD Results Comparison")
    print("=" * 85)
    header = f"{'Experiment':<30} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}"
    print(header)
    print("-" * 85)
    for name, m in rows:
        p = m.get("metrics/precision(B)", 0)
        r = m.get("metrics/recall(B)", 0)
        m50 = m.get("metrics/mAP50(B)", 0)
        m95 = m.get("metrics/mAP50-95(B)", 0)
        print(f"{name:<30} {p:>10.4f} {r:>10.4f} {m50:>10.4f} {m95:>10.4f}")
    print("=" * 85)

    # Decision gate
    baseline_map = rows[0][1].get("metrics/mAP50(B)", 0) if rows else 0
    pkd_map = p1_metrics.get("metrics/mAP50(B)", 0)
    delta = pkd_map - baseline_map

    print(f"\nDecision Gate: P1-RKD mAP50 = {pkd_map:.4f}, P0-A mAP50 = {baseline_map:.4f}, delta = {delta:+.4f}")
    if delta > 0.02:
        print(">>> PASS: KD signal exists! Proceed to feature-level KD (Phase 2).")
    elif delta > -0.02:
        print(">>> MARGINAL: Response KD alone insufficient. Try feature-level KD next.")
    else:
        print(">>> FAIL: KD appears harmful. Investigate teacher-student mismatch.")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: PKD Minimum Viable Verification")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--kd-weight", type=float, default=1.0, help="KD loss weight")
    parser.add_argument("--kd-temperature", type=float, default=4.0, help="KD temperature")
    parser.add_argument("--teacher-weights", default=None, help="Path to teacher model weights")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights", default=None, help="Weights for eval-only mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
