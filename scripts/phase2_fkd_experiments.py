"""
Phase 2: Feature-level KD Ablation
===================================
Runs three KD mode comparisons (seed=0):
    P2-RKD   : response KD only   (kd_mode="response")
    P2-FKD   : feature KD only    (kd_mode="feature")
    P2-RFKD  : response + feature (kd_mode="response+feature")

Compares against P0-A baseline and P0-C oracle.

Usage:
    python scripts/phase2_fkd_experiments.py --device 0 --exp all
    python scripts/phase2_fkd_experiments.py --device 0 --exp P2-FKD
    python scripts/phase2_fkd_experiments.py --device 0 --exp P2-RFKD --eval-only
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
PROJECT_DIR = ROOT / "runs_phase2"

# ---------- Experiment configs ----------
EXPERIMENTS = {
    "P2-RKD": {
        "kd_mode": "response",
        "kd_weight": 1.0,
        "fkd_weight": 0.0,
        "kd_temperature": 4.0,
    },
    "P2-FKD": {
        "kd_mode": "feature",
        "kd_weight": 0.0,
        "fkd_weight": 0.5,
        "kd_temperature": 4.0,
    },
    "P2-RFKD": {
        "kd_mode": "response+feature",
        "kd_weight": 1.0,
        "fkd_weight": 0.5,
        "kd_temperature": 4.0,
    },
}


def train_experiment(exp_id: str, seed: int, device: str) -> Path:
    """Train one PKD experiment."""
    import numpy as np
    import torch
    import random as _random

    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)

    cfg = EXPERIMENTS[exp_id]
    run_name = f"{exp_id}_seed{seed}"

    overrides = dict(MASK_RESCUE_TRAIN_DEFAULTS)
    overrides.update({
        "data": DATA_7CH_YAML,
        "device": device,
        "project": str(PROJECT_DIR),
        "name": run_name,
        "seed": seed,
        "exist_ok": True,
        "teacher_weights": str(TEACHER_WEIGHTS),
        **cfg,
    })

    model = YOLO(str(STUDENT_MODEL_YAML))
    model.train(trainer=PKDTrainer, **overrides)

    best_pt = PROJECT_DIR / run_name / "weights" / "best.pt"
    print(f"\n[{exp_id}] Training complete. Best weights: {best_pt}")
    return best_pt


def evaluate_model(weights: str, data_yaml: str, device: str = "0") -> dict:
    """Evaluate a model with monkey-patched NPY dataset."""
    from ultralytics.data import build as yolo_build
    from ultralytics.models.yolo.detect import val as detect_val
    from ultralytics_plugins.mask_rescue_dataset_fixed import MaskRescueNPYDataset
    from ultralytics.utils import colorstr

    _orig_build = yolo_build.build_yolo_dataset

    def _patched_build(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
        dataset_root = Path(data.get("path", "")).resolve()
        channels = data.get("channels", 3)
        return MaskRescueNPYDataset(
            img_path=img_path, imgsz=cfg.imgsz, batch_size=batch,
            augment=False, hyp=cfg, rect=rect, stride=stride, pad=0.5,
            prefix=colorstr(f"{mode}: "), dataset_root=dataset_root,
            channels=channels, task=cfg.task, data=data,
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


def run_single(exp_id: str, args):
    """Train + evaluate a single experiment."""
    seed = args.seed
    device = args.device
    run_name = f"{exp_id}_seed{seed}"

    if args.eval_only:
        weights_path = args.weights or str(PROJECT_DIR / run_name / "weights" / "best.pt")
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"No weights at {weights_path}. Train first or provide --weights.")
    else:
        weights_path = str(train_experiment(exp_id, seed, device))

    # Evaluate on 6ch test set
    print(f"\n[{exp_id}] Evaluating on 6ch test set...")
    metrics = evaluate_model(weights_path, DATA_6CH_YAML, device=device)

    # Save results
    eval_dir = PROJECT_DIR / f"{run_name}_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cfg = EXPERIMENTS[exp_id]
    result = {
        "exp_id": exp_id,
        "seed": seed,
        **cfg,
        "teacher_weights": str(TEACHER_WEIGHTS),
        "student_weights": weights_path,
        "metrics": metrics,
    }
    result_path = eval_dir / "phase2_metrics.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{exp_id}] Results saved to {result_path}")
    return result


def print_summary(results: list[dict]):
    """Print comparison table with Phase 0 baselines."""
    rows = []

    # Load P0-A baseline
    p0a_path = ROOT / "runs_phase0" / "P0-A_seed0_eval" / "phase0_metrics.json"
    if p0a_path.exists():
        with open(p0a_path) as f:
            rows.append(("P0-A (6ch baseline)", json.load(f)["metrics"]))

    # Phase 2 results
    for r in results:
        rows.append((f"{r['exp_id']} ({r['kd_mode']})", r["metrics"]))

    # Load P0-C oracle
    p0c_path = ROOT / "runs_phase0" / "P0-C_seed0_eval" / "phase0_metrics.json"
    if p0c_path.exists():
        with open(p0c_path) as f:
            rows.append(("P0-C (7ch oracle)", json.load(f)["metrics"]))

    # Also include P1-RKD if available
    p1_path = ROOT / "runs_phase1" / "P1-RKD_seed0_eval" / "phase1_metrics.json"
    if p1_path.exists():
        with open(p1_path) as f:
            p1 = json.load(f)
            rows.insert(1, ("P1-RKD (Phase 1 ref)", p1["metrics"]))

    print("\n" + "=" * 90)
    print("Phase 2: Feature KD Ablation Results")
    print("=" * 90)
    header = f"{'Experiment':<32} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}"
    print(header)
    print("-" * 90)
    for name, m in rows:
        p = m.get("metrics/precision(B)", 0)
        r = m.get("metrics/recall(B)", 0)
        m50 = m.get("metrics/mAP50(B)", 0)
        m95 = m.get("metrics/mAP50-95(B)", 0)
        print(f"{name:<32} {p:>10.4f} {r:>10.4f} {m50:>10.4f} {m95:>10.4f}")
    print("=" * 90)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Feature KD Ablation")
    parser.add_argument("--exp", default="all", choices=["P2-RKD", "P2-FKD", "P2-RFKD", "all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.exp == "all":
        exp_ids = ["P2-RKD", "P2-FKD", "P2-RFKD"]
    else:
        exp_ids = [args.exp]

    results = []
    for eid in exp_ids:
        r = run_single(eid, args)
        results.append(r)

    print_summary(results)
