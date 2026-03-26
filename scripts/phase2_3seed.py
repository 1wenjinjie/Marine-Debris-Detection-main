"""
Phase 2 — 3-seed stability verification for P0-A (baseline) and P2-RKD (best method).
Skips already-completed seeds.

Usage:
    python scripts/phase2_3seed.py --device 0
    python scripts/phase2_3seed.py --device 0 --eval-only
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

import numpy as np
import random as _random
import torch
from ultralytics import YOLO

from scripts.training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS

# ---------- Paths ----------
TEACHER_WEIGHTS = ROOT / "runs_phase0" / "P0-C_seed0" / "weights" / "best.pt"
STUDENT_MODEL_YAML = str(ROOT / "configs" / "models" / "yolov11n_6ch.yaml")
DATA_7CH_YAML = str(ROOT / "configs" / "datasets" / "marida_rescue_7ch.yaml")
DATA_6CH_YAML = str(ROOT / "configs" / "datasets" / "marida_rescue_split_6ch.yaml")

SEEDS = [0, 1, 2]


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)


def _evaluate(weights: str, data_yaml: str, device: str) -> dict:
    """Evaluate with monkey-patched NPY dataset."""
    from ultralytics.data import build as yolo_build
    from ultralytics.models.yolo.detect import val as detect_val
    from ultralytics_plugins.mask_rescue_dataset_fixed import MaskRescueNPYDataset
    from ultralytics.utils import colorstr

    _orig = yolo_build.build_yolo_dataset

    def _patched(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
        return MaskRescueNPYDataset(
            img_path=img_path, imgsz=cfg.imgsz, batch_size=batch,
            augment=False, hyp=cfg, rect=rect, stride=stride, pad=0.5,
            prefix=colorstr(f"{mode}: "),
            dataset_root=Path(data.get("path", "")).resolve(),
            channels=data.get("channels", 3), task=cfg.task, data=data,
        )

    yolo_build.build_yolo_dataset = _patched
    detect_val.build_yolo_dataset = _patched
    try:
        model = YOLO(weights)
        r = model.val(data=data_yaml, device=device, batch=8, imgsz=512, verbose=False)
        m = r.results_dict
        return {k: float(m.get(k, 0)) for k in [
            "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)"
        ]}
    finally:
        yolo_build.build_yolo_dataset = _orig
        detect_val.build_yolo_dataset = _orig


# ====================== P0-A Baseline ======================
def train_p0a(seed: int, device: str) -> Path:
    from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

    _set_seed(seed)
    project = str(ROOT / "runs_phase0")
    run_name = f"P0-A_seed{seed}"
    overrides = dict(MASK_RESCUE_TRAIN_DEFAULTS)
    overrides.update({
        "data": DATA_6CH_YAML,
        "device": device,
        "project": project,
        "name": run_name,
        "seed": seed,
        "exist_ok": True,
    })
    model = YOLO(STUDENT_MODEL_YAML)
    model.train(trainer=MaskRescueTrainer, **overrides)
    return Path(project) / run_name / "weights" / "best.pt"


# ====================== P2-RKD ======================
def train_rkd(seed: int, device: str) -> Path:
    from ultralytics_plugins.pkd_trainer import PKDTrainer

    _set_seed(seed)
    project = str(ROOT / "runs_phase2")
    run_name = f"P2-RKD_seed{seed}"
    overrides = dict(MASK_RESCUE_TRAIN_DEFAULTS)
    overrides.update({
        "data": DATA_7CH_YAML,
        "device": device,
        "project": project,
        "name": run_name,
        "seed": seed,
        "exist_ok": True,
        "teacher_weights": str(TEACHER_WEIGHTS),
        "kd_weight": 1.0,
        "kd_temperature": 4.0,
        "fkd_weight": 0.0,
        "kd_mode": "response",
    })
    model = YOLO(STUDENT_MODEL_YAML)
    model.train(trainer=PKDTrainer, **overrides)
    return Path(project) / run_name / "weights" / "best.pt"


# ====================== Main ======================
def run_all(device: str, eval_only: bool):
    results = {"P0-A": {}, "P2-RKD": {}}

    for seed in SEEDS:
        # --- P0-A ---
        p0a_eval_path = ROOT / "runs_phase0" / f"P0-A_seed{seed}_eval" / "3seed_metrics.json"
        p0a_weights = ROOT / "runs_phase0" / f"P0-A_seed{seed}" / "weights" / "best.pt"

        # Also check legacy result files from Phase 0
        p0a_legacy = ROOT / "runs_phase0" / f"P0-A_seed{seed}_eval" / "phase0_metrics.json"
        if p0a_eval_path.exists():
            with open(p0a_eval_path) as f:
                results["P0-A"][seed] = json.load(f)["metrics"]
            print(f"[P0-A seed={seed}] Loaded cached results.")
        elif p0a_legacy.exists():
            with open(p0a_legacy) as f:
                results["P0-A"][seed] = json.load(f)["metrics"]
            print(f"[P0-A seed={seed}] Loaded from Phase 0 results.")
        else:
            if not p0a_weights.exists() and not eval_only:
                print(f"\n[P0-A seed={seed}] Training...")
                train_p0a(seed, device)
            if p0a_weights.exists():
                print(f"[P0-A seed={seed}] Evaluating...")
                metrics = _evaluate(str(p0a_weights), DATA_6CH_YAML, device)
                results["P0-A"][seed] = metrics
                p0a_eval_path.parent.mkdir(parents=True, exist_ok=True)
                with open(p0a_eval_path, "w") as f:
                    json.dump({"exp": "P0-A", "seed": seed, "metrics": metrics}, f, indent=2)
            else:
                print(f"[P0-A seed={seed}] SKIP — no weights found.")

        # --- P2-RKD ---
        rkd_eval_path = ROOT / "runs_phase2" / f"P2-RKD_seed{seed}_eval" / "3seed_metrics.json"
        rkd_weights = ROOT / "runs_phase2" / f"P2-RKD_seed{seed}" / "weights" / "best.pt"

        rkd_legacy = ROOT / "runs_phase2" / f"P2-RKD_seed{seed}_eval" / "phase2_metrics.json"
        if rkd_eval_path.exists():
            with open(rkd_eval_path) as f:
                results["P2-RKD"][seed] = json.load(f)["metrics"]
            print(f"[P2-RKD seed={seed}] Loaded cached results.")
        elif rkd_legacy.exists():
            with open(rkd_legacy) as f:
                results["P2-RKD"][seed] = json.load(f)["metrics"]
            print(f"[P2-RKD seed={seed}] Loaded from Phase 2 results.")
        else:
            if not rkd_weights.exists() and not eval_only:
                print(f"\n[P2-RKD seed={seed}] Training...")
                train_rkd(seed, device)
            if rkd_weights.exists():
                print(f"[P2-RKD seed={seed}] Evaluating...")
                metrics = _evaluate(str(rkd_weights), DATA_6CH_YAML, device)
                results["P2-RKD"][seed] = metrics
                rkd_eval_path.parent.mkdir(parents=True, exist_ok=True)
                with open(rkd_eval_path, "w") as f:
                    json.dump({"exp": "P2-RKD", "seed": seed, "metrics": metrics}, f, indent=2)
            else:
                print(f"[P2-RKD seed={seed}] SKIP — no weights found.")

    # Print summary
    print_summary(results)

    # Save combined results
    out_path = ROOT / "runs_phase2" / "3seed_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


def print_summary(results: dict):
    print("\n" + "=" * 95)
    print("3-Seed Stability Verification")
    print("=" * 95)

    keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    short = ["Precision", "Recall", "mAP50", "mAP50-95"]

    header = f"{'Experiment':<20} {'seed':<6}" + "".join(f"{s:>12}" for s in short)
    print(header)
    print("-" * 95)

    for exp_name in ["P0-A", "P2-RKD"]:
        for seed in SEEDS:
            if seed in results.get(exp_name, {}):
                m = results[exp_name][seed]
                vals = "".join(f"{m.get(k, 0):>12.4f}" for k in keys)
                print(f"{exp_name:<20} {seed:<6}{vals}")
            else:
                print(f"{exp_name:<20} {seed:<6}{'—':>12}" * 4)

        # Mean ± std
        seed_metrics = [results[exp_name][s] for s in SEEDS if s in results.get(exp_name, {})]
        if len(seed_metrics) >= 2:
            means = {k: np.mean([m[k] for m in seed_metrics]) for k in keys}
            stds = {k: np.std([m[k] for m in seed_metrics]) for k in keys}
            row = "".join(f"{means[k]:>7.4f}±{stds[k]:.4f}" for k in keys)
            print(f"  {'mean±std':<18} {'':>4}{row}")
        print()

    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    run_all(args.device, args.eval_only)
