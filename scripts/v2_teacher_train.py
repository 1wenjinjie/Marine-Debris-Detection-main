"""
MARIDA v2 teacher training: 7ch model (YOLOv11n) with knowledge heatmap.

Evaluates two variants after training:
  P0-C (oracle): full 7ch at test time — upper bound for knowledge channel value
  P0-B (LUPI):   zero channel 7 at test time — simulates deployment without GT

Gate (checked on P0-C):
  Marine_Debris recall >= 0.15 AND Marine_Debris AP50 > 0.05

Decision logic:
  - P0-C passes gate  -> knowledge channel has value, proceed to KD
  - P0-C fails gate   -> Plan B (background ratio / positive resampling)
  - P0-B is explanatory only (distribution shift risk means P0-B < baseline
    does NOT invalidate the approach)

Usage:
  # Single seed (recommended first)
  python scripts/v2_teacher_train.py --seed 0

  # Multi-seed
  python scripts/v2_teacher_train.py --seed 0 1 2

  # Skip training, evaluate existing weights only
  python scripts/v2_teacher_train.py --seed 0 --eval-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.data import build as yolo_build
from ultralytics.models.yolo.detect import val as detect_val
from ultralytics.utils import colorstr

from training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS
from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

DATA_7CH = str(ROOT / "configs" / "datasets" / "marida_v2_7ch.yaml")
MODEL_7CH = str(ROOT / "configs" / "models" / "yolov11n_v2_7ch.yaml")
RUNS_DIR = ROOT / "runs_v2"

CLASS_NAMES = ["Marine_Debris", "Algae_Organic", "Ship"]

# Decision gate thresholds (applied to P0-C oracle)
GATE_DEBRIS_RECALL = 0.15
GATE_DEBRIS_AP50 = 0.05


# ---------------------------------------------------------------------------
# P0-B: Zero-knowledge dataset for LUPI evaluation
# ---------------------------------------------------------------------------

class ZeroKnowledgeNPYDataset(MaskRescueNPYDataset):
    """MaskRescueNPYDataset that zeros the last channel (knowledge heatmap).

    Used for P0-B evaluation: the model was trained with 7ch, but at test
    time the knowledge channel is unavailable (simulated by zeroing).
    """

    def _load_array(self, path: Path) -> np.ndarray:
        arr = super()._load_array(path)
        arr[..., -1] = 0.0  # zero the knowledge heatmap channel
        return arr


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _make_patched_build(dataset_root: Path, channels: int, channel_stats: str | None,
                        zero_knowledge: bool = False):
    """Create a patched build_yolo_dataset for monkey-patching."""
    ds_cls = ZeroKnowledgeNPYDataset if zero_knowledge else MaskRescueNPYDataset

    def _patched(cfg, img_path, batch, data, mode="train", rect=False, stride=32, **kw):
        return ds_cls(
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
            channel_stats=channel_stats,
            task=cfg.task,
            data=data,
        )
    return _patched


def _run_eval(weights: str, data_yaml: str, channels: int,
              zero_knowledge: bool = False, split: str = "test",
              imgsz: int = 512, batch: int = 8, device: str = "0",
              conf: float = 0.001, iou: float = 0.6) -> dict:
    """Run evaluation with optional zero-knowledge mode (P0-B)."""
    data_cfg = yaml.safe_load(Path(data_yaml).read_text())
    dataset_root = Path(data_cfg.get("path", "")).resolve()

    stats_path = dataset_root / f"channel_stats_{channels}ch.json"
    channel_stats = str(stats_path) if stats_path.exists() else None

    _orig_build = yolo_build.build_yolo_dataset
    _orig_det = detect_val.build_yolo_dataset
    patched = _make_patched_build(dataset_root, channels, channel_stats,
                                  zero_knowledge=zero_knowledge)
    yolo_build.build_yolo_dataset = patched
    detect_val.build_yolo_dataset = patched

    try:
        model = YOLO(weights)
        results = model.val(
            data=data_yaml,
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            conf=conf,
            iou=iou,
            verbose=True,
        )
    finally:
        yolo_build.build_yolo_dataset = _orig_build
        detect_val.build_yolo_dataset = _orig_det

    nc = len(CLASS_NAMES)
    variant = "P0-B_zero_knowledge" if zero_knowledge else "P0-C_oracle"
    metrics = {
        "weights": str(weights),
        "data": str(data_yaml),
        "channels": channels,
        "variant": variant,
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class": {},
    }

    for i in range(nc):
        metrics["per_class"][CLASS_NAMES[i]] = {
            "AP50": float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
            "AP50_95": float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
            "precision": float(results.box.p[i]) if i < len(results.box.p) else 0.0,
            "recall": float(results.box.r[i]) if i < len(results.box.r) else 0.0,
        }

    return metrics


def _print_metrics(metrics: dict, label: str) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print()
    print("  Per-class:")
    for name, cls_m in metrics["per_class"].items():
        print(f"    {name:20s}: AP50={cls_m['AP50']:.4f}  "
              f"P={cls_m['precision']:.4f}  R={cls_m['recall']:.4f}")


def _check_gate(p0c_metrics: dict) -> tuple[bool, str]:
    """Check P0-C oracle against decision gate.

    Returns (passed, reason_string).
    """
    debris = p0c_metrics["per_class"].get("Marine_Debris", {})
    debris_recall = debris.get("recall", 0.0)
    debris_ap50 = debris.get("AP50", 0.0)

    recall_ok = debris_recall >= GATE_DEBRIS_RECALL
    ap50_ok = debris_ap50 > GATE_DEBRIS_AP50

    reason_parts = []
    if recall_ok:
        reason_parts.append(f"recall={debris_recall:.4f} >= {GATE_DEBRIS_RECALL}")
    else:
        reason_parts.append(f"recall={debris_recall:.4f} < {GATE_DEBRIS_RECALL} [FAIL]")
    if ap50_ok:
        reason_parts.append(f"AP50={debris_ap50:.4f} > {GATE_DEBRIS_AP50}")
    else:
        reason_parts.append(f"AP50={debris_ap50:.4f} <= {GATE_DEBRIS_AP50} [FAIL]")

    passed = recall_ok and ap50_ok
    return passed, "; ".join(reason_parts)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_seed(seed: int, eval_only: bool = False) -> dict:
    """Train a 7ch teacher for one seed, then evaluate P0-C and P0-B."""
    run_name = f"teacher_7ch_s{seed}"
    project = str(RUNS_DIR)
    best_weights = str(RUNS_DIR / run_name / "weights" / "best.pt")

    # --- Training ---
    if not eval_only:
        overrides = {
            **MASK_RESCUE_TRAIN_DEFAULTS,
            "model": MODEL_7CH,
            "data": DATA_7CH,
            "project": project,
            "name": run_name,
            "seed": seed,
            "exist_ok": True,
            "device": "0",
        }

        print(f"\n{'='*60}")
        print(f"Training: {run_name} (seed={seed})")
        print(f"{'='*60}")

        trainer = MaskRescueTrainer(overrides=overrides)
        trainer.train()
    else:
        if not Path(best_weights).exists():
            raise FileNotFoundError(
                f"--eval-only but weights not found: {best_weights}")
        print(f"\n[eval-only] Skipping training, using: {best_weights}")

    # --- P0-C: Oracle evaluation (full 7ch at test) ---
    print(f"\n--- P0-C (oracle): evaluating {run_name} with full 7ch ---")
    p0c = _run_eval(weights=best_weights, data_yaml=DATA_7CH, channels=7)
    p0c["seed"] = seed
    p0c["run_name"] = run_name

    p0c_path = RUNS_DIR / run_name / "eval_test_p0c.json"
    p0c_path.parent.mkdir(parents=True, exist_ok=True)
    p0c_path.write_text(json.dumps(p0c, indent=2))
    _print_metrics(p0c, f"P0-C Oracle — {run_name}")

    # --- Gate check ---
    passed, reason = _check_gate(p0c)
    gate_status = "PASS" if passed else "FAIL"
    print(f"\n  >> GATE [{gate_status}]: {reason}")

    # --- P0-B: LUPI evaluation (zero knowledge channel at test) ---
    print(f"\n--- P0-B (LUPI): evaluating {run_name} with ch7=0 ---")
    p0b = _run_eval(weights=best_weights, data_yaml=DATA_7CH, channels=7,
                    zero_knowledge=True)
    p0b["seed"] = seed
    p0b["run_name"] = run_name

    p0b_path = RUNS_DIR / run_name / "eval_test_p0b.json"
    p0b_path.write_text(json.dumps(p0b, indent=2))
    _print_metrics(p0b, f"P0-B LUPI (ch7=0) — {run_name}")

    return {
        "seed": seed,
        "run_name": run_name,
        "p0c": p0c,
        "p0b": p0b,
        "gate_passed": passed,
        "gate_reason": reason,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARIDA v2 teacher training (7ch)")
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate existing weights only")
    args = parser.parse_args()

    all_results = []
    for seed in args.seed:
        result = train_one_seed(seed, eval_only=args.eval_only)
        all_results.append(result)

    # --- Multi-seed summary ---
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"MULTI-SEED SUMMARY ({len(all_results)} seeds)")
        print(f"{'='*60}")

        for variant_key, variant_label in [("p0c", "P0-C Oracle"), ("p0b", "P0-B LUPI")]:
            print(f"\n  {variant_label}:")
            for metric in ["mAP50", "mAP50_95", "precision", "recall"]:
                vals = [r[variant_key][metric] for r in all_results]
                print(f"    {metric:15s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
                      f"({', '.join(f'{v:.4f}' for v in vals)})")

            # Per-class Marine_Debris detail
            debris_vals = {
                m: [r[variant_key]["per_class"]["Marine_Debris"][m] for r in all_results]
                for m in ["AP50", "recall", "precision"]
            }
            print(f"    --- Marine_Debris ---")
            for m, vals in debris_vals.items():
                print(f"      {m:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

        summary = {
            "experiment": "v2_teacher_7ch",
            "seeds": args.seed,
            "gate_threshold": {
                "Marine_Debris_recall": GATE_DEBRIS_RECALL,
                "Marine_Debris_AP50": GATE_DEBRIS_AP50,
            },
            "results": all_results,
        }

        # Add aggregated stats
        for variant_key in ["p0c", "p0b"]:
            summary[f"{variant_key}_summary"] = {
                metric: {
                    "mean": float(np.mean([r[variant_key][metric] for r in all_results])),
                    "std": float(np.std([r[variant_key][metric] for r in all_results])),
                }
                for metric in ["mAP50", "mAP50_95", "precision", "recall"]
            }

        summary_path = RUNS_DIR / "teacher_7ch_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  Summary: {summary_path}")

    # --- Final gate verdict ---
    print(f"\n{'='*60}")
    print("GATE RESULTS")
    print(f"{'='*60}")
    print(f"  Threshold: Marine_Debris recall >= {GATE_DEBRIS_RECALL}, "
          f"AP50 > {GATE_DEBRIS_AP50}")
    for r in all_results:
        status = "PASS" if r["gate_passed"] else "FAIL"
        print(f"  seed={r['seed']}: [{status}] {r['gate_reason']}")

    any_passed = any(r["gate_passed"] for r in all_results)
    all_passed = all(r["gate_passed"] for r in all_results)

    if all_passed:
        print("\n  >> ALL seeds passed. Knowledge channel has value.")
        print("     Next: proceed to KD experiments (v2_pkd_train.py)")
    elif any_passed:
        print("\n  >> PARTIAL pass. Check per-seed variance before KD.")
    else:
        print("\n  >> ALL seeds FAILED. Knowledge channel insufficient at this scale.")
        print("     Next: Plan B — background ratio control / positive resampling")


if __name__ == "__main__":
    main()
