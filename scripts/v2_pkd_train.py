"""
MARIDA v2 Privileged Knowledge Distillation training.

Student (6ch spectral) learns from Teacher (7ch with knowledge heatmap)
via response-level KD on classification logits.

Architecture:
  - Student: YOLOv11n, ch=6 (same as baseline)
  - Teacher: YOLOv11n, ch=7 (pre-trained, frozen)
  - Data: 7ch dataset (student slices first 6, teacher uses all 7)
  - Loss: (1 - alpha) * detection_loss + alpha * KD_loss
  - KD: BCE-based on classification logits at all anchor positions
  - Temperature: T=4 (spreads teacher's confident predictions)

Usage:
  python scripts/v2_pkd_train.py --seed 0
  python scripts/v2_pkd_train.py --seed 0 --alpha 0.3 --temperature 8
  python scripts/v2_pkd_train.py --seed 0 --eval-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

# Paths
DATA_7CH = str(ROOT / "configs" / "datasets" / "marida_v2_7ch.yaml")
DATA_6CH = str(ROOT / "configs" / "datasets" / "marida_v2_6ch.yaml")
MODEL_6CH = str(ROOT / "configs" / "models" / "yolov11n_v2_6ch.yaml")
DEFAULT_TEACHER = str(ROOT / "runs_v2" / "teacher_7ch_s0" / "weights" / "best.pt")
RUNS_DIR = ROOT / "runs_v2"

CLASS_NAMES = ["Marine_Debris", "Algae_Organic", "Ship"]


# ---------------------------------------------------------------------------
# KD Loss
# ---------------------------------------------------------------------------

def _kd_cls_loss(student_feats, teacher_feats, temperature, nc):
    """BCE-based KD loss on classification logits at every anchor position.

    YOLO uses independent sigmoids per class (multi-label), not softmax.
    So KD uses binary cross-entropy: student logits vs teacher soft probs.

    Args:
        student_feats: list of [B, no, H, W] per detection scale (training mode)
        teacher_feats: list of [B, no, H, W] per detection scale
        temperature: softening temperature (higher = softer targets)
        nc: number of classes

    Returns:
        Scalar KD loss (mean over all elements).
    """
    total_loss = 0.0
    n_elements = 0

    for s_feat, t_feat in zip(student_feats, teacher_feats):
        # Classification logits are the last nc channels
        s_cls = s_feat[:, -nc:, :, :]  # [B, nc, H, W]
        t_cls = t_feat[:, -nc:, :, :]

        # Temperature scaling
        s_scaled = s_cls / temperature
        t_soft = torch.sigmoid(t_cls / temperature).detach()

        # Binary cross-entropy: student learns teacher's soft distribution
        loss = F.binary_cross_entropy_with_logits(
            s_scaled, t_soft, reduction="sum"
        )
        # T^2 compensates for reduced gradient magnitude from temperature
        total_loss += loss * (temperature ** 2)
        n_elements += s_cls.numel()

    return total_loss / max(n_elements, 1)


# ---------------------------------------------------------------------------
# PKD Detection Model (subclass for clean KD integration)
# ---------------------------------------------------------------------------

class PKDDetectionModel(DetectionModel):
    """DetectionModel with channel slicing and KD loss for PKD training.

    During training:
      - Slices input to first `ch` channels for student forward pass
      - Feeds full input (with knowledge channel) to frozen teacher
      - Combines hard detection loss with KD classification loss

    During inference (predict):
      - Slices input to first `ch` channels if input has extra channels
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store teacher in a plain list so PyTorch does NOT register it
        # as a submodule (prevents trainer from unfreezing teacher params).
        self._pkd_teacher_ref = []
        self._pkd_alpha = 0.5
        self._pkd_temperature = 4.0

    def predict(self, x, *args, **kwargs):
        """Forward for inference — slice to expected channels if needed."""
        ch = self.yaml.get("channels", 6)
        if hasattr(x, "shape") and x.shape[1] > ch:
            x = x[:, :ch]
        return super().predict(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        """Combined hard loss + KD loss."""
        if not self._pkd_teacher_ref or not self.training:
            # No teacher or in eval mode (validation): standard detection loss only.
            # During validation, preds are pre-computed from predict() which slices channels.
            return super().loss(batch, preds)

        teacher = self._pkd_teacher_ref[0]
        img = batch["img"]
        ch = self.yaml.get("channels", 6)
        img_student = img[:, :ch] if img.shape[1] > ch else img

        # Student forward on spectral channels only
        if preds is None:
            preds = self.predict(img_student)

        # Hard detection loss
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        hard_loss, loss_items = self.criterion(preds, batch)

        # Teacher forward on full input (with knowledge channel)
        with torch.no_grad():
            t_out = teacher(img)
            # eval mode returns (decoded, raw_feats); training mode returns raw_feats
            t_preds = t_out[1] if isinstance(t_out, tuple) else t_out

        # KD classification loss
        nc = self.model[-1].nc
        kd_loss = _kd_cls_loss(
            preds, t_preds, self._pkd_temperature, nc
        )

        # Combined: (1-alpha)*hard + alpha*kd, both scaled by batch size
        alpha = self._pkd_alpha
        bs = img.shape[0]
        total_loss = (1.0 - alpha) * hard_loss + alpha * kd_loss * bs

        return total_loss, loss_items


# ---------------------------------------------------------------------------
# PKD Trainer
# ---------------------------------------------------------------------------

class PKDTrainer(MaskRescueTrainer):
    """Trainer for Privileged Knowledge Distillation.

    Uses 7ch data (student slices first 6, teacher uses all 7).
    Student model is ch=6, saved weights are standard 6ch — directly
    evaluable with v2_evaluate.py on the 6ch test set.
    """

    def __init__(self, teacher_weights, alpha=0.5, temperature=4.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher_weights = teacher_weights
        self.kd_alpha = alpha
        self.kd_temperature = temperature

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Create PKDDetectionModel instead of standard DetectionModel.

        Student is always 6ch (spectral only). The 7ch data config is used
        for the dataloader — the model slices channels internally.
        """
        model = PKDDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=6,  # Student is 6ch; dataloader provides 7ch, we slice in loss/predict
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def setup_model(self):
        """Load student model and attach frozen teacher."""
        ckpt = super().setup_model()

        student = unwrap_model(self.model)
        LOGGER.info(
            f"PKD: Loading teacher from {self.teacher_weights}"
        )
        teacher_yolo = YOLO(self.teacher_weights)
        teacher_nn = teacher_yolo.model.to(self.device).eval()
        for p in teacher_nn.parameters():
            p.requires_grad_(False)

        student._pkd_teacher_ref = [teacher_nn]  # List prevents nn.Module registration
        student._pkd_alpha = self.kd_alpha
        student._pkd_temperature = self.kd_temperature

        t_params = sum(p.numel() for p in teacher_nn.parameters())
        s_params = sum(p.numel() for p in student.parameters())
        LOGGER.info(
            f"PKD: teacher={t_params/1e6:.2f}M params (frozen), "
            f"student={s_params/1e6:.2f}M params, "
            f"alpha={self.kd_alpha}, T={self.kd_temperature}"
        )
        return ckpt


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_one_seed(seed: int, teacher_weights: str,
                   alpha: float, temperature: float,
                   eval_only: bool = False) -> dict:
    """Train one PKD seed and evaluate on 6ch test set."""
    run_name = f"pkd_a{alpha}_t{temperature}_s{seed}"
    project = str(RUNS_DIR)
    best_weights = str(RUNS_DIR / run_name / "weights" / "best.pt")

    if not eval_only:
        overrides = {
            **MASK_RESCUE_TRAIN_DEFAULTS,
            "model": MODEL_6CH,     # Student architecture (ch=6)
            "data": DATA_7CH,       # 7ch data (student slices, teacher uses all)
            "project": project,
            "name": run_name,
            "seed": seed,
            "exist_ok": True,
            "device": "0",
        }

        print(f"\n{'='*60}")
        print(f"PKD Training: {run_name}")
        print(f"  Teacher: {teacher_weights}")
        print(f"  alpha={alpha}, T={temperature}, seed={seed}")
        print(f"{'='*60}")

        trainer = PKDTrainer(
            teacher_weights=teacher_weights,
            alpha=alpha,
            temperature=temperature,
            overrides=overrides,
        )
        trainer.train()
    else:
        if not Path(best_weights).exists():
            raise FileNotFoundError(
                f"--eval-only but weights not found: {best_weights}")
        print(f"\n[eval-only] Using: {best_weights}")

    # Evaluate student on 6ch test set (deployment scenario)
    print(f"\nEvaluating {run_name} on 6ch test set...")
    from scripts.v2_evaluate import evaluate
    metrics = evaluate(
        weights=best_weights,
        data_yaml=DATA_6CH,   # 6ch test data (no knowledge channel)
        channels=6,
        split="test",
    )
    metrics["seed"] = seed
    metrics["run_name"] = run_name
    metrics["kd_alpha"] = alpha
    metrics["kd_temperature"] = temperature
    metrics["teacher_weights"] = teacher_weights

    out_path = RUNS_DIR / run_name / "eval_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    # Print results
    print(f"\n{'='*60}")
    print(f"  PKD Student Results — {run_name}")
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

    # Check debris signal
    debris = metrics["per_class"].get("Marine_Debris", {})
    debris_recall = debris.get("recall", 0.0)
    debris_ap50 = debris.get("AP50", 0.0)
    has_signal = debris_recall > 0.0 or debris_ap50 > 0.01

    if has_signal:
        print(f"\n  >> SIGNAL DETECTED: Marine_Debris recall={debris_recall:.4f}, AP50={debris_ap50:.4f}")
        if debris_recall >= 0.154 and debris_ap50 > 0.05:
            print("  >> PAPER-GRADE: meets publication threshold!")
        else:
            print("  >> Exploratory: non-zero signal, not yet paper-grade.")
    else:
        print(f"\n  >> NO SIGNAL: Marine_Debris recall={debris_recall:.4f}, AP50={debris_ap50:.4f}")
        print("     Consider: adjust alpha/T, or try weakened-shortcut teacher.")

    return metrics


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description="MARIDA v2 PKD training")
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--teacher", type=str, default=DEFAULT_TEACHER,
                        help="Path to teacher weights")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="KD loss weight (0=pure hard, 1=pure KD)")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="KD temperature (higher = softer targets)")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    all_results = []
    for seed in args.seed:
        result = train_one_seed(
            seed=seed,
            teacher_weights=args.teacher,
            alpha=args.alpha,
            temperature=args.temperature,
            eval_only=args.eval_only,
        )
        all_results.append(result)

    # Multi-seed summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"PKD MULTI-SEED SUMMARY ({len(all_results)} seeds)")
        print(f"  alpha={args.alpha}, T={args.temperature}")
        print(f"{'='*60}")

        for metric in ["mAP50", "mAP50_95", "precision", "recall"]:
            vals = [r[metric] for r in all_results]
            print(f"  {metric:15s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
                  f"({', '.join(f'{v:.4f}' for v in vals)})")

        # Per-class Marine_Debris
        print(f"\n  Marine_Debris:")
        for m in ["AP50", "recall", "precision"]:
            vals = [r["per_class"]["Marine_Debris"][m] for r in all_results]
            print(f"    {m:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

        summary = {
            "experiment": "v2_pkd",
            "alpha": args.alpha,
            "temperature": args.temperature,
            "teacher": args.teacher,
            "seeds": args.seed,
            "results": all_results,
            "summary": {
                metric: {
                    "mean": float(np.mean([r[metric] for r in all_results])),
                    "std": float(np.std([r[metric] for r in all_results])),
                }
                for metric in ["mAP50", "mAP50_95", "precision", "recall"]
            },
        }
        tag = f"a{args.alpha}_t{args.temperature}"
        summary_path = RUNS_DIR / f"pkd_{tag}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
