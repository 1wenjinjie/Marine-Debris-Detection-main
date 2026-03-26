"""Privileged Knowledge Distillation (PKD) Trainer.

Trains a 6ch student model using KD from a 7ch privileged teacher.
The teacher has access to GT-derived heatmap (channel 7); the student sees only 6 spectral bands.

Supports three KD modes:
    - "response": soft-label KD on detection head class logits
    - "feature":  L2 alignment on neck/FPN outputs (layers 16, 19, 22)
    - "response+feature": both combined

Usage:
    trainer = PKDTrainer(overrides={
        "data": "marida_rescue_7ch.yaml",
        "teacher_weights": "runs_phase0/P0-C_seed0/weights/best.pt",
        "kd_weight": 1.0,
        "kd_temperature": 4.0,
        "fkd_weight": 0.5,
        "kd_mode": "response+feature",
        ...
    })
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from copy import copy
from typing import Any

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from .mask_rescue_trainer import MaskRescueTrainer


# ---------------------------------------------------------------------------
# Neck feature extraction helper
# ---------------------------------------------------------------------------
NECK_LAYERS = (16, 19, 22)  # P3/8, P4/16, P5/32 in YOLOv11n


class _FeatureCapture:
    """Register forward hooks on specific layers to capture their outputs."""

    def __init__(self, model: torch.nn.Module, layer_indices: tuple[int, ...] = NECK_LAYERS):
        self.features: dict[int, torch.Tensor] = {}
        self._hooks = []
        for idx in layer_indices:
            hook = model.model[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, idx: int):
        def hook_fn(module, inp, out):
            self.features[idx] = out
        return hook_fn

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Validator (slices 7ch → 6ch for student)
# ---------------------------------------------------------------------------
class PKDValidator(DetectionValidator):
    """Validator that slices 7ch input to 6ch for the student model.

    Handles the channel mismatch: data YAML says 7ch (for teacher), but
    the student model expects 6ch. Patches self.data["channels"] after
    it's populated by the parent __call__ but before warmup runs.
    """

    def __call__(self, trainer=None, model=None):
        """Patch channels to 6 so warmup creates the right dummy tensor.

        The parent's __call__ sets self.data from trainer.data (during-training)
        or via check_det_dataset (final_eval), then warmups with data["channels"].
        We patch all paths so warmup sees 6, not 7.
        """
        # During-training val: parent copies trainer.data → self.data
        if trainer is not None and hasattr(trainer, "data") and trainer.data is not None:
            trainer.data["channels"] = 6

        # final_eval path: parent calls check_det_dataset which is imported at
        # module level in ultralytics.engine.validator. Patch that reference.
        import ultralytics.engine.validator as _val_mod
        _orig_check = _val_mod.check_det_dataset
        def _patched_check(*a, **kw):
            d = _orig_check(*a, **kw)
            d["channels"] = 6
            return d
        _val_mod.check_det_dataset = _patched_check

        try:
            result = super().__call__(trainer=trainer, model=model)
        finally:
            _val_mod.check_det_dataset = _orig_check
            # Restore 7ch for training dataloader
            if trainer is not None and hasattr(trainer, "data") and trainer.data is not None:
                trainer.data["channels"] = 7
        return result

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess(batch)
        if batch["img"].shape[1] > 6:
            batch["img"] = batch["img"][:, :6, :, :]
        return batch


# ---------------------------------------------------------------------------
# Combined KD Loss
# ---------------------------------------------------------------------------
class PKDLoss:
    """Wraps v8DetectionLoss and adds response-level and/or feature-level KD."""

    def __init__(
        self,
        base_criterion,
        trainer: "PKDTrainer",
        kd_weight: float,
        kd_temperature: float,
        fkd_weight: float,
        kd_mode: str,
    ):
        self.base = base_criterion
        self.trainer = trainer
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.fkd_weight = fkd_weight
        self.kd_mode = kd_mode
        # Copy attributes that ultralytics loss internals may access
        if hasattr(base_criterion, "stride"):
            self.stride = base_criterion.stride
        if hasattr(base_criterion, "reg_max"):
            self.reg_max = base_criterion.reg_max

    def __call__(self, preds, batch):
        """Compute combined detection + KD loss."""
        base_loss, loss_items = self.base(preds, batch)

        teacher_preds = self.trainer._teacher_preds
        if teacher_preds is None:
            return base_loss, loss_items

        kd_total = torch.tensor(0.0, device=base_loss.device)

        # Response-level KD
        if "response" in self.kd_mode:
            rkd = self._response_kd_loss(preds, teacher_preds)
            kd_total = kd_total + self.kd_weight * rkd

        # Feature-level KD
        if "feature" in self.kd_mode:
            fkd = self._feature_kd_loss()
            kd_total = kd_total + self.fkd_weight * fkd

        return base_loss + kd_total, loss_items

    # ---- Response-level KD (binary soft labels) ----
    def _response_kd_loss(self, student_preds, teacher_preds):
        T = self.kd_temperature
        nc = self.base.nc
        total_kd = 0.0
        n_points = 0

        for s_feat, t_feat in zip(student_preds, teacher_preds):
            B, no, H, W = s_feat.shape
            s_cls = s_feat[:, -nc:, :, :].reshape(B, nc, -1)
            t_cls = t_feat[:, -nc:, :, :].reshape(B, nc, -1)

            t_soft = torch.sigmoid(t_cls.detach() / T)
            kd = F.binary_cross_entropy_with_logits(
                s_cls / T, t_soft, reduction="sum"
            )
            total_kd += kd
            n_points += B * nc * H * W

        return (total_kd / max(n_points, 1)) * (T ** 2)

    # ---- Feature-level KD (L2 on neck outputs) ----
    def _feature_kd_loss(self):
        t_feats = self.trainer._teacher_capture.features if self.trainer._teacher_capture else {}
        s_feats = self.trainer._student_capture.features if self.trainer._student_capture else {}
        if not t_feats or not s_feats:
            return torch.tensor(0.0)

        total = 0.0
        n_layers = 0
        for idx in NECK_LAYERS:
            if idx in t_feats and idx in s_feats:
                t_f = t_feats[idx].detach()
                s_f = s_feats[idx]
                # Normalize along channel dim before L2 (prevents scale mismatch)
                t_f = F.normalize(t_f, dim=1)
                s_f = F.normalize(s_f, dim=1)
                total = total + F.mse_loss(s_f, t_f)
                n_layers += 1

        return total / max(n_layers, 1)


# ---------------------------------------------------------------------------
# PKD Trainer
# ---------------------------------------------------------------------------
class PKDTrainer(MaskRescueTrainer):
    """MaskRescue trainer with Privileged Knowledge Distillation.

    Loads a 7ch teacher, trains a 6ch student.
    Dataloader provides 7ch; teacher gets all 7ch, student gets first 6ch.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        # Extract PKD-specific params
        self.teacher_weights = overrides.pop("teacher_weights", None)
        self.kd_weight = float(overrides.pop("kd_weight", 1.0))
        self.kd_temperature = float(overrides.pop("kd_temperature", 4.0))
        self.fkd_weight = float(overrides.pop("fkd_weight", 0.5))
        self.kd_mode = overrides.pop("kd_mode", "response")  # "response", "feature", "response+feature"

        if cfg is None:
            from ultralytics.cfg import DEFAULT_CFG
            cfg = DEFAULT_CFG
        super().__init__(cfg, overrides, _callbacks)

        self.teacher_model = None
        self._teacher_preds = None
        self._teacher_capture: _FeatureCapture | None = None
        self._student_capture: _FeatureCapture | None = None

    def _setup_train(self):
        """Set up training: load teacher, install hooks, wrap loss."""
        super()._setup_train()

        if self.teacher_weights is None:
            raise ValueError("PKDTrainer requires 'teacher_weights' in overrides.")

        # Load teacher model (7ch)
        LOGGER.info(f"PKD: loading teacher from {self.teacher_weights}")
        teacher_yolo = YOLO(self.teacher_weights)
        self.teacher_model = teacher_yolo.model.to(self.device).float()
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        # Force detect head to return raw logits (training-mode output)
        self.teacher_model.model[-1].training = True

        LOGGER.info(
            f"PKD: teacher loaded | mode={self.kd_mode}, "
            f"kd_weight={self.kd_weight}, kd_temperature={self.kd_temperature}, "
            f"fkd_weight={self.fkd_weight}"
        )

        # Install neck feature hooks if feature KD is enabled
        if "feature" in self.kd_mode:
            self._teacher_capture = _FeatureCapture(self.teacher_model)
            student_model = unwrap_model(self.model)
            self._student_capture = _FeatureCapture(student_model)
            LOGGER.info(f"PKD: feature hooks installed on layers {NECK_LAYERS}")

        # Replace student criterion with PKD-aware loss
        student_model = unwrap_model(self.model)
        base_criterion = student_model.init_criterion()
        student_model.criterion = PKDLoss(
            base_criterion, self,
            self.kd_weight, self.kd_temperature,
            self.fkd_weight, self.kd_mode,
        )

    def preprocess_batch(self, batch):
        """Teacher forward on 7ch, capture features, slice to 6ch for student."""
        batch = super().preprocess_batch(batch)

        # Clear previous features
        if self._teacher_capture:
            self._teacher_capture.clear()
        if self._student_capture:
            self._student_capture.clear()

        # Teacher forward on full 7ch (hooks capture neck features automatically)
        with torch.no_grad():
            self._teacher_preds = self.teacher_model(batch["img"])

        # Slice to 6ch for student
        batch["img"] = batch["img"][:, :6, :, :]

        return batch

    def get_validator(self):
        """Return a PKDValidator that slices 7ch to 6ch during validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return PKDValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build a 6ch student model (data YAML says 7ch for teacher)."""
        model = DetectionModel(cfg, nc=self.data["nc"], ch=6, verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
