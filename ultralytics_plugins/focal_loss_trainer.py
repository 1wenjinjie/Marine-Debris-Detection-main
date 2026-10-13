"""
Focal loss trainer for MARIDA v2.

Replaces the standard BCE classification loss in v8DetectionLoss with a
focal-modulated variant.  This down-weights easy negatives and up-weights
hard positives, targeting the confidence calibration problem observed with
Marine_Debris (TPs at conf < 0.006).

Usage: instantiate FocalLossTrainer instead of MaskRescueTrainer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import LOGGER

from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer


class FocalBCE(nn.Module):
    """Drop-in replacement for nn.BCEWithLogitsLoss(reduction='none').

    Returns per-element (unreduced) focal loss so the caller's
    `.sum() / target_scores_sum` normalisation works unchanged.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        pred_prob = pred.sigmoid()
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        modulating = (1.0 - p_t) ** self.gamma
        loss = bce * modulating

        if self.alpha > 0:
            alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss = loss * alpha_t

        return loss


class FocalLossTrainer(MaskRescueTrainer):
    """MaskRescueTrainer with focal classification loss."""

    def __init__(self, focal_gamma: float = 1.5, focal_alpha: float = 0.25,
                 **kwargs):
        super().__init__(**kwargs)
        self._focal_gamma = focal_gamma
        self._focal_alpha = focal_alpha
        self.add_callback("on_train_start", FocalLossTrainer._patch_focal)

    def _patch_focal(self, trainer=None):
        target = trainer if trainer is not None else self
        model = target.model

        original_init = model.init_criterion

        gamma = self._focal_gamma
        alpha = self._focal_alpha

        def focal_init_criterion():
            criterion = original_init()
            criterion.bce = FocalBCE(gamma=gamma, alpha=alpha)
            LOGGER.info(f"Patched cls loss: BCE -> FocalBCE(gamma={gamma}, alpha={alpha})")
            return criterion

        model.init_criterion = focal_init_criterion

        if getattr(model, "criterion", None) is not None:
            model.criterion.bce = FocalBCE(gamma=gamma, alpha=alpha)
            LOGGER.info(f"Patched existing criterion.bce -> FocalBCE(gamma={gamma}, alpha={alpha})")
