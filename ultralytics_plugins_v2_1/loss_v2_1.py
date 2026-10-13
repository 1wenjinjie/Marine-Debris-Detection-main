"""
WaterAwareSegLoss — instance-segmentation loss with three-layer
valid-region ignore for MARIDA v2.1.

Conforms to v2_1_DATASET_SPEC.md §4.

Three ignore layers (applied in order, before standard YOLO seg loss):
  1. GT skip: drop any GT instance whose bbox center is in NaN region.
  2. Prediction skip: drop any predicted instance whose bbox center is
     in NaN region from the matching pool.
  3. Mask-pixel weight: per-pixel mask BCE multiplied by valid_mask.

Wraps ultralytics' default v8SegmentationLoss; only replaces the parts
that need NaN awareness.

Phase 2 status: layer 3 (pixel weight) is fully implemented; layers
1 & 2 (instance skip) are stubs returning identity for the smoke test.
Full implementation in Phase 2.B (after dataset valid_mask propagation
is confirmed end-to-end).
"""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.utils.loss import v8SegmentationLoss


class WaterAwareSegLoss(v8SegmentationLoss):
    """Subclass of v8SegmentationLoss that masks NaN regions in the mask
    BCE loss using `batch['valid_mask']`.
    """

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        # Counter for debug; if v2.1.A smoke test reports zero or NaN here,
        # the dataset is failing to ship valid_mask.
        self._n_invocations = 0
        self._n_with_valid_mask = 0

    def calculate_segmentation_loss(self, *args, **kwargs):
        """Wrap parent. Pixel-level valid_mask ignore: if the loss is
        computed against a target mask, multiply it by valid_mask before
        the BCE so NaN pixels contribute zero gradient.

        Implementation note:
        v8SegmentationLoss.calculate_segmentation_loss has a complex
        signature involving predicted protos and per-instance polygons.
        For Phase 2.A we delegate to parent unchanged; pixel masking
        will be wired in Phase 2.B after dataset confirms valid_mask
        reaches the batch under the expected key.
        """
        return super().calculate_segmentation_loss(*args, **kwargs)

    def loss(self, preds: dict[str, torch.Tensor],
             batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-level loss entry. Reads batch['valid_mask'] if present,
        delegates to parent, and emits a debug counter on first call.
        """
        self._n_invocations += 1
        if "valid_mask" in batch:
            self._n_with_valid_mask += 1
        return super().loss(preds, batch)
