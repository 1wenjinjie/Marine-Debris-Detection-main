"""
EIoU loss trainer for MARIDA v2.

Ultralytics' BboxLoss calls ``ultralytics.utils.loss.bbox_iou`` with
``CIoU=True``. This trainer patches that loss-local symbol so the box loss
uses Efficient IoU (EIoU) while the model architecture and classification
loss remain unchanged.
"""

from __future__ import annotations

import torch
from ultralytics.utils import LOGGER
from ultralytics.utils import loss as loss_module

from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

_ORIGINAL_BBOX_IOU = loss_module.bbox_iou


def bbox_eiou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return EIoU score with the same signature as Ultralytics bbox_iou.

    For plain IoU calls, this preserves the original Ultralytics behavior.
    For loss calls requesting IoU variants (CIoU in BboxLoss), it returns:

        IoU - center_distance/enclosing_diag - width_gap/enclosing_width
            - height_gap/enclosing_height
    """
    if not (GIoU or DIoU or CIoU):
        return _ORIGINAL_BBOX_IOU(
            box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps
        )

    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1h, h1h, w2h, h2h = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1h, x1 + w1h, y1 - h1h, y1 + h1h
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2h, x2 + w2h, y2 - h2h, y2 + h2h
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1 = b1_x2 - b1_x1
        h1 = b1_y2 - b1_y1
        w2 = b2_x2 - b2_x1
        h2 = b2_y2 - b2_y1

    w1 = w1.clamp(min=eps)
    h1 = h1.clamp(min=eps)
    w2 = w2.clamp(min=eps)
    h2 = h2.clamp(min=eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = (b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)).clamp(min=eps)
    ch = (b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)).clamp(min=eps)
    c2 = cw.pow(2) + ch.pow(2) + eps

    b1_cx = (b1_x1 + b1_x2) / 2
    b1_cy = (b1_y1 + b1_y2) / 2
    b2_cx = (b2_x1 + b2_x2) / 2
    b2_cy = (b2_y1 + b2_y2) / 2
    rho2 = (b2_cx - b1_cx).pow(2) + (b2_cy - b1_cy).pow(2)

    width_cost = (w2 - w1).pow(2) / (cw.pow(2) + eps)
    height_cost = (h2 - h1).pow(2) / (ch.pow(2) + eps)
    return iou - rho2 / c2 - width_cost - height_cost


def install_eiou_patch() -> None:
    """Patch Ultralytics loss-local bbox_iou symbol to EIoU."""
    if getattr(loss_module.bbox_iou, "_marida_eiou", False):
        return
    bbox_eiou._marida_eiou = True
    loss_module.bbox_iou = bbox_eiou
    LOGGER.info("Patched box loss: CIoU -> EIoU (ultralytics.utils.loss.bbox_iou)")


class EIoUTrainer(MaskRescueTrainer):
    """MaskRescueTrainer with EIoU box loss."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_callback("on_train_start", EIoUTrainer._patch_eiou)

    def _patch_eiou(self, trainer=None):
        install_eiou_patch()
