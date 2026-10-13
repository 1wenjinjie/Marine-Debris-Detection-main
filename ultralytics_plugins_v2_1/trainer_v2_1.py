"""
MaskRescueSegTrainer_v2_1 — wraps ultralytics SegmentationTrainer for
v2.1: 9-channel input, COCO weight inflation, WaterAwareSegLoss.

Conforms to v2_1_DATASET_SPEC.md §3 + §18.

What it does:
  - On model build: inflates the first conv from 3 → 9 channels using
    `weight_inflate.inflate_first_conv`.
  - Wires `MaskRescueSegDataset_v2_1` as the data class.
  - Swaps the default v8SegmentationLoss for `WaterAwareSegLoss`.
  - (TODO Phase 2.B) Applies differential LR for Xavier-initialised
    channels in the first 100 steps.

Phase 2.A scope: model + dataset + loss class wiring only. Weighted
sampler, differential LR scheduling, and full WaterAwareSegLoss layers
are Phase 2.B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import LOGGER

from .dataset_v2_1 import MaskRescueSegDataset_v2_1
from .loss_v2_1 import WaterAwareSegLoss
from .weight_inflate import inflate_first_conv


class MaskRescueSegTrainer_v2_1(SegmentationTrainer):
    """v2.1 segmentation trainer with multi-channel input + valid-aware loss."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self._v2_1_channels = int(
            (overrides or {}).get("channels",
                                    self.data.get("channels", 9) if isinstance(self.data, dict) else 9)
        )

    # ---------------- model setup ----------------
    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Construct model and inflate first conv to `channels`."""
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        try:
            inflated = inflate_first_conv(model, in_channels=self._v2_1_channels)
            LOGGER.info(
                f"v2.1 trainer: inflated first conv to {inflated.in_channels} channels"
            )
        except Exception as exc:
            LOGGER.error(f"v2.1 trainer: weight inflation FAILED: {exc}")
            raise
        return model

    # ---------------- dataset wiring ----------------
    def build_dataset(self, img_path, mode: str = "train", batch: int | None = None):
        """Use v2.1 dataset class instead of stock YOLODataset."""
        # Replicate parent's gs / hyp resolution
        gs = max(int(self.model.stride.max() if self.model else 0), 32)
        return MaskRescueSegDataset_v2_1(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task="segment",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            channels=self._v2_1_channels,
            channel_stats=(self.data.get("channel_stats")
                            if isinstance(self.data, dict) else None),
            valid_mask_subdir=(self.data.get("valid_mask_subdir", "valid_masks")
                                if isinstance(self.data, dict) else "valid_masks"),
            instance_mask_subdir=(self.data.get("instance_mask_subdir",
                                                  "instance_masks_highmod")
                                   if isinstance(self.data, dict)
                                   else "instance_masks_highmod"),
            train_dilate=(mode == "train"),
        )

    # ---------------- loss wiring ----------------
    def init_criterion(self):
        """Return the v2.1 water-aware seg loss."""
        return WaterAwareSegLoss(self.model)
