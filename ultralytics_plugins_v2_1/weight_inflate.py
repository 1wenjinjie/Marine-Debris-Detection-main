"""
COCO 3-channel -> 9-channel first-conv weight inflation.

Conforms to v2_1_DATASET_SPEC.md §18.

Strategy per output channel of the first conv:
    new_w[:, 0] = coco_w[:, 0]                 # B4 Red    <- COCO R
    new_w[:, 1] = coco_w[:, 1]                 # B3 Green  <- COCO G
    new_w[:, 2] = coco_w[:, 2]                 # B2 Blue   <- COCO B
    new_w[:, 3] = 0.5 * (coco_w[:, 0] + coco_w[:, 1])
                                                # B6 Red-Edge2 <- mean(R,G)
    new_w[:, 4] = coco_w[:, 0]                 # B8 NIR    <- COCO R (R nearest in wl)
    new_w[:, 5] = Xavier                       # B11 SWIR1
    new_w[:, 6] = Xavier                       # B12 SWIR2
    new_w[:, 7] = Xavier                       # FDI
    new_w[:, 8] = Xavier                       # NDVI

After assignment: multiply the entire inflated tensor by sqrt(3/9)
to keep activation magnitudes after the first conv approximately
matched to the COCO-pretrained norm.

Differential learning rate for the first 100 steps is *not* applied
here; that is the trainer's responsibility (see `trainer_v2_1.py`).

Usage:
    from ultralytics import YOLO
    from ultralytics_plugins_v2_1 import inflate_first_conv

    yolo = YOLO("yolo11n-seg.pt")    # COCO pretrained, 3-channel input
    inflate_first_conv(yolo.model, in_channels=9)
    # ... continue training
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# 0-indexed channel mapping: which COCO channel(s) initialise each new ch.
# None means Xavier.
_DEFAULT_INHERITANCE_9CH: list[Optional[tuple[int, ...]]] = [
    (0,),       # ch 0: B4 Red       <- COCO R
    (1,),       # ch 1: B3 Green     <- COCO G
    (2,),       # ch 2: B2 Blue      <- COCO B
    (0, 1),     # ch 3: B6 RedEdge2  <- mean(R, G)
    (0,),       # ch 4: B8 NIR       <- COCO R (closest wl)
    None,       # ch 5: B11 SWIR1    <- Xavier
    None,       # ch 6: B12 SWIR2    <- Xavier
    None,       # ch 7: FDI          <- Xavier
    None,       # ch 8: NDVI         <- Xavier
]


def _find_first_conv(model: nn.Module) -> nn.Conv2d:
    """Locate the first conv that touches the input image.

    For an ultralytics YOLO segmentation model this is `model.model[0].conv`.
    Walk the module hierarchy and return the first nn.Conv2d encountered.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise RuntimeError("No nn.Conv2d found in the model")


def inflate_first_conv(
    model: nn.Module,
    in_channels: int = 9,
    inheritance: Optional[list[Optional[tuple[int, ...]]]] = None,
    rescale: bool = True,
) -> nn.Conv2d:
    """Inflate the first conv of `model` from 3 in_channels to
    `in_channels` channels, in-place. Returns the (replaced) conv module.

    Args:
        model: an ultralytics YOLO model (`.model`).
        in_channels: target input channel count.
        inheritance: per-output-channel COCO-channel-index lists. None
            entries get Xavier initialization. Defaults to
            _DEFAULT_INHERITANCE_9CH.
        rescale: apply sqrt(3 / in_channels) magnitude correction.
            Default True. Disable for unit tests only.
    """
    if inheritance is None:
        inheritance = _DEFAULT_INHERITANCE_9CH
    if len(inheritance) != in_channels:
        raise ValueError(
            f"inheritance length {len(inheritance)} != in_channels {in_channels}")

    conv = _find_first_conv(model)
    if conv.in_channels == in_channels:
        # Already inflated; treat as no-op
        return conv
    if conv.in_channels != 3:
        raise ValueError(
            f"Expected first conv with 3 in_channels, got {conv.in_channels}"
        )

    out_channels, _, kh, kw = conv.weight.shape
    new_w = torch.zeros(
        (out_channels, in_channels, kh, kw),
        dtype=conv.weight.dtype, device=conv.weight.device,
    )

    coco_w = conv.weight.data
    for new_ch, src in enumerate(inheritance):
        if src is None:
            # Xavier (uniform)
            nn.init.xavier_uniform_(new_w[:, new_ch:new_ch + 1])
        else:
            slabs = [coco_w[:, s] for s in src]
            new_w[:, new_ch] = sum(slabs) / len(slabs)

    if rescale:
        # sqrt(3 / in_channels) keeps the per-output-element variance
        # roughly invariant after summation over the new in-channel axis.
        new_w *= math.sqrt(3.0 / in_channels)

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )
    new_conv.weight.data = new_w
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    new_conv = new_conv.to(device=conv.weight.device, dtype=conv.weight.dtype)

    # Replace conv in parent. Walk the model again, find parent.
    parent_module = None
    target_attr = None
    for mod in model.modules():
        for name, child in mod.named_children():
            if child is conv:
                parent_module = mod
                target_attr = name
                break
        if parent_module is not None:
            break
    if parent_module is None:
        raise RuntimeError("Could not find parent of first conv for replacement")
    setattr(parent_module, target_attr, new_conv)
    return new_conv


def get_xavier_channel_indices(
    inheritance: Optional[list[Optional[tuple[int, ...]]]] = None,
) -> list[int]:
    """Return indices of newly Xavier-initialized input channels.

    Trainer uses this to apply differential lr for the first ~100 steps
    (see v2_1_DATASET_SPEC.md §18).
    """
    if inheritance is None:
        inheritance = _DEFAULT_INHERITANCE_9CH
    return [i for i, s in enumerate(inheritance) if s is None]
