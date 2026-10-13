"""
v2.1 plugin package.

ENTIRELY INDEPENDENT from `ultralytics_plugins/` (v2.0). Do not share
state, do not import from it. Both can coexist in the same Python
process without interference.

Side effects on import:
- Registers any custom modules required by v2.1 model YAMLs.
- Does NOT alter v2.0 plugin behaviour.

Public exports:
- MaskRescueSegDataset_v2_1
- WaterAwareSegLoss
- MaskRescueSegTrainer_v2_1
- inflate_first_conv  (COCO 3-channel → 9-channel)
"""

from .register_modules import register_v2_1_modules

# Trigger v2.1-only module registration on package import
register_v2_1_modules()

# Patch ultralytics IMG_FORMATS to include .npy at package-import time,
# BEFORE any downstream module imports ultralytics datasets. This must
# run before users do `from ultralytics import YOLO`.
from .dataset_v2_1 import _patch_img_formats_once as _pifo
_pifo()
del _pifo

# Public API
from .weight_inflate import inflate_first_conv, get_xavier_channel_indices  # noqa: E402

# Lazy import to avoid pulling torch / ultralytics until needed
__all__ = [
    "MaskRescueSegDataset_v2_1",
    "WaterAwareSegLoss",
    "MaskRescueSegTrainer_v2_1",
    "inflate_first_conv",
    "get_xavier_channel_indices",
]


def __getattr__(name):
    if name == "MaskRescueSegDataset_v2_1":
        from .dataset_v2_1 import MaskRescueSegDataset_v2_1
        return MaskRescueSegDataset_v2_1
    if name == "WaterAwareSegLoss":
        from .loss_v2_1 import WaterAwareSegLoss
        return WaterAwareSegLoss
    if name == "MaskRescueSegTrainer_v2_1":
        from .trainer_v2_1 import MaskRescueSegTrainer_v2_1
        return MaskRescueSegTrainer_v2_1
    raise AttributeError(f"module 'ultralytics_plugins_v2_1' has no attribute {name!r}")
