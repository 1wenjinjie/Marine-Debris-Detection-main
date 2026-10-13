"""
MaskRescueSegDataset_v2_1 — instance segmentation dataset for MARIDA v2.1.

Conforms to v2_1_DATASET_SPEC.md. Independent from v2.0 plugin.

Key behaviours:
  - Loads 9-channel float32 .npy tiles from data/processed/marida_v2_1/<split>/images/
  - Loads YOLO-format polygon labels from labels_highmod/ (default) or
    labels_highonly/ / labels_all/ (ablation runs)
  - Applies p01/p99 normalization from train-water stats
  - Optional train-time dilation 3x3 on instance masks before re-deriving
    polygons (SPEC §19)
  - Streams valid_mask alongside each sample for downstream WaterAwareSegLoss
  - Forces task='segment'; expects polygon labels of format
        cls x1 y1 x2 y2 ...  (normalised [0, 1])

NOTE: Phase 2.A — minimal version. NaN handling, weighted sampling,
and per-channel fallback stats are TODO for Phase 2.B+.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import ultralytics.data.utils as _udata_utils
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER


# Make ultralytics accept .npy as an image extension AND verify .npy
# files without delegating to PIL.
#
# Two patches needed:
#  (1) IMG_FORMATS frozenset must contain "npy" in every module that
#      imported it (base.py, build.py, loaders.py, split.py,
#      converter.py, utils.py).
#  (2) verify_image_label() must skip PIL for .npy files and use numpy
#      to read shape directly. This is rebound on the dataset.py
#      module (and others) since they imported by value.
def _patch_img_formats_once() -> None:
    flag = "_v2_1_img_formats_patched"
    if getattr(_udata_utils, flag, False):
        return
    new_formats = frozenset(set(_udata_utils.IMG_FORMATS) | {"npy"})
    _udata_utils.IMG_FORMATS = new_formats
    try:
        from ultralytics.data.utils import VID_FORMATS
        _udata_utils.FORMATS_HELP_MSG = (
            f"Supported formats are:\nimages: {new_formats}\nvideos: {VID_FORMATS}"
        )
    except Exception:
        pass

    # Save original verify_image_label for delegation
    _orig_verify = _udata_utils.verify_image_label

    def _v2_1_verify_image_label(args):
        """NPY-aware verify_image_label.

        For .npy files: skip PIL verification, read shape via numpy.
        For any other extension: delegate to the original function.
        """
        im_file = args[0]
        if not str(im_file).lower().endswith(".npy"):
            return _orig_verify(args)

        # Unpack
        (im_file, lb_file, prefix, keypoint, num_cls,
         nkpt, ndim, single_cls) = args
        import os
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""
        segments, keypoints = [], None
        try:
            arr = np.load(im_file, mmap_mode="r")
            if arr.ndim < 2:
                raise ValueError(f"npy ndim {arr.ndim} < 2")
            shape = (int(arr.shape[0]), int(arr.shape[1]))   # (H, W)
            if not (shape[0] > 9 and shape[1] > 9):
                raise ValueError(f"image size {shape} <10 pixels")

            # ---- label parsing (mirrors original) ----
            from ultralytics.utils.ops import segments2boxes

            if os.path.isfile(lb_file):
                nf = 1
                with open(lb_file, encoding="utf-8") as f:
                    lb_rows = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb_rows) and (not keypoint):
                    classes = np.array([x[0] for x in lb_rows], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2)
                                 for x in lb_rows]
                    lb = np.concatenate(
                        (classes.reshape(-1, 1), segments2boxes(segments)), 1)
                else:
                    lb = np.array(lb_rows, dtype=np.float32) if lb_rows else \
                          np.zeros((0, 5), dtype=np.float32)
                if (nl := len(lb)):
                    if keypoint:
                        assert lb.shape[1] == (5 + nkpt * ndim), \
                            f"labels require {(5 + nkpt * ndim)} columns each"
                        points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                    else:
                        assert lb.shape[1] == 5, \
                            f"labels require 5 columns, got {lb.shape[1]}"
                        points = lb[:, 1:]
                    if not (points.max() <= 1.01):
                        raise ValueError(
                            f"non-normalized coords: {points[points > 1.01]}")
                    if not (lb.min() >= -0.01):
                        raise ValueError(
                            f"negative coords/labels: {lb[lb < -0.01]}")
                    max_cls = 0 if single_cls else lb[:, 0].max()
                    if not (max_cls < num_cls):
                        raise ValueError(
                            f"label class {int(max_cls)} >= num_cls {num_cls}")
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:
                        lb = lb[i]
                        if segments:
                            segments = [segments[x] for x in i]
                        msg = f"{prefix}{im_file}: {nl - len(i)} duplicate labels removed"
                else:
                    ne = 1
                    lb = np.zeros(
                        (0, (5 + nkpt * ndim) if keypoint else 5),
                        dtype=np.float32)
            else:
                nm = 1
                lb = np.zeros(
                    (0, (5 + nkpt * ndim) if keypoint else 5),
                    dtype=np.float32)

            if keypoint:
                keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
                if ndim == 2:
                    kpt_mask = np.where(
                        (keypoints[..., 0] < 0) | (keypoints[..., 1] < 0),
                        0.0, 1.0).astype(np.float32)
                    keypoints = np.concatenate(
                        [keypoints, kpt_mask[..., None]], axis=-1)
            lb = lb[:, :5]
            return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f"{prefix}{im_file}: ignoring corrupt npy/label: {e}"
            return [None, None, None, None, None, nm, nf, ne, nc, msg]

    # Now rebind both IMG_FORMATS and verify_image_label in every
    # ultralytics.data.* module that imported them.
    import ultralytics.data as _udata
    import importlib
    import pkgutil
    _udata_utils.verify_image_label = _v2_1_verify_image_label
    for _, modname, _ in pkgutil.iter_modules(_udata.__path__):
        full = f"ultralytics.data.{modname}"
        try:
            mod = importlib.import_module(full)
        except Exception:
            continue
        if hasattr(mod, "IMG_FORMATS"):
            mod.IMG_FORMATS = new_formats
        if hasattr(mod, "FORMATS_HELP_MSG"):
            mod.FORMATS_HELP_MSG = _udata_utils.FORMATS_HELP_MSG
        if hasattr(mod, "verify_image_label"):
            mod.verify_image_label = _v2_1_verify_image_label
    setattr(_udata_utils, flag, True)
    LOGGER.info("v2.1 plugin: patched IMG_FORMATS + verify_image_label "
                "across ultralytics.data.* for .npy support")


class MaskRescueSegDataset_v2_1(YOLODataset):
    """v2.1 instance-segmentation dataset for 9-channel MARIDA tiles."""

    def __init__(
        self,
        *args,
        channels: int = 9,
        channel_stats: str | Path | None = None,
        valid_mask_subdir: str = "valid_masks",
        instance_mask_subdir: str = "instance_masks_highmod",
        train_dilate: bool = True,
        dilate_kernel: int = 3,
        **kwargs,
    ):
        _patch_img_formats_once()

        self.channels = int(channels)
        self.valid_mask_subdir = valid_mask_subdir
        self.instance_mask_subdir = instance_mask_subdir
        self.train_dilate = bool(train_dilate)
        self.dilate_kernel = int(dilate_kernel)

        self._p01: np.ndarray | None = None
        self._p99: np.ndarray | None = None
        if channel_stats is not None:
            self._load_stats(Path(channel_stats))

        # Force segmentation task; required by parent YOLODataset internals.
        kwargs.setdefault("task", "segment")
        super().__init__(*args, **kwargs)

        # Whether current split is the train split (for dilation control)
        prefix = (kwargs.get("prefix") or "").strip().lower()
        self._is_train_split = "train" in prefix

        LOGGER.info(
            f"v2.1 SegDataset active: channels={self.channels}  "
            f"label_set_subdir={self.instance_mask_subdir}  "
            f"dilate={self.train_dilate}  (applied to both train and val for "
            f"GT-distribution consistency)"
        )

    # ------------------------------------------------------------------
    # Stats loading + normalization
    # ------------------------------------------------------------------
    def _load_stats(self, stats_path: Path) -> None:
        payload = json.loads(stats_path.read_text())
        p01 = payload["percentiles"]["p01"]
        p99 = payload["percentiles"]["p99"]
        if len(p01) != self.channels or len(p99) != self.channels:
            raise ValueError(
                f"channel_stats has {len(p01)} channels, dataset expects {self.channels}"
            )
        self._p01 = np.array(p01, dtype=np.float32).reshape(1, 1, -1)
        self._p99 = np.array(p99, dtype=np.float32).reshape(1, 1, -1)
        LOGGER.info(f"v2.1 SegDataset loaded stats from {stats_path}")

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Clip to [p01, p99] then linearly map to [0, 255] uint8-like float."""
        if self._p01 is None or self._p99 is None:
            # No stats — pass-through with NaN -> 0, clipped to [0, 1] * 255
            arr = np.nan_to_num(arr, nan=0.0)
            return np.clip(arr * 255.0, 0, 255).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        span = np.maximum(self._p99 - self._p01, 1e-6)
        arr = np.clip(arr, self._p01, self._p99)
        arr = (arr - self._p01) / span * 255.0
        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def load_image(self, i: int, rect_mode: bool = True):
        """Load one .npy as (H, W, C) float32, normalize to ~[0, 255]."""
        path = self.im_files[i]
        arr = np.load(path)
        if arr.ndim != 3 or arr.shape[-1] != self.channels:
            raise ValueError(
                f"Expected (H, W, {self.channels}) at {path}, got {arr.shape}"
            )
        arr = self._normalize(arr)
        h0, w0 = arr.shape[:2]
        # Resize to imgsz only if rect_mode is False — letterbox handles
        # the imgsz transition elsewhere in the pipeline.
        if not rect_mode and self.imgsz != max(h0, w0):
            scale = self.imgsz / max(h0, w0)
            new_h = int(round(h0 * scale))
            new_w = int(round(w0 * scale))
            arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = arr.shape[:2]
        # Cache for ultralytics internal mechanism (mimics base behaviour)
        self.ims[i] = arr
        self.im_hw0[i] = (h0, w0)
        self.im_hw[i] = (h, w)
        return arr, (h0, w0), (h, w)

    # ------------------------------------------------------------------
    # Label loading with optional dilation
    # ------------------------------------------------------------------
    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Returns the standard ultralytics label dict, optionally augmented
        with a valid_mask numpy array (under key `valid_mask`).
        """
        label = super().get_image_and_label(index)
        # Try to load valid_mask
        npy_path = Path(self.im_files[index])
        vm_path = npy_path.parent.parent / self.valid_mask_subdir / npy_path.name
        if vm_path.exists():
            try:
                vm = np.load(vm_path)
                if vm.dtype != bool:
                    vm = vm.astype(bool)
                label["valid_mask"] = vm
            except Exception as exc:
                LOGGER.warning(f"valid_mask load failed for {vm_path}: {exc}")
        # Apply mask dilation to BOTH train and val.
        # Rationale (Phase 3.1 finding 2026-05-XX): applying dilation
        # only to train created a train/val GT-distribution mismatch
        # (val saw raw 1-px GT but the model learned dilated 3x3 GT),
        # causing val/seg_loss to explode to 1e6 by epoch 6.
        # The fix: treat dilation as a permanent GT relabeling — both
        # train and val see the same dilated GT. Paper §3 must
        # disclose this, and final test-set evaluation can optionally
        # report a non-dilated raster IoU as a sanity diagnostic.
        if self.train_dilate and "segments" in label:
            label["segments"] = _dilate_polygons_inplace(
                label["segments"],
                kernel=self.dilate_kernel,
                imgsz=max(label.get("ori_shape", (self.imgsz, self.imgsz))),
            )
        return label


def _dilate_polygons_inplace(segments: list[np.ndarray], kernel: int,
                              imgsz: int) -> list[np.ndarray]:
    """Outward-pad each polygon by `kernel // 2` pixels in normalised
    coordinate space. This is a polygon-domain proxy for raster mask
    dilation by a kernel×kernel structuring element.

    segments: list of np.ndarray of shape (n_vert, 2) in normalised [0,1].
    """
    if not segments or kernel <= 1:
        return segments
    pad = (kernel - 1) / 2.0 / max(imgsz, 1)
    out = []
    for seg in segments:
        if seg.size == 0:
            out.append(seg)
            continue
        centroid = seg.mean(axis=0, keepdims=True)
        direction = seg - centroid
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-6)
        unit = direction / norm
        padded = seg + unit * pad
        # clamp to [0, 1]
        padded = np.clip(padded, 0.0, 1.0)
        out.append(padded.astype(seg.dtype))
    return out
