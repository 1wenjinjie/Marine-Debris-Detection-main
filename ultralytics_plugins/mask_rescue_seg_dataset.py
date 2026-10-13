"""Segmentation variant of MaskRescueNPYDataset.

Loads YOLO seg polygon labels (`<cls> x1 y1 x2 y2 ... xn yn` normalized) and
synthesizes bboxes from polygon extents. Reuses MaskRescueNPYDataset for npy
image loading, p01/p99 normalization, letterbox, etc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np

from .mask_rescue_dataset import MaskRescueNPYDataset


def _read_yolo_seg_txt(path: Path):
    """Parse YOLO seg label file.

    Returns: (classes_2d, bboxes_xywh_norm, segments_list, cls_id_list)
    - classes_2d: (N, 1) float32
    - bboxes_xywh_norm: (N, 4) float32
    - segments_list: list of (M_i, 2) float32 arrays, normalized
    - cls_id_list: list of int
    """
    path = Path(path)
    if not path.exists():
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            [],
            [],
        )
    cls_rows: list[list[float]] = []
    bboxes: list[list[float]] = []
    segments: list[np.ndarray] = []
    cls_ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        # YOLO seg requires >=3 polygon vertices: cls + 6 numbers (3 xy pairs)
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        if coords.shape[0] < 3:
            continue
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        bw = x_max - x_min
        bh = y_max - y_min
        cls_rows.append([float(cls)])
        bboxes.append([float(cx), float(cy), float(bw), float(bh)])
        segments.append(coords)
        cls_ids.append(cls)
    if not cls_rows:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            [],
            [],
        )
    return (
        np.asarray(cls_rows, dtype=np.float32),
        np.asarray(bboxes, dtype=np.float32),
        segments,
        cls_ids,
    )


class MaskRescueSegNPYDataset(MaskRescueNPYDataset):
    """Same as MaskRescueNPYDataset but reads polygon labels into segments."""

    def get_labels(self) -> List[dict[str, Any]]:
        from ultralytics.utils import LOGGER

        self.label_files = [str(self._label_path(Path(p))) for p in self.im_files]
        labels: list[dict[str, Any]] = []
        num_classes = int(self.data.get("nc", len(self.data.get("names", [])))) if hasattr(self, "data") else 0

        for im_path_str, lb_path_str in zip(self.im_files, self.label_files):
            im_path = Path(im_path_str)
            try:
                arr = self._load_array(im_path)
            except Exception as exc:
                LOGGER.warning(f"{self.prefix}Skipping {im_path}: {exc}")
                continue

            h, w = arr.shape[:2]
            lb_path = Path(lb_path_str)
            if not lb_path.is_absolute():
                lb_path = (self.dataset_root / lb_path).resolve()

            cls_arr, boxes_arr, segments_list, cls_ids = _read_yolo_seg_txt(lb_path)

            labels.append({
                "im_file": str(im_path),
                "shape": (h, w),
                "cls": cls_arr,
                "bboxes": boxes_arr,
                "segments": segments_list,
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
            })
            if num_classes > 0:
                hist = np.bincount(cls_ids, minlength=num_classes).astype(np.float32)
            else:
                hist = np.zeros(0, dtype=np.float32)
            self.sample_class_hist[str(im_path.resolve())] = hist

        if not labels:
            raise RuntimeError("No valid .npy samples could be loaded for MaskRescueSegNPYDataset")
        return labels
