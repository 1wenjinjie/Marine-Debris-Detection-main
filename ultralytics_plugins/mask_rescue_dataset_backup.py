"""Ultralytics-compatible dataset for MARIDA mask rescue 7-channel .npy inputs."""
from __future__ import annotations

import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rasterio # Added for TIFF support
from ultralytics.data.augment import Compose, CopyPaste, Format, Instances, LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

from datasets.multispectral_yolo_dataset import _read_yolo_txt


DEFAULT_CHANNEL_STATS = {
    5: Path("marida_runs/channel_stats_5ch.json"),
    6: Path("marida_runs/channel_stats_6ch.json"),
    7: Path("marida_runs/channel_stats_7ch.json"),
    13: Path("marida_runs/channel_stats_13ch.json"),
}


def _bbox_ioa_np(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Compute intersection-over-area (IoA) of box1 over box2 using numpy."""
    if box1.size == 0 or box2.size == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)
    box1 = box1.astype(np.float32)
    box2 = box2.astype(np.float32)
    inter = (
        np.minimum(box1[:, None, 2], box2[:, 2])
        - np.maximum(box1[:, None, 0], box2[:, 0])
    ).clip(min=0) * (
        np.minimum(box1[:, None, 3], box2[:, 3])
        - np.maximum(box1[:, None, 1], box2[:, 1])
    ).clip(min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) + 1e-7
    return inter / area1[:, None]


class ClassAwareCopyPaste(CopyPaste):
    """Copy-paste that only pastes instances from allowed class ids."""

    def __init__(self, p: float = 0.5, allowed_classes: Optional[List[int]] = None) -> None:
        super().__init__(p=p)
        self.allowed_classes = {int(c) for c in allowed_classes} if allowed_classes else set()

    def __call__(self, labels):
        # Fall back to base behavior when no filter is set
        if not self.allowed_classes:
            return super().__call__(labels)

        im = labels["img"]
        cls = labels.get("cls")
        if cls is None or len(cls) == 0:
            return labels

        h, w = im.shape[:2]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)

        if self.p and len(instances.segments):
            _, w_img, _ = im.shape
            im_new = np.zeros(im.shape, np.uint8)

            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w_img)

            ioa = _bbox_ioa_np(ins_flip.bboxes, instances.bboxes)
            candidates = []
            for idx in range(len(instances)):
                if int(cls[idx]) not in self.allowed_classes:
                    continue
                if (ioa[idx] < 0.30).all():
                    candidates.append(idx)

            if candidates:
                k = max(1, round(self.p * len(candidates))) if self.p > 0 else 0
                k = min(k, len(candidates))
                for j in random.sample(candidates, k=k):
                    cls = np.concatenate((cls, cls[[j]]), axis=0)
                    instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                    cv2.drawContours(im_new, instances.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

                result = cv2.flip(im, 1)
                mask = cv2.flip(im_new, 1).astype(bool)
                im[mask] = result[mask]

        labels["img"] = im
        labels["cls"] = cls
        labels["instances"] = instances
        return labels


class MaskRescueNPYDataset(YOLODataset):
    """Dataset that consumes manifest-driven 7-channel `.npy` tensors."""

    ARRAY_EXTENSIONS = {".npy", ".npz", ".tif", ".tiff"}

    def __init__(
        self,
        *args,
        dataset_root: Path | str = Path("mask_bbox_rescue_split"),
        channels: int = 3,
        channel_stats: str | Path | None = None,
        knowledge_summary: str | Path | None = None,
        **kwargs,
    ):
        print("DEBUG: Initializing MaskRescueNPYDataset (Modified)")
        self.dataset_root = Path(dataset_root).resolve()
        self.channels = channels
        self.nan_fix_samples = 0
        self.nan_fix_values = 0
        self._range_low = None
        self._range_high = None
        self._range_scale = None
        if knowledge_summary:
            ks_path = Path(knowledge_summary)
            if not ks_path.is_absolute():
                ks_path = (self.dataset_root / ks_path).resolve()
            self.knowledge_summary_path = ks_path
        else:
            self.knowledge_summary_path = (self.dataset_root / "knowledge_generation_summary.json")
        self.sample_metadata: Dict[str, dict] = {}
        self.sample_metadata_by_name: Dict[str, dict] = {}
        self.sample_weights: Optional[List[float]] = None
        self.sample_weight_stats: Optional[dict] = None
        self.sample_weight_examples: Optional[dict] = None
        self.sample_class_hist: Dict[str, np.ndarray] = {}
        self.class_balance_weights: Optional[np.ndarray] = None
        self.class_balance_power: float = float(os.environ.get("MASKRESCUE_CLASS_BALANCE_POWER", 1.0))
        self.class_balance_floor: float = float(os.environ.get("MASKRESCUE_CLASS_BALANCE_FLOOR", 0.25))
        self.class_balance_ceiling: float = float(os.environ.get("MASKRESCUE_CLASS_BALANCE_CEILING", 3.0))
        self._class_balance_path = kwargs.pop("class_balance_path", None)
        if self._class_balance_path is None:
            candidate = self.dataset_root / "class_balance_weights.json"
            if candidate.exists():
                self._class_balance_path = candidate
        stats_path = channel_stats or DEFAULT_CHANNEL_STATS.get(channels)
        if stats_path:
            stats_path = Path(stats_path)
            if stats_path.exists():
                try:
                    with stats_path.open("r", encoding="utf-8") as f:
                        stats_payload = json.load(f)
                    p01 = stats_payload.get("percentiles", {}).get("p01")
                    p99 = stats_payload.get("percentiles", {}).get("p99")
                    if p01 and p99 and len(p01) == len(p99) == channels:
                        low = np.array(p01, dtype=np.float32).reshape(1, 1, -1)
                        high = np.array(p99, dtype=np.float32).reshape(1, 1, -1)
                        span = np.maximum(high - low, 1e-6)
                        scale = 255.0 / span
                        self._range_low = low
                        self._range_high = high
                        self._range_scale = scale
                        LOGGER.info(
                            f"Loaded channel stats from {stats_path} (global clip [{low.flatten().min():.4f}, {high.flatten().max():.4f}])"
                        )
                    else:
                        LOGGER.warning(
                            f"Channel stats in {stats_path} are invalid for {channels} channels; falling back to default scaling."
                        )
                except Exception as exc:
                    LOGGER.warning(
                        f"Failed to read channel stats at {stats_path}: {exc}. Falling back to default scaling."
                    )
            else:
                LOGGER.warning(f"Channel stats file {stats_path} not found; using default scaling.")

        data_cfg = kwargs.get('data') if isinstance(kwargs.get('data'), dict) else None
        declared_channels = data_cfg.get('channels') if data_cfg else None
        if declared_channels is not None and declared_channels != channels:
            raise ValueError(
                f"MaskRescueNPYDataset requested {channels} channels but YAML declares {declared_channels}."
            )
        self.declared_channels = declared_channels

        self.use_segments = kwargs.pop('use_segments', False)
        self.use_keypoints = kwargs.pop('use_keypoints', False)

        super().__init__(*args, **kwargs)
        if not hasattr(self, "labels"):
            self.labels = self.get_labels()
        if self._class_balance_path:
            self._load_class_balance(self._class_balance_path)
        self._load_knowledge_summary()
        self._align_sample_weights()
        LOGGER.info(
            f"{self.prefix}MaskRescueNPYDataset active with channels={self.channels}" +
            (f" (YAML declared {self.declared_channels})" if self.declared_channels is not None else "")
        )
        print(f"DEBUG: Transforms list: {self.transforms.transforms}")

    def get_img_files(self, img_path):
        """Resolve Ultralytics manifest or directory inputs into .npy/.tif tensors."""
        path = Path(img_path)
        if not path.is_absolute():
            path = (self.dataset_root / path).resolve()

        if path.is_file() and path.suffix.lower() == ".txt":
            entries: List[str] = []
            missing: List[str] = []
            for raw in path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = (path.parent / candidate).resolve()
                if not candidate.exists():
                    alt = (self.dataset_root / raw).resolve()
                    if alt.exists():
                        candidate = alt
                    else:
                        missing.append(raw)
                        continue
                entries.append(str(candidate))

            if missing:
                sample = ", ".join(missing[:3])
                LOGGER.warning(
                    f"{getattr(self, 'prefix', '')}Manifest {path} missing {len(missing)} entries (e.g., {sample})"
                )
            if not entries:
                raise FileNotFoundError(f"{path}: manifest resolved to 0 tensors")
            return entries

        if path.is_dir():
            preferred = {".npy", ".npz"}
            files = [
                str(p)
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in preferred
            ]
            if not files:
                files = [
                    str(p)
                    for p in sorted(path.rglob("*"))
                    if p.is_file() and p.suffix.lower() in self.ARRAY_EXTENSIONS
                ]
            if not files:
                raise FileNotFoundError(f"{path}: directory contains no supported tensors {self.ARRAY_EXTENSIONS}")
            return files

        if path.is_file() and path.suffix.lower() in self.ARRAY_EXTENSIONS:
            return [str(path)]

        return super().get_img_files(str(path))

    def get_labels(self) -> List[dict[str, Any]]:
        self.label_files = [str(self._label_path(Path(p))) for p in self.im_files]
        labels: List[dict[str, Any]] = []
        valid = 0
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
            entries = _read_yolo_txt(lb_path)
            if entries:
                label_arr = np.asarray(entries, dtype=np.float32)
                cls = label_arr[:, :1]
                boxes = label_arr[:, 1:5]
                cls_ids = label_arr[:, 0].astype(int).tolist()
            else:
                cls = np.zeros((0, 1), dtype=np.float32)
                boxes = np.zeros((0, 4), dtype=np.float32)
                cls_ids = []

            labels.append(
                {
                    "im_file": str(im_path),
                    "shape": (h, w),
                    "cls": cls,
                    "bboxes": boxes,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
            valid += 1
            if num_classes > 0:
                hist = np.bincount(cls_ids, minlength=num_classes).astype(np.float32)
            else:
                hist = np.zeros(0, dtype=np.float32)
            self.sample_class_hist[str(im_path.resolve())] = hist

        if not labels:
            raise RuntimeError("No valid .npy samples could be loaded for MaskRescueNPYDataset")
        return labels

    def load_image(self, i: int, rect_mode: bool = True):
        im = self.ims[i]
        path = Path(self.im_files[i])
        if im is None:
            im = self._load_array(path)
            h0, w0 = im.shape[:2]
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            if im.ndim == 2:
                im = im[..., None]

            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def build_transforms(self, hyp=None):
         env_cp = os.environ.get("MASKRESCUE_COPY_PASTE_CLASSES")
         env_cp_list = None
         if env_cp:
             env_cp_list = [int(x) for x in env_cp.split(",") if x.strip()]
 
         cp_classes = getattr(hyp, "copy_paste_classes", None)
         if cp_classes is None:
             cp_classes = env_cp_list
 
         #  DEBUG: ��ϸ�����Ϣ
         hyp_copy_paste = getattr(hyp, "copy_paste", 0.0)
         print(f"[DEBUG build_transforms]")
         print(f"  env_cp (raw): {env_cp}")
         print(f"  env_cp_list: {env_cp_list}")
         print(f"  hyp.copy_paste_classes: {getattr(hyp, 'copy_paste_classes', None)}")
         print(f"  hyp.copy_paste: {hyp_copy_paste}")
         print(f"  cp_classes (final): {cp_classes}")
         print(f"  bool(cp_classes): {bool(cp_classes)}")
         print(f"  hyp.copy_paste > 0: {hyp_copy_paste > 0.0}")
         
         use_classaware_cp = bool(cp_classes) and hyp_copy_paste > 0.0
         print(f"   use_classaware_cp: {use_classaware_cp}")
        parsed_cp_classes = None
        if use_classaware_cp:
            if isinstance(cp_classes, str):
                parsed_cp_classes = [int(x) for x in cp_classes.split(",") if x.strip()]
            elif isinstance(cp_classes, (list, tuple, set)):
                parsed_cp_classes = [int(x) for x in cp_classes]

        if use_classaware_cp:
            # Reuse the standard Ultralytics pipeline, swapping CopyPaste with our class-aware variant.
            transforms = super().build_transforms(hyp)
            patched = []
            for op in transforms.transforms:
                if isinstance(op, CopyPaste):
                    patched.append(ClassAwareCopyPaste(p=op.p, allowed_classes=parsed_cp_classes))
                else:
                    patched.append(op)
            return Compose(patched)

        # Default lightweight pipeline (no mosaic/mixup/copy-paste)
        ops = [
            LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=self.augment and not self.rect),
            Format(
                bbox_format="xywh",
                normalize=False,  # Data is already 0-1 float32 from _load_array
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                bgr=0.0,
            ),
            self._normalize_batch_bboxes,  # Normalize bboxes AFTER Format because Format(normalize=False) returns pixels
        ]
        return Compose(ops)

    def _normalize_batch_bboxes(self, batch):
        """Normalizes bounding boxes in the formatted batch from pixels to 0-1."""
        if 'bboxes' in batch and len(batch['bboxes']) > 0:
            # batch['img'] is (C, H, W)
            h, w = batch['img'].shape[1:]
            if batch['bboxes'].max() > 1.0:
                batch['bboxes'][:, [0, 2]] /= w
                batch['bboxes'][:, [1, 3]] /= h
        return batch

    def _normalize_bboxes(self, labels):
        # Deprecated
        return labels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_array(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == '.tif':
            with rasterio.open(path) as src:
                arr = src.read() # (C, H, W)
                arr = np.transpose(arr, (1, 2, 0)) # (H, W, C)
        else:
            arr = np.load(path, allow_pickle=False)
        
        if arr.ndim != 3:
            raise ValueError(f"Array at {path} must be 3-D, got {arr.shape}")
        if arr.shape[0] in (6, 7) and arr.shape[0] <= arr.shape[-1]:
            arr = arr.transpose(1, 2, 0)
        
        # Slice channels if needed
        if arr.shape[-1] > self.channels:
            # CRITICAL FIX: If input is 11-band Sentinel-2, select specific bands
            if arr.shape[-1] >= 11:
                if self.channels == 6:
                    # Red(B4), Green(B3), Blue(B2), NIR(B8), SWIR1(B11), SWIR2(B12)
                    # Indices: 3, 2, 1, 7, 9, 10
                    arr = arr[..., [3, 2, 1, 7, 9, 10]]
                elif self.channels == 3:
                    # RGB = Red(B4), Green(B3), Blue(B2)
                    # Indices: 3, 2, 1
                    arr = arr[..., [3, 2, 1]]
                else:
                    arr = arr[..., :self.channels]
            else:
                # Default behavior (first N channels)
                arr = arr[..., :self.channels]
        
        if self._range_low is not None and arr.shape[2] == self._range_low.shape[2]:
            arr = np.clip(arr, self._range_low, self._range_high)
            arr = (arr - self._range_low) * self._range_scale
        else:
            arr = arr * 1500.0
            arr = np.clip(arr, 0.0, 255.0)
        # ------------------------------------

        finite_mask = np.isfinite(arr)
        if not finite_mask.all():
            invalid = np.count_nonzero(~finite_mask)
            self.nan_fix_samples += 1
            self.nan_fix_values += invalid
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
            np.clip(arr, 0.0, 255.0, out=arr)
            if self.nan_fix_samples <= 3:
                LOGGER.warning(
                    f"{self.prefix}Replaced {invalid} invalid pixel values in {path.name}"
                )
        if arr.shape[2] != self.channels:
            raise ValueError(f"Array at {path} expected {self.channels} channels, got {arr.shape[2]}")
        return arr.astype(np.float32)

    def _label_path(self, img_path: Path) -> Path:
        parts = list(img_path.parts)
        if "images" in parts:
            idx = parts.index("images")
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
        return img_path.with_suffix(".txt")

    # ------------------------------------------------------------------
    # Knowledge summary helpers
    # ------------------------------------------------------------------
    def _load_knowledge_summary(self) -> None:
        path = self.knowledge_summary_path
        if not path or not path.exists():
            LOGGER.info(f"{self.prefix}Knowledge summary not found at {path}; confidence weighting disabled.")
            return
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            LOGGER.warning(f"{self.prefix}Failed to parse knowledge summary {path}: {exc}")
            return

        details = payload.get("details") or []
        mapping: Dict[str, dict] = {}
        name_mapping: Dict[str, dict] = {}
        for entry in details:
            rel = entry.get("relative_path")
            if not rel:
                continue
            key = Path(rel).as_posix()
            mapping[key] = entry
            name_mapping[Path(rel).name] = entry

        if not mapping:
            LOGGER.warning(f"{self.prefix}Knowledge summary {path} contained no usable entries.")
            return

        self.sample_metadata = mapping
        self.sample_metadata_by_name = name_mapping
        LOGGER.info(f"{self.prefix}Loaded knowledge summary with {len(mapping)} entries from {path}")

    def _relative_key(self, path: Path) -> str:
        path = Path(path).resolve()
        try:
            rel = path.relative_to(self.dataset_root)
        except ValueError:
            rel = Path(path.name)
        return rel.as_posix()

    def _lookup_metadata(self, path: Path) -> Optional[dict]:
        key = self._relative_key(path)
        if key in self.sample_metadata:
            return self.sample_metadata[key]
        return self.sample_metadata_by_name.get(Path(path).name)

    def _align_sample_weights(self) -> None:
        if not self.sample_metadata:
            return
        weights: List[float] = []
        missing = 0
        for path_str in self.im_files:
            meta = self._lookup_metadata(Path(path_str))
            if meta is None:
                missing += 1
            weight = float(meta.get("confidence_weight", 1.0)) if meta else 1.0
            weight *= self._compute_class_balance_multiplier(Path(path_str))
            weights.append(max(weight, 1e-3))

        self.sample_weights = weights
        if missing:
            LOGGER.warning(
                f"{self.prefix}Missing confidence entries for {missing}/{len(self.im_files)} samples; default weight=1"
            )
        self._summarize_sample_weights(np.array(weights, dtype=np.float32))
        LOGGER.info(f"{self.prefix}Confidence weights prepared for {len(weights)} samples")

    def get_sample_weight(self, index: int) -> float:
        if not self.sample_weights:
            return 1.0
        return float(self.sample_weights[min(max(index, 0), len(self.sample_weights) - 1)])

    def _summarize_sample_weights(self, weight_arr: np.ndarray) -> None:
        if weight_arr.size == 0:
            self.sample_weight_stats = None
            self.sample_weight_examples = None
            return

        stats = {
            "min": float(np.min(weight_arr)),
            "max": float(np.max(weight_arr)),
            "mean": float(np.mean(weight_arr)),
            "std": float(np.std(weight_arr)),
            "median": float(np.median(weight_arr)),
        }
        self.sample_weight_stats = stats

        highlight = min(5, len(weight_arr))
        sorted_idx = np.argsort(weight_arr)
        lowest = [(Path(self.im_files[i]).name, float(weight_arr[i])) for i in sorted_idx[:highlight]]
        highest = [(Path(self.im_files[i]).name, float(weight_arr[i])) for i in sorted_idx[-highlight:][::-1]]
        self.sample_weight_examples = {"lowest": lowest, "highest": highest}

        LOGGER.info(
            f"{self.prefix}Confidence weight stats -> min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, median={stats['median']:.4f}"
        )
        LOGGER.info(
            f"{self.prefix}Lowest weights: {self._format_weight_examples(lowest)} | Highest: {self._format_weight_examples(highest)}"
        )

    @staticmethod
    def _format_weight_examples(pairs: List[Tuple[str, float]]) -> str:
        return ", ".join(f"{name}:{weight:.4f}" for name, weight in pairs)

    # ------------------------------------------------------------------
    # Class balance helpers
    # ------------------------------------------------------------------
    def _load_class_balance(self, path: Path | None) -> None:
        if not path:
            return
        path = Path(path)
        if not path.exists():
            LOGGER.info(f"{self.prefix}Class balance file not found at {path}")
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning(f"{self.prefix}Failed to read class balance weights from {path}: {exc}")
            return
        weights = payload.get("recommended_weights") or payload.get("weights")
        if isinstance(weights, dict):
            ordered = []
            for name in self.data.get("names", []):
                ordered.append(float(weights.get(name, 1.0)))
            weights = ordered
        if not isinstance(weights, list):
            LOGGER.warning(f"{self.prefix}Class balance file {path} missing weight list; skipping")
            return
        self.class_balance_weights = np.asarray(weights, dtype=np.float32)
        self.class_balance_power = float(payload.get("recommended_power", self.class_balance_power))
        self.class_balance_floor = float(payload.get("min_multiplier", self.class_balance_floor))
        self.class_balance_ceiling = float(payload.get("max_multiplier", self.class_balance_ceiling))
        LOGGER.info(
            f"{self.prefix}Loaded class balance weights from {path} (power={self.class_balance_power}, clamp=[{self.class_balance_floor}, {self.class_balance_ceiling}])"
        )

    def _compute_class_balance_multiplier(self, path: Path) -> float:
        if self.class_balance_weights is None:
            return 1.0
        key = str(Path(path).resolve())
        hist = self.sample_class_hist.get(key)
        if hist is None or hist.size == 0:
            return 1.0
        active = hist > 0
        if not np.any(active):
            return 1.0
        active_weights = self.class_balance_weights[: hist.size][active]
        if active_weights.size == 0:
            return 1.0
        multiplier = float(np.mean(active_weights))
        multiplier = float(np.clip(multiplier, self.class_balance_floor, self.class_balance_ceiling))
        return float(multiplier ** self.class_balance_power)

