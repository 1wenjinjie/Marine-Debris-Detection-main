"""Run the pre-registered MPB experiment matrix without automatic test access.

The default workflow is deliberately staged:

  train -> validation metrics/operating points only
  lock  -> freeze all checkpoint and threshold decisions
  test  -> one explicit, guarded internal re-analysis

Historical MARIDA test results have already been inspected, so even the guarded
test stage remains retrospective internal validation. External scenes are still
required for a confirmatory claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
MPB_RUNTIME_DIR = ROOT / "data" / "processed" / "marida_mpb_v1" / "_runtime"
MPB_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(MPB_RUNTIME_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.utils.prepare_multichannel_yolo_weights import expand_conv_weight, prepare_weights
from scripts.v2_evaluate import evaluate
from scripts.mpb.protocol_integrity import dataset_fingerprint
from training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer


CONFIG_DIR = ROOT / "configs" / "datasets" / "mpb"
PROTOCOL_ROOT = ROOT / "data" / "processed" / "marida_mpb_v1"
PROTOCOL_LOCK = PROTOCOL_ROOT / "protocol_lock.json"
SOURCE_MANIFEST = PROTOCOL_ROOT / "execution_source_manifest.json"
SOURCE_ARCHIVE = PROTOCOL_ROOT / "execution_source_archive.zip"
ENVIRONMENT_MANIFEST = PROTOCOL_ROOT / "execution_environment.json"
RUNS_DIR = ROOT / "runs_mpb"
INIT_DIR = PROTOCOL_ROOT / "initialization"
SEEDS = (0, 1, 2)
PRIMARY_DEPLOYABLE_CELL = "full_scene_ndwi7"
THRESHOLD_GRID = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15,
    0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99,
)
MATCH_IOU = 0.50
NMS_IOU = 0.60
MAX_DET = 300
MIN_INFERENCE_CONF = 0.001
SMOKE_VERSION = 12
BOOTSTRAP_ITERATIONS = 10000
_CONTENT_HASH_CACHE: dict[tuple[object, ...], str] = {}
_DATASET_FINGERPRINT_CACHE: dict[Path, dict[str, object]] = {}
_INITIALIZATION_CACHE: dict[int, Path] | None = None


class MPBTrainer(MaskRescueTrainer):
    """Keep Ultralytics checkpoint saving independent of optional Polars."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        dataset = super().build_dataset(img_path, mode, batch)
        dataset_root = Path(str(self.data["path"]))
        channels = int(self.data["channels"])
        expected_low, expected_high = _load_stats(dataset_root, channels)
        if (
            dataset._range_low is None
            or dataset._range_high is None
            or not np.array_equal(dataset._range_low, expected_low)
            or not np.array_equal(dataset._range_high, expected_high)
        ):
            raise RuntimeError(
                f"{mode} loaded channel statistics outside the locked dataset root: {dataset_root}"
            )
        return dataset

    def read_results_csv(self) -> dict[str, list[object]]:
        try:
            with Path(self.csv).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            return {}
        if not rows:
            return {}

        columns: dict[str, list[object]] = {name: [] for name in rows[0]}
        for row in rows:
            for name in columns:
                value = row.get(name, "")
                try:
                    parsed: object = float(value) if value not in (None, "") else value
                except ValueError:
                    parsed = value
                columns[name].append(parsed)
        return columns


def _ensure_ultralytics_font() -> Path:
    """Provide Ultralytics an offline font without modifying the Python environment."""
    from ultralytics.utils import USER_CONFIG_DIR

    target = USER_CONFIG_DIR / "Arial.ttf"
    if target.exists() and target.stat().st_size:
        return target

    windows_dir = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates = (
        windows_dir / "Fonts" / "arial.ttf",
        windows_dir / "Fonts" / "Arial.ttf",
    )
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "Arial.ttf is unavailable locally; refusing a network-dependent training run"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


@dataclass(frozen=True)
class Cell:
    name: str
    config: str
    channels: int
    role: str
    deployable: bool


CELLS = {
    "v1_clean_ndwi": Cell(
        "v1_clean_ndwi", "v1_audit_clean_ndwi.yaml", 7,
        "2x2 audit: scene-disjoint x deployable NDWI", True,
    ),
    "v1_clean_oracle": Cell(
        "v1_clean_oracle", "v1_audit_clean_oracle.yaml", 7,
        "2x2 audit: scene-disjoint x GT-derived oracle", False,
    ),
    "v1_shared_ndwi": Cell(
        "v1_shared_ndwi", "v1_audit_shared_ndwi.yaml", 7,
        "2x2 audit: shared-scene sibling exposure x deployable NDWI", False,
    ),
    "v1_shared_oracle": Cell(
        "v1_shared_oracle", "v1_audit_shared_oracle.yaml", 7,
        "2x2 audit: shared-scene sibling exposure x GT-derived oracle", False,
    ),
    "full_scene_6ch": Cell(
        "full_scene_6ch", "scene_6ch_min4.yaml", 6,
        "Full-universe six-channel control", True,
    ),
    "full_scene_ndwi7": Cell(
        "full_scene_ndwi7", "scene_7ch_ndwi_min4.yaml", 7,
        "Full-universe deployable NDWI seven-channel model", True,
    ),
    "full_scene_shuffled7": Cell(
        "full_scene_shuffled7", "scene_7ch_ndwi_shuffled_min4.yaml", 7,
        "Full-universe spatially shuffled NDWI negative control", False,
    ),
    **{
        f"min_train{value}": Cell(
            f"min_train{value}",
            f"scene_6ch_minsens_train{value}_eval4.yaml",
            6,
            f"MIN_PIXELS sensitivity: train MIN={value}, fixed val/test MIN=4",
            False,
        )
        for value in (1, 2, 8)
    },
    **{
        f"min_matched{value}": Cell(
            f"min_matched{value}",
            f"scene_6ch_min{value}.yaml",
            6,
            f"Supplementary target-definition sensitivity: train/val/test MIN={value}",
            False,
        )
        for value in (1, 2, 8)
    },
}
SUITES = {
    "factorial": ("v1_clean_ndwi", "v1_clean_oracle", "v1_shared_ndwi", "v1_shared_oracle"),
    "channels": ("full_scene_6ch", "full_scene_ndwi7", "full_scene_shuffled7"),
    "min": ("min_train1", "min_train2", "full_scene_6ch", "min_train8"),
    "min_matched": ("min_matched1", "min_matched2", "full_scene_6ch", "min_matched8"),
}
SUITES["all"] = tuple(
    dict.fromkeys(
        (*SUITES["factorial"], *SUITES["channels"], *SUITES["min"], *SUITES["min_matched"])
    )
)


LOCKED_HYPERPARAMS = {
    **MASK_RESCUE_TRAIN_DEFAULTS,
    "epochs": 50,
    "batch": 8,
    "imgsz": 512,
    "optimizer": "AdamW",
    "lr0": 0.0005,
    "cos_lr": True,
    "patience": 20,
    "deterministic": True,
    "pretrained": True,
    "val": True,
    "plots": False,
    "cache": False,
    "amp": True,
}


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_contract() -> dict[str, object]:
    """Fingerprint every local implementation file and the numerical runtime."""
    import ultralytics

    relative_paths = (
        "scripts/mpb/prepare_protocol.py",
        "scripts/mpb/run_experiments.py",
        "scripts/mpb/protocol_integrity.py",
        "scripts/v2_evaluate.py",
        "scripts/utils/prepare_multichannel_yolo_weights.py",
        "scripts/training/mask_rescue_hyperparams.py",
        "ultralytics_plugins/__init__.py",
        "ultralytics_plugins/register_extra_modules.py",
        "ultralytics_plugins/mask_rescue_trainer.py",
        "ultralytics_plugins/mask_rescue_dataset.py",
    )
    files = {}
    for relative_path in relative_paths:
        path = ROOT / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Code-contract file is missing: {path}")
        files[relative_path] = _sha256(path)

    if not SOURCE_MANIFEST.exists() or not SOURCE_ARCHIVE.exists() or not ENVIRONMENT_MANIFEST.exists():
        raise FileNotFoundError("Execution source/environment snapshot is incomplete; rebuild the protocol")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source_manifest.get("stage") != "build":
        raise RuntimeError("Formal execution requires a build-stage source snapshot")
    if source_manifest.get("source_file_sha256") != files:
        raise RuntimeError("Current implementation differs from the locked execution source snapshot")
    for relative_path, expected_hash in source_manifest.get("dataset_config_sha256", {}).items():
        config_path = ROOT / str(relative_path)
        if not config_path.exists() or _sha256(config_path) != expected_hash:
            raise RuntimeError(f"Generated dataset config differs from source snapshot: {config_path}")
    if source_manifest.get("environment_manifest_sha256") != _sha256(ENVIRONMENT_MANIFEST):
        raise RuntimeError("Execution environment manifest differs from the source snapshot")

    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }
    frozen_runtime = json.loads(ENVIRONMENT_MANIFEST.read_text(encoding="utf-8")).get(
        "runtime_versions", {}
    )
    if any(frozen_runtime.get(name) != version for name, version in runtime.items()):
        raise RuntimeError(
            f"Numerical runtime differs from the locked environment: current={runtime}, "
            f"locked={frozen_runtime}"
        )

    payload: dict[str, object] = {
        "files": files,
        "runtime": runtime,
        "execution_snapshot": {
            "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
            "source_archive_sha256": _sha256(SOURCE_ARCHIVE),
            "environment_manifest_sha256": _sha256(ENVIRONMENT_MANIFEST),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _protocol() -> dict[str, object]:
    if not PROTOCOL_LOCK.exists():
        raise FileNotFoundError(f"Build the MPB protocol first: {PROTOCOL_LOCK}")
    payload = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    if payload.get("protocol") != "marida_mpb_v1":
        raise RuntimeError("Unexpected protocol lock")
    builder = ROOT / str(payload.get("builder", ""))
    if not builder.exists() or payload.get("builder_sha256") != _sha256(builder):
        raise RuntimeError("Protocol builder changed after the protocol lock was written")
    for relative_path, expected_hash in payload.get("artifact_hashes", {}).items():
        artifact = ROOT / str(relative_path)
        if not artifact.exists() or _sha256(artifact) != expected_hash:
            raise RuntimeError(f"Locked protocol artifact changed: {artifact}")
    return payload


def _variant_contract(cell: Cell) -> dict[str, object]:
    expected_name = Path(cell.config).stem
    matches = [
        variant
        for variant in _protocol().get("variants", [])
        if variant.get("name") == expected_name
    ]
    if len(matches) != 1 or "artifact_fingerprint" not in matches[0]:
        raise RuntimeError(f"Missing unique locked artifact fingerprint for {cell.config}")
    return matches[0]


def _verified_dataset(cell: Cell, refresh: bool = False) -> dict[str, object]:
    config_path = _config_path(cell).resolve()
    if refresh or config_path not in _DATASET_FINGERPRINT_CACHE:
        _DATASET_FINGERPRINT_CACHE[config_path] = dataset_fingerprint(
            config_path, _CONTENT_HASH_CACHE
        )
    actual = _DATASET_FINGERPRINT_CACHE[config_path]
    expected = _variant_contract(cell)["artifact_fingerprint"]
    if actual["sha256"] != expected["sha256"]:
        raise RuntimeError(
            f"Dataset artifacts changed after protocol lock for {cell.name}: "
            f"expected {expected['sha256']}, found {actual['sha256']}"
        )
    return actual


def _ensure_initialization() -> dict[int, Path]:
    global _INITIALIZATION_CACHE
    if _INITIALIZATION_CACHE is not None:
        return _INITIALIZATION_CACHE

    from ultralytics import YOLO

    source = ROOT / "yolo11n.pt"
    if not source.exists():
        raise FileNotFoundError(source)
    INIT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        channels: INIT_DIR / f"yolo11n_{channels}ch_coco_mean.pt"
        for channels in (6, 7)
    }
    for channels, output in outputs.items():
        if not output.exists():
            prepare_weights(source, output, channels=channels, mode="mean")

    source_state = YOLO(str(source)).model.state_dict()
    first_conv_key = next(
        name for name, value in source_state.items()
        if name.endswith("model.0.conv.weight") and value.ndim == 4
    )
    expanded_states: dict[int, dict[str, torch.Tensor]] = {}
    for channels, output in outputs.items():
        state = YOLO(str(output)).model.state_dict()
        expanded_states[channels] = state
        if set(state) != set(source_state):
            raise RuntimeError(f"Initialization state keys differ from COCO source: {output}")
        expected_first = (
            expand_conv_weight(source_state[first_conv_key], channels, "mean")
            .half()
            .to(dtype=source_state[first_conv_key].dtype)
        )
        if not torch.equal(state[first_conv_key], expected_first):
            raise RuntimeError(f"First convolution violates mean-fill contract: {output}")
        for name, source_value in source_state.items():
            if name != first_conv_key and not torch.equal(state[name], source_value):
                raise RuntimeError(f"Pretrained tensor changed during channel expansion: {output}::{name}")
    if not torch.equal(
        expanded_states[6][first_conv_key], expanded_states[7][first_conv_key][:, :6]
    ):
        raise RuntimeError("Six- and seven-channel initializations do not share identical first six filters")

    builder_path = ROOT / "scripts" / "utils" / "prepare_multichannel_yolo_weights.py"
    contract = {
        "source": str(source),
        "source_sha256": _sha256(source),
        "builder": str(builder_path),
        "builder_sha256": _sha256(builder_path),
        "fill_mode": "mean",
        "serialization": "Ultralytics model.save FP16 checkpoint, reloaded as FP32",
        "first_conv_key": first_conv_key,
        "non_first_layer_tensors_equal_to_source": True,
        "six_vs_seven_first_six_filters_equal": True,
        "outputs": {
            str(channels): {
                "path": str(output),
                "sha256": _sha256(output),
                "first_conv_shape": list(expanded_states[channels][first_conv_key].shape),
            }
            for channels, output in outputs.items()
        },
    }
    _json_dump(INIT_DIR / "initialization_contract.json", contract)
    _INITIALIZATION_CACHE = outputs
    return outputs


def _run_name(cell: Cell, seed: int) -> str:
    return f"mpb__{cell.name}__s{seed}"


def _config_path(cell: Cell) -> Path:
    path = CONFIG_DIR / cell.config
    if not path.exists():
        raise FileNotFoundError(f"Missing generated dataset config: {path}")
    return path


def _load_dataset_config(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _manifest_image_paths(data_config: Path, split: str) -> list[Path]:
    data = _load_dataset_config(data_config)
    dataset_root = Path(str(data["path"]))
    manifest = Path(str(data[split]))
    if not manifest.is_absolute():
        manifest = dataset_root / manifest
    return [Path(line.strip()) for line in manifest.read_text().splitlines() if line.strip()]


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for folder in ("images", "images_7ch"):
        if folder in parts:
            parts[parts.index(folder)] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"Cannot derive label path from {image_path}")


def _load_stats(dataset_root: Path, channels: int) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads((dataset_root / f"channel_stats_{channels}ch.json").read_text())
    if payload.get("fit_split") not in {"train_only", "factorial_train_intersection_only"}:
        raise RuntimeError(f"Refusing non-train-only channel stats in {dataset_root}")
    low = np.asarray(payload["percentiles"]["p01"], dtype=np.float32).reshape(1, 1, channels)
    high = np.asarray(payload["percentiles"]["p99"], dtype=np.float32).reshape(1, 1, channels)
    return low, high


def _preprocess(image_path: Path, low: np.ndarray, high: np.ndarray, imgsz: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    array = np.load(image_path, allow_pickle=False).astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC array at {image_path}, got {array.shape}")
    array = np.clip(array, low, high)
    range_scale = np.float32(255.0) / np.maximum(high - low, np.float32(1e-6))
    array = (array - low) * range_scale
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    height, width = array.shape[:2]
    # MaskRescueNPYDataset.load_image first resizes the longest side to imgsz,
    # including scale-up, before its non-augmenting LetterBox pads the short side.
    # Threshold and bootstrap metrics must reproduce that two-step geometry.
    scale = imgsz / max(height, width)
    resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(array, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = imgsz - resized_w, imgsz - resized_h
    left, top = pad_w // 2, pad_h // 2
    padded = np.pad(
        resized,
        ((top, pad_h - top), (left, pad_w - left), (0, 0)),
        mode="constant",
        constant_values=114.0,
    )
    return padded.astype(np.float32), (left, top, resized_w, resized_h)


def _load_gt(label_path: Path, geometry: tuple[int, int, int, int]) -> np.ndarray:
    left, top, resized_w, resized_h = geometry
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or int(float(parts[0])) != 0:
            continue
        cx, cy, width, height = map(float, parts[1:5])
        boxes.append(
            [
                left + (cx - width / 2) * resized_w,
                top + (cy - height / 2) * resized_h,
                left + (cx + width / 2) * resized_w,
                top + (cy + height / 2) * resized_h,
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _iou_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(predictions) == 0 or len(targets) == 0:
        return np.zeros((len(predictions), len(targets)), dtype=np.float32)
    left_top = np.maximum(predictions[:, None, :2], targets[None, :, :2])
    right_bottom = np.minimum(predictions[:, None, 2:], targets[None, :, 2:])
    intersection = np.prod(np.clip(right_bottom - left_top, 0, None), axis=2)
    pred_area = np.prod(np.clip(predictions[:, 2:] - predictions[:, :2], 0, None), axis=1)
    target_area = np.prod(np.clip(targets[:, 2:] - targets[:, :2], 0, None), axis=1)
    return intersection / np.maximum(pred_area[:, None] + target_area[None, :] - intersection, 1e-7)


def _raw_predict(model: torch.nn.Module, tensor: torch.Tensor, nc: int) -> np.ndarray:
    from ultralytics.utils.nms import non_max_suppression

    model.eval()
    with torch.no_grad():
        output = model(tensor)
    predictions = output[0] if isinstance(output, (tuple, list)) else output
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    result = non_max_suppression(
        predictions,
        conf_thres=MIN_INFERENCE_CONF,
        iou_thres=NMS_IOU,
        nc=nc,
        max_det=MAX_DET,
    )[0]
    if result is None or not len(result):
        return np.zeros((0, 6), dtype=np.float32)
    return result.detach().cpu().numpy().astype(np.float32)


def _scene_mapping() -> dict[str, str]:
    path = PROTOCOL_ROOT / "audit" / "patch_inventory.csv"
    mapping: dict[str, str] = {}
    if path.exists():
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                mapping[row["stem"]] = row["scene"]
    return mapping


def _patch_audit_mapping() -> dict[str, dict[str, str]]:
    path = PROTOCOL_ROOT / "audit" / "patch_inventory.csv"
    mapping: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                mapping[row["stem"]] = row
    return mapping


def _component_areas() -> dict[str, list[int]]:
    path = PROTOCOL_ROOT / "audit" / "marine_debris_components.csv"
    values: dict[str, list[int]] = defaultdict(list)
    if path.exists():
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                values[row["stem"]].append(int(row["area_pixels"]))
    return values


def collect_predictions(
    weights: Path,
    data_config: Path,
    split: str,
    imgsz: int = 512,
    device: str = "0",
    allow_test: bool = False,
) -> list[dict[str, object]]:
    from ultralytics import YOLO

    if split == "test" and not allow_test:
        raise PermissionError("Raw test prediction collection requires the guarded final-test stage")

    data = _load_dataset_config(data_config)
    dataset_root = Path(str(data["path"]))
    channels = int(data["channels"])
    variant_metadata_path = dataset_root / "variant_metadata.json"
    variant_metadata = (
        json.loads(variant_metadata_path.read_text(encoding="utf-8"))
        if variant_metadata_path.exists() else {}
    )
    evaluation_min_pixels = int(variant_metadata.get("val_test_min_pixels_marine_debris", 1))
    image_paths = _manifest_image_paths(data_config, split)
    low, high = _load_stats(dataset_root, channels)
    scene_map = _scene_mapping()
    patch_audit = _patch_audit_mapping()
    areas = _component_areas()

    yolo = YOLO(str(weights))
    torch_device = (
        torch.device("cpu")
        if device == "cpu" or not torch.cuda.is_available()
        else torch.device(f"cuda:{device.split(',')[0]}")
    )
    model = yolo.model.to(torch_device).eval()
    nc = int(model.yaml.get("nc", data.get("nc", 3)))
    entries: list[dict[str, object]] = []

    for index, image_path in enumerate(image_paths, start=1):
        array, geometry = _preprocess(image_path, low, high, imgsz)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0).to(torch_device)
        predictions = _raw_predict(model, tensor, nc)
        debris_predictions = predictions[predictions[:, 5].astype(int) == 0]
        ground_truth = _load_gt(_label_path(image_path), geometry)
        stem = image_path.stem
        gt_areas = [area for area in areas.get(stem, []) if area >= evaluation_min_pixels]
        if len(gt_areas) != len(ground_truth):
            gt_areas = None
        entries.append(
            {
                "stem": stem,
                "scene": scene_map.get(stem, stem.rsplit("_", 1)[0]),
                "predictions": debris_predictions,
                "ground_truth": ground_truth,
                "gt_areas": gt_areas,
                "negative_category_min4": patch_audit.get(stem, {}).get(
                    "negative_category_min4", "not_available"
                ),
            }
        )
        if index % 50 == 0 or index == len(image_paths):
            print(f"  inference {split}: {index}/{len(image_paths)} patches")
    return entries


def _match_entry(entry: dict[str, object], threshold: float) -> tuple[int, int, int, list[int]]:
    predictions = np.asarray(entry["predictions"], dtype=np.float32)
    targets = np.asarray(entry["ground_truth"], dtype=np.float32)
    predictions = predictions[predictions[:, 4] >= threshold]
    predictions = predictions[np.argsort(-predictions[:, 4])] if len(predictions) else predictions
    ious = _iou_matrix(predictions[:, :4], targets)
    matched_targets: set[int] = set()
    matched_indices: list[int] = []
    true_positives = 0
    for prediction_index in range(len(predictions)):
        if not len(targets):
            break
        target_order = np.argsort(-ious[prediction_index])
        for target_index in target_order:
            if ious[prediction_index, target_index] < MATCH_IOU:
                break
            if int(target_index) not in matched_targets:
                matched_targets.add(int(target_index))
                matched_indices.append(int(target_index))
                true_positives += 1
                break
    false_positives = len(predictions) - true_positives
    false_negatives = len(targets) - true_positives
    return true_positives, false_positives, false_negatives, matched_indices


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def operating_metrics(entries: Iterable[dict[str, object]], threshold: float) -> dict[str, object]:
    selected = list(entries)
    tp = fp = fn = 0
    negative_patch_fp = 0
    negative_patches = 0
    size_totals = Counter()
    size_tp = Counter()
    size_metadata_available = True
    scene_counts: dict[str, Counter] = defaultdict(Counter)
    negative_strata: dict[str, Counter] = defaultdict(Counter)

    def size_bin(area: int) -> str:
        if area <= 1:
            return "1"
        if area <= 3:
            return "2-3"
        if area <= 7:
            return "4-7"
        return "8+"

    for entry in selected:
        entry_tp, entry_fp, entry_fn, matched = _match_entry(entry, threshold)
        tp += entry_tp
        fp += entry_fp
        fn += entry_fn
        scene = str(entry["scene"])
        scene_counts[scene].update(tp=entry_tp, fp=entry_fp, fn=entry_fn, patches=1)
        raw_gt_areas = entry["gt_areas"]
        if raw_gt_areas is None:
            size_metadata_available = False
        else:
            gt_areas = list(raw_gt_areas)
            for area in gt_areas:
                size_totals[size_bin(int(area))] += 1
            for target_index in matched:
                size_tp[size_bin(int(gt_areas[target_index]))] += 1
        if len(np.asarray(entry["ground_truth"])) == 0:
            negative_patches += 1
            negative_patch_fp += entry_fp
            category = str(entry.get("negative_category_min4", "not_available"))
            negative_strata[category].update(patches=1, false_positives=entry_fp)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "patches": len(selected),
        "scenes": len(scene_counts),
        "alerts_per_100_patches": 100.0 * _safe_div(tp + fp, len(selected)),
        "false_alarms_per_100_patches": 100.0 * _safe_div(fp, len(selected)),
        "false_alarms_per_100_label_negative_patches": 100.0 * _safe_div(negative_patch_fp, negative_patches),
        "negative_patches": negative_patches,
        "label_negative_strata_min4_audit": {
            category: {
                "patches": counts["patches"],
                "false_positives": counts["false_positives"],
                "false_alarms_per_100_patches": 100.0 * _safe_div(
                    counts["false_positives"], counts["patches"]
                ),
            }
            for category, counts in sorted(negative_strata.items())
        },
        "size_stratified_recall": {
            "available": size_metadata_available,
            "reason": (
                None
                if size_metadata_available
                else "The source component ledger does not reproduce the historical v1 label-conversion rule."
            ),
            "bins": (
                {
                    name: {
                        "tp": size_tp[name],
                        "gt": size_totals[name],
                        "recall": _safe_div(size_tp[name], size_totals[name]),
                    }
                    for name in ("1", "2-3", "4-7", "8+")
                    if size_totals[name] > 0
                }
                if size_metadata_available else None
            ),
        },
        "per_scene": {
            scene: {
                **dict(counts),
                "precision": _safe_div(counts["tp"], counts["tp"] + counts["fp"]),
                "recall": _safe_div(counts["tp"], counts["tp"] + counts["fn"]),
                "false_alarms_per_patch": _safe_div(counts["fp"], counts["patches"]),
            }
            for scene, counts in sorted(scene_counts.items())
        },
    }


def ap50(entries: Iterable[dict[str, object]]) -> float:
    prediction_rows: list[tuple[float, int]] = []
    total_gt = 0
    for entry in entries:
        predictions = np.asarray(entry["predictions"], dtype=np.float32)
        predictions = predictions[np.argsort(-predictions[:, 4])] if len(predictions) else predictions
        targets = np.asarray(entry["ground_truth"], dtype=np.float32)
        total_gt += len(targets)
        ious = _iou_matrix(predictions[:, :4], targets)
        matched: set[int] = set()
        for prediction_index, prediction in enumerate(predictions):
            is_tp = 0
            for target_index in np.argsort(-ious[prediction_index]):
                if ious[prediction_index, target_index] < MATCH_IOU:
                    break
                if int(target_index) not in matched:
                    matched.add(int(target_index))
                    is_tp = 1
                    break
            prediction_rows.append((float(prediction[4]), is_tp))
    if total_gt == 0 or not prediction_rows:
        return 0.0
    prediction_rows.sort(key=lambda row: row[0], reverse=True)
    true_positives = np.cumsum([row[1] for row in prediction_rows])
    false_positives = np.cumsum([1 - row[1] for row in prediction_rows])
    recall = true_positives / total_gt
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    interpolated = [float(np.max(precision[recall >= value])) if np.any(recall >= value) else 0.0 for value in np.linspace(0, 1, 101)]
    return float(np.mean(interpolated))


def bootstrap_scene_ci(
    entries: list[dict[str, object]], threshold: float, iterations: int = 10000, seed: int = 20260818
) -> dict[str, object]:
    by_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        by_scene[str(entry["scene"])].append(entry)
    scenes = sorted(by_scene)
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        replicate: list[dict[str, object]] = []
        for copy_index, scene in enumerate(sampled):
            for entry in by_scene[str(scene)]:
                replicate.append({**entry, "scene": f"{scene}__bootstrap_{copy_index}"})
        metrics = operating_metrics(replicate, threshold)
        distributions["ap50"].append(ap50(replicate))
        for name in ("precision", "recall", "f1", "false_alarms_per_100_patches"):
            distributions[name].append(float(metrics[name]))
    return {
        "unit": "acquisition_scene",
        "iterations": iterations,
        "seed": seed,
        "intervals": {
            name: {
                "median": float(np.median(values)),
                "low_95": float(np.percentile(values, 2.5)),
                "high_95": float(np.percentile(values, 97.5)),
            }
            for name, values in distributions.items()
        },
    }


def _write_prediction_ledger(path: Path, entries: list[dict[str, object]]) -> None:
    _json_dump(
        path,
        {
            "format": "mpb-marine-debris-predictions-v2",
            "entries": [
                {
                    "stem": entry["stem"],
                    "scene": entry["scene"],
                    "predictions_xyxy_conf_cls": np.asarray(entry["predictions"]).tolist(),
                    "ground_truth_xyxy": np.asarray(entry["ground_truth"]).tolist(),
                    "gt_component_areas": entry["gt_areas"],
                    "negative_category_min4": entry.get("negative_category_min4", "not_available"),
                }
                for entry in entries
            ],
        },
    )


def _read_prediction_ledger(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") not in {
        "mpb-marine-debris-predictions-v1", "mpb-marine-debris-predictions-v2"
    }:
        raise RuntimeError(f"Unexpected prediction ledger format: {path}")
    return [
        {
            "stem": row["stem"],
            "scene": row["scene"],
            "predictions": np.asarray(row["predictions_xyxy_conf_cls"], dtype=np.float32).reshape(-1, 6),
            "ground_truth": np.asarray(row["ground_truth_xyxy"], dtype=np.float32).reshape(-1, 4),
            "gt_areas": row["gt_component_areas"],
            "negative_category_min4": row.get("negative_category_min4", "not_available"),
        }
        for row in payload["entries"]
    ]


def _metric_bundle(entries: list[dict[str, object]], threshold: float) -> dict[str, float]:
    operating = operating_metrics(entries, threshold)
    return {
        "ap50": ap50(entries),
        "precision": float(operating["precision"]),
        "recall": float(operating["recall"]),
        "f1": float(operating["f1"]),
        "false_alarms_per_100_patches": float(operating["false_alarms_per_100_patches"]),
    }


def _resample_scenes(
    entries: list[dict[str, object]], sampled_scenes: np.ndarray
) -> list[dict[str, object]]:
    by_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        by_scene[str(entry["scene"])].append(entry)
    replicate = []
    for copy_index, scene in enumerate(sampled_scenes):
        for entry in by_scene[str(scene)]:
            replicate.append({**entry, "scene": f"{scene}__bootstrap_{copy_index}"})
    return replicate


def paired_contrast_summary(
    entries_by_run: dict[tuple[str, int], list[dict[str, object]]],
    thresholds: dict[tuple[str, int], float],
    cells: tuple[str, ...],
    contrasts: dict[str, dict[str, float]],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 20260818,
) -> dict[str, object]:
    scene_sets = {
        frozenset(str(entry["scene"]) for entry in entries_by_run[(cell, run_seed)])
        for cell in cells
        for run_seed in SEEDS
    }
    if len(scene_sets) != 1:
        raise RuntimeError(f"Paired contrast cells do not share the same scenes: {cells}")
    scenes = np.asarray(sorted(next(iter(scene_sets))))
    metric_names = (
        "ap50",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_100_patches",
    )

    full_metrics = {
        key: _metric_bundle(entries, thresholds[key])
        for key, entries in entries_by_run.items()
        if key[0] in cells
    }

    def calculate_contrasts(metrics: dict[tuple[str, int], dict[str, float]]) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for contrast_name, coefficients in contrasts.items():
            output[contrast_name] = {
                metric: float(
                    sum(
                        coefficient
                        * np.mean([metrics[(cell, run_seed)][metric] for run_seed in SEEDS])
                        for cell, coefficient in coefficients.items()
                    )
                )
                for metric in metric_names
            }
        return output

    point_estimates = calculate_contrasts(full_metrics)
    seed_level = {
        contrast_name: {
            str(run_seed): {
                metric: float(
                    sum(
                        coefficient * full_metrics[(cell, run_seed)][metric]
                        for cell, coefficient in coefficients.items()
                    )
                )
                for metric in metric_names
            }
            for run_seed in SEEDS
        }
        for contrast_name, coefficients in contrasts.items()
    }

    rng = np.random.default_rng(seed)
    distributions: dict[str, dict[str, list[float]]] = {
        contrast_name: {metric: [] for metric in metric_names}
        for contrast_name in contrasts
    }
    for _ in range(iterations):
        sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        replicate_metrics = {
            (cell, run_seed): _metric_bundle(
                _resample_scenes(entries_by_run[(cell, run_seed)], sampled_scenes),
                thresholds[(cell, run_seed)],
            )
            for cell in cells
            for run_seed in SEEDS
        }
        replicate_contrasts = calculate_contrasts(replicate_metrics)
        for contrast_name, metric_values in replicate_contrasts.items():
            for metric, value in metric_values.items():
                distributions[contrast_name][metric].append(value)

    return {
        "unit": "paired acquisition-scene cluster bootstrap",
        "seeds": list(SEEDS),
        "iterations": iterations,
        "bootstrap_seed": seed,
        "cells": list(cells),
        "contrast_coefficients": contrasts,
        "point_estimates_mean_across_paired_seeds": point_estimates,
        "seed_level_effects": seed_level,
        "scene_bootstrap_95_ci": {
            contrast_name: {
                metric: {
                    "low": float(np.percentile(values, 2.5)),
                    "high": float(np.percentile(values, 97.5)),
                }
                for metric, values in metric_distributions.items()
            }
            for contrast_name, metric_distributions in distributions.items()
        },
    }


def validation_operating_points(entries: list[dict[str, object]]) -> dict[str, object]:
    curve = [operating_metrics(entries, threshold) for threshold in THRESHOLD_GRID]
    balanced = max(curve, key=lambda row: (row["f1"], -row["tp"] - row["fp"], row["threshold"]))
    recall_candidates = [row for row in curve if row["recall"] >= 0.80]
    precision_candidates = [row for row in curve if row["precision"] >= 0.80]
    high_recall = max(recall_candidates, key=lambda row: (row["precision"], row["recall"], row["threshold"])) if recall_candidates else None
    high_precision = max(precision_candidates, key=lambda row: (row["recall"], row["precision"], row["threshold"])) if precision_candidates else None
    return {
        "selection_split": "val",
        "matching_iou": MATCH_IOU,
        "nms_iou": NMS_IOU,
        "max_det": MAX_DET,
        "threshold_grid": list(THRESHOLD_GRID),
        "curve": curve,
        "selected": {
            "balanced": balanced,
            "high_recall_r_ge_0.80": high_recall,
            "high_precision_p_ge_0.80": high_precision,
        },
    }


def train_one(cell: Cell, seed: int, device: str, smoke: bool = False) -> None:
    protocol = _protocol()
    initializations = _ensure_initialization()
    code_contract = _code_contract()
    data_config = _config_path(cell)
    data_fingerprint = _verified_dataset(cell)
    project = RUNS_DIR / ("smoke" if smoke else "full")
    run_name = _run_name(cell, seed) + (f"__smoke_v{SMOKE_VERSION}" if smoke else "")
    run_dir = project / run_name
    if (run_dir / "validation_complete.json").exists():
        manifest = json.loads((run_dir / "mpb_run_manifest.json").read_text(encoding="utf-8"))
        completion = json.loads((run_dir / "validation_complete.json").read_text(encoding="utf-8"))
        weights = run_dir / "weights" / "best.pt"
        val_metrics_path = run_dir / "eval_val.json"
        val_operating_path = run_dir / "val_operating_points.json"
        val_ledger_path = run_dir / "val_predictions.json"
        reusable = (
            manifest.get("protocol_lock_sha256") == _sha256(PROTOCOL_LOCK)
            and manifest.get("dataset_config_sha256") == _sha256(data_config)
            and manifest.get("dataset_artifact_fingerprint", {}).get("sha256") == data_fingerprint["sha256"]
            and manifest.get("initialization_sha256") == _sha256(initializations[cell.channels])
            and manifest.get("code_contract", {}).get("sha256") == code_contract["sha256"]
            and completion.get("best_weights_sha256") == _sha256(weights)
            and completion.get("validation_metrics_sha256") == _sha256(val_metrics_path)
            and completion.get("operating_points_sha256") == _sha256(val_operating_path)
            and completion.get("validation_ledger_sha256") == _sha256(val_ledger_path)
            and completion.get("test_accessed") is False
        )
        if not reusable:
            raise RuntimeError(f"Completed run belongs to a stale or modified protocol: {run_dir}")
        print(f"[skip] validation already complete: {run_name}")
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Partial run exists; inspect or resume explicitly before continuing: {run_dir}")

    hyperparams = {**LOCKED_HYPERPARAMS}
    if smoke:
        hyperparams.update(epochs=1, patience=1, plots=False, workers=0)
    overrides = {
        **hyperparams,
        "model": str(initializations[cell.channels]),
        "data": str(data_config),
        "project": str(project),
        "name": run_name,
        "seed": seed,
        "device": device,
        "exist_ok": True,
    }
    manifest = {
        "run_name": run_name,
        "cell": asdict(cell),
        "seed": seed,
        "smoke": smoke,
        "protocol_lock_sha256": _sha256(PROTOCOL_LOCK),
        "dataset_config_sha256": _sha256(data_config),
        "dataset_artifact_fingerprint": data_fingerprint,
        "initialization": str(initializations[cell.channels]),
        "initialization_sha256": _sha256(initializations[cell.channels]),
        "initialization_contract_sha256": _sha256(INIT_DIR / "initialization_contract.json"),
        "code_contract": code_contract,
        "hyperparameters": hyperparams,
        "test_accessed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "historical_test_status": protocol["test_status"],
    }
    _json_dump(run_dir / "mpb_run_manifest.json", manifest)

    trainer = MPBTrainer(overrides=overrides)
    trainer.train()
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(best_weights)

    val_ap = evaluate(
        weights=str(best_weights),
        data_yaml=str(data_config),
        channels=cell.channels,
        split="val",
        imgsz=int(hyperparams["imgsz"]),
        batch=int(hyperparams["batch"]),
        device=device,
        conf=MIN_INFERENCE_CONF,
        iou=NMS_IOU,
        workers=0,
        project=str(run_dir / "ultralytics_eval"),
        name="val_ap",
    )
    _json_dump(run_dir / "eval_val.json", val_ap)
    entries = collect_predictions(
        best_weights, data_config, split="val", imgsz=int(hyperparams["imgsz"]), device=device
    )
    val_ledger_path = run_dir / "val_predictions.json"
    _write_prediction_ledger(val_ledger_path, entries)
    operating = validation_operating_points(entries)
    val_operating_path = run_dir / "val_operating_points.json"
    _json_dump(val_operating_path, operating)
    val_metrics_path = run_dir / "eval_val.json"
    _json_dump(
        run_dir / "validation_complete.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "best_weights_sha256": _sha256(best_weights),
            "validation_metrics_sha256": _sha256(val_metrics_path),
            "operating_points_sha256": _sha256(val_operating_path),
            "validation_ledger_sha256": _sha256(val_ledger_path),
            "test_accessed": False,
        },
    )


def expected_runs(suite: str, seeds: Iterable[int]) -> list[tuple[Cell, int, Path]]:
    return [
        (CELLS[cell_name], seed, RUNS_DIR / "full" / _run_name(CELLS[cell_name], seed))
        for cell_name in SUITES[suite]
        for seed in seeds
    ]


def lock_decisions(suite: str, seeds: tuple[int, ...]) -> Path:
    if suite != "all" or seeds != SEEDS:
        raise RuntimeError(f"Decision lock requires suite=all and exact seeds={SEEDS}")
    protocol = _protocol()
    initializations = _ensure_initialization()
    code_contract = _code_contract()
    if (RUNS_DIR / "test_access_all.json").exists():
        raise RuntimeError("The one-shot test stage has already started; decisions can no longer be changed")
    runs = expected_runs(suite, seeds)
    missing = [str(path) for _, _, path in runs if not (path / "validation_complete.json").exists()]
    if missing:
        raise RuntimeError("Cannot lock decisions; validation is incomplete:\n" + "\n".join(missing))
    verified_datasets = {
        cell.name: _verified_dataset(cell, refresh=True)
        for cell_name in SUITES[suite]
        for cell in (CELLS[cell_name],)
    }
    decisions = []
    for cell, seed, run_dir in runs:
        manifest = json.loads((run_dir / "mpb_run_manifest.json").read_text(encoding="utf-8"))
        completion = json.loads((run_dir / "validation_complete.json").read_text(encoding="utf-8"))
        weights = run_dir / "weights" / "best.pt"
        current_weights_hash = _sha256(weights)
        expected_dataset = verified_datasets[cell.name]
        if completion.get("best_weights_sha256") != current_weights_hash:
            raise RuntimeError(f"Checkpoint changed after validation: {weights}")
        if completion.get("test_accessed") is not False or manifest.get("test_accessed") is not False:
            raise RuntimeError(f"Unexpected test access before decision lock: {run_dir}")
        if manifest.get("cell") != asdict(cell) or manifest.get("seed") != seed:
            raise RuntimeError(f"Run identity differs from the pre-registered cell/seed: {run_dir}")
        if manifest.get("smoke") is not False:
            raise RuntimeError(f"Smoke output cannot enter the formal decision lock: {run_dir}")
        if manifest.get("hyperparameters") != LOCKED_HYPERPARAMS:
            raise RuntimeError(f"Hyperparameters differ from the pre-registered contract: {run_dir}")
        if manifest.get("code_contract") != code_contract:
            raise RuntimeError(f"Implementation/runtime differs from the training manifest: {run_dir}")
        if manifest.get("protocol_lock_sha256") != _sha256(PROTOCOL_LOCK):
            raise RuntimeError(f"Protocol differs from training manifest: {run_dir}")
        if manifest.get("dataset_config_sha256") != _sha256(_config_path(cell)):
            raise RuntimeError(f"Dataset config differs from training manifest: {run_dir}")
        if manifest.get("dataset_artifact_fingerprint", {}).get("sha256") != expected_dataset["sha256"]:
            raise RuntimeError(f"Dataset content differs from training manifest: {run_dir}")
        expected_initialization = initializations[cell.channels]
        if Path(manifest.get("initialization", "")).resolve() != expected_initialization.resolve():
            raise RuntimeError(f"Unexpected initialization path in training manifest: {run_dir}")
        if manifest.get("initialization_sha256") != _sha256(expected_initialization):
            raise RuntimeError(f"Initialization changed after training: {run_dir}")
        if manifest.get("initialization_contract_sha256") != _sha256(INIT_DIR / "initialization_contract.json"):
            raise RuntimeError(f"Initialization contract changed after training: {run_dir}")
        val_operating_path = run_dir / "val_operating_points.json"
        val_metrics_path = run_dir / "eval_val.json"
        val_ledger_path = run_dir / "val_predictions.json"
        if completion.get("validation_metrics_sha256") != _sha256(val_metrics_path):
            raise RuntimeError(f"Validation metrics changed after completion: {run_dir}")
        if completion.get("operating_points_sha256") != _sha256(val_operating_path):
            raise RuntimeError(f"Validation operating points changed after completion: {run_dir}")
        if completion.get("validation_ledger_sha256") != _sha256(val_ledger_path):
            raise RuntimeError(f"Validation prediction ledger changed after completion: {run_dir}")
        operating = json.loads(val_operating_path.read_text(encoding="utf-8"))
        val_metrics = json.loads(val_metrics_path.read_text(encoding="utf-8"))
        if operating.get("selection_split") != "val" or val_metrics.get("split") != "val":
            raise RuntimeError(f"Non-validation result encountered during lock: {run_dir}")
        val_entries = _read_prediction_ledger(val_ledger_path)
        expected_val_stems = [path.stem for path in _manifest_image_paths(_config_path(cell), "val")]
        ledger_stems = [str(entry["stem"]) for entry in val_entries]
        if len(ledger_stems) != len(set(ledger_stems)) or sorted(ledger_stems) != sorted(expected_val_stems):
            raise RuntimeError(f"Validation ledger does not match the locked val manifest: {run_dir}")
        recomputed_operating = validation_operating_points(val_entries)
        if recomputed_operating != operating:
            raise RuntimeError(f"Validation operating points do not reproduce from the locked ledger: {run_dir}")
        decisions.append(
            {
                "cell": cell.name,
                "seed": seed,
                "run_dir": str(run_dir),
                "weights": str(weights),
                "weights_sha256": current_weights_hash,
                "dataset_config_sha256": _sha256(_config_path(cell)),
                "dataset_artifact_fingerprint": expected_dataset,
                "code_contract_sha256": code_contract["sha256"],
                "run_manifest_sha256": _sha256(run_dir / "mpb_run_manifest.json"),
                "validation_complete_sha256": _sha256(run_dir / "validation_complete.json"),
                "validation_metrics_sha256": _sha256(run_dir / "eval_val.json"),
                "operating_points_sha256": _sha256(run_dir / "val_operating_points.json"),
                "validation_ledger_sha256": _sha256(run_dir / "val_predictions.json"),
                "balanced_threshold": operating["selected"]["balanced"]["threshold"],
                "high_recall_threshold": (
                    operating["selected"]["high_recall_r_ge_0.80"]["threshold"]
                    if operating["selected"]["high_recall_r_ge_0.80"] else None
                ),
                "high_precision_threshold": (
                    operating["selected"]["high_precision_p_ge_0.80"]["threshold"]
                    if operating["selected"]["high_precision_p_ge_0.80"] else None
                ),
                "validation_metrics": val_metrics,
            }
        )
    lock = {
        "suite": suite,
        "seeds": list(seeds),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_lock_sha256": _sha256(PROTOCOL_LOCK),
        "analysis_hierarchy": protocol["analysis_hierarchy"],
        "code_contract": code_contract,
        "primary_deployable_cell_predeclared": PRIMARY_DEPLOYABLE_CELL,
        "checkpoint_rule": "Ultralytics best.pt selected by validation fitness only",
        "threshold_rule": "Validation-only, strict one-to-one IoU>=0.5 matching on the predeclared grid",
        "test_used_for_decisions": False,
        "historical_test_caveat": "MARIDA remains retrospective internal validation because earlier project stages inspected its test labels/results.",
        "decisions": decisions,
    }
    path = RUNS_DIR / f"decision_lock_{suite}.json"
    _json_dump(path, lock)
    print(f"Decision lock written: {path}")
    print(f"Decision lock SHA-256: {_sha256(path)}")
    return path


def summarize_locked_results(suite: str = "all", allow_pending_marker: bool = False) -> Path:
    if suite != "all":
        raise RuntimeError("Only the complete all-suite test can be summarized")
    marker_path = RUNS_DIR / "test_access_all.json"
    lock_path = RUNS_DIR / "decision_lock_all.json"
    if not marker_path.exists() or not lock_path.exists():
        raise FileNotFoundError("Completed test marker and decision lock are required")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    final_status = "retrospective_internal_validation_not_external_confirmation"
    allowed_statuses = {final_status}
    if allow_pending_marker:
        allowed_statuses.add("inference_complete_summary_pending")
    if marker.get("status") not in allowed_statuses:
        raise RuntimeError("The one-shot test stage has not completed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if marker.get("decision_lock_sha256") != _sha256(lock_path):
        raise RuntimeError("Decision lock changed after test execution")
    protocol = _protocol()
    current_code_contract = _code_contract()
    if lock.get("protocol_lock_sha256") != _sha256(PROTOCOL_LOCK):
        raise RuntimeError("Protocol differs from the completed test lock")
    if lock.get("analysis_hierarchy") != protocol.get("analysis_hierarchy"):
        raise RuntimeError("Analysis hierarchy differs from the completed test lock")
    if lock.get("code_contract") != current_code_contract:
        raise RuntimeError("Analysis implementation/runtime changed after test")
    sealed_output_manifest: dict[str, object] | None = None
    if marker.get("status") == final_status:
        output_manifest = RUNS_DIR / "test_output_manifest_all.json"
        if (
            not output_manifest.exists()
            or marker.get("test_output_manifest_sha256") != _sha256(output_manifest)
        ):
            raise RuntimeError("Final test output manifest is missing or changed")
        sealed_output_manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
        files = sealed_output_manifest.get("files")
        if not isinstance(files, dict):
            raise RuntimeError("Final test output manifest has no file contract")
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if (
            sealed_output_manifest.get("suite") != suite
            or sealed_output_manifest.get("decision_lock_sha256") != _sha256(lock_path)
            or sealed_output_manifest.get("files_contract_sha256")
            != hashlib.sha256(canonical).hexdigest()
        ):
            raise RuntimeError("Final test output manifest contract is invalid")
        expected_paths = {
            path.relative_to(ROOT).as_posix()
            for path in _test_output_paths(
                suite, lock["decisions"], RUNS_DIR / "paired_contrasts_all.json"
            )
        }
        if set(files) != expected_paths:
            raise RuntimeError("Final test output manifest has an incomplete or unexpected file set")
        for relative_path, expected_hash in files.items():
            path = ROOT / str(relative_path)
            if not path.exists() or _sha256(path) != expected_hash:
                raise RuntimeError(f"Frozen test output changed: {path}")

    entries_by_run: dict[tuple[str, int], list[dict[str, object]]] = {}
    thresholds: dict[tuple[str, int], float] = {}
    for decision in lock["decisions"]:
        key = (str(decision["cell"]), int(decision["seed"]))
        run_dir = Path(str(decision["run_dir"]))
        result_path = run_dir / "eval_test_locked.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        ledger_path = run_dir / "test_predictions_locked.json"
        if result.get("prediction_ledger_sha256") != _sha256(ledger_path):
            raise RuntimeError(f"Prediction ledger changed after test: {ledger_path}")
        entries_by_run[key] = _read_prediction_ledger(ledger_path)
        thresholds[key] = float(decision["balanced_threshold"])

    factorial_cells = (
        "v1_clean_ndwi",
        "v1_clean_oracle",
        "v1_shared_ndwi",
        "v1_shared_oracle",
    )
    channel_cells = ("full_scene_6ch", "full_scene_ndwi7", "full_scene_shuffled7")
    min_cells = ("min_train1", "min_train2", "full_scene_6ch", "min_train8")
    matched_cells = ("min_matched1", "min_matched2", "full_scene_6ch", "min_matched8")
    summary = {
        "scope": "retrospective internal MARIDA validation; not external confirmation",
        "decision_lock_sha256": _sha256(lock_path),
        "factorial": paired_contrast_summary(
            entries_by_run,
            thresholds,
            factorial_cells,
            {
                "scene_exposure_main_effect": {
                    "v1_clean_ndwi": -0.5,
                    "v1_clean_oracle": -0.5,
                    "v1_shared_ndwi": 0.5,
                    "v1_shared_oracle": 0.5,
                },
                "oracle_prior_main_effect": {
                    "v1_clean_ndwi": -0.5,
                    "v1_clean_oracle": 0.5,
                    "v1_shared_ndwi": -0.5,
                    "v1_shared_oracle": 0.5,
                },
                "interaction": {
                    "v1_clean_ndwi": 1.0,
                    "v1_clean_oracle": -1.0,
                    "v1_shared_ndwi": -1.0,
                    "v1_shared_oracle": 1.0,
                },
            },
            seed=20260818,
        ),
        "channels": paired_contrast_summary(
            entries_by_run,
            thresholds,
            channel_cells,
            {
                "ndwi7_minus_6ch": {"full_scene_ndwi7": 1.0, "full_scene_6ch": -1.0},
                "shuffled7_minus_6ch": {"full_scene_shuffled7": 1.0, "full_scene_6ch": -1.0},
                "ndwi7_minus_shuffled7": {"full_scene_ndwi7": 1.0, "full_scene_shuffled7": -1.0},
            },
            seed=20260819,
        ),
        "min_fixed_estimand": paired_contrast_summary(
            entries_by_run,
            thresholds,
            min_cells,
            {
                "train_min1_minus_train_min4": {"min_train1": 1.0, "full_scene_6ch": -1.0},
                "train_min2_minus_train_min4": {"min_train2": 1.0, "full_scene_6ch": -1.0},
                "train_min8_minus_train_min4": {"min_train8": 1.0, "full_scene_6ch": -1.0},
            },
            seed=20260820,
        ),
        "min_matched_estimands": {
            "interpretation": (
                "Descriptive target-definition sensitivity only. Metrics across MIN values have different "
                "ground-truth populations and must not be ranked as model improvements."
            ),
            "cells": {
                cell: {
                    "ground_truth_instances": len(
                        np.concatenate(
                            [np.asarray(entry["ground_truth"]).reshape(-1, 4) for entry in entries_by_run[(cell, SEEDS[0])]],
                            axis=0,
                        )
                    ),
                    "patches": len(entries_by_run[(cell, SEEDS[0])]),
                    "scenes": len({entry["scene"] for entry in entries_by_run[(cell, SEEDS[0])]}),
                    "seed_metrics": {
                        str(run_seed): _metric_bundle(
                            entries_by_run[(cell, run_seed)], thresholds[(cell, run_seed)]
                        )
                        for run_seed in SEEDS
                    },
                }
                for cell in matched_cells
            },
        },
    }
    output = RUNS_DIR / "paired_contrasts_all.json"
    _json_dump(output, summary)
    if (
        sealed_output_manifest is not None
        and sealed_output_manifest["files"].get(output.relative_to(ROOT).as_posix())
        != _sha256(output)
    ):
        raise RuntimeError("Recomputed paired summary differs from the frozen test output")
    return output


def _test_output_paths(
    suite: str,
    decisions: list[dict[str, object]],
    summary_path: Path,
) -> list[Path]:
    paths = [
        Path(str(decision["run_dir"])) / name
        for decision in decisions
        for name in ("eval_test_locked.json", "test_predictions_locked.json")
    ]
    return [*paths, RUNS_DIR / f"test_results_{suite}.json", summary_path]


def _write_test_output_manifest(
    suite: str,
    lock_path: Path,
    decisions: list[dict[str, object]],
    summary_path: Path,
) -> Path:
    files = {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in _test_output_paths(suite, decisions, summary_path)
    }
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = {
        "suite": suite,
        "decision_lock_sha256": _sha256(lock_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "files_contract_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    output = RUNS_DIR / f"test_output_manifest_{suite}.json"
    _json_dump(output, payload)
    return output


def test_locked_suite(
    suite: str,
    seeds: tuple[int, ...],
    device: str,
    confirm: bool,
    confirmed_lock_sha256: str | None,
    resume: bool = False,
) -> None:
    if suite != "all":
        raise RuntimeError("Test access is only permitted once for the complete pre-registered suite")
    if seeds != SEEDS:
        raise RuntimeError(f"Test access requires the exact pre-registered seeds {SEEDS}")
    initializations = _ensure_initialization()
    if not confirm:
        raise RuntimeError("Test remains locked. Re-run with --confirm-final-test only after reviewing the decision lock.")
    lock_path = RUNS_DIR / f"decision_lock_{suite}.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing decision lock: {lock_path}")
    actual_lock_sha256 = _sha256(lock_path)
    if (
        confirmed_lock_sha256 is None
        or len(confirmed_lock_sha256) != 64
        or confirmed_lock_sha256.lower() != actual_lock_sha256
    ):
        raise RuntimeError(
            "Test remains locked. Supply the exact SHA-256 printed by the lock stage with "
            "--confirm-decision-lock-sha256 after reviewing that immutable decision file."
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["protocol_lock_sha256"] != _sha256(PROTOCOL_LOCK):
        raise RuntimeError("Protocol changed after decision lock; test access refused")
    if tuple(lock["seeds"]) != seeds:
        raise RuntimeError("Seed set differs from decision lock")
    current_code_contract = _code_contract()
    if lock.get("code_contract") != current_code_contract:
        raise RuntimeError("Implementation/runtime changed after decision lock; test access refused")
    expected_pairs = {
        (cell.name, seed)
        for cell, seed, _ in expected_runs(suite, seeds)
    }
    decision_pairs = [(decision["cell"], decision["seed"]) for decision in lock["decisions"]]
    if len(decision_pairs) != len(set(decision_pairs)) or set(decision_pairs) != expected_pairs:
        raise RuntimeError("Decision lock does not contain the exact pre-registered run matrix")
    marker = RUNS_DIR / f"test_access_{suite}.json"
    existing_marker = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
    if existing_marker is not None:
        if existing_marker.get("status") == "retrospective_internal_validation_not_external_confirmation":
            raise RuntimeError(f"One-shot test stage already completed: {marker}")
        if not resume:
            raise RuntimeError(
                f"An interrupted one-shot test exists: {marker}. "
                "Use --resume-final-test to continue only unfinished runs under the same lock."
            )
        if existing_marker.get("decision_lock_sha256") != _sha256(lock_path):
            raise RuntimeError("Interrupted test belongs to a different decision lock")
        if existing_marker.get("user_confirmed_decision_lock_sha256") != confirmed_lock_sha256.lower():
            raise RuntimeError("Interrupted test was not opened with this confirmed decision-lock digest")
    elif resume:
        raise RuntimeError("No interrupted final-test marker exists to resume")

    verified_datasets = {
        cell.name: _verified_dataset(cell, refresh=True)
        for cell_name in SUITES[suite]
        for cell in (CELLS[cell_name],)
    }
    for decision in lock["decisions"]:
        cell = CELLS[decision["cell"]]
        run_dir = Path(decision["run_dir"])
        weights = Path(decision["weights"])
        expected_run_dir = RUNS_DIR / "full" / _run_name(cell, int(decision["seed"]))
        if run_dir.resolve() != expected_run_dir.resolve():
            raise RuntimeError(f"Decision lock points to a non-formal run directory: {run_dir}")
        if weights.resolve() != (run_dir / "weights" / "best.pt").resolve():
            raise RuntimeError(f"Decision lock points to an unexpected checkpoint: {weights}")
        data_config = _config_path(cell)
        current_dataset = verified_datasets[cell.name]
        if _sha256(weights) != decision["weights_sha256"]:
            raise RuntimeError(f"Checkpoint changed after lock: {weights}")
        if decision.get("dataset_config_sha256") != _sha256(data_config):
            raise RuntimeError(f"Dataset config changed after lock: {data_config}")
        if decision.get("dataset_artifact_fingerprint", {}).get("sha256") != current_dataset["sha256"]:
            raise RuntimeError(f"Dataset content changed after lock: {data_config}")
        locked_artifacts = {
            "run_manifest_sha256": run_dir / "mpb_run_manifest.json",
            "validation_complete_sha256": run_dir / "validation_complete.json",
            "validation_metrics_sha256": run_dir / "eval_val.json",
            "operating_points_sha256": run_dir / "val_operating_points.json",
            "validation_ledger_sha256": run_dir / "val_predictions.json",
        }
        for key, artifact_path in locked_artifacts.items():
            if decision.get(key) != _sha256(artifact_path):
                raise RuntimeError(f"Validation artifact changed after lock: {artifact_path}")
        if decision.get("code_contract_sha256") != current_code_contract["sha256"]:
            raise RuntimeError(f"Code contract differs from locked training run: {run_dir}")

        manifest = json.loads((run_dir / "mpb_run_manifest.json").read_text(encoding="utf-8"))
        expected_initialization = initializations[cell.channels]
        if (
            manifest.get("cell") != asdict(cell)
            or manifest.get("seed") != decision["seed"]
            or manifest.get("smoke") is not False
            or manifest.get("hyperparameters") != LOCKED_HYPERPARAMS
            or manifest.get("protocol_lock_sha256") != _sha256(PROTOCOL_LOCK)
            or manifest.get("dataset_config_sha256") != _sha256(data_config)
            or manifest.get("dataset_artifact_fingerprint", {}).get("sha256")
            != current_dataset["sha256"]
            or manifest.get("code_contract") != current_code_contract
            or Path(str(manifest.get("initialization", ""))).resolve()
            != expected_initialization.resolve()
            or manifest.get("initialization_sha256") != _sha256(expected_initialization)
            or manifest.get("initialization_contract_sha256")
            != _sha256(INIT_DIR / "initialization_contract.json")
            or manifest.get("test_accessed") is not False
        ):
            raise RuntimeError(f"Formal training manifest changed before test: {run_dir}")

        completion = json.loads((run_dir / "validation_complete.json").read_text(encoding="utf-8"))
        if (
            completion.get("best_weights_sha256") != _sha256(weights)
            or completion.get("validation_metrics_sha256") != _sha256(run_dir / "eval_val.json")
            or completion.get("operating_points_sha256")
            != _sha256(run_dir / "val_operating_points.json")
            or completion.get("validation_ledger_sha256")
            != _sha256(run_dir / "val_predictions.json")
            or completion.get("test_accessed") is not False
        ):
            raise RuntimeError(f"Validation completion contract changed before test: {run_dir}")
        val_metrics = json.loads((run_dir / "eval_val.json").read_text(encoding="utf-8"))
        operating = json.loads(
            (run_dir / "val_operating_points.json").read_text(encoding="utf-8")
        )
        val_entries = _read_prediction_ledger(run_dir / "val_predictions.json")
        expected_val_stems = [
            path.stem for path in _manifest_image_paths(data_config, "val")
        ]
        ledger_stems = [str(entry["stem"]) for entry in val_entries]
        if (
            val_metrics.get("split") != "val"
            or operating.get("selection_split") != "val"
            or len(ledger_stems) != len(set(ledger_stems))
            or sorted(ledger_stems) != sorted(expected_val_stems)
        ):
            raise RuntimeError(f"Validation evidence no longer matches the locked val split: {run_dir}")
        recomputed_operating = validation_operating_points(val_entries)
        if recomputed_operating != operating:
            raise RuntimeError(f"Validation thresholds no longer reproduce from the ledger: {run_dir}")
        expected_thresholds = {
            "balanced_threshold": operating["selected"]["balanced"]["threshold"],
            "high_recall_threshold": (
                operating["selected"]["high_recall_r_ge_0.80"]["threshold"]
                if operating["selected"]["high_recall_r_ge_0.80"] else None
            ),
            "high_precision_threshold": (
                operating["selected"]["high_precision_p_ge_0.80"]["threshold"]
                if operating["selected"]["high_precision_p_ge_0.80"] else None
            ),
        }
        for threshold_name, expected_threshold in expected_thresholds.items():
            if decision.get(threshold_name) != expected_threshold:
                raise RuntimeError(
                    f"Decision-lock threshold is not validation-derived: {run_dir}, {threshold_name}"
                )

    marker_payload = existing_marker or {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "suite": suite,
            "decision_lock_sha256": _sha256(lock_path),
            "user_confirmed_decision_lock_sha256": confirmed_lock_sha256.lower(),
            "status": "started_same_lock_resume_only",
            "completed_runs": {},
        }
    _json_dump(marker, marker_payload)

    results = []
    for decision in lock["decisions"]:
        cell = CELLS[decision["cell"]]
        run_dir = Path(decision["run_dir"])
        weights = Path(decision["weights"])
        data_config = _config_path(cell)
        run_id = f"{cell.name}__s{decision['seed']}"
        result_path = run_dir / "eval_test_locked.json"
        ledger_path = run_dir / "test_predictions_locked.json"
        completed = marker_payload.get("completed_runs", {}).get(run_id)
        if completed is not None:
            if (
                completed.get("result_sha256") != _sha256(result_path)
                or completed.get("ledger_sha256") != _sha256(ledger_path)
            ):
                raise RuntimeError(f"Completed test artifact changed before resume: {run_id}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("prediction_ledger_sha256") != _sha256(ledger_path):
                raise RuntimeError(f"Prediction ledger changed before resume: {run_id}")
            results.append(result)
            continue

        test_ap = evaluate(
            weights=str(weights), data_yaml=str(data_config), channels=cell.channels,
            split="test", imgsz=512, batch=8, device=device,
            conf=MIN_INFERENCE_CONF, iou=NMS_IOU, workers=0,
            project=str(run_dir / "ultralytics_eval"), name="test_ap", allow_test=True,
        )
        entries = collect_predictions(
            weights, data_config, split="test", imgsz=512, device=device, allow_test=True
        )
        locked_thresholds = {
            "balanced": decision["balanced_threshold"],
            "high_recall_r_ge_0.80": decision["high_recall_threshold"],
            "high_precision_p_ge_0.80": decision["high_precision_threshold"],
        }
        operating_points = {
            name: (
                operating_metrics(entries, float(threshold))
                if threshold is not None else None
            )
            for name, threshold in locked_thresholds.items()
        }
        _write_prediction_ledger(ledger_path, entries)
        result = {
            "cell": cell.name,
            "seed": decision["seed"],
            "threshold_source": "validation_locked",
            "locked_thresholds": locked_thresholds,
            "ap_metrics": test_ap,
            "operating_points": operating_points,
            "operating_metrics": operating_points["balanced"],
            "marine_debris_ap50_recomputed": ap50(entries),
            "prediction_ledger_sha256": _sha256(ledger_path),
            "scene_bootstrap": bootstrap_scene_ci(
                entries, float(decision["balanced_threshold"]), iterations=BOOTSTRAP_ITERATIONS,
                seed=20260818 + int(decision["seed"]),
            ),
        }
        _json_dump(result_path, result)
        results.append(result)
        marker_payload.setdefault("completed_runs", {})[run_id] = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "result_sha256": _sha256(result_path),
            "ledger_sha256": _sha256(ledger_path),
        }
        _json_dump(marker, marker_payload)

    _json_dump(RUNS_DIR / f"test_results_{suite}.json", results)
    marker_payload["status"] = "inference_complete_summary_pending"
    marker_payload["inference_completed_utc"] = datetime.now(timezone.utc).isoformat()
    _json_dump(marker, marker_payload)
    output = summarize_locked_results(suite, allow_pending_marker=True)
    output_manifest = _write_test_output_manifest(suite, lock_path, lock["decisions"], output)
    _json_dump(
        marker,
        {
            **marker_payload,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "runs": len(results),
            "test_output_manifest_sha256": _sha256(output_manifest),
            "status": "retrospective_internal_validation_not_external_confirmation",
        },
    )
    print(f"Paired contrast summary written: {output}")


def print_status(suite: str, seeds: tuple[int, ...]) -> None:
    for cell, seed, run_dir in expected_runs(suite, seeds):
        if (run_dir / "eval_test_locked.json").exists():
            status = "test-complete"
        elif (run_dir / "validation_complete.json").exists():
            status = "validation-complete"
        elif run_dir.exists():
            status = "partial"
        else:
            status = "pending"
        print(f"{cell.name:28s} seed={seed}  {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the locked MPB experiments")
    parser.add_argument(
        "--stage",
        choices=("train", "lock", "test", "summarize", "status", "smoke"),
        default="status",
    )
    parser.add_argument("--suite", choices=tuple(SUITES), default="all")
    parser.add_argument("--seed", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--confirm-decision-lock-sha256")
    parser.add_argument("--resume-final-test", action="store_true")
    parser.add_argument("--smoke-cell", choices=tuple(CELLS))
    args = parser.parse_args()
    seeds = tuple(args.seed)
    if any(seed not in SEEDS for seed in seeds):
        raise ValueError(f"Only pre-registered seeds {SEEDS} are permitted")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not permitted")

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _ensure_ultralytics_font()
    _protocol()
    if args.stage == "status":
        print_status(args.suite, seeds)
    elif args.stage == "smoke":
        smoke_cell = CELLS[args.smoke_cell] if args.smoke_cell else CELLS[SUITES[args.suite][0]]
        train_one(smoke_cell, seeds[0], args.device, smoke=True)
    elif args.stage == "train":
        for cell_name in SUITES[args.suite]:
            for seed in seeds:
                train_one(CELLS[cell_name], seed, args.device)
    elif args.stage == "lock":
        lock_decisions(args.suite, seeds)
    elif args.stage == "test":
        test_locked_suite(
            args.suite,
            seeds,
            args.device,
            args.confirm_final_test,
            args.confirm_decision_lock_sha256,
            resume=args.resume_final_test,
        )
    elif args.stage == "summarize":
        print(f"Paired contrast summary written: {summarize_locked_results(args.suite)}")


if __name__ == "__main__":
    main()
