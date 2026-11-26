"""Prepare the locked MARIDA protocol used for the MPB revision.

The script is intentionally separate from the historical v1/v2 builders. It
uses all 63 scenes and all 1,381 patches, produces matched scene-level and
patch-level splits, and keeps every generated artifact under a new root.

Stages:
  audit  - scan labels, freeze splits, and write data/composition audits
  build  - additionally materialize 6ch/7ch arrays and YOLO dataset variants
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rasterio
import yaml
from scipy import ndimage
from tqdm import tqdm

try:
    from .protocol_integrity import dataset_fingerprint
except ImportError:  # Direct script execution.
    from protocol_integrity import dataset_fingerprint


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "MARIDA" / "patches"
OUT_DIR = ROOT / "data" / "processed" / "marida_mpb_v1"
CONFIG_DIR = ROOT / "configs" / "datasets" / "mpb"
SOURCE_MANIFEST = OUT_DIR / "execution_source_manifest.json"
SOURCE_ARCHIVE = OUT_DIR / "execution_source_archive.zip"
ENVIRONMENT_MANIFEST = OUT_DIR / "execution_environment.json"

EXECUTION_SOURCE_FILES = (
    "scripts/mpb/prepare_protocol.py",
    "scripts/mpb/protocol_integrity.py",
    "scripts/mpb/run_experiments.py",
    "scripts/v2_evaluate.py",
    "scripts/utils/prepare_multichannel_yolo_weights.py",
    "scripts/training/mask_rescue_hyperparams.py",
    "ultralytics_plugins/__init__.py",
    "ultralytics_plugins/register_extra_modules.py",
    "ultralytics_plugins/mask_rescue_trainer.py",
    "ultralytics_plugins/mask_rescue_dataset.py",
)

SEED = 20260818
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = np.array([0.70, 0.15, 0.15], dtype=np.float64)
SCENE_CAPACITIES = (44, 9, 10)
MIN_PIXELS = (1, 2, 4, 8)
PRIMARY_MIN_PIXELS = 4
BBOX_PADDING = 2
CONNECTIVITY = 4
BAND_INDICES = (4, 3, 2, 8, 10, 11)
BAND_NAMES = ("B4_Red", "B3_Green", "B2_Blue", "B8_NIR", "B11_SWIR1", "B12_SWIR2")
CLASS_NAMES = ("Marine_Debris", "Algae_Organic", "Ship")
V1_CLASS_NAMES = (
    "Marine_Debris",
    "Dense_Sargassum",
    "Sparse_Sargassum",
    "Natural_Organic",
    "Ship",
    "Foam",
    "Wakes",
)
CLASS_MAP = {1: 0, 2: 1, 3: 1, 4: 1, 5: 2}
RAW_CLASS_NAMES = {
    0: "Unannotated",
    1: "Marine_Debris",
    2: "Dense_Sargassum",
    3: "Sparse_Sargassum",
    4: "Natural_Organic_Material",
    5: "Ship",
    6: "Clouds",
    7: "Marine_Water",
    8: "Sediment_Laden_Water",
    9: "Foam",
    10: "Turbid_Water",
    11: "Shallow_Water",
    12: "Waves",
    13: "Cloud_Shadow",
    14: "Wakes",
    15: "Mixed_Water",
}
HISTORICAL_V1_CLASSES = {1, 2, 3, 4, 5, 9, 14}
STRUCTURE = ndimage.generate_binary_structure(2, 1)


@dataclass
class Component:
    original_class: int
    yolo_class: int
    area: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence_pixels: dict[int, int] = field(default_factory=dict)

    def yolo_row(self, height: int, width: int) -> tuple[int, float, float, float, float]:
        x0 = max(0, self.x_min - BBOX_PADDING)
        y0 = max(0, self.y_min - BBOX_PADDING)
        x1 = min(width - 1, self.x_max + BBOX_PADDING)
        y1 = min(height - 1, self.y_max + BBOX_PADDING)
        box_w = x1 - x0 + 1
        box_h = y1 - y0 + 1
        cx = (x0 + x1 + 1) / 2.0 / width
        cy = (y0 + y1 + 1) / 2.0 / height
        return self.yolo_class, cx, cy, box_w / width, box_h / height


@dataclass
class PatchRecord:
    stem: str
    scene: str
    image_path: Path
    class_path: Path
    confidence_path: Path
    height: int
    width: int
    class_pixels: dict[int, int]
    debris_confidence_pixels: dict[int, int]
    components: list[Component]

    def components_for(self, min_pixels: int) -> list[Component]:
        return [
            comp
            for comp in self.components
            if comp.area >= (min_pixels if comp.original_class == 1 else PRIMARY_MIN_PIXELS)
        ]

    def boxes_for(self, min_pixels: int) -> list[tuple[int, float, float, float, float]]:
        return [comp.yolo_row(self.height, self.width) for comp in self.components_for(min_pixels)]

    def debris_count(self, min_pixels: int) -> int:
        return sum(comp.original_class == 1 and comp.area >= min_pixels for comp in self.components)


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def _write_execution_snapshot(variants: list[dict[str, object]]) -> None:
    """Freeze the exact local implementation, generated configs, and runtime inventory."""
    import torch
    import ultralytics

    source_paths = [ROOT / relative_path for relative_path in EXECUTION_SOURCE_FILES]
    missing = [str(path) for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Execution source is incomplete:\n" + "\n".join(missing))

    config_paths = sorted(
        {Path(str(variant["config"])).resolve() for variant in variants},
        key=lambda path: path.as_posix(),
    )
    if variants and len(config_paths) != len(variants):
        raise RuntimeError("Every locked variant must have one unique dataset config")
    for path in config_paths:
        if not path.exists() or CONFIG_DIR.resolve() not in path.parents:
            raise RuntimeError(f"Variant config is missing or outside the MPB config root: {path}")

    installed_distributions = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed_distributions.append(
                {"name": name, "version": distribution.version}
            )
    installed_distributions.sort(key=lambda item: (item["name"].lower(), item["version"]))

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "rasterio": rasterio.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "pyyaml": importlib.metadata.version("pyyaml"),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "devices": cuda_devices,
        },
        "relevant_environment_variables": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
                "KMP_DUPLICATE_LIB_OK",
            )
        },
        "installed_distributions": installed_distributions,
    }
    _json_dump(ENVIRONMENT_MANIFEST, environment)

    source_hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    config_hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in config_paths
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "marida_mpb_v1",
        "stage": "build" if variants else "audit",
        "purpose": (
            "Recoverable execution snapshot for a dirty working tree; protocol_lock.json hashes "
            "this manifest, its deterministic ZIP archive, and the environment inventory."
        ),
        "source_file_sha256": source_hashes,
        "dataset_config_sha256": config_hashes,
        "environment_manifest": ENVIRONMENT_MANIFEST.relative_to(ROOT).as_posix(),
        "environment_manifest_sha256": hashlib.sha256(
            ENVIRONMENT_MANIFEST.read_bytes()
        ).hexdigest(),
    }
    _json_dump(SOURCE_MANIFEST, manifest)

    archive_paths = [
        *source_paths,
        *config_paths,
        ENVIRONMENT_MANIFEST,
        SOURCE_MANIFEST,
    ]
    temporary_archive = SOURCE_ARCHIVE.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(archive_paths, key=lambda item: item.relative_to(ROOT).as_posix()):
            relative_path = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    temporary_archive.replace(SOURCE_ARCHIVE)


def _scan_one(class_path: Path) -> PatchRecord:
    stem = class_path.name.removesuffix("_cl.tif")
    image_path = class_path.with_name(f"{stem}.tif")
    confidence_path = class_path.with_name(f"{stem}_conf.tif")
    if not image_path.exists() or not confidence_path.exists():
        raise FileNotFoundError(f"Incomplete MARIDA triplet for {stem}")

    with rasterio.open(class_path) as src:
        class_mask = src.read(1)
    with rasterio.open(confidence_path) as src:
        confidence_mask = src.read(1)
    if class_mask.shape != confidence_mask.shape:
        raise ValueError(f"Mask shape mismatch for {stem}")

    unique, counts = np.unique(class_mask, return_counts=True)
    class_pixels = {int(k): int(v) for k, v in zip(unique, counts)}
    debris_confidence_pixels = {
        level: int(np.count_nonzero((class_mask == 1) & (confidence_mask == level)))
        for level in (1, 2, 3)
    }

    components: list[Component] = []
    for original_class, yolo_class in CLASS_MAP.items():
        labeled, n_components = ndimage.label(class_mask == original_class, structure=STRUCTURE)
        for component_id in range(1, n_components + 1):
            component_mask = labeled == component_id
            ys, xs = np.where(component_mask)
            conf_values, conf_counts = np.unique(confidence_mask[component_mask], return_counts=True)
            components.append(
                Component(
                    original_class=original_class,
                    yolo_class=yolo_class,
                    area=int(len(xs)),
                    x_min=int(xs.min()),
                    y_min=int(ys.min()),
                    x_max=int(xs.max()),
                    y_max=int(ys.max()),
                    confidence_pixels={int(k): int(v) for k, v in zip(conf_values, conf_counts)},
                )
            )

    height, width = class_mask.shape
    return PatchRecord(
        stem=stem,
        scene=class_path.parent.name,
        image_path=image_path,
        class_path=class_path,
        confidence_path=confidence_path,
        height=height,
        width=width,
        class_pixels=class_pixels,
        debris_confidence_pixels=debris_confidence_pixels,
        components=components,
    )


def scan_records() -> list[PatchRecord]:
    class_paths = sorted(RAW_DIR.rglob("*_cl.tif"))
    records = [_scan_one(path) for path in tqdm(class_paths, desc="Scanning MARIDA masks")]
    stems = [record.stem for record in records]
    if len(records) != 1381 or len(set(stems)) != len(stems):
        raise RuntimeError(f"Expected 1,381 unique patches, found {len(records)}")
    scenes = {record.scene for record in records}
    if len(scenes) != 63:
        raise RuntimeError(f"Expected 63 scenes, found {len(scenes)}")
    return records


def _feature_vector(record: PatchRecord) -> np.ndarray:
    primary = record.components_for(PRIMARY_MIN_PIXELS)
    class_counts = Counter(comp.yolo_class for comp in primary)
    values = [1.0]
    values.extend(float(record.debris_count(value)) for value in MIN_PIXELS)
    values.extend(float(record.debris_count(value) > 0) for value in MIN_PIXELS)
    values.extend(float(class_counts.get(value, 0)) for value in range(len(CLASS_NAMES)))
    values.extend(float(record.class_pixels.get(value, 0) > 0) for value in range(6, 16))
    return np.asarray(values, dtype=np.float64)


FEATURE_NAMES = (
    "patches",
    "md_boxes_min1",
    "md_boxes_min2",
    "md_boxes_min4",
    "md_boxes_min8",
    "md_positive_patches_min1",
    "md_positive_patches_min2",
    "md_positive_patches_min4",
    "md_positive_patches_min8",
    "boxes_marine_debris_min4",
    "boxes_algae_organic_min4",
    "boxes_ship_min4",
    "patch_has_cloud",
    "patch_has_marine_water",
    "patch_has_sediment",
    "patch_has_foam",
    "patch_has_turbid_water",
    "patch_has_shallow_water",
    "patch_has_waves",
    "patch_has_cloud_shadow",
    "patch_has_wakes",
    "patch_has_mixed_water",
)
FEATURE_WEIGHTS = np.asarray(
    [1.0, 0.4, 0.8, 2.0, 4.0, 0.2, 0.4, 1.0, 2.0, 2.0, 0.7, 0.7]
    + [0.15] * 10,
    dtype=np.float64,
)


def _assignment_score(features: np.ndarray, assignment: np.ndarray) -> float:
    totals = features.sum(axis=0)
    sums = np.vstack([features[assignment == split_id].sum(axis=0) for split_id in range(3)])
    targets = SPLIT_RATIOS[:, None] * totals[None, :]
    scales = np.maximum(targets, 1.0)
    score = float(np.sum(FEATURE_WEIGHTS[None, :] * ((sums - targets) / scales) ** 2))

    min8_index = FEATURE_NAMES.index("md_boxes_min8")
    min8_positive_index = FEATURE_NAMES.index("md_positive_patches_min8")
    min4_index = FEATURE_NAMES.index("md_boxes_min4")
    for split_id in (1, 2):
        score += 50.0 * max(0.0, 4.0 - sums[split_id, min8_index]) ** 2
        score += 50.0 * max(0.0, 2.0 - sums[split_id, min8_positive_index]) ** 2
        score += 5.0 * max(0.0, 20.0 - sums[split_id, min4_index]) ** 2
    return score


def make_scene_split(records: list[PatchRecord]) -> tuple[dict[str, str], float]:
    by_scene: dict[str, list[PatchRecord]] = defaultdict(list)
    for record in records:
        by_scene[record.scene].append(record)
    scenes = sorted(by_scene)
    features = np.vstack([sum((_feature_vector(r) for r in by_scene[scene]), start=np.zeros(len(FEATURE_NAMES))) for scene in scenes])
    rng = np.random.default_rng(SEED)
    largest_min8_scene = int(np.argmax(features[:, FEATURE_NAMES.index("md_boxes_min8")]))

    best_assignment: np.ndarray | None = None
    best_score = float("inf")
    for _ in range(80):
        movable = [index for index in range(len(scenes)) if index != largest_min8_scene]
        rng.shuffle(movable)
        assignment = np.full(len(scenes), -1, dtype=np.int8)
        assignment[largest_min8_scene] = 0
        train_extra = SCENE_CAPACITIES[0] - 1
        assignment[movable[:train_extra]] = 0
        assignment[movable[train_extra : train_extra + SCENE_CAPACITIES[1]]] = 1
        assignment[movable[train_extra + SCENE_CAPACITIES[1] :]] = 2
        current_score = _assignment_score(features, assignment)

        for _ in range(4000):
            left, right = rng.integers(0, len(scenes), size=2)
            if left == largest_min8_scene or assignment[left] == assignment[right]:
                continue
            candidate = assignment.copy()
            candidate[left], candidate[right] = candidate[right], candidate[left]
            candidate_score = _assignment_score(features, candidate)
            if candidate_score < current_score:
                assignment, current_score = candidate, candidate_score
        if current_score < best_score:
            best_assignment, best_score = assignment.copy(), current_score

    if best_assignment is None:
        raise RuntimeError("Scene split optimization failed")
    counts = Counter(int(value) for value in best_assignment)
    if tuple(counts[index] for index in range(3)) != SCENE_CAPACITIES:
        raise AssertionError(f"Wrong scene capacities: {counts}")
    mapping = {scene: SPLIT_NAMES[int(best_assignment[index])] for index, scene in enumerate(scenes)}
    return mapping, best_score


def make_patch_split(
    records: list[PatchRecord], scene_split: dict[str, str]
) -> tuple[dict[str, str], float]:
    capacities = tuple(
        sum(scene_split[record.scene] == split for record in records) for split in SPLIT_NAMES
    )
    features = np.vstack([_feature_vector(record) for record in records])
    rng = np.random.default_rng(SEED + 1)
    best_assignment: np.ndarray | None = None
    best_score = float("inf")

    for _ in range(3000):
        order = rng.permutation(len(records))
        assignment = np.empty(len(records), dtype=np.int8)
        start = 0
        for split_id, capacity in enumerate(capacities):
            assignment[order[start : start + capacity]] = split_id
            start += capacity
        score = _assignment_score(features, assignment)
        if score < best_score:
            best_assignment, best_score = assignment.copy(), score

    if best_assignment is None:
        raise RuntimeError("Patch split optimization failed")
    current_score = best_score
    for _ in range(50000):
        left, right = rng.integers(0, len(records), size=2)
        if best_assignment[left] == best_assignment[right]:
            continue
        candidate = best_assignment.copy()
        candidate[left], candidate[right] = candidate[right], candidate[left]
        candidate_score = _assignment_score(features, candidate)
        if candidate_score < current_score:
            best_assignment, current_score = candidate, candidate_score

    mapping = {
        record.stem: SPLIT_NAMES[int(best_assignment[index])]
        for index, record in enumerate(records)
    }
    for split, capacity in zip(SPLIT_NAMES, capacities):
        if sum(value == split for value in mapping.values()) != capacity:
            raise AssertionError(f"Wrong patch capacity for {split}")
    return mapping, current_score


def _negative_category(record: PatchRecord, min_pixels: int) -> str:
    components = record.components_for(min_pixels)
    if any(component.yolo_class == 0 for component in components):
        return "marine_debris_positive"
    if record.class_pixels.get(1, 0) > 0:
        return "subthreshold_marine_debris"
    if any(component.yolo_class in {1, 2} for component in components):
        return "hard_confounder_algae_or_ship"
    if any(record.class_pixels.get(value, 0) for value in (9, 12, 14)):
        return "visual_confounder_foam_waves_wakes"
    if any(record.class_pixels.get(value, 0) for value in (7, 8, 10, 11, 15)):
        return "environmental_water_background"
    if any(record.class_pixels.get(value, 0) for value in (6, 13)):
        return "cloud_or_shadow"
    return "unannotated_or_other"


def _summarize_split(
    records: list[PatchRecord], assignment: dict[str, str], key: str
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split in SPLIT_NAMES:
        selected = [record for record in records if assignment[getattr(record, key)] == split]
        primary_components = [component for record in selected for component in record.components_for(PRIMARY_MIN_PIXELS)]
        raw_class_patch_counts = {
            RAW_CLASS_NAMES[value]: sum(record.class_pixels.get(value, 0) > 0 for record in selected)
            for value in range(16)
        }
        summary[split] = {
            "patches": len(selected),
            "scenes": len({record.scene for record in selected}),
            "empty_detection_labels_min4": sum(not record.components_for(PRIMARY_MIN_PIXELS) for record in selected),
            "class_instances_min4": {
                CLASS_NAMES[class_id]: sum(component.yolo_class == class_id for component in primary_components)
                for class_id in range(len(CLASS_NAMES))
            },
            "marine_debris_sensitivity": {
                f"min{value}": {
                    "instances": sum(record.debris_count(value) for record in selected),
                    "positive_patches": sum(record.debris_count(value) > 0 for record in selected),
                }
                for value in MIN_PIXELS
            },
            "negative_composition_exclusive_min4": dict(
                Counter(_negative_category(record, PRIMARY_MIN_PIXELS) for record in selected)
            ),
            "raw_class_patch_counts_nonexclusive": raw_class_patch_counts,
            "debris_confidence_pixels": {
                str(level): sum(record.debris_confidence_pixels[level] for record in selected)
                for level in (1, 2, 3)
            },
        }
    return summary


def _historical_subset(records: list[PatchRecord]) -> tuple[set[str], dict[str, object]]:
    historical_root = ROOT / "data" / "processed" / "mask_bbox_rescue_split_physidx_v2split"
    stems: set[str] = set()
    for split in SPLIT_NAMES:
        manifest = historical_root / f"{split}.txt"
        if manifest.exists():
            stems.update(Path(line.strip()).stem for line in manifest.read_text().splitlines() if line.strip())

    record_by_stem = {record.stem: record for record in records}
    excluded = [record for record in records if record.stem not in stems]
    no_historical_target_pixels = sum(
        not any(record.class_pixels.get(value, 0) for value in HISTORICAL_V1_CLASSES)
        for record in excluded
    )
    return stems, {
        "historical_root": str(historical_root),
        "included_patches": len(stems),
        "included_scenes": len({record_by_stem[stem].scene for stem in stems if stem in record_by_stem}),
        "excluded_patches": len(excluded),
        "excluded_without_any_v1_target_pixels": no_historical_target_pixels,
        "excluded_with_v1_target_pixels_but_filtered_by_old_builder": len(excluded) - no_historical_target_pixels,
        "interpretation": (
            "The historical builder retained only patches with at least one accepted seven-class box. "
            "It was therefore target-enriched and omitted operational background patches."
        ),
    }


def write_audits(
    records: list[PatchRecord],
    scene_split: dict[str, str],
    patch_split: dict[str, str],
    scene_score: float,
    patch_score: float,
) -> dict[str, object]:
    audit_dir = OUT_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    historical_stems, historical_summary = _historical_subset(records)

    inventory_fields = [
        "stem",
        "scene",
        "scene_split",
        "patch_split",
        "historical_514_subset",
        "negative_category_min4",
        *[f"md_instances_min{value}" for value in MIN_PIXELS],
        *[f"raw_pixels_class_{value}_{RAW_CLASS_NAMES[value]}" for value in range(16)],
        "md_confidence_high_pixels",
        "md_confidence_moderate_pixels",
        "md_confidence_low_pixels",
    ]
    with (audit_dir / "patch_inventory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory_fields)
        writer.writeheader()
        for record in records:
            row: dict[str, object] = {
                "stem": record.stem,
                "scene": record.scene,
                "scene_split": scene_split[record.scene],
                "patch_split": patch_split[record.stem],
                "historical_514_subset": int(record.stem in historical_stems),
                "negative_category_min4": _negative_category(record, PRIMARY_MIN_PIXELS),
                "md_confidence_high_pixels": record.debris_confidence_pixels[1],
                "md_confidence_moderate_pixels": record.debris_confidence_pixels[2],
                "md_confidence_low_pixels": record.debris_confidence_pixels[3],
            }
            row.update({f"md_instances_min{value}": record.debris_count(value) for value in MIN_PIXELS})
            row.update(
                {
                    f"raw_pixels_class_{value}_{RAW_CLASS_NAMES[value]}": record.class_pixels.get(value, 0)
                    for value in range(16)
                }
            )
            writer.writerow(row)

    component_fields = [
        "component_id",
        "stem",
        "scene",
        "scene_split",
        "original_class",
        "class_name",
        "area_pixels",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "confidence_high_pixels",
        "confidence_moderate_pixels",
        "confidence_low_pixels",
        "dominant_confidence",
    ]
    debris_components: list[dict[str, object]] = []
    component_id = 0
    for record in records:
        for component in record.components:
            if component.original_class != 1:
                continue
            component_id += 1
            debris_components.append(
                {
                    "component_id": component_id,
                    "stem": record.stem,
                    "scene": record.scene,
                    "scene_split": scene_split[record.scene],
                    "original_class": component.original_class,
                    "class_name": RAW_CLASS_NAMES[component.original_class],
                    "area_pixels": component.area,
                    "x_min": component.x_min,
                    "y_min": component.y_min,
                    "x_max": component.x_max,
                    "y_max": component.y_max,
                    "confidence_high_pixels": component.confidence_pixels.get(1, 0),
                    "confidence_moderate_pixels": component.confidence_pixels.get(2, 0),
                    "confidence_low_pixels": component.confidence_pixels.get(3, 0),
                    "dominant_confidence": min(
                        (1, 2, 3),
                        key=lambda level: (-component.confidence_pixels.get(level, 0), level),
                    ),
                }
            )
    with (audit_dir / "marine_debris_components.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=component_fields)
        writer.writeheader()
        writer.writerows(debris_components)

    scene_sets = {
        split: {scene for scene, assigned in scene_split.items() if assigned == split}
        for split in SPLIT_NAMES
    }
    patch_scene_sets = {
        split: {record.scene for record in records if patch_split[record.stem] == split}
        for split in SPLIT_NAMES
    }
    overlaps = {
        f"{left}_{right}": {
            "scene_split": len(scene_sets[left] & scene_sets[right]),
            "patch_split": len(patch_scene_sets[left] & patch_scene_sets[right]),
        }
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    }
    if any(value["scene_split"] for value in overlaps.values()):
        raise AssertionError("Scene-disjoint split contains scene overlap")
    if not all(value["patch_split"] > 0 for value in overlaps.values()):
        raise AssertionError("Patch-random control unexpectedly lacks scene overlap")

    source_fingerprint_rows = [
        f"{record.image_path.relative_to(ROOT).as_posix()}|{record.image_path.stat().st_size}|"
        f"{record.class_path.stat().st_size}|{record.confidence_path.stat().st_size}"
        for record in records
    ]
    summary = {
        "protocol": "marida_mpb_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "MARIDA",
            "patches": len(records),
            "scenes": len({record.scene for record in records}),
            "native_shape": sorted({(record.height, record.width) for record in records}),
            "raw_inventory_fingerprint_sha256": _sha256_text("\n".join(source_fingerprint_rows)),
            "reflectance_note": "MARIDA ACOLITE Rayleigh-corrected Sentinel-2 imagery; do not relabel as generic L2A surface reflectance.",
        },
        "label_definition": {
            "classes": list(CLASS_NAMES),
            "class_map": {str(key): value for key, value in CLASS_MAP.items()},
            "confidence_policy": "All released MARIDA confidence levels are retained; their composition is reported separately.",
            "connectivity": CONNECTIVITY,
            "bbox_padding_native_pixels": BBOX_PADDING,
            "primary_min_pixels": PRIMARY_MIN_PIXELS,
            "sensitivity_min_pixels": list(MIN_PIXELS),
            "sensitivity_rule": (
                "Only the Marine_Debris training-label threshold changes; algae/organic and ship remain at MIN_PIXELS=4, "
                "and every sensitivity model is selected/evaluated against the same MIN_PIXELS=4 validation/test estimand."
            ),
        },
        "split_design": {
            "seed": SEED,
            "target_ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS.tolist())),
            "scene_capacities": dict(zip(SPLIT_NAMES, SCENE_CAPACITIES)),
            "scene_split_score": scene_score,
            "patch_split_score": patch_score,
            "overlaps": overlaps,
            "scene_disjoint": _summarize_split(records, scene_split, "scene"),
            "patch_random": _summarize_split(records, patch_split, "stem"),
        },
        "historical_514_subset": historical_summary,
        "test_status": (
            "Internal re-analysis only: historical work inspected MARIDA test data. "
            "No MARIDA repartition can restore a genuinely unseen test. Final confirmation requires the independent external scene in item 6."
        ),
        "component_conversion_audit": {
            "component_ledger": str(audit_dir / "marine_debris_components.csv"),
            "scope": "All MARIDA Marine_Debris components across train, validation, and test scenes",
            "purpose": (
                "Automated accounting of connected components, component area, bbox geometry, confidence "
                "composition, and MIN_PIXELS sensitivity. No human expert-review result is claimed."
            ),
        },
    }
    _json_dump(audit_dir / "data_audit.json", summary)
    _json_dump(
        audit_dir / "split_manifest.json",
        {
            "scene_disjoint": scene_split,
            "patch_random": patch_split,
            "seed": SEED,
            "test_locked": True,
        },
    )
    return summary


def _ndwi_inverse(image_6ch: np.ndarray) -> np.ndarray:
    green = image_6ch[..., 1]
    nir = image_6ch[..., 3]
    denominator = green + nir
    ndwi = np.divide(
        green - nir,
        denominator,
        out=np.zeros_like(green, dtype=np.float32),
        where=np.abs(denominator) > 1e-6,
    )
    return np.clip((1.0 - ndwi) / 2.0, 0.0, 1.0).astype(np.float32)


def materialize_common_arrays(records: list[PatchRecord]) -> dict[str, Path]:
    common = OUT_DIR / "common"
    roots = {
        "6ch": common / "images_6ch",
        "ndwi7": common / "images_7ch_ndwi",
        "shuffled_ndwi7": common / "images_7ch_ndwi_shuffled",
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)

    for record in tqdm(records, desc="Materializing spectral/prior arrays"):
        outputs = {name: path / f"{record.stem}.npy" for name, path in roots.items()}
        if all(path.exists() for path in outputs.values()):
            continue
        with rasterio.open(record.image_path) as src:
            image_6ch = src.read(BAND_INDICES).transpose(1, 2, 0).astype(np.float32)
        if not outputs["6ch"].exists():
            np.save(outputs["6ch"], image_6ch)
        if not outputs["ndwi7"].exists():
            ndwi7 = np.concatenate([image_6ch, _ndwi_inverse(image_6ch)[..., None]], axis=2)
            np.save(outputs["ndwi7"], ndwi7.astype(np.float32))
        if not outputs["shuffled_ndwi7"].exists():
            prior = _ndwi_inverse(image_6ch)
            seed = int(hashlib.sha256(record.stem.encode("utf-8")).hexdigest()[:16], 16)
            shuffled = np.random.default_rng(seed).permutation(prior.reshape(-1)).reshape(prior.shape)
            shuffled_ndwi7 = np.concatenate([image_6ch, shuffled[..., None]], axis=2)
            np.save(outputs["shuffled_ndwi7"], shuffled_ndwi7.astype(np.float32))
    return roots


def _compute_stats(paths: Iterable[Path], channels: int, fixed_prior_range: bool = False) -> dict[str, object]:
    samples = []
    file_count = 0
    for path in tqdm(list(paths), desc=f"Computing train-only {channels}ch stats", leave=False):
        array = np.load(path, allow_pickle=False).astype(np.float32)
        flat = array.reshape(-1, channels)[::64]
        flat = flat[np.isfinite(flat).all(axis=1)]
        samples.append(flat)
        file_count += 1
    pool = np.concatenate(samples, axis=0)
    p01 = np.percentile(pool, 1, axis=0).astype(float)
    p99 = np.percentile(pool, 99, axis=0).astype(float)
    if fixed_prior_range:
        p01[-1], p99[-1] = 0.0, 1.0
    return {
        "channels": channels,
        "channel_names": list(BAND_NAMES) + ([] if channels == 6 else ["prior"]),
        "fit_split": "train_only",
        "files": file_count,
        "sample_stride_pixels": 64,
        "sampled_valid_pixels": int(len(pool)),
        "percentiles": {"p01": p01.tolist(), "p99": p99.tolist()},
    }


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_labels(path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    text = "".join(f"{cls} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}\n" for cls, cx, cy, width, height in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_variants(
    records: list[PatchRecord],
    scene_split: dict[str, str],
    patch_split: dict[str, str],
    common_roots: dict[str, Path],
) -> list[dict[str, object]]:
    variants = [
        ("scene_6ch_min4", "scene", "6ch", 6, 4, 4, False),
        *[
            (f"scene_6ch_minsens_train{value}_eval4", "scene", "6ch", 6, value, 4, False)
            for value in MIN_PIXELS if value != PRIMARY_MIN_PIXELS
        ],
        *[
            (f"scene_6ch_min{value}", "scene", "6ch", 6, value, value, False)
            for value in MIN_PIXELS if value != PRIMARY_MIN_PIXELS
        ],
        ("scene_7ch_ndwi_min4", "scene", "ndwi7", 7, 4, 4, False),
        ("scene_7ch_ndwi_shuffled_min4", "scene", "shuffled_ndwi7", 7, 4, 4, False),
    ]
    record_by_stem = {record.stem: record for record in records}
    stats_cache: dict[tuple[str, str], dict[str, object]] = {}
    metadata: list[dict[str, object]] = []
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for name, split_kind, source_kind, channels, train_min_pixels, eval_min_pixels, fixed_prior_range in variants:
        variant_root = OUT_DIR / "datasets" / name
        assignment = scene_split if split_kind == "scene" else patch_split
        assignment_key = "scene" if split_kind == "scene" else "stem"
        image_folder = "images" if channels == 6 else "images_7ch"
        manifests: dict[str, list[str]] = {split: [] for split in SPLIT_NAMES}

        for record in tqdm(records, desc=f"Linking {name}", leave=False):
            split = assignment[getattr(record, assignment_key)]
            image_target = variant_root / split / image_folder / f"{record.stem}.npy"
            _link_or_copy(common_roots[source_kind] / f"{record.stem}.npy", image_target)
            label_min_pixels = train_min_pixels if split == "train" else eval_min_pixels
            _write_labels(
                variant_root / split / "labels" / f"{record.stem}.txt",
                record.boxes_for(label_min_pixels),
            )
            manifests[split].append(str(image_target.resolve()))

        for split, lines in manifests.items():
            (variant_root / f"{split}.txt").write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")

        stats_key = (split_kind, source_kind)
        if stats_key not in stats_cache:
            train_stems = [
                record.stem
                for record in records
                if assignment[getattr(record, assignment_key)] == "train"
            ]
            stats_cache[stats_key] = _compute_stats(
                (common_roots[source_kind] / f"{stem}.npy" for stem in train_stems),
                channels,
                fixed_prior_range=fixed_prior_range,
            )
        stats = stats_cache[stats_key]
        _json_dump(variant_root / f"channel_stats_{channels}ch.json", stats)

        yaml_payload = {
            "path": str(variant_root.resolve()),
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
            "channels": channels,
            "nc": len(CLASS_NAMES),
            "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        }
        config_path = CONFIG_DIR / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")
        variant_meta = {
            "name": name,
            "split_kind": split_kind,
            "source_kind": source_kind,
            "channels": channels,
            "train_min_pixels_marine_debris": train_min_pixels,
            "val_test_min_pixels_marine_debris": eval_min_pixels,
            "min_pixels_other_classes": PRIMARY_MIN_PIXELS,
            "dataset_root": str(variant_root),
            "config": str(config_path),
            "manifests": {split: len(lines) for split, lines in manifests.items()},
            "train_stats_sha256": _sha256_text(json.dumps(stats, sort_keys=True)),
        }
        _json_dump(variant_root / "variant_metadata.json", variant_meta)
        metadata.append(variant_meta)

    return metadata


def _stem_hash(stem: str) -> str:
    return hashlib.sha256(stem.encode("utf-8")).hexdigest()


def _manifest_stems(path: Path) -> list[str]:
    return [Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()]


def _v1_patch_feature(label_path: Path, ndwi_path: Path) -> tuple[np.ndarray, list[str]]:
    rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = np.asarray([[float(value) for value in row[:5]] for row in rows], dtype=np.float32).reshape(-1, 5)
    class_ids = labels[:, 0].astype(int) if len(labels) else np.empty(0, dtype=int)
    class_counts = np.bincount(class_ids, minlength=len(V1_CLASS_NAMES)).astype(np.float64)
    class_presence = (class_counts > 0).astype(np.float64)
    debris = labels[class_ids == 0] if len(labels) else np.empty((0, 5), dtype=np.float32)
    debris_areas = debris[:, 3] * debris[:, 4] if len(debris) else np.empty(0, dtype=np.float32)
    area_features = np.asarray(
        [
            float(np.sum(debris_areas)),
            float(np.max(debris_areas)) if len(debris_areas) else 0.0,
            float(np.sum(debris_areas <= (8.0 / 256.0) ** 2)),
            float(np.sum(debris_areas > (8.0 / 256.0) ** 2)),
        ],
        dtype=np.float64,
    )
    image = np.load(ndwi_path, allow_pickle=False).astype(np.float32)[::16, ::16]
    flat = image.reshape(-1, image.shape[-1])
    flat[~np.isfinite(flat)] = np.nan
    image_means = np.nanmean(flat, axis=0).astype(np.float64)
    image_stds = np.nanstd(flat, axis=0).astype(np.float64)
    image_means = np.nan_to_num(image_means)
    image_stds = np.nan_to_num(image_stds)
    names = (
        [f"boxes_{name}" for name in V1_CLASS_NAMES]
        + [f"patch_has_{name}" for name in V1_CLASS_NAMES]
        + ["md_bbox_area_sum", "md_bbox_area_max", "md_boxes_small", "md_boxes_large"]
        + [f"mean_ch{index + 1}" for index in range(image.shape[-1])]
        + [f"std_ch{index + 1}" for index in range(image.shape[-1])]
    )
    return np.concatenate([class_counts, class_presence, area_features, image_means, image_stds]), names


def _choose_removed_train_scenes(
    scene_counts: dict[str, int],
    scene_features: dict[str, np.ndarray],
    target_features: np.ndarray,
    n_scenes: int,
    n_patches: int,
    beam_width: int = 128,
) -> tuple[list[str], float]:
    ordered = sorted(scene_counts, key=_stem_hash)
    zero = np.zeros_like(target_features)
    states: dict[tuple[int, int], list[tuple[tuple[str, ...], np.ndarray]]] = {
        (0, 0): [(tuple(), zero)]
    }
    for scene in ordered:
        count = scene_counts[scene]
        updates = {key: list(values) for key, values in states.items()}
        for (chosen_scenes, chosen_patches), candidates in states.items():
            key = (chosen_scenes + 1, chosen_patches + count)
            if key[0] > n_scenes or key[1] > n_patches:
                continue
            bucket = updates.setdefault(key, [])
            for selected, feature_sum in candidates:
                bucket.append(((*selected, scene), feature_sum + scene_features[scene]))
        for (chosen_scenes, chosen_patches), candidates in updates.items():
            fraction = chosen_patches / max(n_patches, 1)
            candidates.sort(
                key=lambda item: (
                    float(np.mean((item[1] - target_features * fraction) ** 2)),
                    item[0],
                )
            )
            updates[(chosen_scenes, chosen_patches)] = candidates[:beam_width]
        states = updates
    key = (n_scenes, n_patches)
    if key not in states:
        raise RuntimeError(f"Cannot match {n_scenes} train scenes and {n_patches} patches")
    selected, feature_sum = min(
        states[key],
        key=lambda item: (float(np.mean((item[1] - target_features) ** 2)), item[0]),
    )
    return list(selected), float(np.sqrt(np.mean((feature_sum - target_features) ** 2)))


def build_v1_factorial_audit() -> list[dict[str, object]]:
    oracle_root = ROOT / "data" / "processed" / "mask_bbox_rescue_split"
    ndwi_root = ROOT / "data" / "processed" / "mask_bbox_rescue_split_physidx"
    split_root = ROOT / "data" / "processed" / "mask_bbox_rescue_split_v2split"
    required = (oracle_root, ndwi_root, split_root)
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"Missing historical v1 source required for controlled 2x2 audit: {required}")

    def source_map(root: Path, folder: str) -> dict[str, Path]:
        paths = list(root.glob(f"*/{folder}/*.npy" if folder.startswith("images") else f"*/{folder}/*.txt"))
        mapping = {path.stem: path for path in paths if not path.stem.startswith("aug_")}
        if len(mapping) < 514:
            raise RuntimeError(f"Expected at least 514 v1 files in {root}/{folder}, found {len(mapping)}")
        return mapping

    image_sources = {
        "ndwi": source_map(ndwi_root, "images"),
        "oracle": source_map(oracle_root, "images"),
    }
    label_sources = source_map(oracle_root, "labels")

    clean_train = _manifest_stems(split_root / "train.txt")
    validation = _manifest_stems(split_root / "val.txt")
    clean_test = _manifest_stems(split_root / "test.txt")
    scene_of = lambda stem: stem.rsplit("_", 1)[0]
    test_by_scene: dict[str, list[str]] = defaultdict(list)
    for stem in clean_test:
        test_by_scene[scene_of(stem)].append(stem)

    audit_test: list[str] = []
    sibling_exposure: list[str] = []
    clean_only_scenes: list[str] = []
    for scene, stems in sorted(test_by_scene.items()):
        if len(stems) == 1:
            clean_only_scenes.append(scene)
            continue
        ordered = sorted(stems, key=_stem_hash)
        audit_count = int(round(0.70 * len(ordered)))
        audit_test.extend(ordered[:audit_count])
        sibling_exposure.extend(ordered[audit_count:])
    if len(audit_test) != 51 or len(sibling_exposure) != 21:
        raise AssertionError(
            f"Expected 51 audit and 21 sibling patches, found {len(audit_test)}/{len(sibling_exposure)}"
        )

    feature_stems = [*clean_train, *sibling_exposure]
    raw_features = []
    feature_names: list[str] | None = None
    for stem in tqdm(feature_stems, desc="Matching v1 factorial training composition", leave=False):
        feature, names = _v1_patch_feature(label_sources[stem], image_sources["ndwi"][stem])
        raw_features.append(feature)
        feature_names = names
    feature_matrix = np.vstack(raw_features)
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = np.maximum(feature_matrix.std(axis=0), 1e-9)
    standardized = (feature_matrix - feature_mean) / feature_std
    matching_weights = np.ones(standardized.shape[1], dtype=np.float64)
    matching_weights[: len(V1_CLASS_NAMES)] = 5.0
    matching_weights[len(V1_CLASS_NAMES) : 2 * len(V1_CLASS_NAMES)] = 3.0
    matching_weights[2 * len(V1_CLASS_NAMES) : 2 * len(V1_CLASS_NAMES) + 4] = 5.0
    feature_by_stem = dict(zip(feature_stems, standardized * matching_weights))
    unweighted_feature_by_stem = dict(zip(feature_stems, standardized))

    train_scene_counts = Counter(scene_of(stem) for stem in clean_train)
    scene_features = {
        scene: sum(
            (feature_by_stem[stem] for stem in clean_train if scene_of(stem) == scene),
            start=np.zeros(standardized.shape[1]),
        )
        for scene in train_scene_counts
    }
    sibling_features = sum(
        (feature_by_stem[stem] for stem in sibling_exposure),
        start=np.zeros(standardized.shape[1]),
    )
    removed_scenes, matching_rmse = _choose_removed_train_scenes(
        dict(train_scene_counts), scene_features, sibling_features, 8, len(sibling_exposure), beam_width=256
    )
    removed_scene_set = set(removed_scenes)
    shared_train = [stem for stem in clean_train if scene_of(stem) not in removed_scene_set]
    shared_train.extend(sibling_exposure)
    removed_stems = [stem for stem in clean_train if scene_of(stem) in removed_scene_set]
    unweighted_difference = (
        sum(
            (unweighted_feature_by_stem[stem] for stem in removed_stems),
            start=np.zeros(standardized.shape[1]),
        )
        - sum(
            (unweighted_feature_by_stem[stem] for stem in sibling_exposure),
            start=np.zeros(standardized.shape[1]),
        )
    )

    split_sets = {
        "clean": {"train": clean_train, "val": validation, "test": audit_test},
        "shared": {"train": shared_train, "val": validation, "test": audit_test},
    }
    for exposure, splits in split_sets.items():
        if len(splits["train"]) != 371 or len({scene_of(stem) for stem in splits["train"]}) != 40:
            raise AssertionError(f"{exposure} training panel is not matched at 371 patches/40 scenes")
        if any(
            set(splits[left]) & set(splits[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise AssertionError(f"Exact patch overlap in {exposure} audit")
    clean_overlap = {scene_of(stem) for stem in clean_train} & {scene_of(stem) for stem in audit_test}
    shared_overlap = {scene_of(stem) for stem in shared_train} & {scene_of(stem) for stem in audit_test}
    if clean_overlap or len(shared_overlap) != 8:
        raise AssertionError(f"Unexpected controlled scene exposure: clean={clean_overlap}, shared={shared_overlap}")
    for exposure, expected_train_test_overlap in (("clean", 0), ("shared", 8)):
        scene_sets = {
            split: {scene_of(stem) for stem in split_sets[exposure][split]}
            for split in SPLIT_NAMES
        }
        overlaps = {
            (left, right): len(scene_sets[left] & scene_sets[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        }
        if overlaps[("train", "test")] != expected_train_test_overlap:
            raise AssertionError(f"Unexpected train/test scene overlap in {exposure}: {overlaps}")
        if overlaps[("train", "val")] or overlaps[("val", "test")]:
            raise AssertionError(f"Unexpected validation scene overlap in {exposure}: {overlaps}")

    for stem in tqdm(sorted(set(clean_train + validation + clean_test)), desc="Checking v1 common spectral inputs", leave=False):
        ndwi = np.load(image_sources["ndwi"][stem], allow_pickle=False)
        oracle = np.load(image_sources["oracle"][stem], allow_pickle=False)
        if ndwi.shape != oracle.shape or not np.array_equal(ndwi[..., :6], oracle[..., :6], equal_nan=True):
            raise RuntimeError(f"NDWI/oracle first-six-channel mismatch for {stem}")
    variants = [
        ("v1_audit_clean_ndwi", "clean", "ndwi", False),
        ("v1_audit_clean_oracle", "clean", "oracle", True),
        ("v1_audit_shared_ndwi", "shared", "ndwi", False),
        ("v1_audit_shared_oracle", "shared", "oracle", True),
    ]
    common_train_stems = sorted(set(clean_train) & set(shared_train))
    if len(common_train_stems) != 350:
        raise AssertionError(
            f"Expected 350 exact training patches shared by both exposure levels, found {len(common_train_stems)}"
        )
    factorial_stats = _compute_stats(
        (image_sources["ndwi"][stem] for stem in common_train_stems),
        channels=7,
        fixed_prior_range=True,
    )
    factorial_stats["fit_split"] = "factorial_train_intersection_only"
    factorial_stats["fit_contract"] = (
        "Common normalization fitted on the 350 exact patches present in both clean and shared training sets; "
        "no exposure-exclusive, validation, or audit-test patch contributes."
    )
    metadata: list[dict[str, object]] = []

    for name, exposure, prior, fixed_prior_range in variants:
        variant_root = OUT_DIR / "datasets" / name
        splits = split_sets[exposure]
        manifests: dict[str, list[str]] = {split: [] for split in SPLIT_NAMES}
        for split, stems in splits.items():
            for stem in stems:
                image_target = variant_root / split / "images_7ch" / f"{stem}.npy"
                label_target = variant_root / split / "labels" / f"{stem}.txt"
                _link_or_copy(image_sources[prior][stem], image_target)
                _link_or_copy(label_sources[stem], label_target)
                manifests[split].append(str(image_target.resolve()))
        for split, lines in manifests.items():
            (variant_root / f"{split}.txt").write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")

        stats = factorial_stats
        _json_dump(variant_root / "channel_stats_7ch.json", stats)
        config_path = CONFIG_DIR / f"{name}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "path": str(variant_root.resolve()),
                    "train": "train.txt",
                    "val": "val.txt",
                    "test": "test.txt",
                    "channels": 7,
                    "nc": len(V1_CLASS_NAMES),
                    "names": {index: class_name for index, class_name in enumerate(V1_CLASS_NAMES)},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        variant_meta = {
            "name": name,
            "scientific_role": "controlled_v1_scene_exposure_x_prior_audit",
            "scene_exposure": exposure,
            "prior": prior,
            "channels": 7,
            "classes": list(V1_CLASS_NAMES),
            "dataset_root": str(variant_root),
            "config": str(config_path),
            "manifests": {split: len(lines) for split, lines in manifests.items()},
            "train_scenes": len({scene_of(stem) for stem in splits["train"]}),
            "test_scenes": len({scene_of(stem) for stem in splits["test"]}),
            "train_test_scene_overlap": len(
                {scene_of(stem) for stem in splits["train"]}
                & {scene_of(stem) for stem in splits["test"]}
            ),
            "normalization_contract": (
                "common 350-patch clean/shared train-intersection first-six statistics; prior fixed to [0,1]"
            ),
            "train_stats_sha256": _sha256_text(json.dumps(stats, sort_keys=True)),
        }
        _json_dump(variant_root / "variant_metadata.json", variant_meta)
        metadata.append(variant_meta)

    def composition(stems: list[str]) -> dict[str, object]:
        box_counts = Counter()
        positive_patches = Counter()
        for stem in stems:
            rows = [
                line.split()
                for line in label_sources[stem].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            present = set()
            for row in rows:
                class_id = int(float(row[0]))
                box_counts[class_id] += 1
                present.add(class_id)
            positive_patches.update(present)
        return {
            "patches": len(stems),
            "boxes": {
                V1_CLASS_NAMES[class_id]: box_counts[class_id]
                for class_id in range(len(V1_CLASS_NAMES))
            },
            "positive_patches": {
                V1_CLASS_NAMES[class_id]: positive_patches[class_id]
                for class_id in range(len(V1_CLASS_NAMES))
            },
        }

    panel = {
        "scope": "Conditional leakage audit on the historical 514-patch target-enriched v1 universe",
        "interpretation_limit": "Not valid for operational prevalence, false-alarm burden, or deployable performance claims",
        "selection_rule": "Within each eligible test scene, stems are SHA-256 sorted; 70% form the fixed audit panel and the remainder form sibling exposure patches. Labels are not used.",
        "clean_only_scenes": clean_only_scenes,
        "audit_test_patches": sorted(audit_test),
        "sibling_exposure_patches": sorted(sibling_exposure),
        "removed_clean_train_scenes": sorted(removed_scenes),
        "training_composition_matching": {
            "method": "deterministic constrained beam matching at exactly 8 scenes and 21 patches",
            "features": feature_names,
            "weighted_standardized_feature_rmse": matching_rmse,
            "unweighted_standardized_feature_rmse": float(
                np.sqrt(np.mean(unweighted_difference ** 2))
            ),
            "maximum_absolute_unweighted_standardized_difference": float(
                np.max(np.abs(unweighted_difference))
            ),
            "weights": {
                "class_box_counts": 5.0,
                "class_presence": 3.0,
                "md_box_size": 5.0,
                "spectral_and_prior_summary": 1.0,
            },
            "normalization": "features standardized across clean-train and sibling-exposure candidate patches",
        },
        "normalization_contract": (
            "All four cells use identical statistics computed from the 350 exact patches in the clean/shared "
            "training intersection; exposure-exclusive, validation and audit-test patches are excluded. "
            "The first six channels share p01/p99 and both prior channels are fixed to [0,1]."
        ),
        "training_composition": {
            "clean": composition(clean_train),
            "shared": composition(shared_train),
            "removed_clean_scenes": composition(
                removed_stems
            ),
            "added_test_scene_siblings": composition(sibling_exposure),
        },
        "variants": metadata,
    }
    _json_dump(OUT_DIR / "audit" / "v1_factorial_panel.json", panel)
    return metadata


def write_protocol_lock(audit: dict[str, object], variants: list[dict[str, object]]) -> None:
    audit_dir = OUT_DIR / "audit"
    tracked_files = [
        audit_dir / "data_audit.json",
        audit_dir / "split_manifest.json",
        audit_dir / "patch_inventory.csv",
        audit_dir / "marine_debris_components.csv",
        SOURCE_MANIFEST,
        SOURCE_ARCHIVE,
        ENVIRONMENT_MANIFEST,
    ]
    factorial_panel = audit_dir / "v1_factorial_panel.json"
    if factorial_panel.exists():
        tracked_files.append(factorial_panel)
    artifact_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked_files
    }
    lock = {
        "protocol": "marida_mpb_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "builder": str(Path(__file__).relative_to(ROOT)),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git": _git_state(),
        "artifact_hashes": artifact_hashes,
        "variants": variants,
        "model_selection": "best checkpoint and every deployment threshold must be selected from validation only",
        "analysis_hierarchy": {
            "primary_endpoint": "Marine_Debris AP50",
            "primary_contrast": "full_scene_ndwi7 minus full_scene_6ch",
            "required_mechanism_control": "full_scene_ndwi7 minus full_scene_shuffled7",
            "primary_support_rule": (
                "The paired acquisition-scene bootstrap 95% interval for NDWI7-6ch must lie above zero; "
                "all three paired seed effects are reported, and a majority sign reversal precludes a benefit claim."
            ),
            "mechanism_support_rule": (
                "A representation-specific interpretation additionally requires the paired scene-bootstrap "
                "95% interval for NDWI7-shuffled7 to lie above zero."
            ),
            "secondary": [
                "conditional v1 2x2 leakage-attribution audit",
                "fixed-MIN=4 estimand sensitivity",
            ],
            "descriptive_only": ["matched target-definition MIN sensitivity"],
            "multiplicity": (
                "Only the predeclared NDWI7-6ch contrast is confirmatory within the internal MARIDA analysis; "
                "all other intervals are effect-estimation or mechanism/robustness evidence."
            ),
            "management_claim": (
                "No operational acceptability or management-value claim is permitted from internal MARIDA alone; "
                "false-alarm burdens are reported descriptively until an external use-case cost is specified."
            ),
        },
        "test_access": "forbidden until a decision_lock.json has been written by the MPB runner",
        "test_status": audit["test_status"],
        "external_confirmation_required": True,
    }
    _json_dump(OUT_DIR / "protocol_lock.json", lock)


def attach_variant_fingerprints(variants: list[dict[str, object]]) -> None:
    hash_cache: dict[tuple[object, ...], str] = {}
    for metadata in tqdm(variants, desc="Hashing locked dataset variants"):
        fingerprint = dataset_fingerprint(Path(str(metadata["config"])), hash_cache)
        metadata["artifact_fingerprint"] = fingerprint
        metadata_path = Path(str(metadata["dataset_root"])) / "variant_metadata.json"
        persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
        persisted["artifact_fingerprint"] = fingerprint
        _json_dump(metadata_path, persisted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the locked MARIDA MPB protocol")
    parser.add_argument("--stage", choices=("audit", "build"), default="audit")
    args = parser.parse_args()

    records = scan_records()
    scene_split, scene_score = make_scene_split(records)
    patch_split, patch_score = make_patch_split(records, scene_split)
    audit = write_audits(records, scene_split, patch_split, scene_score, patch_score)
    variants: list[dict[str, object]] = []
    if args.stage == "build":
        common_roots = materialize_common_arrays(records)
        variants = build_variants(records, scene_split, patch_split, common_roots)
        variants.extend(build_v1_factorial_audit())
        attach_variant_fingerprints(variants)
    _write_execution_snapshot(variants)
    write_protocol_lock(audit, variants)

    print("\nMARIDA MPB protocol prepared")
    print(f"  stage:   {args.stage}")
    print(f"  output:  {OUT_DIR}")
    print(f"  patches: {len(records)}")
    print(f"  scenes:  {len({record.scene for record in records})}")
    print("  test:    locked; internal re-analysis only")


if __name__ == "__main__":
    main()
