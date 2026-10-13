"""Materialize MADOS crops into MARIDA-compatible YOLO .npy tiles for zero-shot external eval.

MADOS ships ACOLITE Rayleigh-corrected rhorc patches (240x240) with sparse _cl masks.
This script stacks B4/B3/B2/B8/B11/B12, optionally appends NDWI-inverse ch7, converts
connected components to YOLO boxes with the locked MIN=4 / pad=2 / 4-connectivity rule,
and writes official MADOS train/val/test manifests without refitting normalization stats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rasterio
import yaml
from scipy import ndimage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MADOS_ROOT = ROOT / "data" / "MADOS"
OUT_ROOT = ROOT / "data" / "processed" / "mados_external_min4"
MARIDA_STATS_ROOT = ROOT / "data" / "processed" / "marida_mpb_v1" / "datasets"
CONFIG_DIR = ROOT / "configs" / "datasets" / "mpb"
MARIDA_SCENE_MANIFEST = ROOT / "data" / "processed" / "marida_mpb_v1" / "audit" / "split_manifest.json"

CROP_ID = re.compile(r"^Scene_(?P<scene>\d+)_(?P<crop>\d+)$")
CLASS_MAP = {1: 0, 2: 1, 3: 1, 4: 1, 5: 2}
MIN_PIXELS = 4
BBOX_PADDING = 2
STRUCTURE = ndimage.generate_binary_structure(2, 1)
TEN_M_CHANNELS = {"B4", "B3", "B2", "B8"}
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FILES = {"train": "train_X.txt", "val": "val_X.txt", "test": "test_X.txt"}


@dataclass
class Component:
    original_class: int
    yolo_class: int
    area: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int

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
class CropRecord:
    crop_id: str
    scene_id: int
    crop_index: int
    split: str
    height: int
    width: int
    class_pixels: dict[int, int]
    components: list[Component] = field(default_factory=list)

    def components_for(self, min_pixels: int) -> list[Component]:
        return [
            component
            for component in self.components
            if component.area >= (min_pixels if component.original_class == 1 else MIN_PIXELS)
        ]

    def debris_count(self, min_pixels: int) -> int:
        return sum(
            component.original_class == 1 and component.area >= min_pixels
            for component in self.components
        )


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _discover_band_tokens(scene_dir: Path) -> dict[str, str]:
    names10 = " ".join(path.name for path in (scene_dir / "10").glob("*.tif"))
    names20 = " ".join(path.name for path in (scene_dir / "20").glob("*.tif"))
    green = "rhorc_559" if "rhorc_559" in names10 else "rhorc_560"
    swir1 = "rhorc_1610" if "rhorc_1610" in names20 else "rhorc_1614"
    swir2 = "rhorc_2186" if "rhorc_2186" in names20 else "rhorc_2202"
    required10 = ("rhorc_665", "rhorc_492", "rhorc_833", green)
    required20 = (swir1, swir2)
    if not all(token in names10 for token in required10) or not all(token in names20 for token in required20):
        raise RuntimeError(f"MADOS scene lacks required rhorc bands: {scene_dir.name}")
    return {"B4": "rhorc_665", "B3": green, "B2": "rhorc_492", "B8": "rhorc_833", "B11": swir1, "B12": swir2}


def _load_split_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split, filename in SPLIT_FILES.items():
        for line in (MADOS_ROOT / "splits" / filename).read_text(encoding="utf-8").splitlines():
            crop_id = line.strip()
            if not crop_id:
                continue
            if crop_id in mapping:
                raise RuntimeError(f"Duplicate MADOS crop id across splits: {crop_id}")
            mapping[crop_id] = split
    if len(mapping) != 2803:
        raise RuntimeError(f"Expected 2803 MADOS crops, found {len(mapping)}")
    return mapping


def _scan_components(class_mask: np.ndarray) -> tuple[dict[int, int], list[Component]]:
    unique, counts = np.unique(class_mask, return_counts=True)
    class_pixels = {int(value): int(count) for value, count in zip(unique, counts)}
    components: list[Component] = []
    for original_class, yolo_class in CLASS_MAP.items():
        labeled, n_components = ndimage.label(class_mask == original_class, structure=STRUCTURE)
        for component_id in range(1, n_components + 1):
            ys, xs = np.where(labeled == component_id)
            components.append(
                Component(
                    original_class=original_class,
                    yolo_class=yolo_class,
                    area=int(len(xs)),
                    x_min=int(xs.min()),
                    y_min=int(ys.min()),
                    x_max=int(xs.max()),
                    y_max=int(ys.max()),
                )
            )
    return class_pixels, components


def _load_crop_array(scene_id: int, crop_index: int, tokens: dict[str, str]) -> np.ndarray:
    scene_dir = MADOS_ROOT / f"Scene_{scene_id}"
    reference_shape: tuple[int, int] | None = None
    arrays: list[np.ndarray] = []
    for channel in ("B4", "B3", "B2", "B8", "B11", "B12"):
        folder = "10" if channel in TEN_M_CHANNELS else "20"
        path = scene_dir / folder / f"Scene_{scene_id}_L2R_{tokens[channel]}_{crop_index}.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        with rasterio.open(path) as source:
            array = source.read(1).astype(np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        if reference_shape is None:
            reference_shape = array.shape
        elif array.shape != reference_shape:
            array = cv2.resize(array, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_LINEAR)
        arrays.append(array)
    stack = np.stack(arrays, axis=-1)
    if stack.shape[:2] != (240, 240):
        raise RuntimeError(f"Unexpected MADOS crop size for Scene_{scene_id}_{crop_index}: {stack.shape}")
    return stack


def _collect_records(split_filter: set[str] | None = None) -> list[CropRecord]:
    split_map = _load_split_map()
    token_cache: dict[int, dict[str, str]] = {}
    records: list[CropRecord] = []
    for crop_id, split in sorted(split_map.items()):
        if split_filter is not None and split not in split_filter:
            continue
        match = CROP_ID.match(crop_id)
        if match is None:
            raise RuntimeError(f"Unexpected MADOS crop id: {crop_id}")
        scene_id = int(match.group("scene"))
        crop_index = int(match.group("crop"))
        scene_dir = MADOS_ROOT / f"Scene_{scene_id}"
        if scene_id not in token_cache:
            token_cache[scene_id] = _discover_band_tokens(scene_dir)
        class_path = scene_dir / "10" / f"Scene_{scene_id}_L2R_cl_{crop_index}.tif"
        if not class_path.exists():
            raise FileNotFoundError(class_path)
        with rasterio.open(class_path) as source:
            class_mask = source.read(1)
        class_pixels, components = _scan_components(class_mask)
        height, width = class_mask.shape
        records.append(
            CropRecord(
                crop_id=crop_id,
                scene_id=scene_id,
                crop_index=crop_index,
                split=split,
                height=height,
                width=width,
                class_pixels=class_pixels,
                components=components,
            )
        )
    return records


def _write_label(path: Path, record: CropRecord) -> None:
    rows = record.components_for(MIN_PIXELS)
    lines = [
        f"{yolo_class} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}"
        for yolo_class, cx, cy, box_w, box_h in (component.yolo_row(record.height, record.width) for component in rows)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _copy_locked_stats() -> None:
    for variant, channels in (("scene_6ch_min4", 6), ("scene_7ch_ndwi_min4", 7)):
        source = MARIDA_STATS_ROOT / variant / f"channel_stats_{channels}ch.json"
        if not source.exists():
            raise FileNotFoundError(source)
        destination_root = OUT_ROOT / "datasets" / variant
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / f"channel_stats_{channels}ch.json"
        if destination.exists() and _sha256(destination) != _sha256(source):
            raise RuntimeError(f"Refusing to overwrite mismatched stats: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)


def _write_dataset_yaml(variant: str, channels: int) -> Path:
    config_path = CONFIG_DIR / f"mados_external_{variant}.yaml"
    payload = {
        "path": str((OUT_ROOT / "datasets" / variant).resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "channels": channels,
        "nc": 3,
        "names": {0: "Marine_Debris", 1: "Algae_Organic", 2: "Ship"},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def _composition_audit(records: list[CropRecord]) -> dict[str, object]:
    by_split: dict[str, list[CropRecord]] = defaultdict(list)
    for record in records:
        by_split[record.split].append(record)
    summary: dict[str, object] = {}
    for split, split_records in sorted(by_split.items()):
        debris_instances = sum(record.debris_count(MIN_PIXELS) for record in split_records)
        debris_tiles = sum(record.debris_count(MIN_PIXELS) > 0 for record in split_records)
        summary[split] = {
            "crops": len(split_records),
            "unique_scenes": len({record.scene_id for record in split_records}),
            "marine_debris_instances_min4": debris_instances,
            "marine_debris_positive_tiles_min4": debris_tiles,
            "empty_detection_labels_min4": sum(not record.components_for(MIN_PIXELS) for record in split_records),
        }
    return summary


def inspect() -> None:
    records = _collect_records()
    composition = _composition_audit(records)
    marida_scenes = len(json.loads(MARIDA_SCENE_MANIFEST.read_text(encoding="utf-8"))["scene_disjoint"])
    print(
        json.dumps(
            {
                "mados_root": str(MADOS_ROOT),
                "crop_count": len(records),
                "scene_count": len({record.scene_id for record in records}),
                "native_crop_pixels": [240, 240],
                "radiometry": "ACOLITE rhorc reflectance (already unitless)",
                "marida_scene_count_for_overlap_note": marida_scenes,
                "georeferencing": "MADOS crops ship without geotransform; geographic overlap with MARIDA cannot be verified programmatically.",
                "composition_min4": composition,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def materialize(splits: tuple[str, ...], overwrite: bool) -> None:
    split_filter = set(splits)
    records = _collect_records(split_filter)
    token_cache: dict[int, dict[str, str]] = {}
    variant_roots = {
        "6ch": OUT_ROOT / "datasets" / "scene_6ch_min4",
        "ndwi7": OUT_ROOT / "datasets" / "scene_7ch_ndwi_min4",
    }
    for record in tqdm(records, desc="Materializing MADOS"):
        scene_dir = MADOS_ROOT / f"Scene_{record.scene_id}"
        if record.scene_id not in token_cache:
            token_cache[record.scene_id] = _discover_band_tokens(scene_dir)
        image_6ch = _load_crop_array(record.scene_id, record.crop_index, token_cache[record.scene_id])
        image_7ch = np.concatenate([image_6ch, _ndwi_inverse(image_6ch)[..., None]], axis=2)
        for split in SPLIT_NAMES:
            if record.split != split:
                continue
            for key, array in (("6ch", image_6ch), ("ndwi7", image_7ch)):
                variant = "scene_6ch_min4" if key == "6ch" else "scene_7ch_ndwi_min4"
                image_path = variant_roots[key] / split / "images" / f"{record.crop_id}.npy"
                label_path = variant_roots[key] / split / "labels" / f"{record.crop_id}.txt"
                if image_path.exists() and not overwrite:
                    if not label_path.exists():
                        raise RuntimeError(f"Partial materialization detected: {image_path}")
                else:
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(image_path, array.astype(np.float32))
                if not label_path.exists() or overwrite:
                    _write_label(label_path, record)
    for key, variant in (("6ch", "scene_6ch_min4"), ("ndwi7", "scene_7ch_ndwi_min4")):
        root = variant_roots[key]
        for split in SPLIT_NAMES:
            split_records = [record for record in records if record.split == split]
            manifest_path = root / f"{split}.txt"
            if split_records:
                lines = [
                    str((root / split / "images" / f"{record.crop_id}.npy").resolve())
                    for record in sorted(split_records, key=lambda item: item.crop_id)
                ]
                manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            elif not manifest_path.exists():
                manifest_path.write_text("", encoding="utf-8")

    _copy_locked_stats()
    config_6 = _write_dataset_yaml("scene_6ch_min4", 6)
    config_7 = _write_dataset_yaml("scene_7ch_ndwi_min4", 7)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MADOS",
        "output_root": str(OUT_ROOT),
        "splits_materialized": sorted(split_filter),
        "crop_count": len(records),
        "native_crop_pixels": [240, 240],
        "min_pixels": MIN_PIXELS,
        "bbox_padding": BBOX_PADDING,
        "connectivity": 4,
        "class_map": CLASS_MAP,
        "radiometry": "ACOLITE rhorc reflectance from MADOS; zero-shot normalization imports MARIDA train-only p01/p99.",
        "normalization_import": {
            "6ch": str(MARIDA_STATS_ROOT / "scene_6ch_min4" / "channel_stats_6ch.json"),
            "7ch": str(MARIDA_STATS_ROOT / "scene_7ch_ndwi_min4" / "channel_stats_7ch.json"),
        },
        "dataset_configs": {
            "6ch": str(config_6),
            "7ch": str(config_7),
        },
        "composition_min4": _composition_audit(records),
        "geographic_exclusion_note": (
            "MADOS crop GeoTIFFs lack geotransforms, so date/MGRS intersection with MARIDA's 63 scenes "
            "cannot be verified locally. External status relies on MADOS being an independent benchmark "
            "dataset and on evaluating only the official held-out MADOS split with frozen MARIDA checkpoints."
        ),
        "marida_scene_manifest_sha256": _sha256(MARIDA_SCENE_MANIFEST) if MARIDA_SCENE_MANIFEST.exists() else None,
    }
    _json_dump(OUT_ROOT / "materialization_manifest.json", manifest)
    print(f"MADOS materialization complete: {OUT_ROOT / 'materialization_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "materialize"), default="inspect")
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated MADOS official splits to materialize (default: all).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage == "inspect":
        inspect()
        return
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    unknown = set(splits) - set(SPLIT_NAMES)
    if unknown:
        raise SystemExit(f"Unknown splits: {sorted(unknown)}")
    materialize(splits, args.overwrite)


if __name__ == "__main__":
    main()
