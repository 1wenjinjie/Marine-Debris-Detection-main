"""Run a pre-specified controlled external check on PLP2021 targets.

PLP2021 contains repeated Sentinel-2 observations of one deployed HDPE target
and, after deployment, one wood target. This script deliberately treats it as a
controlled external target-transfer check, not as an exhaustive natural-debris
benchmark. It imports the already locked MARIDA model checkpoints and
validation-selected thresholds and never uses PLP labels to select either.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rasterio
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.mpb.run_experiments import MAX_DET, MIN_INFERENCE_CONF, NMS_IOU, _raw_predict


PLP_ROOT = ROOT / "data" / "PLP" / "PLP2021"
SCENES_ROOT = PLP_ROOT / "Sentinel-2"
TARGET_DBF = PLP_ROOT / "PLP2021_targets.dbf"
TARGET_SHP = PLP_ROOT / "PLP2021_targets.shp"
TARGET_SUPPORT_FILES = (
    TARGET_DBF,
    TARGET_SHP,
    PLP_ROOT / "PLP2021_targets.shx",
    PLP_ROOT / "PLP2021_targets.prj",
)
RUNS_ROOT = ROOT / "runs_mpb"
DECISION_LOCK = RUNS_ROOT / "decision_lock_all.json"
TEST_MARKER = RUNS_ROOT / "test_access_all.json"
TEST_OUTPUT_MANIFEST = RUNS_ROOT / "test_output_manifest_all.json"
OUT_DIR = RUNS_ROOT / "external_plp2021"
PROTOCOL_PATH = OUT_DIR / "external_protocol.json"
SUMMARY_PATH = OUT_DIR / "summary.json"
SUMMARY_MARKDOWN_PATH = OUT_DIR / "summary.md"
TARGET_CSV_PATH = OUT_DIR / "targets.csv"

PRIMARY_CELL = "full_scene_ndwi7"
EXPECTED_SEEDS = (0, 1, 2)
CHANNEL_NAMES = ("B4", "B3", "B2", "B8", "B11", "B12")
PLP_REFLECTANCE_DIVISOR = 10000.0
TARGET_RADIUS_METERS = 14.0
PRIMARY_CENTER_ALLOWANCE_METERS = 10.0
WIDE_CENTER_ALLOWANCE_METERS = 20.0
IMAGE_SIZE = 512
NATIVE_CROP_SIZE = (256, 256)
DATE_PATTERN = re.compile(r"^(?P<date>\d{8})(?:T|(?:\d{6})_PLP2021_native256)")


@dataclass(frozen=True)
class Target:
    target_id: int
    date: str
    material: str
    note: str
    radius_m: float
    easting: float
    northing: float


@dataclass(frozen=True)
class Scene:
    date: str
    path: Path
    width: int
    height: int
    crs: str
    res_x_m: float
    res_y_m: float
    bounds: tuple[float, float, float, float]
    reflectance_units: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_dbf(path: Path) -> list[dict[str, str]]:
    with path.open("rb") as handle:
        header = handle.read(32)
        if len(header) != 32:
            raise RuntimeError(f"Invalid DBF header: {path}")
        record_count = struct.unpack("<I", header[4:8])[0]
        header_length = struct.unpack("<H", header[8:10])[0]
        record_length = struct.unpack("<H", header[10:12])[0]
        fields: list[tuple[str, int]] = []
        while handle.tell() < header_length - 1:
            descriptor = handle.read(32)
            if len(descriptor) != 32:
                raise RuntimeError(f"Truncated DBF field descriptor: {path}")
            name = descriptor[:11].split(b"\0")[0].decode("latin1")
            fields.append((name, descriptor[16]))
        handle.seek(header_length)
        rows: list[dict[str, str]] = []
        for _ in range(record_count):
            record = handle.read(record_length)
            if len(record) != record_length:
                raise RuntimeError(f"Truncated DBF record: {path}")
            if record[:1] == b"*":
                continue
            offset = 1
            row: dict[str, str] = {}
            for name, width in fields:
                row[name] = record[offset : offset + width].decode("latin1").strip()
                offset += width
            rows.append(row)
    return rows


def _read_point_shp(path: Path) -> list[tuple[float, float]]:
    with path.open("rb") as handle:
        header = handle.read(100)
        if len(header) != 100 or struct.unpack("<I", header[32:36])[0] != 1:
            raise RuntimeError(f"PLP target geometry must be an ESRI Point shapefile: {path}")
        points: list[tuple[float, float]] = []
        while record_header := handle.read(8):
            if len(record_header) != 8:
                raise RuntimeError(f"Truncated SHP record header: {path}")
            _, content_words = struct.unpack(">2I", record_header)
            content = handle.read(content_words * 2)
            if len(content) != content_words * 2:
                raise RuntimeError(f"Truncated SHP record: {path}")
            if struct.unpack("<I", content[:4])[0] != 1:
                raise RuntimeError(f"PLP target record is not a point: {path}")
            points.append(struct.unpack("<2d", content[4:20]))
    return points


def _load_targets() -> list[Target]:
    for path in TARGET_SUPPORT_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
    rows = _read_dbf(TARGET_DBF)
    points = _read_point_shp(TARGET_SHP)
    if len(rows) != len(points):
        raise RuntimeError("PLP DBF and SHP record counts differ")
    targets: list[Target] = []
    for row, (easting, northing) in zip(rows, points):
        required = ("id", "date", "type", "note", "radius")
        missing = [name for name in required if name not in row]
        if missing:
            raise RuntimeError(f"PLP target DBF misses fields: {missing}")
        material = row["type"]
        if material not in {"HDPE", "wood"}:
            raise RuntimeError(f"Unexpected PLP target material: {material!r}")
        date = row["date"]
        if not re.fullmatch(r"\d{8}", date):
            raise RuntimeError(f"Unexpected PLP target date: {date!r}")
        targets.append(
            Target(
                target_id=int(row["id"]),
                date=date,
                material=material,
                note=row["note"],
                radius_m=float(row["radius"]),
                easting=float(easting),
                northing=float(northing),
            )
        )
    if not targets:
        raise RuntimeError("PLP target file has no rows")
    return targets


def _load_scenes(scene_root: Path) -> dict[str, Scene]:
    if not scene_root.exists():
        raise FileNotFoundError(scene_root)
    scenes: dict[str, Scene] = {}
    reference_crs: str | None = None
    for path in sorted(scene_root.glob("*.tif")):
        match = DATE_PATTERN.match(path.name)
        if match is None:
            raise RuntimeError(f"Cannot determine PLP acquisition date: {path.name}")
        date = match.group("date")
        if date in scenes:
            raise RuntimeError(f"Duplicate PLP acquisition date: {date}")
        with rasterio.open(path) as source:
            descriptions = tuple(source.descriptions)
            if source.count not in (6, 12) or any(name not in descriptions for name in CHANNEL_NAMES):
                raise RuntimeError(f"PLP scene lacks required Sentinel-2 bands: {path}")
            if source.count == 6 and descriptions != CHANNEL_NAMES:
                raise RuntimeError(f"Native PLP six-band scene has an unexpected band order: {path}")
            if source.count == 6:
                reflectance_units = source.tags().get("reflectance_units")
                if reflectance_units != "DN_divide_by_10000":
                    raise RuntimeError(
                        "Native PLP six-band scenes must declare reflectance_units=DN_divide_by_10000: "
                        f"{path}"
                    )
            else:
                reflectance_units = "bundled_PLP_DN_divide_by_10000"
            if source.crs is None:
                raise RuntimeError(f"PLP scene has no CRS: {path}")
            crs = str(source.crs)
            if reference_crs is None:
                reference_crs = crs
            elif crs != reference_crs:
                raise RuntimeError(f"PLP scenes use inconsistent CRS: {path}")
            scenes[date] = Scene(
                date=date,
                path=path,
                width=source.width,
                height=source.height,
                crs=crs,
                res_x_m=float(abs(source.res[0])),
                res_y_m=float(abs(source.res[1])),
                bounds=(
                    float(source.bounds.left),
                    float(source.bounds.bottom),
                    float(source.bounds.right),
                    float(source.bounds.top),
                ),
                reflectance_units=reflectance_units,
            )
    if not scenes:
        raise RuntimeError(f"No GeoTIFF scenes found: {scene_root}")
    return scenes


def _assert_native_crop_geometry(scenes: Iterable[Scene]) -> None:
    invalid = [
        f"{scene.path.name}: {scene.width}x{scene.height} at {scene.res_x_m:g}x{scene.res_y_m:g} m"
        for scene in scenes
        if (
            (scene.width, scene.height) != NATIVE_CROP_SIZE
            or not math.isclose(scene.res_x_m, 10.0, abs_tol=1e-6)
            or not math.isclose(scene.res_y_m, 10.0, abs_tol=1e-6)
        )
    ]
    if invalid:
        raise RuntimeError(
            "PLP inference is blocked because the supplied scenes are not native 256x256-pixel crops. "
            "The bundled PLP2021 files are 134x133 and would change the target scale when resized. "
            "Prepare same-date, native-10m 256x256 Sentinel-2 crops centered on the supplied targets first:\n"
            + "\n".join(invalid)
        )


def _load_materialization_provenance(
    scene_root: Path,
    scenes: dict[str, Scene],
) -> dict[str, object]:
    manifest_path = scene_root.parent / "materialization_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "PLP inference requires the native-crop materialization manifest next to the scene directory. "
            "Use materialize_plp2021_native256.py; do not substitute manually resized imagery."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = {
        scene.path.relative_to(ROOT).as_posix(): _sha256(scene.path)
        for scene in scenes.values()
    }
    if (
        manifest.get("native_crop_pixels") != list(NATIVE_CROP_SIZE)
        or manifest.get("output_scene_files_sha256") != expected_files
        or manifest.get("scene_count") != len(scenes)
    ):
        raise RuntimeError("PLP native-crop materialization manifest does not match the supplied scene files")
    radiometry = manifest.get("source_radiometry")
    if not isinstance(radiometry, str) or not radiometry:
        raise RuntimeError("PLP materialization manifest does not state source radiometry")
    return {
        "path": manifest_path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(manifest_path),
        "source_radiometry": radiometry,
        "source_collection": manifest.get("source_collection"),
    }


def _validate_target_coverage(targets: Iterable[Target], scenes: dict[str, Scene]) -> None:
    for target in targets:
        scene = scenes.get(target.date)
        if scene is None:
            raise RuntimeError(f"PLP target date has no matching GeoTIFF: {target.date}")
        left, bottom, right, top = scene.bounds
        if not (left <= target.easting <= right and bottom <= target.northing <= top):
            raise RuntimeError(f"PLP target lies outside its scene: {target}")


def _load_locked_decisions() -> list[dict[str, object]]:
    if not DECISION_LOCK.exists():
        raise FileNotFoundError(DECISION_LOCK)
    lock = json.loads(DECISION_LOCK.read_text(encoding="utf-8"))
    if lock.get("suite") != "all" or lock.get("test_used_for_decisions") is not False:
        raise RuntimeError("PLP evaluation requires the immutable validation-only all-suite decision lock")
    decisions = [
        decision
        for decision in lock.get("decisions", [])
        if decision.get("cell") == PRIMARY_CELL
    ]
    decisions.sort(key=lambda decision: int(decision["seed"]))
    if tuple(int(decision["seed"]) for decision in decisions) != EXPECTED_SEEDS:
        raise RuntimeError(f"Expected the predeclared {PRIMARY_CELL} seeds {EXPECTED_SEEDS}")
    for decision in decisions:
        weights = Path(str(decision["weights"]))
        if not weights.exists() or _sha256(weights) != decision.get("weights_sha256"):
            raise RuntimeError(f"Locked primary weights are missing or changed: {weights}")
        if not isinstance(decision.get("balanced_threshold"), (int, float)):
            raise RuntimeError(f"Missing validation-selected balanced threshold: {decision}")
    return decisions


def _assert_internal_test_is_frozen() -> None:
    if not TEST_MARKER.exists() or not TEST_OUTPUT_MANIFEST.exists():
        raise RuntimeError("Complete the locked internal MARIDA test before opening the external PLP run")
    marker = json.loads(TEST_MARKER.read_text(encoding="utf-8"))
    if marker.get("status") != "retrospective_internal_validation_not_external_confirmation":
        raise RuntimeError("The internal MARIDA test is not complete and frozen")
    if marker.get("decision_lock_sha256") != _sha256(DECISION_LOCK):
        raise RuntimeError("Internal test marker does not belong to the current decision lock")
    if marker.get("test_output_manifest_sha256") != _sha256(TEST_OUTPUT_MANIFEST):
        raise RuntimeError("Internal test output manifest changed")


def _load_normalization(decisions: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, int]:
    config_paths = {
        Path(str(decision["validation_metrics"]["data"]))
        for decision in decisions
    }
    if len(config_paths) != 1:
        raise RuntimeError("Primary seeds do not share one locked data configuration")
    data_config = yaml.safe_load(next(iter(config_paths)).read_text(encoding="utf-8"))
    if int(data_config["channels"]) != 7:
        raise RuntimeError("PLP external run expects the seven-channel primary model")
    dataset_root = Path(str(data_config["path"]))
    stats_path = dataset_root / "channel_stats_7ch.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("fit_split") != "train_only":
        raise RuntimeError("External normalization must use train-only primary statistics")
    if tuple(stats.get("channel_names", [])) != (
        "B4_Red", "B3_Green", "B2_Blue", "B8_NIR", "B11_SWIR1", "B12_SWIR2", "prior"
    ):
        raise RuntimeError("Unexpected seven-channel normalization contract")
    low = np.asarray(stats["percentiles"]["p01"], dtype=np.float32).reshape(1, 1, 7)
    high = np.asarray(stats["percentiles"]["p99"], dtype=np.float32).reshape(1, 1, 7)
    image_sizes = {int(decision["validation_metrics"]["imgsz"]) for decision in decisions}
    if len(image_sizes) != 1 or image_sizes != {IMAGE_SIZE}:
        raise RuntimeError("Primary seeds do not share the expected locked image size")
    return low, high, IMAGE_SIZE


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


def _load_scene_array(scene: Scene) -> tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(scene.path) as source:
        band_index = {name: index + 1 for index, name in enumerate(source.descriptions)}
        array = source.read([band_index[name] for name in CHANNEL_NAMES]).astype(np.float32)
        transform = source.transform
    image_6ch = array.transpose(1, 2, 0) / np.float32(PLP_REFLECTANCE_DIVISOR)
    image_7ch = np.concatenate((image_6ch, _ndwi_inverse(image_6ch)[..., None]), axis=2)
    return image_7ch.astype(np.float32), transform


def _preprocess_array(
    array: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    imgsz: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if array.ndim != 3 or array.shape[-1] != 7:
        raise ValueError(f"Expected HWC seven-channel array, got {array.shape}")
    array = np.clip(array, low, high)
    range_scale = np.float32(255.0) / np.maximum(high - low, np.float32(1e-6))
    array = (array - low) * range_scale
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    height, width = array.shape[:2]
    scale = imgsz / max(height, width)
    resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(array, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_width, pad_height = imgsz - resized_width, imgsz - resized_height
    left, top = pad_width // 2, pad_height // 2
    padded = np.pad(
        resized,
        ((top, pad_height - top), (left, pad_width - left), (0, 0)),
        mode="constant",
        constant_values=114.0,
    )
    return padded.astype(np.float32), (left, top, resized_width, resized_height)


def _boxes_to_native(
    boxes: np.ndarray,
    geometry: tuple[int, int, int, int],
    width: int,
    height: int,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    left, top, resized_width, resized_height = geometry
    native = boxes.astype(np.float32).copy()
    native[:, [0, 2]] = (native[:, [0, 2]] - left) / (resized_width / width)
    native[:, [1, 3]] = (native[:, [1, 3]] - top) / (resized_height / height)
    native[:, [0, 2]] = np.clip(native[:, [0, 2]], 0.0, float(width))
    native[:, [1, 3]] = np.clip(native[:, [1, 3]], 0.0, float(height))
    return native


def _world_center(transform: rasterio.Affine, box: np.ndarray) -> tuple[float, float]:
    center_x = float((box[0] + box[2]) / 2.0)
    center_y = float((box[1] + box[3]) / 2.0)
    easting, northing = transform * (center_x, center_y)
    return float(easting), float(northing)


def _source_files(scenes: dict[str, Scene]) -> dict[str, str]:
    paths = [*TARGET_SUPPORT_FILES, *(scene.path for _, scene in sorted(scenes.items()))]
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


def _input_contract(
    scenes: dict[str, Scene],
    targets: list[Target],
    decisions: list[dict[str, object]],
    low: np.ndarray,
    high: np.ndarray,
    materialization: dict[str, object],
) -> dict[str, object]:
    target_counts: dict[str, int] = defaultdict(int)
    for target in targets:
        target_counts[target.material] += 1
    return {
        "dataset": "PLP2021",
        "plp_root": str(PLP_ROOT),
        "source_files_sha256": _source_files(scenes),
        "target_counts": dict(sorted(target_counts.items())),
        "scene_count": len(scenes),
        "crs": next(iter(scenes.values())).crs,
        "scene_reflectance_units": sorted({scene.reflectance_units for scene in scenes.values()}),
        "native_crop_materialization": materialization,
        "decision_lock": str(DECISION_LOCK),
        "decision_lock_sha256": _sha256(DECISION_LOCK),
        "primary_cell": PRIMARY_CELL,
        "seeds": [int(decision["seed"]) for decision in decisions],
        "weights_sha256": {str(decision["seed"]): decision["weights_sha256"] for decision in decisions},
        "balanced_thresholds_validation_only": {
            str(decision["seed"]): float(decision["balanced_threshold"])
            for decision in decisions
        },
        "preprocessing": {
            "channels": list(CHANNEL_NAMES) + ["(1-NDWI)/2"],
            "plp_reflectance_divisor": PLP_REFLECTANCE_DIVISOR,
            "normalization_p01": low.reshape(-1).tolist(),
            "normalization_p99": high.reshape(-1).tolist(),
            "imgsz": IMAGE_SIZE,
            "min_inference_confidence": MIN_INFERENCE_CONF,
            "nms_iou": NMS_IOU,
            "max_detections": MAX_DET,
        },
        "matching": {
            "one_to_one_assignment": "Within each acquisition date, use maximum-cardinality minimum-distance target-prediction assignment.",
            "primary": "Marine_Debris box center within target radius plus one native 10 m pixel",
            "strict_sensitivity": "Marine_Debris box center within the supplied 14 m PLP target radius",
            "wide_sensitivity": "Marine_Debris box center within target radius plus two native 10 m pixels",
            "target_radius_m": TARGET_RADIUS_METERS,
            "primary_radius_m": TARGET_RADIUS_METERS + PRIMARY_CENTER_ALLOWANCE_METERS,
            "wide_radius_m": TARGET_RADIUS_METERS + WIDE_CENTER_ALLOWANCE_METERS,
        },
        "source_code_sha256": {
            "external_runner": _sha256(Path(__file__).resolve()),
            "internal_runner": _sha256(ROOT / "scripts" / "mpb" / "run_experiments.py"),
            "protocol_builder": _sha256(ROOT / "scripts" / "mpb" / "prepare_protocol.py"),
        },
    }


def _write_or_validate_protocol(contract: dict[str, object]) -> str:
    contract_sha = _canonical_sha256(contract)
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing.get("input_contract_sha256") != contract_sha:
            raise RuntimeError("Existing PLP output belongs to a different input contract; do not overwrite it")
        return contract_sha
    _json_dump(
        PROTOCOL_PATH,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_contract_sha256": contract_sha,
            "input_contract": contract,
            "scope": (
                "Controlled external target-transfer check on repeated artificial PLP2021 targets; "
                "not an exhaustive natural-debris prevalence or false-positive benchmark."
            ),
            "selection_rule": (
                "Weights and balanced thresholds are imported from the immutable MARIDA validation-only decision lock. "
                "PLP labels are not used for checkpoint, threshold, or architecture selection."
            ),
        },
    )
    return contract_sha


def _target_rows(targets: Iterable[Target], scenes: dict[str, Scene]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in sorted(targets, key=lambda item: (item.date, item.target_id)):
        scene = scenes[target.date]
        with rasterio.open(scene.path) as source:
            row, column = source.index(target.easting, target.northing)
        rows.append(
            {
                "date": target.date,
                "target_id": target.target_id,
                "material": target.material,
                "note": target.note,
                "radius_m": target.radius_m,
                "easting": target.easting,
                "northing": target.northing,
                "scene": scene.path.name,
                "pixel_row": row,
                "pixel_column": column,
            }
        )
    return rows


def _write_target_csv(targets: Iterable[Target], scenes: dict[str, Scene]) -> None:
    rows = _target_rows(targets, scenes)
    TARGET_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_seed(
    decision: dict[str, object],
    scenes: dict[str, Scene],
    low: np.ndarray,
    high: np.ndarray,
    imgsz: int,
    device: str,
    contract_sha: str,
    threshold: float | None = None,
    output_path: Path | None = None,
    threshold_source: str = "validation_locked_balanced",
) -> Path:
    seed = int(decision["seed"])
    output_path = output_path or (OUT_DIR / f"seed_{seed}_predictions.json")
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("input_contract_sha256") != contract_sha:
            raise RuntimeError(f"Existing PLP seed output does not match its input contract: {output_path}")
        return output_path
    weights = Path(str(decision["weights"]))
    threshold = float(decision["balanced_threshold"] if threshold is None else threshold)
    torch_device = torch.device("cpu") if device == "cpu" or not torch.cuda.is_available() else torch.device(f"cuda:{device}")
    yolo = YOLO(str(weights))
    model = yolo.model.to(torch_device).eval()
    class_count = int(model.yaml.get("nc", 3))
    predictions: list[dict[str, object]] = []
    raw_range_audit: dict[str, list[float]] = {}
    for index, (date, scene) in enumerate(sorted(scenes.items()), start=1):
        image, transform = _load_scene_array(scene)
        raw_range_audit[date] = [
            float(np.nanpercentile(image[..., channel], 1))
            for channel in range(image.shape[-1])
        ]
        prepared, geometry = _preprocess_array(image, low, high, imgsz)
        tensor = torch.from_numpy(prepared.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0).to(torch_device)
        detections = _raw_predict(model, tensor, class_count)
        debris = detections[detections[:, 5].astype(int) == 0]
        debris = debris[debris[:, 4] >= threshold]
        boxes = _boxes_to_native(debris[:, :4], geometry, scene.width, scene.height)
        for box, detection in zip(boxes, debris):
            easting, northing = _world_center(transform, box)
            predictions.append(
                {
                    "date": date,
                    "scene": scene.path.name,
                    "confidence": float(detection[4]),
                    "class_id": int(detection[5]),
                    "box_pixels_xyxy": [float(value) for value in box],
                    "center_easting": easting,
                    "center_northing": northing,
                }
            )
        print(f"PLP seed={seed}: {index}/{len(scenes)} scenes")
    _json_dump(
        output_path,
        {
            "input_contract_sha256": contract_sha,
            "seed": seed,
            "weights": str(weights),
            "weights_sha256": _sha256(weights),
            "threshold_source": threshold_source,
            "threshold": threshold,
            "predictions": predictions,
            "post_scale_p01_by_scene": raw_range_audit,
        },
    )
    return output_path


def _nearest_distance(target: Target, predictions: Iterable[dict[str, object]]) -> tuple[float | None, float | None]:
    nearest_distance: float | None = None
    nearest_confidence: float | None = None
    for prediction in predictions:
        distance = math.hypot(
            float(prediction["center_easting"]) - target.easting,
            float(prediction["center_northing"]) - target.northing,
        )
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_confidence = float(prediction["confidence"])
    return nearest_distance, nearest_confidence


def _rate(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "detected": numerator,
        "total": denominator,
        "rate": float(numerator / denominator) if denominator else None,
    }


def _one_to_one_matches(
    targets: list[Target],
    predictions: list[dict[str, object]],
    center_allowance_m: float,
) -> dict[int, tuple[int, float]]:
    """Return a maximum-cardinality, minimum-distance assignment for one date."""
    if not targets or not predictions:
        return {}
    unmatched_cost = 10_000.0
    invalid_cost = 1_000_000.0
    costs = np.full((len(targets), len(predictions) + len(targets)), invalid_cost, dtype=np.float64)
    costs[:, len(predictions) :] = unmatched_cost
    distances = np.full((len(targets), len(predictions)), np.inf, dtype=np.float64)
    for target_index, target in enumerate(targets):
        maximum_distance = target.radius_m + center_allowance_m
        for prediction_index, prediction in enumerate(predictions):
            distance = math.hypot(
                float(prediction["center_easting"]) - target.easting,
                float(prediction["center_northing"]) - target.northing,
            )
            if distance <= maximum_distance:
                costs[target_index, prediction_index] = distance
                distances[target_index, prediction_index] = distance
    target_indices, prediction_indices = linear_sum_assignment(costs)
    return {
        int(target_index): (int(prediction_index), float(distances[target_index, prediction_index]))
        for target_index, prediction_index in zip(target_indices, prediction_indices)
        if prediction_index < len(predictions) and np.isfinite(distances[target_index, prediction_index])
    }


def _summarize_seed(
    payload: dict[str, object],
    targets: list[Target],
    scenes: dict[str, Scene],
) -> dict[str, object]:
    by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for prediction in payload["predictions"]:
        by_date[str(prediction["date"])].append(prediction)
    ordered_targets = sorted(targets, key=lambda item: (item.date, item.target_id))
    matches_by_allowance: dict[float, dict[int, tuple[int, float]]] = {
        allowance: {}
        for allowance in (0.0, PRIMARY_CENTER_ALLOWANCE_METERS, WIDE_CENTER_ALLOWANCE_METERS)
    }
    for date, predictions in by_date.items():
        date_indices = [index for index, target in enumerate(ordered_targets) if target.date == date]
        date_targets = [ordered_targets[index] for index in date_indices]
        for allowance, matches in matches_by_allowance.items():
            matches.update(
                {
                    date_indices[target_index]: match
                    for target_index, match in _one_to_one_matches(
                        date_targets, predictions, allowance
                    ).items()
                }
            )
    target_results: list[dict[str, object]] = []
    for target_index, target in enumerate(ordered_targets):
        distance, confidence = _nearest_distance(target, by_date[target.date])
        strict_match = matches_by_allowance[0.0].get(target_index)
        primary_match = matches_by_allowance[PRIMARY_CENTER_ALLOWANCE_METERS].get(target_index)
        wide_match = matches_by_allowance[WIDE_CENTER_ALLOWANCE_METERS].get(target_index)
        target_results.append(
            {
                "date": target.date,
                "target_id": target.target_id,
                "material": target.material,
                "note": target.note,
                "radius_m": target.radius_m,
                "nearest_detection_distance_m": distance,
                "nearest_detection_confidence": confidence,
                "detected_within_strict_14m": strict_match is not None,
                "detected_within_primary_24m": primary_match is not None,
                "detected_within_wide_34m": wide_match is not None,
                "strict_match_distance_m": strict_match[1] if strict_match is not None else None,
                "primary_match_distance_m": primary_match[1] if primary_match is not None else None,
                "wide_match_distance_m": wide_match[1] if wide_match is not None else None,
            }
        )
    def count(material: str, field: str, nominal_only: bool = False) -> dict[str, object]:
        selected = [
            row for row in target_results
            if row["material"] == material and (not nominal_only or not row["note"])
        ]
        return _rate(sum(bool(row[field]) for row in selected), len(selected))
    primary_hdpe = count("HDPE", "detected_within_primary_24m")
    primary_hdpe_nominal = count("HDPE", "detected_within_primary_24m", nominal_only=True)
    strict_hdpe = count("HDPE", "detected_within_strict_14m")
    wide_hdpe = count("HDPE", "detected_within_wide_34m")
    primary_wood = count("wood", "detected_within_primary_24m")
    nominal_area_km2 = sum(
        scene.width * scene.height * scene.res_x_m * scene.res_y_m / 1_000_000.0
        for scene in scenes.values()
    )
    matched_primary_predictions = {
        (ordered_targets[target_index].date, prediction_index)
        for target_index, (prediction_index, _) in matches_by_allowance[
            PRIMARY_CENTER_ALLOWANCE_METERS
        ].items()
    }
    total_predictions = sum(len(predictions) for predictions in by_date.values())
    unmatched = total_predictions - len(matched_primary_predictions)
    return {
        "seed": int(payload["seed"]),
        "validation_locked_threshold": float(payload["threshold"]),
        "hdpe_target_recovery_primary_24m": primary_hdpe,
        "hdpe_target_recovery_nominal_only_24m": primary_hdpe_nominal,
        "hdpe_target_recovery_strict_sensitivity_14m": strict_hdpe,
        "hdpe_target_recovery_wide_sensitivity_34m": wide_hdpe,
        "wood_target_marine_debris_assignments_24m": primary_wood,
        "marine_debris_predictions": total_predictions,
        "unmatched_marine_debris_alerts": unmatched,
        "scene_observation_area_km2": nominal_area_km2,
        "unmatched_alerts_per_km2": float(unmatched / nominal_area_km2),
        "target_results": target_results,
    }


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# PLP2021 Controlled External Target-Transfer Check",
        "",
        "This is a controlled external check on repeated artificial floating targets in the Gulf of Gera, Lesvos. It is not an exhaustive natural-debris benchmark or a prevalence-based false-positive study.",
        "",
        "## Fixed Design",
        "",
        "- 21 Sentinel-2 acquisitions, 21 HDPE target observations, and 15 wood target observations.",
        "- The three `full_scene_ndwi7` checkpoints and their balanced thresholds were imported from the MARIDA validation-only decision lock.",
        "- Native Sentinel-2 L2A digital numbers were divided by 10,000 before the locked B4/B3/B2/B8/B11/B12 plus `(1-NDWI)/2` preprocessing and train-only normalization. This is an external radiometric domain shift relative to MARIDA's ACOLITE Rayleigh-corrected inputs.",
        "- Primary endpoint: a one-to-one Marine_Debris box-center match within 24 m (14 m target radius plus one native 10 m pixel). The 14 m and 34 m results are pre-specified sensitivities, not optimized endpoints.",
        "- The two HDPE records marked as partly/possibly underwater remain in the all-observation primary result and are separately shown in the nominal-condition sensitivity result.",
        "",
        "## Results",
        "",
        "| Seed | Validation threshold | HDPE recovery, all | HDPE recovery, nominal only | HDPE, 14 m | HDPE, 34 m | Wood assignments | Unmatched alerts/km2 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_seed"]:
        def fmt_rate(value: dict[str, object]) -> str:
            return f"{value['detected']}/{value['total']} ({100.0 * float(value['rate']):.1f}%)"
        lines.append(
            "| {seed} | {threshold:.3f} | {hdpe} | {nominal} | {strict} | {wide} | {wood} | {alerts:.3f} |".format(
                seed=row["seed"],
                threshold=row["validation_locked_threshold"],
                hdpe=fmt_rate(row["hdpe_target_recovery_primary_24m"]),
                nominal=fmt_rate(row["hdpe_target_recovery_nominal_only_24m"]),
                strict=fmt_rate(row["hdpe_target_recovery_strict_sensitivity_14m"]),
                wide=fmt_rate(row["hdpe_target_recovery_wide_sensitivity_34m"]),
                wood=fmt_rate(row["wood_target_marine_debris_assignments_24m"]),
                alerts=row["unmatched_alerts_per_km2"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- Repeated observations of the same physical targets are not treated as independent scenes for hypothesis testing.",
            "- HDPE target recovery supports transfer to this controlled artificial target, not automatic confirmation of performance on naturally occurring mixed marine litter.",
            "- Wood assignments are a known-material confusion check. Unmatched alerts cannot be called false positives because the rest of each scene is not exhaustively annotated.",
            "",
            "Sources: [PLP2021 Zenodo record](https://zenodo.org/records/7085112) and [PLP2021 project description](https://plp.aegean.gr/plastic-litter-project-2021/).",
            "",
        ]
    )
    return "\n".join(lines)


def diagnose(device: str, scene_root: Path) -> None:
    """Explanatory existence check at conf=0.001. Does not select a new operating point."""
    _assert_internal_test_is_frozen()
    if not PROTOCOL_PATH.exists() or not SUMMARY_PATH.exists():
        raise RuntimeError("Run the locked PLP operating-point stage before the diagnostic")
    targets = _load_targets()
    scenes = _load_scenes(scene_root)
    _validate_target_coverage(targets, scenes)
    _assert_native_crop_geometry(scenes.values())
    decisions = _load_locked_decisions()
    low, high, imgsz = _load_normalization(decisions)
    contract_sha = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["input_contract_sha256"]
    diag_dir = OUT_DIR / "diagnostics"
    outputs = [
        _run_seed(
            decision,
            scenes,
            low,
            high,
            imgsz,
            device,
            contract_sha,
            threshold=MIN_INFERENCE_CONF,
            output_path=diag_dir / f"seed_{int(decision['seed'])}_conf001_predictions.json",
            threshold_source="diagnostic_min_inference_conf_not_an_operating_point",
        )
        for decision in decisions
    ]
    per_seed = [
        _summarize_seed(json.loads(path.read_text(encoding="utf-8")), targets, scenes)
        for path in outputs
    ]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "explanatory PLP existence check; not a new threshold or confirmatory result",
        "threshold": MIN_INFERENCE_CONF,
        "locked_operating_point_summary": str(SUMMARY_PATH.relative_to(ROOT).as_posix()),
        "per_seed": per_seed,
    }
    _json_dump(diag_dir / "conf001_summary.json", payload)
    print(f"PLP diagnostic written: {diag_dir / 'conf001_summary.json'}")
    for row in per_seed:
        hdpe = row["hdpe_target_recovery_primary_24m"]
        print(
            f"seed={row['seed']} conf={MIN_INFERENCE_CONF} "
            f"HDPE {hdpe['detected']}/{hdpe['total']} "
            f"md_pred={row['marine_debris_predictions']}"
        )


def run(device: str, scene_root: Path) -> None:
    _assert_internal_test_is_frozen()
    targets = _load_targets()
    scenes = _load_scenes(scene_root)
    _validate_target_coverage(targets, scenes)
    _assert_native_crop_geometry(scenes.values())
    materialization = _load_materialization_provenance(scene_root, scenes)
    decisions = _load_locked_decisions()
    low, high, imgsz = _load_normalization(decisions)
    contract = _input_contract(scenes, targets, decisions, low, high, materialization)
    contract_sha = _write_or_validate_protocol(contract)
    _write_target_csv(targets, scenes)
    outputs = [
        _run_seed(decision, scenes, low, high, imgsz, device, contract_sha)
        for decision in decisions
    ]
    per_seed = [
        _summarize_seed(json.loads(path.read_text(encoding="utf-8")), targets, scenes)
        for path in outputs
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_contract_sha256": contract_sha,
        "scope": "controlled external target-transfer check; not exhaustive natural-debris confirmation",
        "per_seed": per_seed,
        "interpretation_limits": [
            "PLP2021 repeatedly observes the same artificial targets, so the target observations are not independent acquisition-scene replicates.",
            "PLP labels were not used for model, checkpoint, architecture, or threshold selection.",
            "Unmatched detections are alerts outside known HDPE target buffers, not verified false positives.",
        ],
    }
    _json_dump(SUMMARY_PATH, summary)
    SUMMARY_MARKDOWN_PATH.write_text(_markdown(summary), encoding="utf-8")
    print(f"PLP external summary written: {SUMMARY_PATH}")
    print(f"PLP external report written: {SUMMARY_MARKDOWN_PATH}")


def inspect(scene_root: Path) -> None:
    targets = _load_targets()
    scenes = _load_scenes(scene_root)
    _validate_target_coverage(targets, scenes)
    counts: dict[str, int] = defaultdict(int)
    for target in targets:
        counts[target.material] += 1
    payload = {
        "scene_count": len(scenes),
        "target_count": len(targets),
        "target_counts": dict(sorted(counts.items())),
        "notes": [asdict(target) for target in targets if target.note],
        "all_target_dates_have_scenes": True,
        "scene_crs": next(iter(scenes.values())).crs,
        "scene_resolution_m": [next(iter(scenes.values())).res_x_m, next(iter(scenes.values())).res_y_m],
        "native_crop_geometry_required_for_run": list(NATIVE_CROP_SIZE),
        "all_scenes_match_native_crop_geometry": all(
            (scene.width, scene.height) == NATIVE_CROP_SIZE for scene in scenes.values()
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "run", "diagnose"), default="inspect")
    parser.add_argument("--device", default="0")
    parser.add_argument("--scene-root", type=Path, default=SCENES_ROOT)
    args = parser.parse_args()
    if args.stage == "inspect":
        inspect(args.scene_root)
    elif args.stage == "diagnose":
        diagnose(args.device, args.scene_root)
    else:
        run(args.device, args.scene_root)


if __name__ == "__main__":
    main()
