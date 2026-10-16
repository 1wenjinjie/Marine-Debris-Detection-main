"""Audit locally verifiable MADOS/MARIDA tile, footprint, and acquisition overlap."""

import argparse
import csv
import json
import re
import warnings
from collections import Counter
from datetime import date
from pathlib import Path

import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform_bounds


ROOT = Path(__file__).resolve().parents[2]
MADOS_CROP = re.compile(r"^Scene_(\d+)_L2R_cl_(\d+)\.tif$", re.IGNORECASE)
MARIDA_PATCH = re.compile(r"^S2_(\d{1,2})-(\d{1,2})-(\d{2})_([0-9]{2}[A-Z]{3})_(\d+)\.tif$", re.IGNORECASE)
MGRS = re.compile(r"(?:^|[^A-Z0-9])(?:T)?([0-9]{2}[C-HJ-NP-X][A-HJ-NP-Z]{2})(?:$|[^A-Z0-9])", re.IGNORECASE)
DATE = re.compile(r"(?<!\d)(20\d{2})[-_/]?([01]?\d)[-_/]?([0-3]?\d)(?!\d)")
FIELDS = ["dataset", "split", "scene", "crop_or_patch", "source_path", "date", "date_source", "mgrs_tile", "tile_source", "bounds_wgs84", "geography_source", "geography_issue", "date_issue", "tile_issue"]
PAIR_FIELDS = ["layer", "mados_split", "mados_scene", "mados_crop", "mados_path", "marida_scene", "marida_patch", "marida_path", "mados_date", "marida_date", "mgrs_tile", "intersection_area_deg2", "bbox_iou"]


def _valid_date(text):
    for match in DATE.finditer(str(text)):
        try:
            return date(*map(int, match.groups())).isoformat()
        except ValueError:
            pass
    return None


def _tile(text):
    match = MGRS.search(str(text))
    return match.group(1).upper() if match else None


def _valid_bounds(bounds):
    if bounds is None or len(bounds) != 4:
        return False
    west, south, east, north = bounds
    return -180 <= west < east <= 180 and -90 <= south < north <= 90


def _read_geotiff(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as image:
            tags = image.tags()
            if image.crs is None:
                return tags, None, "missing_crs"
            if image.transform.is_identity:
                return tags, None, "identity_transform_no_usable_footprint"
            try:
                bounds = transform_bounds(image.crs, "EPSG:4326", *image.bounds, densify_pts=21)
            except Exception as exc:
                return tags, None, f"coordinate_transform_failed:{type(exc).__name__}"
            if not _valid_bounds(bounds):
                return tags, None, "invalid_wgs84_bounds"
            return tags, [round(float(value), 9) for value in bounds], None


def _tag_metadata(tags, kind):
    if kind == "date":
        preferred = ("date", "acquisition", "sensing", "datetime", "timestamp", "product")
        parser = _valid_date
    else:
        preferred = ("mgrs", "tile", "product")
        parser = _tile
    for key, value in tags.items():
        if any(token in key.lower() for token in preferred):
            found = parser(value)
            if found:
                return found, f"GeoTIFF tag:{key}"
    return None, None


def _load_scene_metadata(path):
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        scene = str(row.get("scene", "")).strip()
        if not re.fullmatch(r"Scene_\d+", scene):
            raise ValueError(f"Invalid scene in {path}: {scene!r}")
        if scene in result:
            raise ValueError(f"Duplicate scene in {path}: {scene}")
        raw_date = str(row.get("date") or "").strip()
        raw_tile = str(row.get("mgrs_tile") or "").strip().upper()
        if raw_date and _valid_date(raw_date) != raw_date:
            raise ValueError(f"Invalid ISO date for {scene}: {raw_date}")
        if raw_tile and not re.fullmatch(r"[0-9]{2}[C-HJ-NP-X][A-HJ-NP-Z]{2}", raw_tile):
            raise ValueError(f"Invalid MGRS tile for {scene}: {raw_tile}")
        result[scene] = {"date": raw_date or None, "mgrs_tile": raw_tile or None}
    return result


def _mados_records(root, scene_metadata, metadata_path):
    split_files = {"train": "train_X.txt", "val": "val_X.txt", "test": "test_X.txt"}
    records = []
    seen = set()
    for split, filename in split_files.items():
        manifest = root / "splits" / filename
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        for crop_id in manifest.read_text(encoding="utf-8-sig").splitlines():
            crop_id = crop_id.strip()
            if not crop_id:
                continue
            match = re.fullmatch(r"Scene_(\d+)_(\d+)", crop_id)
            if not match:
                raise ValueError(f"Invalid crop ID in {manifest}: {crop_id}")
            if crop_id in seen:
                raise ValueError(f"Duplicate crop across official splits: {crop_id}")
            seen.add(crop_id)
            scene = f"Scene_{match.group(1)}"
            path = root / scene / "10" / f"{scene}_L2R_cl_{match.group(2)}.tif"
            tags, bounds, geography_issue = _read_geotiff(path) if path.is_file() else ({}, None, "missing_class_tif")
            metadata = scene_metadata.get(scene, {})
            tif_date, tif_date_source = _tag_metadata(tags, "date")
            tif_tile, tif_tile_source = _tag_metadata(tags, "tile")
            known_date = metadata.get("date") or tif_date
            known_tile = metadata.get("mgrs_tile") or tif_tile
            records.append({
                "dataset": "MADOS", "split": split, "scene": scene,
                "crop_or_patch": crop_id, "source_path": str(path.resolve()),
                "date": known_date, "date_source": str(metadata_path.resolve()) if metadata.get("date") else tif_date_source,
                "mgrs_tile": known_tile, "tile_source": str(metadata_path.resolve()) if metadata.get("mgrs_tile") else tif_tile_source,
                "bounds_wgs84": bounds, "geography_source": str(path.resolve()) if bounds else None,
                "geography_issue": geography_issue, "date_issue": None if known_date else "no_acquisition_date_or_scene_mapping",
                "tile_issue": None if known_tile else "no_mgrs_tile_or_scene_mapping",
            })
    return records


def _marida_records(root):
    records = []
    for path in sorted(root.rglob("*.tif")):
        if path.name.endswith(("_cl.tif", "_conf.tif")):
            continue
        match = MARIDA_PATCH.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unrecognised MARIDA patch name: {path}")
        day, month, year, tile, patch = match.groups()
        acquisition = date(2000 + int(year), int(month), int(day)).isoformat()
        tags, bounds, geography_issue = _read_geotiff(path)
        scene = path.parent.name
        records.append({
            "dataset": "MARIDA", "split": "raw_all", "scene": scene,
            "crop_or_patch": path.stem, "source_path": str(path.resolve()),
            "date": acquisition, "date_source": "patch_filename", "mgrs_tile": tile.upper(),
            "tile_source": "patch_filename", "bounds_wgs84": bounds,
            "geography_source": str(path.resolve()) if bounds else None,
            "geography_issue": geography_issue, "date_issue": None, "tile_issue": None,
        })
    if not records:
        raise FileNotFoundError(f"No MARIDA source patches found under {root}")
    return records


def _intersection(first, second):
    west = max(first[0], second[0])
    south = max(first[1], second[1])
    east = min(first[2], second[2])
    north = min(first[3], second[3])
    if west >= east or south >= north:
        return None
    intersection = (east - west) * (north - south)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection, intersection / (first_area + second_area - intersection)


def _pairs(mados_test, marida):
    matches = []
    for mados in mados_test:
        for reference in marida:
            same_tile = bool(mados["mgrs_tile"] and reference["mgrs_tile"] and mados["mgrs_tile"] == reference["mgrs_tile"])
            spatial = _intersection(mados["bounds_wgs84"], reference["bounds_wgs84"]) if mados["bounds_wgs84"] and reference["bounds_wgs84"] else None
            same_day_location = bool(spatial and mados["date"] and reference["date"] and mados["date"] == reference["date"])
            for layer, matched in (("same_mgrs_tile", same_tile), ("spatial_intersection", bool(spatial)), ("same_day_same_location", same_day_location)):
                if not matched:
                    continue
                matches.append({
                    "layer": layer, "mados_split": mados["split"], "mados_scene": mados["scene"],
                    "mados_crop": mados["crop_or_patch"], "mados_path": mados["source_path"],
                    "marida_scene": reference["scene"], "marida_patch": reference["crop_or_patch"],
                    "marida_path": reference["source_path"], "mados_date": mados["date"],
                    "marida_date": reference["date"], "mgrs_tile": mados["mgrs_tile"] if same_tile else None,
                    "intersection_area_deg2": round(spatial[0], 12) if spatial else None,
                    "bbox_iou": round(spatial[1], 9) if spatial else None,
                })
    return matches


def _layer_summary(mados_test, marida, pairs):
    required = {
        "same_mgrs_tile": ("mgrs_tile",),
        "spatial_intersection": ("bounds_wgs84",),
        "same_day_same_location": ("date", "bounds_wgs84"),
    }
    result = {}
    for layer, fields in required.items():
        mados_checkable = sum(all(record[field] for field in fields) for record in mados_test)
        marida_checkable = sum(all(record[field] for field in fields) for record in marida)
        matched = sum(pair["layer"] == layer for pair in pairs)
        result[layer] = {
            "mados_test_checkable": mados_checkable,
            "mados_test_unverifiable": len(mados_test) - mados_checkable,
            "marida_checkable": marida_checkable,
            "comparable_pairs": mados_checkable * marida_checkable,
            "candidate_pairs": matched,
            "conclusion": "unverifiable" if mados_checkable == 0 else "candidate_overlap_found" if matched else "partially_verified_no_candidate" if mados_checkable < len(mados_test) else "no_candidate_in_checked_test",
        }
    return result


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})


def run(mados_root, marida_root, output_root, scene_metadata_path=None):
    if not mados_root.is_dir() or not marida_root.is_dir():
        raise FileNotFoundError("MADOS and MARIDA input directories must exist")
    scene_metadata = _load_scene_metadata(scene_metadata_path)
    mados = _mados_records(mados_root, scene_metadata, scene_metadata_path)
    marida = _marida_records(marida_root)
    test = [record for record in mados if record["split"] == "test"]
    pairs = _pairs(test, marida)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "mados_crops.csv", mados, FIELDS)
    _write_csv(output_root / "marida_patches.csv", marida, FIELDS)
    _write_csv(output_root / "candidate_pairs.csv", pairs, PAIR_FIELDS)
    summary = {
        "inputs": {
            "mados_root": str(mados_root.resolve()), "marida_root": str(marida_root.resolve()),
            "scene_metadata_csv": str(scene_metadata_path.resolve()) if scene_metadata_path else None,
            "mados_official_crops": len(mados), "mados_by_split": dict(Counter(record["split"] for record in mados)),
            "mados_scenes": len({record["scene"] for record in mados}),
            "marida_raw_patches": len(marida), "marida_scenes": len({record["scene"] for record in marida}),
        },
        "scope": "Official MADOS test_X crops versus all local raw MARIDA patches; all official MADOS crops are checked for metadata quality.",
        "layers": _layer_summary(test, marida, pairs),
        "missing_mados_test": {
            field: dict(Counter(record[field] for record in test if record[field]))
            for field in ("geography_issue", "date_issue", "tile_issue")
        },
        "missing_marida": {
            field: dict(Counter(record[field] for record in marida if record[field]))
            for field in ("geography_issue", "date_issue", "tile_issue")
        },
        "interpretation": [
            "EPSG:4326 paired with an identity transform is not a usable geographic footprint.",
            "An unverifiable layer is not evidence of zero overlap or confirmed geographic independence.",
            "MADOS source publication reports using MARIDA for its Marine Debris component; dataset-level independence cannot be assumed from different file names.",
            "Candidate pairs and missing reasons with source paths are in the companion CSV files.",
        ],
    }
    path = output_root / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mados-root", type=Path, default=ROOT / "data" / "MADOS")
    parser.add_argument("--marida-root", type=Path, default=ROOT / "data" / "raw" / "MARIDA" / "patches")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs_mpb" / "overlap_audit")
    parser.add_argument("--scene-metadata", type=Path, help="Optional CSV with scene,date,mgrs_tile columns for independently verified MADOS scene metadata")
    args = parser.parse_args()
    path, summary = run(args.mados_root, args.marida_root, args.output_root, args.scene_metadata)
    print(f"Overlap audit written: {path}")
    for layer, result in summary["layers"].items():
        print(f"{layer}: {result['conclusion']} ({result['mados_test_checkable']}/{summary['inputs']['mados_by_split'].get('test', 0)} MADOS test crops checkable; {result['candidate_pairs']} candidate pairs)")


if __name__ == "__main__":
    main()
