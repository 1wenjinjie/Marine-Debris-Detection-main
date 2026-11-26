"""Materialize native 256x256 PLP2021 crops from same-date Sentinel-2 L2A COGs.

The bundled PLP2021 GeoTIFFs are 134x133 pixels. Resizing them before inference
would alter the physical scale seen by the MARIDA model, so this utility queries
the matching original Sentinel-2 L2A item for every labelled date and writes a
native 10 m, 256x256 crop centred on the date-specific HDPE target. The output
is deliberately labelled as an external L2A radiometric transfer condition; it
is not silently presented as MARIDA's ACOLITE Rayleigh-corrected input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mpb.run_plp2021_external_validation import (
    CHANNEL_NAMES,
    NATIVE_CROP_SIZE,
    PLP_ROOT,
    TARGET_SUPPORT_FILES,
    Target,
    _load_targets,
    _sha256,
)

LOCAL_SCENES_ROOT = PLP_ROOT / "Sentinel-2"
OUT_ROOT = ROOT / "data" / "PLP" / "PLP2021_native256"
OUT_SCENES = OUT_ROOT / "Sentinel-2"
MANIFEST_PATH = OUT_ROOT / "materialization_manifest.json"
STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
COLLECTION = "sentinel-2-l2a"
ASSETS = {
    "B4": "B04",
    "B3": "B03",
    "B2": "B02",
    "B8": "B08",
    "B11": "B11",
    "B12": "B12",
}
TEN_METER_ASSETS = {"B4", "B3", "B2", "B8"}
TARGET_CRS = "EPSG:32635"
SOURCE_RADIOMETRY = (
    "Sentinel-2 Level-2A digital numbers from Microsoft Planetary Computer; "
    "the external runner divides them by 10000. This remains a radiometric domain shift "
    "from MARIDA ACOLITE Rayleigh-corrected imagery."
)


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _http_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method, url, timeout=90, **kwargs)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return payload


def _target_by_date() -> dict[str, Target]:
    grouped: dict[str, list[Target]] = {}
    for target in _load_targets():
        if target.material == "HDPE":
            grouped.setdefault(target.date, []).append(target)
    if not grouped:
        raise RuntimeError("PLP2021 labels have no HDPE targets")
    duplicate_dates = {date: values for date, values in grouped.items() if len(values) != 1}
    if duplicate_dates:
        raise RuntimeError(f"Expected exactly one HDPE target per date: {duplicate_dates}")
    return {date: values[0] for date, values in grouped.items()}


def _expected_mgrs_tile() -> str:
    tiles = set()
    for path in LOCAL_SCENES_ROOT.glob("*.tif"):
        match = re.search(r"_T(?P<tile>\d{2}[A-Z]{3})\.tif$", path.name)
        if match is not None:
            tiles.add(match.group("tile"))
    if len(tiles) != 1:
        raise RuntimeError(f"Could not establish one PLP2021 MGRS tile from local scenes: {sorted(tiles)}")
    return next(iter(tiles))


def _item_for_target(target: Target, expected_tile: str) -> dict[str, Any]:
    longitude, latitude = transform(TARGET_CRS, "EPSG:4326", [target.easting], [target.northing])
    day = f"{target.date[:4]}-{target.date[4:6]}-{target.date[6:]}"
    payload = _http_json(
        "POST",
        STAC_SEARCH_URL,
        json={
            "collections": [COLLECTION],
            "bbox": [longitude[0] - 0.01, latitude[0] - 0.01, longitude[0] + 0.01, latitude[0] + 0.01],
            "datetime": f"{day}T00:00:00Z/{day}T23:59:59Z",
            "limit": 10,
        },
    )
    features = payload.get("features")
    if not isinstance(features, list):
        raise RuntimeError(f"Planetary Computer STAC response has no features for {target.date}")
    matches = [
        feature
        for feature in features
        if feature.get("properties", {}).get("s2:mgrs_tile") == expected_tile
    ]
    if len(matches) != 1:
        identifiers = [str(feature.get("id")) for feature in features]
        raise RuntimeError(
            f"Expected one {expected_tile} Sentinel-2 L2A item for {target.date}; got {identifiers}"
        )
    item = matches[0]
    assets = item.get("assets")
    if not isinstance(assets, dict) or any(asset not in assets for asset in ASSETS.values()):
        raise RuntimeError(f"STAC item lacks one of {sorted(ASSETS.values())}: {item.get('id')}")
    return item


def _signed_href(asset_href: str) -> str:
    payload = _http_json("GET", SIGN_URL, params={"href": asset_href})
    signed = payload.get("href")
    if not isinstance(signed, str) or not signed:
        raise RuntimeError("Planetary Computer signing response has no href")
    return signed


def _crop_from_item(item: dict[str, Any], target: Target) -> tuple[np.ndarray, rasterio.Affine, Any, dict[str, str]]:
    assets = item["assets"]
    raw_hrefs = {channel: str(assets[asset]["href"]) for channel, asset in ASSETS.items()}
    signed_hrefs = {channel: _signed_href(href) for channel, href in raw_hrefs.items()}
    height, width = NATIVE_CROP_SIZE
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(signed_hrefs["B4"]) as reference:
            if reference.crs is None or str(reference.crs) != TARGET_CRS:
                raise RuntimeError(f"Unexpected B04 CRS for {item['id']}: {reference.crs}")
            if not math.isclose(abs(reference.res[0]), 10.0, abs_tol=1e-6) or not math.isclose(
                abs(reference.res[1]), 10.0, abs_tol=1e-6
            ):
                raise RuntimeError(f"B04 is not a native 10 m asset: {item['id']}")
            row, column = reference.index(target.easting, target.northing)
            window = Window(column - width // 2, row - height // 2, width, height)
            if window.col_off < 0 or window.row_off < 0 or window.col_off + width > reference.width or window.row_off + height > reference.height:
                raise RuntimeError(f"Native crop would exceed source bounds for {item['id']}")
            output_transform = reference.window_transform(window)
            arrays: list[np.ndarray] = []
            for channel in CHANNEL_NAMES:
                with rasterio.open(signed_hrefs[channel]) as source:
                    if source.crs != reference.crs:
                        raise RuntimeError(f"Band CRS mismatch for {channel} in {item['id']}")
                    if channel in TEN_METER_ASSETS:
                        if source.transform != reference.transform or source.width != reference.width:
                            raise RuntimeError(f"Native grid mismatch for {channel} in {item['id']}")
                        array = source.read(1, window=window)
                    else:
                        with WarpedVRT(
                            source,
                            crs=reference.crs,
                            transform=output_transform,
                            width=width,
                            height=height,
                            resampling=Resampling.bilinear,
                        ) as warped:
                            array = warped.read(1)
                    if array.shape != (height, width) or not np.isfinite(array).all():
                        raise RuntimeError(f"Invalid crop array for {channel} in {item['id']}")
                    arrays.append(array.astype(np.uint16, copy=False))
            stack = np.stack(arrays, axis=0)
            return stack, output_transform, reference.crs, raw_hrefs


def _output_name(item: dict[str, Any], date: str, tile: str) -> str:
    timestamp = str(item.get("properties", {}).get("datetime", ""))
    compact = re.sub(r"[^0-9]", "", timestamp)[:14]
    if len(compact) != 14 or not compact.startswith(date):
        raise RuntimeError(f"Unexpected STAC datetime for {item.get('id')}: {timestamp!r}")
    return f"{compact}_PLP2021_native256_T{tile}.tif"


def _write_scene(
    output_path: Path,
    stack: np.ndarray,
    transform_: rasterio.Affine,
    crs: Any,
    item: dict[str, Any],
    target: Target,
    raw_hrefs: dict[str, str],
) -> None:
    temporary = output_path.with_suffix(".partial.tif")
    if temporary.exists():
        temporary.unlink()
    profile = {
        "driver": "GTiff",
        "height": stack.shape[1],
        "width": stack.shape[2],
        "count": stack.shape[0],
        "dtype": "uint16",
        "crs": crs,
        "transform": transform_,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(stack)
        destination.descriptions = CHANNEL_NAMES
        destination.update_tags(
            dataset="PLP2021",
            materialization_method="same-date native 10 m crop from Planetary Computer STAC Sentinel-2 L2A",
            reflectance_units="DN_divide_by_10000",
            source_radiometry=SOURCE_RADIOMETRY,
            source_stac_id=str(item["id"]),
            source_datetime=str(item.get("properties", {}).get("datetime")),
            crop_center_easting_m=f"{target.easting:.6f}",
            crop_center_northing_m=f"{target.northing:.6f}",
            B11_B12_resampling="bilinear to B04 10 m grid",
        )
        for index, channel in enumerate(CHANNEL_NAMES, start=1):
            destination.update_tags(index, source_asset_href=raw_hrefs[channel])
    temporary.replace(output_path)
    with rasterio.open(output_path) as check:
        if (
            (check.width, check.height) != NATIVE_CROP_SIZE
            or check.count != len(CHANNEL_NAMES)
            or tuple(check.descriptions) != CHANNEL_NAMES
            or check.tags().get("reflectance_units") != "DN_divide_by_10000"
        ):
            raise RuntimeError(f"Post-write verification failed: {output_path}")


def _scene_file_hashes(paths: list[Path]) -> dict[str, str]:
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in sorted(paths)}


def inspect() -> None:
    targets = _target_by_date()
    expected_tile = _expected_mgrs_tile()
    items = {
        date: _item_for_target(target, expected_tile)
        for date, target in sorted(targets.items())
    }
    print(
        json.dumps(
            {
                "target_dates": len(targets),
                "expected_mgrs_tile": expected_tile,
                "native_crop_pixels": list(NATIVE_CROP_SIZE),
                "source_collection": COLLECTION,
                "items": {
                    date: {
                        "id": item["id"],
                        "datetime": item.get("properties", {}).get("datetime"),
                    }
                    for date, item in items.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def materialize(overwrite: bool) -> None:
    targets = _target_by_date()
    expected_tile = _expected_mgrs_tile()
    OUT_SCENES.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    outputs: list[Path] = []
    for index, (date, target) in enumerate(sorted(targets.items()), start=1):
        item = _item_for_target(target, expected_tile)
        output_path = OUT_SCENES / _output_name(item, date, expected_tile)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing crop: {output_path}")
        stack, transform_, crs, raw_hrefs = _crop_from_item(item, target)
        _write_scene(output_path, stack, transform_, crs, item, target, raw_hrefs)
        outputs.append(output_path)
        records.append(
            {
                "date": date,
                "target_id": target.target_id,
                "target_material": target.material,
                "target_easting_m": target.easting,
                "target_northing_m": target.northing,
                "source_stac_id": item["id"],
                "source_datetime": item.get("properties", {}).get("datetime"),
                "source_asset_hrefs": raw_hrefs,
                "output_file": output_path.relative_to(ROOT).as_posix(),
                "output_sha256": _sha256(output_path),
                "output_transform": list(transform_),
            }
        )
        print(f"PLP native crop: {index}/{len(targets)} {date}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "PLP2021",
        "source_collection": COLLECTION,
        "source_radiometry": SOURCE_RADIOMETRY,
        "native_crop_pixels": list(NATIVE_CROP_SIZE),
        "native_resolution_m": 10.0,
        "channel_order": list(CHANNEL_NAMES),
        "reflectance_units": "DN_divide_by_10000",
        "scene_count": len(outputs),
        "label_support_files_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path) for path in TARGET_SUPPORT_FILES
        },
        "output_scene_files_sha256": _scene_file_hashes(outputs),
        "records": records,
    }
    _json_dump(MANIFEST_PATH, manifest)
    print(f"PLP native-crop manifest written: {MANIFEST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "materialize"), default="inspect")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage == "inspect":
        inspect()
    else:
        materialize(args.overwrite)


if __name__ == "__main__":
    main()
