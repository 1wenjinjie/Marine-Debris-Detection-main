"""
Build a geographic index mapping each MARIDA v2 .npy tile to lat/lon
metadata extracted from the original Sentinel-2 GeoTIFFs.

Output: data/processed/marida_v2/tile_geo_index.json
{
  "S2_20-4-18_30VWH_4": {
    "scene": "S2_20-4-18_30VWH",
    "date": "2018-04-20",
    "tile_index": 4,
    "crs": "EPSG:32630",
    "shape": [256, 256],
    "px_size_m": 10.0,
    "utm_bounds": [left, bottom, right, top],
    "center_latlon": [lat, lon],
    "corner_latlon": {
      "tl": [lat, lon], "tr": [lat, lon],
      "bl": [lat, lon], "br": [lat, lon]
    }
  },
  ...
}

Usage:
  python scripts/v2_build_geo_index.py
  python scripts/v2_build_geo_index.py --tile S2_20-4-18_30VWH_4   # one tile
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import rasterio
from rasterio.warp import transform as rio_transform

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "MARIDA" / "patches"
V2_DIR = ROOT / "data" / "processed" / "marida_v2"
OUT_PATH = V2_DIR / "tile_geo_index.json"

SCENE_DATE_RE = re.compile(r"^S2_(\d+)-(\d+)-(\d+)_([A-Z0-9]+)$")


def parse_scene_date(scene: str) -> str:
    """Convert 'S2_20-4-18_30VWH' -> '2018-04-20'."""
    m = SCENE_DATE_RE.match(scene)
    if not m:
        return ""
    d, mo, y2, _ = m.groups()
    year = 2000 + int(y2) if int(y2) < 70 else 1900 + int(y2)
    return f"{year:04d}-{int(mo):02d}-{int(d):02d}"


def split_tile_name(tile_name: str) -> tuple[str, int]:
    """'S2_20-4-18_30VWH_4' -> ('S2_20-4-18_30VWH', 4)."""
    m = re.match(r"^(S2_\d+-\d+-\d+_[A-Z0-9]+)_(\d+)$", tile_name)
    if not m:
        raise ValueError(f"Cannot parse tile name: {tile_name}")
    return m.group(1), int(m.group(2))


def index_one_tile(tile_name: str) -> dict | None:
    """Read original .tif and return geo metadata; returns None if tif missing."""
    scene, idx = split_tile_name(tile_name)
    tif = RAW_DIR / scene / f"{tile_name}.tif"
    if not tif.exists():
        return None

    with rasterio.open(tif) as src:
        crs = src.crs
        h, w = src.shape
        bounds = src.bounds
        cx, cy = src.xy(h // 2, w // 2)
        # Corners in UTM (top-left, top-right, bottom-left, bottom-right pixel centers)
        corners_px = {
            "tl": (0, 0),
            "tr": (0, w - 1),
            "bl": (h - 1, 0),
            "br": (h - 1, w - 1),
        }
        corner_utm = {k: src.xy(r, c) for k, (r, c) in corners_px.items()}
        # Reproject center + corners to WGS84 in one batched call
        xs = [cx] + [corner_utm[k][0] for k in ("tl", "tr", "bl", "br")]
        ys = [cy] + [corner_utm[k][1] for k in ("tl", "tr", "bl", "br")]
        lons, lats = rio_transform(crs, "EPSG:4326", xs, ys)

        return {
            "scene": scene,
            "date": parse_scene_date(scene),
            "tile_index": idx,
            "crs": str(crs),
            "shape": [h, w],
            "px_size_m": float(abs(src.transform.a)),
            "utm_bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "center_latlon": [float(lats[0]), float(lons[0])],
            "corner_latlon": {
                "tl": [float(lats[1]), float(lons[1])],
                "tr": [float(lats[2]), float(lons[2])],
                "bl": [float(lats[3]), float(lons[3])],
                "br": [float(lats[4]), float(lons[4])],
            },
        }


def list_v2_tiles() -> list[str]:
    """Collect all unique tile names from train/val/test image dirs."""
    names: set[str] = set()
    for split in ("train", "val", "test"):
        d = V2_DIR / split / "images"
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix == ".npy":
                names.add(p.stem)
    return sorted(names)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build geo index for MARIDA v2 tiles")
    ap.add_argument("--tile", default="", help="Index a single tile name (debug)")
    ap.add_argument("--out", default=str(OUT_PATH), help="Output JSON path")
    args = ap.parse_args()

    if args.tile:
        info = index_one_tile(args.tile)
        if info is None:
            print(f"ERROR: original .tif not found for {args.tile}")
            return 1
        print(json.dumps({args.tile: info}, indent=2))
        return 0

    tiles = list_v2_tiles()
    if not tiles:
        print(f"ERROR: no .npy files found under {V2_DIR}")
        return 1

    print(f"Indexing {len(tiles)} tiles from {V2_DIR}...")
    index: dict[str, dict] = {}
    missing: list[str] = []
    for i, name in enumerate(tiles, 1):
        info = index_one_tile(name)
        if info is None:
            missing.append(name)
            continue
        index[name] = info
        if i % 100 == 0:
            print(f"  {i}/{len(tiles)} indexed...")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2))

    print(f"\nDone.")
    print(f"  indexed: {len(index)} / {len(tiles)}")
    print(f"  missing: {len(missing)}")
    if missing:
        print(f"  first missing: {missing[:5]}")
    print(f"  output:  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
