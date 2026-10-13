"""表 3: per-tile detection performance bucketed by distance-to-coast.

Reads `docs/paper/v1_test_per_tile.csv` (output of v1_per_tile_test_inference.py),
computes distance to nearest coastline (Natural Earth 1:50M) for each tile,
buckets into coastal (<10 km) / nearshore (10-50 km) / offshore (>50 km),
and outputs:

- Markdown table: `docs/paper/v1_coast_distance_table.md`
- JSON detail:    `docs/paper/v1_coast_distance_table.json`
- Per-tile CSV with distance column: `docs/paper/v1_test_per_tile_with_coast.csv`

Caveats (write into table caption):
- 73 test tiles split into 3 buckets — per-bucket sample size is small
- Descriptive only; do NOT claim "statistically significant difference between buckets"
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points, unary_union
from pyproj import Geod
import cartopy.io.shapereader as shpreader

IN_CSV = ROOT / "docs/paper/v1_test_per_tile.csv"
OUT_CSV = ROOT / "docs/paper/v1_test_per_tile_with_coast.csv"
OUT_MD = ROOT / "docs/paper/v1_coast_distance_table.md"
OUT_JSON = ROOT / "docs/paper/v1_coast_distance_table.json"

BUCKETS = [
    ("coastal (<5 km)", 0.0, 5.0),
    ("nearshore (5-20 km)", 5.0, 20.0),
    ("offshore (>20 km)", 20.0, np.inf),
]


def load_coastline_union():
    print("[geo] loading Natural Earth 1:50M coastline ...", flush=True)
    shp_path = shpreader.natural_earth(resolution="50m", category="physical", name="coastline")
    reader = shpreader.Reader(shp_path)
    geoms = [rec.geometry for rec in reader.records() if rec.geometry is not None]
    print(f"[geo] {len(geoms)} coastline features, building unary_union ...", flush=True)
    union = unary_union(geoms)
    print(f"[geo] union geometry type: {union.geom_type}", flush=True)
    return union


def distance_to_coast_km(lat: float, lon: float, coast, geod: Geod) -> float:
    p = Point(lon, lat)
    near = nearest_points(p, coast)[1]
    _, _, dist_m = geod.inv(lon, lat, near.x, near.y)
    return dist_m / 1000.0


def aggregate_bucket(sub: pd.DataFrame) -> dict:
    tp = int(sub["n_tp"].sum())
    fp = int(sub["n_fp"].sum())
    fn = int(sub["n_fn"].sum())
    n_gt = int(sub["n_gt"].sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "n_tiles": len(sub),
        "n_gt": n_gt,
        "n_tp": tp,
        "n_fp": fp,
        "n_fn": fn,
        "P": round(p, 3),
        "R": round(r, 3),
        "F1": round(f1, 3),
        "dist_min_km": round(float(sub["dist_coast_km"].min()), 1) if len(sub) else None,
        "dist_max_km": round(float(sub["dist_coast_km"].max()), 1) if len(sub) else None,
        "dist_median_km": round(float(sub["dist_coast_km"].median()), 1) if len(sub) else None,
    }


def main():
    df = pd.read_csv(IN_CSV)
    print(f"[load] {len(df)} test tiles from {IN_CSV.name}")

    coast = load_coastline_union()
    geod = Geod(ellps="WGS84")

    print("[geo] computing distance-to-coast for each tile ...", flush=True)
    distances = []
    for i, row in df.iterrows():
        d = distance_to_coast_km(float(row["lat"]), float(row["lon"]), coast, geod)
        distances.append(d)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(df)}]", flush=True)
    df["dist_coast_km"] = distances
    df.to_csv(OUT_CSV, index=False)
    print(f"[save] {OUT_CSV}")

    print("\n[summary] distance distribution (km):")
    print(df["dist_coast_km"].describe().to_string())

    buckets = {}
    for label, lo, hi in BUCKETS:
        sub = df[(df["dist_coast_km"] >= lo) & (df["dist_coast_km"] < hi)]
        buckets[label] = aggregate_bucket(sub)

    overall = aggregate_bucket(df)

    payload = {
        "source_csv": str(IN_CSV.relative_to(ROOT)),
        "n_tiles": len(df),
        "conf_thr": float(df["conf_thr"].iloc[0]) if len(df) else None,
        "bucket_definitions": [
            {"label": label, "min_km": float(lo) if np.isfinite(lo) else None,
             "max_km": float(hi) if np.isfinite(hi) else None}
            for label, lo, hi in BUCKETS
        ],
        "buckets": buckets,
        "overall": overall,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] {OUT_JSON}")

    md_lines = [
        "# Table 3: Marine_Debris detection by distance-to-coast bucket",
        "",
        f"**Source:** {IN_CSV.relative_to(ROOT)} (BiFPN @ imgsz=1024, seed 0; "
        f"deployment conf τ = {payload['conf_thr']}; IoU ≥ 0.5)",
        "",
        f"**Coastline reference:** Natural Earth 1:50M physical/coastline",
        "",
        f"**Caveat:** 73 test tiles split into 3 buckets — per-bucket samples are small. "
        f"Numbers are descriptive; do **not** claim statistical significance between buckets.",
        "",
        "| Bucket | n tiles | n GT | TP | FP | FN | P | R | F1 | dist median (km) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, lo, hi in BUCKETS:
        b = buckets[label]
        md_lines.append(
            f"| {label} | {b['n_tiles']} | {b['n_gt']} | {b['n_tp']} | "
            f"{b['n_fp']} | {b['n_fn']} | {b['P']:.3f} | {b['R']:.3f} | {b['F1']:.3f} | "
            f"{b['dist_median_km']} |"
        )
    md_lines.append(
        f"| **all (reference)** | **{overall['n_tiles']}** | **{overall['n_gt']}** | "
        f"**{overall['n_tp']}** | **{overall['n_fp']}** | **{overall['n_fn']}** | "
        f"**{overall['P']:.3f}** | **{overall['R']:.3f}** | **{overall['F1']:.3f}** | "
        f"{overall['dist_median_km']} |"
    )

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[save] {OUT_MD}")

    print("\n[summary] per-bucket F1:")
    for label, _, _ in BUCKETS:
        b = buckets[label]
        print(f"  {label:<22}  n={b['n_tiles']:>3}  GT={b['n_gt']:>4}  "
              f"F1={b['F1']:.3f}  (P={b['P']:.3f} R={b['R']:.3f})")


if __name__ == "__main__":
    main()
