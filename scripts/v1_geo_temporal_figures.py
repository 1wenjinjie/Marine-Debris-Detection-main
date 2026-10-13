"""Spatial + temporal distribution figures (老师 plan 图 9 / 图 10).

图 9 (spatial): cartopy world-map scatter of all v1-honest tiles, colored by split,
                marker size proportional to Marine_Debris GT count per tile.
图 10 (temporal): two-panel bar chart — by month-of-year (left) and by year (right),
                  stacked by split. Descriptive only; sparse sampling precludes
                  formal trend analysis.

Both read tile names from the v1 v2split manifests and look up center lat/lon and
date in `data/processed/marida_v2/tile_geo_index.json` (same MARIDA tiles).
GT counts read directly from YOLO label files (class 0 = Marine_Debris in v1 7-class).

No inference required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

MANIFEST_ROOT = ROOT / "data/processed/mask_bbox_rescue_split_physidx_v2split"
IMAGE_DATA_ROOT = ROOT / "data/processed/mask_bbox_rescue_split_physidx"
GEO_INDEX_PATH = ROOT / "data/processed/marida_v2/tile_geo_index.json"
FIG_DIR = ROOT / "docs/paper/figures"
CSV_OUT = ROOT / "docs/paper/v1_tile_geo_temporal.csv"

SPLITS = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
SPLIT_COLOR = {"train": "tab:blue", "val": "tab:orange", "test": "tab:red"}
MARINE_DEBRIS_CLS = 0


def load_per_tile_data() -> pd.DataFrame:
    geo = json.loads(GEO_INDEX_PATH.read_text(encoding="utf-8"))
    rows = []
    missing_geo = []
    for split, fname in SPLITS.items():
        manifest = MANIFEST_ROOT / fname
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            tile_name = Path(line).stem
            label_path = IMAGE_DATA_ROOT / line.replace("/images/", "/labels/").replace(".npy", ".txt")
            n_debris = 0
            if label_path.exists():
                for L in label_path.read_text(encoding="utf-8").splitlines():
                    parts = L.split()
                    if parts and int(parts[0]) == MARINE_DEBRIS_CLS:
                        n_debris += 1
            g = geo.get(tile_name)
            if g is None:
                missing_geo.append(tile_name)
                continue
            lat, lon = g["center_latlon"]
            rows.append({
                "tile": tile_name,
                "split": split,
                "lat": float(lat),
                "lon": float(lon),
                "date": g["date"],
                "n_debris": n_debris,
            })
    df = pd.DataFrame(rows)
    if missing_geo:
        print(f"[warn] {len(missing_geo)} tiles missing from geo index "
              f"(first 5: {missing_geo[:5]})")
    return df


def plot_spatial(df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(14, 7), dpi=140)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f3eee4", edgecolor="#7a7a7a", linewidth=0.4)
    ax.add_feature(cfeature.OCEAN, facecolor="#e6eef5")
    ax.add_feature(cfeature.COASTLINE, edgecolor="#555", linewidth=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, color="gray")

    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        sizes = np.maximum(10, sub["n_debris"].values * 8 + 12)
        n_debris_total = int(sub["n_debris"].sum())
        ax.scatter(
            sub["lon"], sub["lat"],
            s=sizes, c=SPLIT_COLOR[split],
            alpha=0.65, edgecolor="black", linewidth=0.4,
            label=f"{split} ({len(sub)} tiles, {n_debris_total} debris)",
            transform=ccrs.PlateCarree(), zorder=4,
        )

    ax.set_title(
        "Spatial distribution of MARIDA tiles (7-class label set, scene-clean split)\n"
        "marker size ∝ Marine_Debris GT count per tile",
        fontsize=11,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_path}")


def plot_temporal(df: pd.DataFrame, out_path: Path) -> None:
    df = df.copy()
    df["dt"] = pd.to_datetime(df["date"])
    df["month"] = df["dt"].dt.month
    df["year"] = df["dt"].dt.year

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    colors = [SPLIT_COLOR[s] for s in ("train", "val", "test")]

    by_month = (
        df.groupby(["month", "split"]).size().unstack(fill_value=0)
        .reindex(range(1, 13), fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
    )
    by_month.plot(kind="bar", stacked=True, ax=axes[0], color=colors,
                  edgecolor="black", width=0.85)
    axes[0].set_xlabel("Month of year")
    axes[0].set_ylabel("Tile count")
    axes[0].set_title("(a) Tile count by month (aggregated across years)")
    axes[0].set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        rotation=0, fontsize=9,
    )
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(title="split", fontsize=9)

    by_year = (
        df.groupby(["year", "split"]).size().unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
    )
    by_year.plot(kind="bar", stacked=True, ax=axes[1], color=colors,
                 edgecolor="black", width=0.85)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Tile count")
    axes[1].set_title("(b) Tile count by year")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(title="split", fontsize=9)

    fig.suptitle(
        "Temporal distribution of MARIDA tiles "
        "(descriptive; sparse sampling precludes formal trend analysis)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_per_tile_data()
    print(f"loaded {len(df)} tiles")
    print(df["split"].value_counts())
    print(f"total Marine_Debris GT across all splits: {int(df['n_debris'].sum())}")

    df.to_csv(CSV_OUT, index=False)
    print(f"[save] {CSV_OUT}")

    plot_spatial(df, FIG_DIR / "v1_spatial_distribution.png")
    plot_temporal(df, FIG_DIR / "v1_temporal_distribution.png")


if __name__ == "__main__":
    main()
