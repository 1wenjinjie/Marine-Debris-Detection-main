"""
Figure 9 (cartopy upgrade): spatial distribution of MARIDA v2 GT
annotations with proper coastlines / land / ocean basemap.

Requires cartopy to be installed in the active Python environment.
Falls back with a clear error message if cartopy is missing.

Reuses build_dataframe / CLASS_NAMES / SPLIT_MARKERS from the matplotlib
version (v2_make_fig9_spatial_map.py).

Outputs:
  runs_v2/figures/fig9_overview_cartopy.png
  runs_v2/figures/fig9_per_class_cartopy.png
  runs_v2/figures/fig9_zoomins_cartopy.png   (regional zoom-ins)

Usage:
  python scripts/v2_make_fig9_spatial_map_cartopy.py
  python scripts/v2_make_fig9_spatial_map_cartopy.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Import data builders from the cartopy-free version
from v2_make_fig9_spatial_map import (
    CLASS_NAMES,
    CLASS_COLORS,
    SPLIT_MARKERS,
    build_dataframe,
)

FIG_DIR = ROOT / "runs_v2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _require_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except ImportError as e:
        print("ERROR: cartopy is not installed in this environment.")
        print("       Install via:  conda install -c conda-forge cartopy")
        print("       Or run the matplotlib-only fallback:")
        print("           python scripts/v2_make_fig9_spatial_map.py")
        raise SystemExit(1) from e


def _setup_geo_ax(ax, ccrs, cfeature, *, extent=None) -> None:
    """Decorate a GeoAxes with land/ocean/coastlines."""
    ax.add_feature(cfeature.OCEAN, facecolor="#dbeafe", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f5f0e6", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#444444", zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#888888",
                   linestyle=":", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.4, linestyle="--", zorder=1)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())


def _scatter_class(ax, ccrs, rows, cls, *, max_size=200, alpha=0.75):
    """Scatter all tiles for a single class on a GeoAxes."""
    color = CLASS_COLORS[cls]
    counts = np.array([r[cls] for r in rows])
    n_tiles_with_gt = int((counts > 0).sum())
    total_gt = int(counts.sum())

    # zero-count tiles in light gray to show full coverage
    zero_idx = np.where(counts == 0)[0]
    if len(zero_idx):
        ax.scatter(
            [rows[i]["lon"] for i in zero_idx],
            [rows[i]["lat"] for i in zero_idx],
            s=4, c="#bbbbbb", alpha=0.35, linewidths=0,
            transform=ccrs.PlateCarree(), zorder=2,
        )

    nonzero_idx = np.where(counts > 0)[0]
    if len(nonzero_idx):
        max_count = max(1, counts[nonzero_idx].max())
        sizes = 25 + (counts[nonzero_idx] / max_count) * (max_size - 25)
        for s_marker, marker in SPLIT_MARKERS.items():
            split_idx = [i for i in nonzero_idx if rows[i]["split"] == s_marker]
            if not split_idx:
                continue
            ax.scatter(
                [rows[i]["lon"] for i in split_idx],
                [rows[i]["lat"] for i in split_idx],
                s=[sizes[list(nonzero_idx).index(i)] for i in split_idx],
                c=color, alpha=alpha, marker=marker,
                edgecolors="black", linewidths=0.5,
                transform=ccrs.PlateCarree(), zorder=3,
                label=f"{s_marker} (n={len(split_idx)})",
            )

    ax.set_title(f"{cls.replace('_', ' ')}\n"
                 f"{n_tiles_with_gt} tiles, {total_gt} GT instances",
                 fontsize=11)
    if len(nonzero_idx):
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    return total_gt


def _global_extent(rows, padding: float = 8.0) -> list[float]:
    if not rows:
        return [-180, 180, -60, 60]
    lons = [r["lon"] for r in rows]
    lats = [r["lat"] for r in rows]
    return [
        max(-180, min(lons) - padding),
        min(180, max(lons) + padding),
        max(-90, min(lats) - padding),
        min(90, max(lats) + padding),
    ]


def make_overview(rows, out_path: Path, ccrs, cfeature) -> None:
    extent = _global_extent(rows)
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    fig.suptitle("Figure 9. MARIDA v2 spatial distribution of GT annotations",
                 fontsize=13, fontweight="bold")

    proj = ccrs.PlateCarree()
    for i, cls in enumerate(CLASS_NAMES):
        ax = fig.add_subplot(2, 2, i + 1, projection=proj)
        _setup_geo_ax(ax, ccrs, cfeature, extent=extent)
        _scatter_class(ax, ccrs, rows, cls)

    # 4th panel: split coverage
    ax = fig.add_subplot(2, 2, 4, projection=proj)
    _setup_geo_ax(ax, ccrs, cfeature, extent=extent)
    for split, marker in SPLIT_MARKERS.items():
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        ax.scatter(
            [r["lon"] for r in sub],
            [r["lat"] for r in sub],
            s=10, alpha=0.6, marker=marker, linewidths=0.3, edgecolors="black",
            transform=ccrs.PlateCarree(), zorder=3,
            label=f"{split} (n={len(sub)})",
        )
    ax.set_title(f"All tiles by split (total n={len(rows)})", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def make_per_class(rows, out_path: Path, ccrs, cfeature) -> None:
    extent = _global_extent(rows)
    fig = plt.figure(figsize=(16, 5.5), constrained_layout=True)
    fig.suptitle("Figure 9 (per-class). MARIDA v2 GT spatial distribution",
                 fontsize=12, fontweight="bold")
    proj = ccrs.PlateCarree()
    for i, cls in enumerate(CLASS_NAMES):
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        _setup_geo_ax(ax, ccrs, cfeature, extent=extent)
        _scatter_class(ax, ccrs, rows, cls)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# Pre-defined regions of interest for zoom-ins (lon_min, lon_max, lat_min, lat_max)
# Picked by inspecting fig9_summary.json + general MARIDA literature.
ROI_REGIONS = [
    ("SE Asia & Indonesia", [95, 130, -15, 25]),
    ("Caribbean & Gulf", [-100, -75, 8, 25]),
    ("Mediterranean & N Sea", [-10, 40, 30, 60]),
]


def make_zoomins(rows, out_path: Path, ccrs, cfeature) -> None:
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    fig.suptitle("Figure 9 (regional zoom-ins). High-density MARIDA v2 sampling areas",
                 fontsize=12, fontweight="bold")
    proj = ccrs.PlateCarree()
    for i, (name, extent) in enumerate(ROI_REGIONS):
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        _setup_geo_ax(ax, ccrs, cfeature, extent=extent)
        # Plot all 3 classes overlaid in this region
        for cls in CLASS_NAMES:
            counts = np.array([r[cls] for r in rows])
            idx = [j for j in range(len(rows))
                   if counts[j] > 0
                   and extent[0] <= rows[j]["lon"] <= extent[1]
                   and extent[2] <= rows[j]["lat"] <= extent[3]]
            if not idx:
                continue
            sizes = 25 + counts[idx] / max(1, counts[idx].max()) * 175
            ax.scatter(
                [rows[j]["lon"] for j in idx],
                [rows[j]["lat"] for j in idx],
                s=sizes, c=CLASS_COLORS[cls], alpha=0.7,
                edgecolors="black", linewidths=0.5,
                transform=ccrs.PlateCarree(), zorder=3,
                label=f"{cls.replace('_', ' ')} (n={len(idx)})",
            )
        ax.set_title(name, fontsize=11)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Cartopy-based Figure 9 generator")
    ap.add_argument("--split", default="", choices=["", "train", "val", "test"])
    ap.add_argument("--out-dir", default=str(FIG_DIR))
    args = ap.parse_args()

    ccrs, cfeature = _require_cartopy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.split}" if args.split else ""

    print("Building rows...")
    rows = build_dataframe(split_filter=args.split or None)["rows"]
    print(f"  {len(rows)} tiles loaded")

    if not rows:
        print("No tiles after filtering; aborting.")
        return 1

    print("\nWriting cartopy figures...")
    make_overview(rows, out_dir / f"fig9_overview_cartopy{suffix}.png", ccrs, cfeature)
    make_per_class(rows, out_dir / f"fig9_per_class_cartopy{suffix}.png", ccrs, cfeature)
    make_zoomins(rows, out_dir / f"fig9_zoomins_cartopy{suffix}.png", ccrs, cfeature)

    return 0


if __name__ == "__main__":
    sys.exit(main())
