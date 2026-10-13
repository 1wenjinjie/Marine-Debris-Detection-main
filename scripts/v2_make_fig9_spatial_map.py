"""
Figure 9: spatial distribution of MARIDA v2 ground-truth annotations.

Reads tile_geo_index.json + label files, plots tile centers in lat/lon
space with per-class scatter (size = GT count, color = class). Produces:

  runs_v2/figures/fig9_overview.png       4-panel overview (3 classes + split)
  runs_v2/figures/fig9_per_class.png      single-row 3-panel, paper-ready
  runs_v2/figures/fig9_summary.json       per-tile counts table

Pure matplotlib (no cartopy / network basemap). Axes are degrees;
coastline overlay can be added later if cartopy is approved.

Usage:
  python scripts/v2_make_fig9_spatial_map.py
  python scripts/v2_make_fig9_spatial_map.py --split test     # only test tiles
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "data" / "processed" / "marida_v2"
INDEX_PATH = V2_DIR / "tile_geo_index.json"
FIG_DIR = ROOT / "runs_v2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Marine_Debris", "Algae_Organic", "Ship"]
CLASS_COLORS = {"Marine_Debris": "#d62728", "Algae_Organic": "#2ca02c", "Ship": "#1f77b4"}
SPLIT_MARKERS = {"train": "o", "val": "s", "test": "^"}


def find_tile_split(tile_name: str) -> str:
    for split in ("train", "val", "test"):
        if (V2_DIR / split / "labels" / f"{tile_name}.txt").exists():
            return split
    return "unknown"


def count_gt_per_tile(tile_name: str, split: str) -> dict[str, int]:
    """Count GT boxes per class for one tile from its YOLO-txt label file."""
    label_path = V2_DIR / split / "labels" / f"{tile_name}.txt"
    counts = {c: 0 for c in CLASS_NAMES}
    if not label_path.exists():
        return counts
    text = label_path.read_text().strip()
    if not text:
        return counts
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue
        if 0 <= cls_id < len(CLASS_NAMES):
            counts[CLASS_NAMES[cls_id]] += 1
    return counts


def build_dataframe(split_filter: str | None = None) -> dict:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Geo index missing: {INDEX_PATH}. "
                                f"Run scripts/v2_build_geo_index.py first.")
    index = json.loads(INDEX_PATH.read_text())

    rows = []
    for tile_name, info in index.items():
        split = find_tile_split(tile_name)
        if split == "unknown":
            continue
        if split_filter and split != split_filter:
            continue
        counts = count_gt_per_tile(tile_name, split)
        lat, lon = info["center_latlon"]
        rows.append({
            "tile": tile_name,
            "scene": info["scene"],
            "split": split,
            "lat": lat,
            "lon": lon,
            **counts,
            "total_gt": sum(counts.values()),
        })
    return {"rows": rows}


def annotate_global_extent(ax, rows: list[dict], padding: float = 5.0) -> None:
    if not rows:
        return
    lats = [r["lat"] for r in rows]
    lons = [r["lon"] for r in rows]
    ax.set_xlim(min(lons) - padding, max(lons) + padding)
    ax.set_ylim(min(lats) - padding, max(lats) + padding)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.set_aspect("auto")


def plot_class_panel(ax, rows: list[dict], cls: str, *, title: str = None, max_size: float = 200) -> int:
    """Scatter all tiles; size proportional to GT count of `cls`, color by class."""
    color = CLASS_COLORS[cls]
    counts = np.array([r[cls] for r in rows])
    n_tiles_with_gt = int((counts > 0).sum())
    total_gt = int(counts.sum())

    # background: zero-count tiles (light gray small dots)
    zero_mask = counts == 0
    if zero_mask.any():
        ax.scatter(
            [rows[i]["lon"] for i in range(len(rows)) if zero_mask[i]],
            [rows[i]["lat"] for i in range(len(rows)) if zero_mask[i]],
            s=4, c="#cccccc", alpha=0.4, linewidths=0, zorder=1,
        )

    # foreground: non-zero tiles, size scales with count
    nonzero_idx = np.where(counts > 0)[0]
    if len(nonzero_idx):
        max_count = max(1, counts[nonzero_idx].max())
        sizes = 20 + (counts[nonzero_idx] / max_count) * (max_size - 20)
        for s_marker, marker in SPLIT_MARKERS.items():
            split_idx = [i for i in nonzero_idx if rows[i]["split"] == s_marker]
            if not split_idx:
                continue
            ax.scatter(
                [rows[i]["lon"] for i in split_idx],
                [rows[i]["lat"] for i in split_idx],
                s=[sizes[list(nonzero_idx).index(i)] for i in split_idx],
                c=color, alpha=0.7, marker=marker,
                edgecolors="black", linewidths=0.5, zorder=2,
                label=f"{s_marker} (n={len(split_idx)})",
            )
    annotate_global_extent(ax, rows)
    if title is None:
        title = f"{cls.replace('_', ' ')}\n{n_tiles_with_gt} tiles, {total_gt} GT instances"
    ax.set_title(title, fontsize=11)
    if len(nonzero_idx):
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    return total_gt


def plot_split_panel(ax, rows: list[dict]) -> None:
    for split, marker in SPLIT_MARKERS.items():
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        ax.scatter(
            [r["lon"] for r in sub],
            [r["lat"] for r in sub],
            s=10, alpha=0.6, marker=marker,
            label=f"{split} (n={len(sub)})", linewidths=0.3, edgecolors="black",
        )
    annotate_global_extent(ax, rows)
    ax.set_title(f"All tiles by split (total n={len(rows)})", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)


def make_overview_figure(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    plot_class_panel(axes[0, 0], rows, "Marine_Debris")
    plot_class_panel(axes[0, 1], rows, "Algae_Organic")
    plot_class_panel(axes[1, 0], rows, "Ship")
    plot_split_panel(axes[1, 1], rows)
    fig.suptitle("Figure 9. MARIDA v2 spatial distribution of ground-truth annotations",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def make_perclass_figure(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    for ax, cls in zip(axes, CLASS_NAMES):
        plot_class_panel(ax, rows, cls)
    fig.suptitle("Figure 9 (per-class). Spatial distribution of MARIDA v2 GT instances",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot MARIDA v2 GT spatial distribution (Figure 9)")
    ap.add_argument("--split", default="", choices=["", "train", "val", "test"],
                    help="Restrict to one split. Default: all splits.")
    ap.add_argument("--out-dir", default=str(FIG_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building rows...")
    data = build_dataframe(split_filter=args.split or None)
    rows = data["rows"]
    print(f"  {len(rows)} tiles loaded")

    if not rows:
        print("No tiles after filtering; aborting.")
        return 1

    suffix = f"_{args.split}" if args.split else ""
    print("\nWriting figures...")
    make_overview_figure(rows, out_dir / f"fig9_overview{suffix}.png")
    make_perclass_figure(rows, out_dir / f"fig9_per_class{suffix}.png")

    summary_path = out_dir / f"fig9_summary{suffix}.json"
    summary_path.write_text(json.dumps({
        "n_tiles": len(rows),
        "total_gt_by_class": {c: sum(r[c] for r in rows) for c in CLASS_NAMES},
        "tiles_with_gt_by_class": {
            c: sum(1 for r in rows if r[c] > 0) for c in CLASS_NAMES
        },
        "split_counts": {s: sum(1 for r in rows if r["split"] == s) for s in SPLIT_MARKERS},
        "rows": rows,
    }, indent=2))
    print(f"  saved {summary_path.name}")

    print("\nClass totals:")
    for c in CLASS_NAMES:
        n_gt = sum(r[c] for r in rows)
        n_tiles = sum(1 for r in rows if r[c] > 0)
        print(f"  {c:18s}: {n_gt:>5d} GT instances across {n_tiles:>4d} tiles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
