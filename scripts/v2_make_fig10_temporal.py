"""
Figure 10: temporal sampling distribution of MARIDA v2.

The MARIDA collection is sparse and irregular — 32 distinct year-months
across 2015–2021, with a strong sampling bias toward Sep/Jan. We
therefore avoid trend-line fitting and instead report two descriptive
panels:

  Panel A: per-class GT instance count by calendar month
           (months 1-12, aggregated across all years)
  Panel B: per-class GT instance count by year
           (showing the longitudinal coverage)

Output:
  runs_v2/figures/fig10_temporal_distribution.png

Per Discussion 5.5, the paper must NOT claim seasonal patterns from
this distribution; we only show coverage, not phenomenology.

Usage:
  python scripts/v2_make_fig10_temporal.py
"""

from __future__ import annotations

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
CLASS_COLORS = {
    "Marine_Debris": "#d62728",
    "Algae_Organic": "#2ca02c",
    "Ship":          "#1f77b4",
}

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def find_tile_split(tile_name: str) -> str:
    for split in ("train", "val", "test"):
        if (V2_DIR / split / "labels" / f"{tile_name}.txt").exists():
            return split
    return "unknown"


def count_gt_per_tile(tile_name: str, split: str) -> dict[str, int]:
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


def aggregate_by_key(index, key_fn):
    """Return {key: {class: gt_count, '_tiles': tile_count}} dict."""
    out = defaultdict(lambda: {c: 0 for c in CLASS_NAMES} | {"_tiles": 0})
    for tile_name, info in index.items():
        date = info.get("date")
        if not date:
            continue
        split = find_tile_split(tile_name)
        if split == "unknown":
            continue
        key = key_fn(date)
        if key is None:
            continue
        counts = count_gt_per_tile(tile_name, split)
        for c in CLASS_NAMES:
            out[key][c] += counts[c]
        out[key]["_tiles"] += 1
    return dict(out)


def stacked_bar(ax, x_labels, data_dict, title: str, xlabel: str) -> None:
    """data_dict[label] = {class_name: count, '_tiles': N}."""
    n_x = len(x_labels)
    bottoms = np.zeros(n_x)
    bars = {}
    for cls in CLASS_NAMES:
        vals = np.array([data_dict.get(lbl, {}).get(cls, 0) for lbl in x_labels])
        bars[cls] = ax.bar(np.arange(n_x), vals, bottom=bottoms,
                            color=CLASS_COLORS[cls], edgecolor="black",
                            linewidth=0.5, alpha=0.92, label=cls.replace("_", " "))
        bottoms += vals
    # Annotate tile count on top
    tile_counts = [data_dict.get(lbl, {}).get("_tiles", 0) for lbl in x_labels]
    ymax = max(bottoms.max() if len(bottoms) else 1, 1)
    for i, t in enumerate(tile_counts):
        if t == 0:
            continue
        ax.text(i, bottoms[i] + ymax * 0.012, f"n={t}",
                ha="center", va="bottom", fontsize=7, color="#444")
    ax.set_xticks(np.arange(n_x))
    ax.set_xticklabels(x_labels, fontsize=9, rotation=0)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Ground-truth instance count", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, ymax * 1.12)


def main() -> int:
    if not INDEX_PATH.exists():
        print(f"ERROR: {INDEX_PATH} not found. Run v2_build_geo_index.py first.")
        return 1
    index = json.loads(INDEX_PATH.read_text())

    # Panel A: by month-of-year (aggregated across years)
    by_month = aggregate_by_key(index, key_fn=lambda d: int(d.split("-")[1]))
    # Panel B: by year
    by_year = aggregate_by_key(index, key_fn=lambda d: d.split("-")[0])
    years_sorted = sorted(by_year.keys())

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("Figure 10. Temporal sampling distribution of MARIDA v2 (descriptive only)",
                 fontsize=13, fontweight="bold")

    # Panel A — keys must match the labels used by stacked_bar
    month_data = {MONTH_LABELS[m - 1]: by_month.get(m, {}) for m in range(1, 13)}
    stacked_bar(axes[0], MONTH_LABELS, month_data,
                title="A. GT instance count by month-of-year (aggregated 2015-2021)",
                xlabel="Calendar month")

    # Panel B
    year_data = {y: by_year[y] for y in years_sorted}
    stacked_bar(axes[1], years_sorted, year_data,
                title="B. GT instance count by year",
                xlabel="Year")

    # Caveat note at the bottom
    fig.text(0.5, -0.025,
             "Note: 'n=' annotations are tile counts. MARIDA v2 sampling is "
             "sparse and uneven; this figure shows coverage, not seasonal "
             "phenomenology — no trend lines are fit.",
             ha="center", va="top", fontsize=9, color="#444", style="italic")

    out_path = FIG_DIR / "fig10_temporal_distribution.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
