"""
Figure 5: ablation study bar chart.

Reads structure_ablation_summary.json + engineering_ablation_summary.json
and produces a single multi-panel bar chart comparing 5 structure-ablation
runs across 4 key metrics (mAP50, Marine_Debris AP50, Algae AP50, Ship AP50)
with 3-seed std error bars.

A second figure focuses on the engineering pipeline (Table 2) to show the
debris TP recovery story.

Outputs:
  runs_v2/figures/fig5_structure_ablation_bars.png
  runs_v2/figures/fig5_engineering_pipeline_bars.png

Usage:
  python scripts/v2_make_fig5_ablation_bars.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs_v2"
FIG_DIR = RUNS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURE_JSON = RUNS_DIR / "structure_ablation_summary.json"
ENGINEERING_JSON = RUNS_DIR / "engineering_ablation_summary.json"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run the corresponding summary script first."
        )
    return json.loads(path.read_text())


def _bar_panel(ax, names, means, stds, *, ylabel: str, baseline_idx: int = 0,
               highlight_color: str = "#d9ead3",
               proposed_color: str = "#ffd966") -> None:
    """One panel: vertical bars with optional std error, baseline highlighted."""
    colors = []
    for i in range(len(names)):
        if i == baseline_idx:
            colors.append("#7fb3d5")  # baseline blue
        else:
            colors.append(proposed_color)

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="#333333", linewidth=0.8, alpha=0.92)
    # Annotate values on top of bars
    for bar, m, s in zip(bars, means, stds):
        h = bar.get_height()
        if s > 0:
            label = f"{m:.3f}±{s:.3f}"
        else:
            label = f"{m:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + (max(means) * 0.025 if max(means) > 0 else 0.005),
                label, ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if max(means) > 0:
        ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.18)


def make_structure_figure() -> None:
    table = _load(STRUCTURE_JSON)
    rows = table["rows"]
    if not rows:
        print("No rows in structure summary.")
        return
    names = [r["display_name"] for r in rows]

    metrics = [
        ("mAP50", "mAP@0.5"),
        ("marine_debris_ap50", "Marine_Debris AP@0.5"),
        ("algae_organic_ap50", "Algae_Organic AP@0.5"),
        ("ship_ap50", "Ship AP@0.5"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    fig.suptitle(
        "Figure 5. Structure ablation on MARIDA v2 test set (imgsz=512, BCE, 3-seed)",
        fontsize=13, fontweight="bold",
    )
    for ax, (key, label) in zip(axes.flat, metrics):
        means = [r.get(key) or 0.0 for r in rows]
        stds = [r.get(f"{key}_std") or 0.0 for r in rows]
        _bar_panel(ax, names, means, stds, ylabel=label, baseline_idx=0)
        ax.set_title(label, fontsize=11)

    out = FIG_DIR / "fig5_structure_ablation_bars.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def make_engineering_figure() -> None:
    table = _load(ENGINEERING_JSON)
    rows = table["rows"]
    if not rows:
        print("No rows in engineering summary.")
        return
    names = [r["display_name"] for r in rows]

    # Two panels: left = mAP50, right = Debris TP at conf=0.05 (the headline story)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(
        "Figure 5b. Engineering pipeline on MARIDA v2 (Marine_Debris recovery)",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: mAP50
    means = [r.get("mAP50") or 0.0 for r in rows]
    stds = [r.get("mAP50_std") or 0.0 for r in rows]
    _bar_panel(axes[0], names, means, stds, ylabel="mAP@0.5", baseline_idx=0)
    axes[0].set_title("Overall mAP@0.5 (decreases as model focuses on rare debris)",
                       fontsize=10)

    # Panel 2: Debris TP@0.05 (the headline)
    tp_means, tp_stds = [], []
    for r in rows:
        d = r.get("debris_tp_by_conf", {}).get("0.05")
        if d and d.get("values"):
            tp_means.append(d.get("mean", d["values"][0]))
            tp_stds.append(d.get("std", 0.0))
        else:
            tp_means.append(0.0)
            tp_stds.append(0.0)
    _bar_panel(axes[1], names, tp_means, tp_stds,
                ylabel="Debris TP @ conf=0.05  (out of 13)", baseline_idx=0)
    axes[1].set_title("Marine_Debris recovery at practical conf=0.05",
                       fontsize=10)
    axes[1].axhline(13, color="#cc3333", linewidth=0.8, linestyle=":",
                     alpha=0.6, label="all 13 instances")
    axes[1].legend(loc="upper left", fontsize=8)

    out = FIG_DIR / "fig5_engineering_pipeline_bars.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def main() -> int:
    print("Generating ablation bar charts...")
    make_structure_figure()
    make_engineering_figure()
    return 0


if __name__ == "__main__":
    sys.exit(main())
