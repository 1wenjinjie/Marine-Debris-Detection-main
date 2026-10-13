"""
Figures 3 and 4: training-curve comparisons across models.

Reads each run's results.csv (Ultralytics' standard training log) and
produces two paper-ready figures with side-by-side ablation views:

  fig3_training_curves_structure.png   (mAP@0.5 / mAP@0.5:0.95 / P / R)
  fig3_training_curves_engineering.png (same metrics, engineering rows)
  fig4_training_curves_combined.png    (compact summary, two columns)

The structure view contrasts the 5 structure-ablation runs at imgsz=512;
the engineering view contrasts baseline / +imgsz=1024 / +focal+1024.

Usage:
  python scripts/v2_make_fig3_training_curves.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs_v2"
FIG_DIR = RUNS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Each entry: (display name, run dir name, color, linestyle, linewidth, zorder)
STRUCTURE_RUNS = [
    ("Baseline (BCE, 512)",     "baseline_6ch_s0", "#1f77b4", "-",  2.2, 5),
    ("+CBAM",                   "cbam_v2_s0",      "#ff7f0e", "-",  1.6, 4),
    ("+BiFPN-lite",             "bifpn_v2_s0",     "#2ca02c", "-",  1.6, 4),
    ("+EIoU",                   "eiou_v2_s0",      "#9467bd", "-",  1.6, 4),
    ("Full (CBAM+BiFPN+EIoU)",  "full_v2_s0",      "#8c564b", "-",  1.6, 4),
]

ENGINEERING_RUNS = [
    ("Baseline (BCE, 512)",         "baseline_6ch_s0",     "#1f77b4", "-",  2.0, 4),
    ("+ imgsz=1024 (BCE)",          "imgsz1024_s0",        "#17becf", "-",  1.8, 4),
    ("+ focal γ=2.0 (i1024)",       "focal_g2.0_i1024_s0", "#d62728", "-",  2.4, 5),
]

METRIC_COLS = {
    "mAP@0.5":      "metrics/mAP50(B)",
    "mAP@0.5:0.95": "metrics/mAP50-95(B)",
    "Precision":    "metrics/precision(B)",
    "Recall":       "metrics/recall(B)",
}


def _load_curve(run_dir_name: str) -> pd.DataFrame | None:
    csv_path = RUNS_DIR / run_dir_name / "results.csv"
    if not csv_path.exists():
        print(f"  [WARN] missing: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _plot_metric_panel(ax, runs, metric_label: str, metric_col: str) -> None:
    for name, dirname, color, ls, lw, z in runs:
        df = _load_curve(dirname)
        if df is None or metric_col not in df.columns:
            continue
        ax.plot(df["epoch"], df[metric_col], color=color, linestyle=ls,
                linewidth=lw, zorder=z, label=name, alpha=0.92)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_title(metric_label, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _make_panel_figure(runs, suptitle: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (label, col) in zip(axes.flat, METRIC_COLS.items()):
        _plot_metric_panel(ax, runs, label, col)
    # One legend for the entire figure (placed below)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 4), fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def _make_combined_summary(out_path: Path) -> None:
    """Two-column compact summary for Figure 4: mAP@0.5 evolution side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    _plot_metric_panel(axes[0], STRUCTURE_RUNS,
                        "Structure ablation (imgsz=512, BCE)",
                        "metrics/mAP50(B)")
    axes[0].set_title("Structure ablation: mAP@0.5 vs epoch", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.85)
    _plot_metric_panel(axes[1], ENGINEERING_RUNS,
                        "Engineering pipeline",
                        "metrics/mAP50(B)")
    axes[1].set_title("Engineering pipeline: mAP@0.5 vs epoch", fontsize=11)
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.85)
    fig.suptitle("Figure 4. Training-curve mAP@0.5 across model variants",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main() -> int:
    print("Generating training-curve comparison figures...")
    _make_panel_figure(STRUCTURE_RUNS,
                        "Figure 3a. Structure ablation training curves (imgsz=512, BCE)",
                        FIG_DIR / "fig3_training_curves_structure.png")
    _make_panel_figure(ENGINEERING_RUNS,
                        "Figure 3b. Engineering pipeline training curves",
                        FIG_DIR / "fig3_training_curves_engineering.png")
    _make_combined_summary(FIG_DIR / "fig4_training_curves_combined.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
