"""
Figure 6: FPS-vs-mAP@0.5 trade-off scatter.

Combines structure-ablation and engineering-pipeline rows into a single
scatter plot. Each point is one model variant; horizontal axis is the
FPS median (RTX 3090 Ti, batch=1, after warmup), vertical axis is
mAP@0.5 mean across seeds with std as the y-error bar.

Output:
  runs_v2/figures/fig6_fps_map_tradeoff.png

Usage:
  python scripts/v2_make_fig6_fps_map_tradeoff.py
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


def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def collect_points():
    """Return list of (label, fps, mAP_mean, mAP_std, group, marker)."""
    points = []

    # ---- Structure ablation rows ----
    structure = load_json(RUNS_DIR / "structure_ablation_summary.json") or {}
    fps_struct = load_json(RUNS_DIR / "fps_bench_summary_structure_ablation.json") or {}

    for row in structure.get("rows", []):
        prefix = row["run_prefix"]
        sample_run = f"{prefix}_s0"
        fps_info = fps_struct.get(sample_run, {})
        fps = fps_info.get("fps_median")
        if fps is None:
            continue
        points.append({
            "label":     row["display_name"],
            "fps":       fps,
            "map_mean":  row.get("mAP50") or 0.0,
            "map_std":   row.get("mAP50_std") or 0.0,
            "group":     "structure",
            "marker":    "o",
            "is_baseline": prefix == "baseline_6ch",
        })

    # ---- Engineering rows ----
    engineering = load_json(RUNS_DIR / "engineering_ablation_summary.json") or {}
    fps_eng = load_json(RUNS_DIR / "fps_bench_summary_engineering.json") or {}

    for row in engineering.get("rows", []):
        # Avoid duplicating baseline (already in structure list)
        display = row["display_name"]
        if display.startswith("Baseline"):
            continue
        # Platt is a post-hoc score remap, identical inference cost AND mAP
        # to the focal row -- it would render as an exact overlap point.
        if "Platt" in display:
            continue
        # Engineering rows don't have run_prefix in spec; pick by imgsz lookup
        imgsz = row.get("imgsz")
        # find a run in fps_eng matching imgsz that is NOT baseline
        match_run = None
        for run_name, info in fps_eng.items():
            if info.get("imgsz") == imgsz and run_name != "baseline_6ch_s0":
                # avoid double-count: prefer focal_g2.0 over imgsz1024 when display says focal
                if "focal" in display and "focal" in run_name:
                    match_run = run_name
                    break
                if "imgsz=1024" in display and run_name == "imgsz1024_s0":
                    match_run = run_name
                    break
        if match_run is None:
            # fallback: any matching imgsz
            for run_name, info in fps_eng.items():
                if info.get("imgsz") == imgsz:
                    match_run = run_name
                    break
        fps = fps_eng.get(match_run, {}).get("fps_median") if match_run else None
        if fps is None:
            continue

        # Platt has same FPS as focal (post-hoc, doesn't change inference)
        points.append({
            "label":     display,
            "fps":       fps,
            "map_mean":  row.get("mAP50") or 0.0,
            "map_std":   row.get("mAP50_std") or 0.0,
            "group":     "engineering",
            "marker":    "s",
            "is_baseline": False,
        })

    return points


def main() -> int:
    points = collect_points()
    if not points:
        print("No points collected; aborting.")
        return 1

    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)

    # Color scheme: structure runs in shades of orange, engineering in shades of green
    color_struct = "#ff7f0e"
    color_eng = "#2ca02c"
    color_baseline = "#1f77b4"

    for p in points:
        if p["is_baseline"]:
            color = color_baseline
            edge = "black"
        elif p["group"] == "structure":
            color = color_struct
            edge = "#a04000"
        else:
            color = color_eng
            edge = "#16691a"

        ax.errorbar(p["fps"], p["map_mean"], yerr=p["map_std"],
                    fmt=p["marker"], markersize=11, color=color,
                    markeredgecolor=edge, markeredgewidth=1.0,
                    capsize=4, elinewidth=1.0, alpha=0.92, zorder=3)
        # Annotation just to the right of the marker
        ax.annotate(p["label"],
                    xy=(p["fps"], p["map_mean"]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color="#222", va="center", ha="left",
                    zorder=4)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Baseline",
                   markerfacecolor=color_baseline, markersize=10,
                   markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", label="Structure ablation",
                   markerfacecolor=color_struct, markersize=10,
                   markeredgecolor="#a04000"),
        plt.Line2D([0], [0], marker="s", color="w", label="Engineering pipeline",
                   markerfacecolor=color_eng, markersize=10,
                   markeredgecolor="#16691a"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.92)

    ax.set_xlabel("FPS (median, RTX 3090 Ti, batch=1)", fontsize=11)
    ax.set_ylabel("mAP@0.5 (3-seed mean, ± std)", fontsize=11)
    ax.set_title("Figure 6. Inference-speed vs detection-quality trade-off",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a faint gridline at baseline mAP for visual reference
    base_pt = next((p for p in points if p["is_baseline"]), None)
    if base_pt is not None:
        ax.axhline(base_pt["map_mean"], color="#1f77b4", linewidth=0.6,
                   linestyle=":", alpha=0.5, zorder=1)
        ax.text(ax.get_xlim()[1] * 0.99, base_pt["map_mean"] + 0.005,
                f"baseline mAP={base_pt['map_mean']:.3f}",
                fontsize=8, color="#1f77b4", ha="right", alpha=0.8)

    out_path = FIG_DIR / "fig6_fps_map_tradeoff.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
