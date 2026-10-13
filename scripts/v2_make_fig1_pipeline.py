"""
Figure 1: overall pipeline of the proposed marine debris detector for
MARIDA v2.

Final pipeline (route C decision, 2026-04-30):
  baseline YOLOv11n + focal loss (gamma=2.0) + imgsz=1024 + post-hoc
  Platt calibration. No CBAM/BiFPN/EIoU -- those proved unstable on
  3-seed evaluation and are reported in the structure-ablation table.

Pure matplotlib (no graphviz / mermaid / drawio). Produces:

  runs_v2/figures/fig1_pipeline.png
  runs_v2/figures/fig1_pipeline.svg

Usage:
  python scripts/v2_make_fig1_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "runs_v2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Color palette
C_INPUT = "#cfe2f3"     # light blue
C_PREPROC = "#fff2cc"   # light yellow
C_BACKBONE = "#d9ead3"  # light green
C_NECK = "#ead1dc"      # light pink
C_HEAD = "#f4cccc"      # light salmon
C_OUTPUT = "#d0e0e3"    # light teal
C_POSTHOC = "#fce5cd"   # light orange (Platt is core, not optional)
C_HIGHLIGHT = "#ffd966" # gold for proposed components
C_TEXT = "#333333"
C_ARROW = "#666666"


def add_box(ax, x, y, w, h, label, *, fc=C_INPUT, ec="#444444",
            sub: str | None = None, fontsize: int = 10, sub_fontsize: int = 8,
            dashed: bool = False) -> None:
    style = "round,pad=0.02,rounding_size=0.05"
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=style,
        linewidth=1.4 if not dashed else 1.0,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(box)
    if sub:
        ax.text(x + w / 2, y + h * 0.62, label,
                ha="center", va="center", fontsize=fontsize, color=C_TEXT, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub,
                ha="center", va="center", fontsize=sub_fontsize, color=C_TEXT, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize, color=C_TEXT, fontweight="bold")


def add_arrow(ax, x1, y1, x2, y2, *, dashed: bool = False) -> None:
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_length=10,head_width=6",
        linewidth=1.5,
        color=C_ARROW,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(arrow)


def add_callout(ax, x, y, w, h, label, *, fc=C_HIGHLIGHT, ec="#806600",
                fontsize: int = 8) -> None:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize, color=C_TEXT, fontweight="bold")


def main() -> int:
    fig, ax = plt.subplots(figsize=(10, 12.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.5)
    ax.set_aspect("equal")
    ax.axis("off")

    cx = 5.0
    bw = 5.6
    bh = 0.85
    gap = 0.42
    bx = cx - bw / 2

    y = 12.0

    # 1. Input
    add_box(ax, bx, y - bh, bw, bh,
            "Sentinel-2 L2A multispectral tile",
            sub="(H × W × 13 bands, float32)",
            fc=C_INPUT)
    y -= bh + gap

    # 2. Channel selection
    add_box(ax, bx, y - bh, bw, bh,
            "Channel selection: B4, B3, B2, B8, B11, B12",
            sub="(6 channels: RGB + NIR + 2× SWIR)",
            fc=C_INPUT)
    y -= bh + gap

    # 3. Preprocessing
    add_box(ax, bx, y - bh, bw, bh,
            "Per-channel p01/p99 normalization → [0, 1]",
            sub="(spectral-safe; no HSV / mosaic / mixup)",
            fc=C_PREPROC)
    y -= bh + gap

    # 4. Resolution choice (HIGHLIGHT - proposed)
    res_h = 0.95
    box = FancyBboxPatch(
        (bx, y - res_h), bw, res_h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.6, edgecolor="#806600", facecolor=C_HIGHLIGHT,
    )
    ax.add_patch(box)
    ax.text(cx, y - res_h * 0.42, "Train at imgsz = 1024 (vs default 512)",
            ha="center", va="center", fontsize=10, color=C_TEXT, fontweight="bold")
    ax.text(cx, y - res_h * 0.78, "preserves small-target spatial signal (12-14 px debris)",
            ha="center", va="center", fontsize=8, color=C_TEXT, style="italic")
    y -= res_h + gap

    # 5. Backbone (vanilla)
    backbone_h = 0.95
    add_box(ax, bx, y - backbone_h, bw, backbone_h,
            "YOLOv11n backbone",
            sub="(C3k2 + SPPF + C2PSA, no attention modules)",
            fc=C_BACKBONE, fontsize=10, sub_fontsize=8)
    y -= backbone_h + gap

    # 6. Neck
    add_box(ax, bx, y - bh, bw, bh,
            "PANet neck (top-down + bottom-up)",
            sub="(P3 / P4 / P5, baseline topology)",
            fc=C_NECK)
    y -= bh + gap

    # 7. Detect head + focal (HIGHLIGHT - proposed)
    head_h = 1.15
    box = FancyBboxPatch(
        (bx, y - head_h), bw, head_h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.6, edgecolor="#806600", facecolor=C_HIGHLIGHT,
    )
    ax.add_patch(box)
    ax.text(cx, y - 0.30, "Detect head (3 scales: stride 8 / 16 / 32)",
            ha="center", va="center", fontsize=10, color=C_TEXT, fontweight="bold")
    ax.text(cx, y - 0.62, "Focal loss (γ=2.0, α=0.25) replacing BCE",
            ha="center", va="center", fontsize=9, color=C_TEXT, fontweight="bold")
    ax.text(cx, y - 0.92, "(suppresses easy-negative gradient dominance for rare debris)",
            ha="center", va="center", fontsize=8, color=C_TEXT, style="italic")
    y -= head_h + gap

    # 8. Output
    output_h = 1.1
    add_box(ax, bx, y - output_h, bw, output_h,
            "Detections per tile",
            sub="3 classes: Marine_Debris, Algae_Organic, Ship\n(boxes + raw scores)",
            fc=C_OUTPUT, fontsize=10, sub_fontsize=8)
    y -= output_h + gap

    # 9. Platt calibration (HIGHLIGHT - proposed, post-hoc but core)
    platt_h = 1.0
    box = FancyBboxPatch(
        (bx, y - platt_h), bw, platt_h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.6, edgecolor="#806600", facecolor=C_HIGHLIGHT,
    )
    ax.add_patch(box)
    ax.text(cx, y - 0.32, "Post-hoc Platt scaling on val-set TP/FP",
            ha="center", va="center", fontsize=10, color=C_TEXT, fontweight="bold")
    ax.text(cx, y - 0.65, "p_calibrated = σ(a · logit(p_raw) + b)",
            ha="center", va="center", fontsize=9, color=C_TEXT, family="monospace")
    ax.text(cx, y - 0.88, "(makes conf=0.1 a usable deployment threshold)",
            ha="center", va="center", fontsize=8, color=C_TEXT, style="italic")

    # Arrows
    stops = [
        (12.0,           12.0 - bh),
        (12.0 - bh - gap, 12.0 - bh - gap - bh),
        (12.0 - 2*(bh+gap), 12.0 - 2*(bh+gap) - bh),
        (12.0 - 3*(bh+gap), 12.0 - 3*(bh+gap) - res_h),
    ]
    # Re-compute precisely from layout above
    # Simpler: store y values as we go
    ys_top = [12.0]
    ys_h = [bh, bh, bh, res_h, backbone_h, bh, head_h, output_h, platt_h]
    cur = 12.0
    pairs = []
    for h_ in ys_h:
        top = cur
        bot = cur - h_
        pairs.append((top, bot))
        cur = bot - gap

    for i in range(len(pairs) - 1):
        _, bot = pairs[i]
        top_next, _ = pairs[i + 1]
        add_arrow(ax, cx, bot, cx, top_next)

    # Legend (small, top-left)
    legend_x, legend_y = 0.1, 0.4
    legend_items = [
        (C_INPUT, "Input / channel selection"),
        (C_PREPROC, "Preprocessing"),
        (C_HIGHLIGHT, "Proposed component"),
        (C_BACKBONE, "Backbone (baseline)"),
        (C_NECK, "Neck"),
        (C_OUTPUT, "Output"),
    ]
    for i, (color, name) in enumerate(legend_items):
        ax.add_patch(patches.Rectangle((legend_x, legend_y + i * 0.32 - 0.1),
                                        0.28, 0.22, facecolor=color, edgecolor="#555"))
        ax.text(legend_x + 0.38, legend_y + i * 0.32, name,
                ha="left", va="center", fontsize=8, color=C_TEXT)

    # Title
    ax.text(cx, 12.4,
            "Figure 1. Proposed pipeline: baseline YOLOv11n + focal loss + imgsz=1024 + Platt calibration",
            ha="center", va="center", fontsize=11, fontweight="bold")

    out_png = FIG_DIR / "fig1_pipeline.png"
    out_svg = FIG_DIR / "fig1_pipeline.svg"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}")
    print(f"  saved {out_svg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
