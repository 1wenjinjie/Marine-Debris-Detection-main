#!/usr/bin/env python3
"""Publication-ready vector plotting script for MPB Figures 2 and 3 (D4 rerun).

Redesigned specifically for Marine Pollution Bulletin readers:
  - Figure 2 (190 mm x 88 mm): 9-fold 2x2 leakage audit as an intuitive 4-step
    inflation bar chart (a) paired with a clean 3-row factorial effect breakdown (b).
  - Figure 3 (190 mm x 86 mm): (a) Ranked 4-system 9-fold scene-CV comparison
    showing val-calibrated and uncalibrated pooled OOF AP50, paired with
    (b) Zero-shot external transfer on MADOS before vs after removing all 282
    MARIDA-overlapping crops.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_py_dir = Path(sys.executable).parent
_lib_bin = _py_dir / "Library" / "bin"
if _lib_bin.is_dir() and str(_lib_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_lib_bin) + os.pathsep + os.environ.get("PATH", "")
if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path(__file__).resolve().parent / ".mpl"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.legend_handler as mhandler
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mpb_typography import configure_typography, flatten_png_to_rgb

MM = 1.0 / 25.4

PALETTE = {
    "c6": "#2E5B88",
    "ndwi": "#3C7A78",
    "oracle": "#B5442F",
    "exposure": "#D29A3A",
    "clean": "#2E5B88",
    "shuffled": "#9E6385",
    "rf": "#64748B",
    "fdi": "#7C8A99",
    "fdi_bar": "#94A3B8",
    "ink": "#1A1A1A",
    "muted": "#52525B",
    "slate": "#64748B",
    "pair_line": "#94A3B8",
    "grid": "#E5E7EB",
    "band": "#EDF1F5",
}

SEED_MARKERS = ["o", "s", "^"]


class _HatchedLegendPatch:
    def __init__(self, facecolor: str, hatch: str = "////", edgecolor: str = "#1E293B") -> None:
        self.facecolor = facecolor
        self.hatch = hatch
        self.edgecolor = edgecolor


class _HatchedPatchHandler(mhandler.HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p_fill = mpatches.Rectangle(
            (xdescent, ydescent), width, height,
            facecolor=orig_handle.facecolor, edgecolor="none", transform=trans,
        )
        p_hatch = mpatches.Rectangle(
            (xdescent, ydescent), width, height,
            facecolor="none", edgecolor="white", hatch=orig_handle.hatch, linewidth=0.0, transform=trans,
        )
        p_edge = mpatches.Rectangle(
            (xdescent, ydescent), width, height,
            facecolor="none", edgecolor=orig_handle.edgecolor, linewidth=0.65, transform=trans,
        )
        return [p_fill, p_hatch, p_edge]


FIG2_CONDITIONS = [
    {
        "key": "clean_ndwi",
        "label": "Clean\nC7 (+NDWI)",
        "mean": 0.1762,
        "ci": (0.1082, 0.2845),
        "seeds": [0.2353, 0.1589, 0.1344],
        "color": PALETTE["clean"],
        "hatched": False,
    },
    {
        "key": "shared_ndwi",
        "label": "+ Same-scene\nsplit",
        "mean": 0.2686,
        "ci": (0.1674, 0.3773),
        "seeds": [0.3205, 0.2775, 0.2077],
        "color": PALETTE["clean"],
        "hatched": True,
    },
    {
        "key": "clean_oracle",
        "label": "+ Label\ninput",
        "mean": 0.3573,
        "ci": (0.2658, 0.4799),
        "seeds": [0.3320, 0.4003, 0.3397],
        "color": PALETTE["oracle"],
        "hatched": False,
    },
    {
        "key": "shared_oracle",
        "label": "Both\nleakages",
        "mean": 0.4621,
        "ci": (0.3282, 0.6000),
        "seeds": [0.5047, 0.4586, 0.4231],
        "color": PALETTE["oracle"],
        "hatched": True,
    },
]

FIG2_EFFECTS = [
    {
        "label": "Label-derived\ninput",
        "mean": 0.1873,
        "ci": (0.0885, 0.2825),
        "seeds": [0.1404, 0.2113, 0.2104],
        "color": PALETTE["oracle"],
        "sig": True,
    },
    {
        "label": "Same-scene\nexposure",
        "mean": 0.0986,
        "ci": (0.0227, 0.1585),
        "seeds": [0.1290, 0.0884, 0.0784],
        "color": PALETTE["exposure"],
        "sig": True,
    },
    {
        "label": "Interaction\n(non-additive)",
        "mean": 0.0125,
        "ci": (-0.1022, 0.1173),
        "seeds": [0.0875, -0.0603, 0.0101],
        "color": PALETTE["slate"],
        "sig": False,
    },
]

FIG3_CV9_SYSTEMS = [
    {
        "label": "C6 (6-band)",
        "mean": 0.1999,
        "ci": (0.1088, 0.3424),
        "seeds": [0.2054, 0.1896, 0.2048],
        "color": PALETTE["c6"],
    },
    {
        "label": "C7 (+NDWI)",
        "mean": 0.1458,
        "ci": (0.0722, 0.2679),
        "seeds": [0.1937, 0.1187, 0.1249],
        "color": PALETTE["ndwi"],
    },
    {
        "label": "Spectral RF\n(MARIDA)",
        "mean": 0.0641,
        "ci": (0.0284, 0.1232),
        "seeds": [0.0698, 0.0567, 0.0660],
        "color": PALETTE["rf"],
    },
    {
        "label": "FDI baseline\n(uncal.)",
        "mean": 0.0040,
        "ci": (0.0019, 0.0079),
        "seeds": None,
        "color": PALETTE["fdi_bar"],
    },
]

# MADOS 101-pt COCO AP50 + 10,000-replicate scene-cluster bootstrap 95% CIs
# (t6_test_vs_marida_trainval = 525 patches matching Fig. 1c)
FIG3_MADOS = {
    "groups": [
        "Official MADOS test\n(728 patches, 90 GT)",
        "Overlap-removed\n(525 patches, 79 GT)",
    ],
    "c6": {
        "means": [0.1110, 0.1178],
        "cis": [(0.0159, 0.2245), (0.0111, 0.2422)],
        "seeds": [[0.1261, 0.1000, 0.1069], [0.1344, 0.1078, 0.1111]],
    },
    "ndwi": {
        "means": [0.1627, 0.1655],
        "cis": [(0.0253, 0.2952), (0.0163, 0.3051)],
        "seeds": [[0.1945, 0.2019, 0.0917], [0.2065, 0.2036, 0.0865]],
    },
}

# Exported for Supplementary Fig. S3 (D4 values)
FIG3_SPLIT10 = {
    "C6 (6-band)": {"mean": 0.2827, "seeds": [0.2716, 0.2968, 0.2797]},
    "C7-shuffled": {"mean": 0.2745, "seeds": [0.3160, 0.2007, 0.3068]},
    "C7 (+NDWI)":  {"mean": 0.2038, "seeds": [0.2182, 0.2693, 0.1238]},
}

FIG3_SPLIT10_DIFFS = [
    {
        "label": "C7 (+NDWI) vs C6",
        "mean": -0.0789,
        "ci": (-0.1423, -0.0015),
        "seeds": [-0.0534, -0.0275, -0.1559],
        "sig": True,
    },
    {
        "label": "C7-shuffled vs C6",
        "mean": -0.0082,
        "ci": (-0.0737, 0.0497),
        "seeds": [0.0444, -0.0961, 0.0271],
        "sig": False,
    },
    {
        "label": "C7 (+NDWI) vs shuffled",
        "mean": -0.0707,
        "ci": (-0.1143, -0.0024),
        "seeds": [-0.0978, 0.0686, -0.1830],
        "sig": True,
    },
]

FOLD_META = [
    (0, 1, 56), (1, 1, 23), (2, 9, 19), (3, 9, 19), (4, 9, 19),
    (5, 9, 19), (6, 9, 19), (7, 8, 19), (8, 8, 19),
]

FOLD_AP50_C6 = [
    [0.3317, 0.3174, 0.5215],
    [0.6160, 0.5857, 0.4850],
    [0.5074, 0.5741, 0.3954],
    [0.2186, 0.1110, 0.1758],
    [0.0407, 0.3338, 0.1553],
    [0.4445, 0.2346, 0.6558],
    [0.1551, 0.1869, 0.0586],
    [0.3193, 0.2754, 0.4225],
    [0.1616, 0.1832, 0.0747],
]

FOLD_AP50_C7 = [
    [0.5156, 0.1451, 0.1780],
    [0.5332, 0.0521, 0.5725],
    [0.4409, 0.2853, 0.2075],
    [0.0772, 0.1240, 0.1437],
    [0.3553, 0.3765, 0.4122],
    [0.4384, 0.3277, 0.2225],
    [0.2152, 0.1323, 0.0448],
    [0.1727, 0.1956, 0.0992],
    [0.1152, 0.1126, 0.1142],
]

FOLD_AP50_FDI = [0.026, 0.077, 0.019, 0.003, 0.003, 0.010, 0.004, 0.007, 0.002]


def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.unicode_minus": True,
        "hatch.linewidth": 0.85,
    })
    configure_typography()


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def fmt_half_up(val: float, signed: bool = False) -> str:
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(str(val)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    if abs(float(q)) < 5e-5:
        return "0.000"
    s = f"{q:+.3f}" if signed else f"{q:.3f}"
    return s.replace("-", "\u2212")


def fmt_signed(val: float) -> str:
    return fmt_half_up(val, signed=True)


# =============================================================================
# FIGURE 2: 190 mm x 88 mm — Leakage Inflation Staircase + 3-Row Effect Summary
# =============================================================================

def build_figure2(out_dir: str, png_dpi: int = 0) -> str:
    fig = plt.figure(figsize=(190.0 * MM, 88.0 * MM), facecolor="white")
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.36, 0.78], wspace=0.38,
        left=0.076, right=0.816, top=0.840, bottom=0.175,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---- Panel (a): 2x2 factorial bar chart (color = prior, hatch = same-scene) ----
    style_axis(ax_a, grid_axis="y")
    xs = np.arange(len(FIG2_CONDITIONS))
    bar_w = 0.54

    clean_val = FIG2_CONDITIONS[0]["mean"]
    both_val = FIG2_CONDITIONS[3]["mean"]
    bx = 3.48

    ax_a.plot([-0.48, bx], [clean_val, clean_val], color="#94A3B8", linestyle="--", linewidth=0.85, zorder=1)
    ax_a.plot([3.0 + bar_w / 2.0, bx], [both_val, both_val], color="#94A3B8", linestyle=":", linewidth=0.80, zorder=1)

    for i, item in enumerate(FIG2_CONDITIONS):
        m = item["mean"]
        lo, hi = item["ci"]
        ax_a.bar(
            i, m, width=bar_w, color=item["color"], alpha=0.90,
            edgecolor="none", zorder=2,
        )
        if item["hatched"]:
            ax_a.bar(
                i, m, width=bar_w, facecolor="none", edgecolor="white",
                hatch="////", linewidth=0.0, zorder=2.4,
            )
        ax_a.bar(
            i, m, width=bar_w, facecolor="none", edgecolor="#1E293B",
            linewidth=0.70, zorder=2.8,
        )
        ax_a.errorbar(
            i, m, yerr=[[m - lo], [hi - m]],
            fmt="none", ecolor="#1E293B", elinewidth=1.15, capsize=2.8, capthick=0.95, zorder=4,
        )
        offsets = [-0.11, 0.0, 0.11]
        for s_idx, sv in enumerate(item["seeds"]):
            ax_a.scatter(
                [i + offsets[s_idx]], [sv], marker="o", s=15,
                facecolor="white", edgecolor="#1E293B", linewidth=0.85, zorder=5,
            )
        ax_a.text(
            i, hi + 0.015, f"{m:.3f}",
            ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=PALETTE["ink"],
        )

    # Right-side vertical bracket from clean baseline (0.176) to combined leakage (0.462)
    cap_w = 0.10
    ax_a.plot([bx - cap_w, bx, bx, bx - cap_w], [clean_val, clean_val, both_val, both_val],
              color=PALETTE["oracle"], linewidth=1.15, zorder=5)
    y_mid = (clean_val + both_val) / 2.0
    ax_a.text(
        bx + 0.08, y_mid + 0.025, "+0.286",
        ha="left", va="bottom", fontsize=7.6, fontweight="bold", color=PALETTE["oracle"],
    )
    ax_a.text(
        bx + 0.08, y_mid - 0.008, "0.462 =\n2.6\u00d7 clean",
        ha="left", va="top", fontsize=6.6, color=PALETTE["muted"], linespacing=1.16,
    )

    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([c["label"] for c in FIG2_CONDITIONS], fontsize=6.8, color=PALETTE["ink"])
    ax_a.set_xlim(-0.48, 4.44)
    ax_a.set_ylim(0.0, 0.705)
    ax_a.yaxis.set_major_locator(mticker.MultipleLocator(0.10))
    ax_a.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_a.set_ylabel("Pooled OOF AP50 (705-patch panel, 128 GT)", fontweight="bold", color=PALETTE["ink"])
    ax_a.text(-0.155, 1.055, "(a)", transform=ax_a.transAxes, fontsize=9.2, fontweight="bold", color=PALETTE["ink"])

    # ---- Panel (b): 3-row factorial effect decomposition (external numeric column) ----
    style_axis(ax_b, grid_axis="x")
    ax_b.axvline(0.0, color=PALETTE["ink"], linestyle="--", linewidth=0.9, zorder=1)

    ys = [2.0, 1.0, 0.0]
    trans_b = ax_b.get_yaxis_transform()
    for y_pos, eff in zip(ys, FIG2_EFFECTS):
        m = eff["mean"]
        lo, hi = eff["ci"]
        col = eff["color"]
        is_sig = eff["sig"]

        ax_b.plot([lo, hi], [y_pos, y_pos], color=col, linewidth=2.2 if is_sig else 1.5, zorder=3)
        ax_b.plot([lo, lo], [y_pos - 0.11, y_pos + 0.11], color=col, linewidth=1.2, zorder=3)
        ax_b.plot([hi, hi], [y_pos - 0.11, y_pos + 0.11], color=col, linewidth=1.2, zorder=3)
        ax_b.scatter(
            [m], [y_pos], marker="D" if is_sig else "o", s=32 if is_sig else 24,
            facecolor=col, edgecolor="#1E293B", linewidth=0.65, zorder=5,
        )
        seed_y_offs = [-0.19, -0.27, -0.35]
        for s_i, sv in enumerate(eff["seeds"]):
            ax_b.scatter(
                [sv], [y_pos + seed_y_offs[s_i]], marker="o", s=14,
                facecolor="white", edgecolor="#1E293B", linewidth=0.85, zorder=4,
            )

        txt = f"{fmt_signed(m)} [{fmt_half_up(lo)}, {fmt_half_up(hi)}]"
        ax_b.text(
            1.045, y_pos, txt, transform=trans_b, clip_on=False,
            va="center", ha="left", fontsize=7.0,
            fontweight="bold" if is_sig else "normal",
            color=PALETTE["ink"] if is_sig else PALETTE["muted"],
        )

    ax_b.set_yticks(ys)
    ax_b.set_yticklabels([e["label"] for e in FIG2_EFFECTS], fontsize=7.2, fontweight="bold", color=PALETTE["ink"])
    ax_b.set_ylim(-0.58, 2.52)
    ax_b.set_xlim(-0.125, 0.320)
    ax_b.set_xticks([-0.10, 0.00, 0.10, 0.20, 0.30])
    ax_b.set_xticklabels(["\u22120.10", "0.00", "+0.10", "+0.20", "+0.30"])
    ax_b.set_xlabel("Effect on pooled AP50 (95% CI)", fontweight="bold", color=PALETTE["ink"])
    ax_b.text(-0.32, 1.055, "(b)", transform=ax_b.transAxes, fontsize=9.2, fontweight="bold", color=PALETTE["ink"])

    # ---- Unified top legend matching BOTH (a) and (b) ----
    hatch_handle = _HatchedLegendPatch(PALETTE["exposure"], hatch="////", edgecolor="#1E293B")
    leg_handles = [
        mpatches.Patch(facecolor=PALETTE["clean"], edgecolor="#1E293B", lw=0.65, label="Optical input"),
        mpatches.Patch(facecolor=PALETTE["oracle"], edgecolor="#1E293B", lw=0.65, label="Label-derived input"),
        hatch_handle,
        mlines.Line2D([], [], color="#1E293B", lw=1.25, label="95% bootstrap CI"),
        mlines.Line2D([], [], color="none", marker="o", markersize=4.3, markerfacecolor="white",
                      markeredgecolor="#1E293B", markeredgewidth=0.85, label="Individual seeds (n = 3)"),
    ]
    leg_labels = [
        "Optical input",
        "Label-derived input",
        "Same-scene exposure",
        "95% bootstrap CI",
        "Individual seeds (n = 3)",
    ]
    fig.legend(
        handles=leg_handles, labels=leg_labels,
        handler_map={_HatchedLegendPatch: _HatchedPatchHandler()},
        loc="upper center", bbox_to_anchor=(0.50, 0.985),
        ncol=5, frameon=False, columnspacing=1.05, handletextpad=0.40, fontsize=6.9,
    )

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "Fig2_CV_Factorial_Audit_MPB.pdf")
    fig.savefig(pdf_path, format="pdf", dpi=600)
    if png_dpi > 0:
        png_path = os.path.join(out_dir, "Fig2_CV_Factorial_Audit_MPB.png")
        fig.savefig(png_path, dpi=png_dpi)
        flatten_png_to_rgb(png_path, fallback_dpi=png_dpi)
    plt.close(fig)
    return pdf_path


# =============================================================================
# FIGURE 3: 190 mm x 86 mm — (a) Internal 9-fold CV + (b) External MADOS Transfer
# =============================================================================

def build_figure3(out_dir: str, png_dpi: int = 0) -> str:
    fig = plt.figure(figsize=(190.0 * MM, 86.0 * MM), facecolor="white")
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.12, 1.00], wspace=0.29,
        left=0.112, right=0.985, top=0.825, bottom=0.170,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---- Panel (a): Ranked 4-system 9-fold CV (single val-calibrated metric) ----
    style_axis(ax_a, grid_axis="x")
    ys = np.array([3.0, 2.0, 1.0, 0.0])
    bar_h = 0.54

    for y_pos, sys_d in zip(ys, FIG3_CV9_SYSTEMS):
        m = sys_d["mean"]
        lo, hi = sys_d["ci"]
        col = sys_d["color"]
        ax_a.barh(
            y_pos, m, height=bar_h, color=col, alpha=0.90,
            edgecolor="#1E293B", linewidth=0.65, zorder=2,
        )
        ax_a.errorbar(
            m, y_pos, xerr=[[m - lo], [hi - m]],
            fmt="none", ecolor="#1E293B", elinewidth=1.15, capsize=2.8, capthick=0.95, zorder=4,
        )
        if sys_d["seeds"] is not None:
            y_offs = [0.13, 0.0, -0.13]
            for s_idx, sv in enumerate(sys_d["seeds"]):
                ax_a.scatter(
                    [sv], [y_pos + y_offs[s_idx]], marker="o", s=15,
                    facecolor="white", edgecolor="#1E293B", linewidth=0.85, zorder=5,
                )

        ax_a.text(
            hi + 0.008, y_pos, f"{m:.3f}",
            va="center", ha="left", fontsize=7.2,
            fontweight="bold", color=PALETTE["ink"],
        )

    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([s["label"] for s in FIG3_CV9_SYSTEMS], fontsize=7.3, fontweight="bold", color=PALETTE["ink"])
    ax_a.set_ylim(-0.55, 3.55)
    ax_a.set_xlim(0.0, 0.405)
    ax_a.xaxis.set_major_locator(mticker.MultipleLocator(0.10))
    ax_a.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_a.set_xlabel("Val-calibrated pooled OOF AP50 (full 9-fold CV, 212 GT)", fontweight="bold", color=PALETTE["ink"])
    ax_a.set_title("Internal 9-fold scene-disjoint CV", fontsize=8.0, fontweight="bold", color=PALETTE["ink"], pad=5)
    ax_a.text(-0.215, 1.07, "(a)", transform=ax_a.transAxes, fontsize=9.2, fontweight="bold", color=PALETTE["ink"])

    # ---- Panel (b): Zero-shot external transfer on MADOS (728 vs 525 crops + 95% CIs) ----
    style_axis(ax_b, grid_axis="y")
    xg = np.array([0.0, 1.0])
    bw = 0.30
    gap = 0.04

    for g_i in range(2):
        for model_key, sign in (("c6", -1), ("ndwi", 1)):
            md = FIG3_MADOS[model_key]
            m_val = md["means"][g_i]
            lo, hi = md["cis"][g_i]
            seeds = md["seeds"][g_i]
            x_pos = xg[g_i] + sign * (bw / 2.0 + gap / 2.0)

            ax_b.bar(
                x_pos, m_val, width=bw, color=PALETTE[model_key], alpha=0.90,
                edgecolor="#1E293B", linewidth=0.65, zorder=2,
            )
            ax_b.errorbar(
                x_pos, m_val, yerr=[[m_val - lo], [hi - m_val]],
                fmt="none", ecolor="#1E293B", elinewidth=1.15, capsize=2.8, capthick=0.95, zorder=4,
            )
            for s_idx, sv in enumerate(seeds):
                ax_b.scatter(
                    [x_pos + (s_idx - 1) * 0.055], [sv], marker="o", s=15,
                    facecolor="white", edgecolor="#1E293B", linewidth=0.85, zorder=5,
                )
            ax_b.text(
                x_pos, hi + 0.009, f"{m_val:.3f}",
                ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=PALETTE["ink"],
            )

    ax_b.set_xticks(xg)
    ax_b.set_xticklabels(FIG3_MADOS["groups"], fontsize=7.1, color=PALETTE["ink"])
    ax_b.set_xlim(-0.52, 1.52)
    ax_b.set_ylim(0.0, 0.360)
    ax_b.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax_b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_b.set_ylabel("Zero-shot AP50 (marine debris)", fontweight="bold", color=PALETTE["ink"])
    ax_b.set_title("External transfer on MADOS (frozen weights)", fontsize=8.0, fontweight="bold", color=PALETTE["ink"], pad=5)
    ax_b.text(-0.185, 1.07, "(b)", transform=ax_b.transAxes, fontsize=9.2, fontweight="bold", color=PALETTE["ink"])

    # ---- Clean unified 4-item top legend ----
    leg_handles = [
        mpatches.Patch(facecolor=PALETTE["c6"], edgecolor="#1E293B", lw=0.65, label="C6 (6-band)"),
        mpatches.Patch(facecolor=PALETTE["ndwi"], edgecolor="#1E293B", lw=0.65, label="C7 (+NDWI)"),
        mlines.Line2D([], [], color="#1E293B", lw=1.25, label="95% scene-bootstrap CI"),
        mlines.Line2D([], [], color="none", marker="o", markersize=4.3, markerfacecolor="white",
                      markeredgecolor="#1E293B", markeredgewidth=0.85, label="Individual seeds (n = 3)"),
    ]
    fig.legend(
        handles=leg_handles, loc="upper center", bbox_to_anchor=(0.52, 0.990),
        ncol=4, frameon=False, columnspacing=1.35, handletextpad=0.40, fontsize=7.0,
    )

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "Fig3_Channel_Baseline_Comparison_MPB.pdf")
    fig.savefig(pdf_path, format="pdf", dpi=600)
    if png_dpi > 0:
        png_path = os.path.join(out_dir, "Fig3_Channel_Baseline_Comparison_MPB.png")
        fig.savefig(png_path, dpi=png_dpi)
        flatten_png_to_rgb(png_path, fallback_dpi=png_dpi)
    plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MPB D4 vector PDFs for Figures 2 and 3.")
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument("--png-dpi", type=int, default=600, help="PNG export DPI.")
    args, _ = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configure_matplotlib_style()

    f2 = build_figure2(args.out_dir, png_dpi=args.png_dpi)
    f3 = build_figure3(args.out_dir, png_dpi=args.png_dpi)
    print(f"Generated vector Figure 2: {f2}")
    print(f"Generated vector Figure 3: {f3}")


if __name__ == "__main__":
    main()
