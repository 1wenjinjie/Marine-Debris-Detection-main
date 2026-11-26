#!/usr/bin/env python3
"""Publication-ready vector plotting script for Figures 1 and 4 (Elsevier MPB).

Generates 190 mm double-column vector PDFs (with editable Type 42 Arial fonts):
  - Figure 1 (190 mm x 184 mm): Audit pathways, real empirical Sentinel-2
    evidence & protocol controls —
      (a) pure vector world map of 63 MARIDA scenes + real 48-patch Tile 16PCC
          UTM layout vs 9-fold scene-grouped CV allocation (S4),
      (b) real MARIDA debris patch (S2_18-9-20_16PCC_17) RGB -> polygon mask ->
          internal L2 distance-transform Oracle diagnostic pathway,
      (c) real composite Sentinel-2 shifted-footprint overlap between MARIDA
          train (S2_22-12-20_18QYF_0) and MADOS test (Scene_143_1) + 3x3 split
          duplication matrix (S3) -> MADOS 728-to-525 deduplication control,
      (d) 5-stage validation-locked evaluation workflow (matching Fig. 1d refs).
  - Figure 4 (190 mm x 114 mm): 9-fold OOF detection & error diagnostics
    (3 streamlined panels: PR envelope, stratified recall, and full-width
    unmatched-box composition; matching sensitivity moved to Fig. S3d).

Usage:
    python3 plot_mpb_fig1_fig4.py --out-dir ./figures_out [--png-dpi 600]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

# Windows Anaconda safe DLL / font cache setup
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
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mpb_typography import configure_typography, flatten_png_to_rgb


# =============================================================================
# 1. VERIFIED MANUSCRIPT DATA (v8.6: Sec 4.1, 4.5 & Supplementary S3, S4, S8, S10)
# =============================================================================

MM = 1.0 / 25.4

PALETTE = {
    "c6": "#2E5B88",
    "ndwi": "#3C7A78",
    "oracle": "#B5442F",
    "fdi": "#7C8A99",
    "purple": "#9E6385",
    "ink": "#1A1A1A",
    "muted": "#52525B",
    "slate": "#64748B",
    "grid": "#E5E7EB",
    "train_blue": "#2E5B88",
    "test_gold": "#D29A3A",
    "val_grey": "#94A3B8",
}

SEED_MARKERS = ["o", "s", "^"]

S4_FOLDS = [
    (0, 1, 48, 56),
    (1, 1, 13, 23),
    (2, 9, 135, 19),
    (3, 9, 283, 19),
    (4, 9, 233, 19),
    (5, 9, 212, 19),
    (6, 9, 230, 19),
    (7, 8, 93, 19),
    (8, 8, 134, 19),
]

S3_MATRIX = [
    [
        (436, 527, 395, "same"),
        (67,  81,  65,  "val_cross"),
        (52,  63,  47,  "rev_test"),
    ],
    [
        (199, 241, 178, "val_cross"),
        (28,  36,  26,  "same"),
        (36,  42,  35,  "rev_test"),
    ],
    [
        (155, 185, 148, "crit_leak"),
        (48,  53,  44,  "val_leak"),
        (79,  100, 77,  "shared_test"),
    ],
]

FIG4_EVIDENCE = Path("runs_v2/_tmp_eval/rerun7_20261016")

SEED_LINESTYLES = ["-", "--", "-."]
STRATA = [
    ("area_4-7", "4\u20137 px"),
    ("area_8-15", "8\u201315 px"),
    ("area_ge16", "\u2265 16 px"),
    ("high", "High"),
    ("moderate", "Moderate"),
    ("low", "Low"),
]
MATCHING_CRITERIA = [
    ("iou_0.50", "\u2265 0.50"),
    ("iou_0.30", "\u2265 0.30"),
    ("iou_0.10", "\u2265 0.10"),
    ("center_1px", "\u2264 1 px"),
    ("center_2px", "\u2264 2 px"),
    ("center_3px", "\u2264 3 px"),
]


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))



# =============================================================================
# 2. STYLING & ASSET LOADING
# =============================================================================

def configure_matplotlib_style() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    sans_order = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    chosen = next((f for f in sans_order if f in available), "DejaVu Sans")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [chosen, "DejaVu Sans"],
        "font.size": 7.8,
        "axes.labelsize": 7.8,
        "axes.titlesize": 8.3,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.3,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")
    if grid_axis:
        ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def load_real_bundle() -> tuple[dict[str, np.ndarray], dict]:
    sdir = Path(__file__).resolve().parent / "assets"
    npz_path = sdir / "mpb_real_imagery_bundle.npz"
    meta_path = sdir / "mpb_real_imagery_meta.json"
    if not npz_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Real imagery bundle missing at {npz_path}; run extract_real_assets.py first."
        )
    npz = dict(np.load(npz_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return npz, meta


# =============================================================================
# 3. FIGURE 1: 190 mm x 184 mm Audit Framework, Real Evidence & Controls (a-d)
# =============================================================================

def build_figure1(out_dir: str, png_dpi: int = 0) -> str:
    """Builds upgraded 190 mm vector Figure 1 from real Sentinel-2 GeoTIFF assets."""
    npz, meta = load_real_bundle()

    fig = plt.figure(figsize=(190.0 * MM, 184.0 * MM), facecolor="white")
    banner_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    banner_ax.set_xlim(0, 1)
    banner_ax.set_ylim(0, 1)
    banner_ax.axis("off")

    # Top Column Headers
    banner_ax.text(0.018, 0.980, "Leakage pathway & empirical evidence",
                   fontsize=8.6, fontweight="bold", color=PALETTE["ink"], va="center")
    banner_ax.text(0.625, 0.980, "Audit control in evaluation protocol",
                   fontsize=8.6, fontweight="bold", color=PALETTE["ink"], va="center")
    banner_ax.plot([0.015, 0.985], [0.964, 0.964], color="#0F172A", linewidth=1.0)
    banner_ax.plot([0.015, 0.985], [0.692, 0.692], color="#CBD5E1", linewidth=0.8)
    banner_ax.plot([0.015, 0.985], [0.460, 0.460], color="#CBD5E1", linewidth=0.8)
    banner_ax.plot([0.015, 0.985], [0.118, 0.118], color="#CBD5E1", linewidth=0.8)

    # =========================================================================
    # ROW (a): Same-scene spatial exposure [y: 0.70 .. 0.955]
    # =========================================================================
    banner_ax.text(0.018, 0.944, "(a) Same-scene spatial exposure",
                   fontsize=8.5, fontweight="bold", color=PALETTE["ink"], va="center")

    # 100% Pure Vector Global Map of 63 MARIDA Scenes
    ax_map = fig.add_axes([0.018, 0.795, 0.214, 0.130], facecolor="#F8FAFC")
    for ring in meta.get("world_rings", []):
        pts = np.asarray(ring, dtype=float)
        ax_map.fill(pts[:, 0], pts[:, 1], facecolor="#E2E8F0", edgecolor="#CBD5E1",
                    linewidth=0.35, zorder=1)
    sc_lons = [s["lon"] for s in meta["scenes_63"]]
    sc_lats = [s["lat"] for s in meta["scenes_63"]]
    ax_map.scatter(sc_lons, sc_lats, s=8, facecolor=PALETTE["ink"], edgecolor="white",
                   linewidth=0.3, alpha=0.85, zorder=3)
    # Highlight Tile 16PCC (Gulf of Honduras: -88.22, 15.89)
    ax_map.scatter([-88.22], [15.89], s=44, facecolor="none", edgecolor=PALETTE["oracle"],
                   linewidth=1.35, zorder=4)
    ax_map.set_xlim(-130, 155)
    ax_map.set_ylim(-48, 66)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for sp in ax_map.spines.values():
        sp.set_edgecolor("#64748B")
        sp.set_linewidth(0.75)

    banner_ax.text(0.018, 0.773, "63 MARIDA scenes (1,381 S2 patches)",
                   fontsize=7.0, color=PALETTE["muted"], va="center")
    banner_ax.text(0.018, 0.750, "Split (patch frame / CV bar):",
                   fontsize=6.9, color=PALETTE["muted"], va="center")
    for idx_s, (lbl, col) in enumerate([
        ("Train", PALETTE["train_blue"]),
        ("Val", PALETTE["val_grey"]),
        ("Test", PALETTE["test_gold"]),
    ]):
        lx = 0.018 + idx_s * 0.072
        banner_ax.add_patch(mpatches.Rectangle(
            (lx, 0.713), 0.016, 0.020, facecolor=col, edgecolor="#1E293B", linewidth=0.75
        ))
        banner_ax.text(lx + 0.020, 0.723, lbl, fontsize=7.0, color=PALETTE["ink"], va="center")

    # 100% Native Vector + Real GeoTIFF Tile 16PCC Map (48 patches in UTM km)
    ax_pcc = fig.add_axes([0.246, 0.712, 0.206, 0.214], facecolor="white")
    pcc_patches = meta["tile_16pcc"]["patches"]
    pcc_thumbs = npz["pcc_thumbs_32"]
    all_b = np.array([p["bounds"] for p in pcc_patches], dtype=float)
    x_origin, y_origin = all_b[:, 0].min(), all_b[:, 1].min()
    x_span = (all_b[:, 2].max() - x_origin) / 1000.0
    y_span = (all_b[:, 3].max() - y_origin) / 1000.0
    split_color_map = {
        "train": PALETTE["train_blue"],
        "val": PALETTE["val_grey"],
        "test": PALETTE["test_gold"],
    }
    for idx_p, p_item in enumerate(pcc_patches):
        bx0, by0, bx1, by1 = p_item["bounds"]
        kx0, kx1 = (bx0 - x_origin) / 1000.0, (bx1 - x_origin) / 1000.0
        ky0, ky1 = (by0 - y_origin) / 1000.0, (by1 - y_origin) / 1000.0
        ax_pcc.imshow(
            pcc_thumbs[idx_p], extent=[kx0, kx1, ky0, ky1],
            origin="upper", interpolation="nearest", zorder=2
        )
        ec = split_color_map[p_item["random_split"]]
        ax_pcc.add_patch(mpatches.Rectangle(
            (kx0, ky0), kx1 - kx0, ky1 - ky0,
            facecolor="none", edgecolor=ec, linewidth=1.15, zorder=3
        ))

    # Crisp vector North arrow & exact 10 km scale bar
    ax_pcc.text(3.2, 10.5, "N", fontsize=7.3, fontweight="bold", color=PALETTE["ink"],
                ha="center", va="bottom", zorder=5)
    ax_pcc.annotate(
        "", xy=(3.2, 10.2), xytext=(3.2, 5.8),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["ink"], lw=1.0, mutation_scale=7),
        zorder=5
    )
    ax_pcc.plot([1.5, 11.5], [2.2, 2.2], color=PALETTE["ink"], linewidth=1.8,
                solid_capstyle="butt", zorder=5)
    ax_pcc.text(6.5, 3.0, "10 km", fontsize=7.0, fontweight="bold", color=PALETTE["ink"],
                ha="center", va="bottom", zorder=5)

    pad_km = 1.6
    ax_pcc.set_xlim(-pad_km, x_span + pad_km)
    ax_pcc.set_ylim(-pad_km, y_span + pad_km)
    ax_pcc.set_xticks([])
    ax_pcc.set_yticks([])
    for sp in ax_pcc.spines.values():
        sp.set_edgecolor("#334155")
        sp.set_linewidth(0.8)

    banner_ax.text(
        0.462, 0.908,
        "Tile 16PCC\n(Gulf of Honduras)\n4 Sep 2016\n\n48 S2 patches\nin one scene;\npatch-random\nsplit mixes train\n& test (< 10 km)",
        fontsize=6.9, color=PALETTE["ink"], va="top", linespacing=1.22
    )

    banner_ax.annotate(
        "", xy=(0.606, 0.820), xytext=(0.570, 0.820),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["ink"], lw=1.1, mutation_scale=9)
    )

    # Right (a): Scene-grouped 9-fold CV allocation
    banner_ax.text(0.625, 0.944, "Scene-grouped 9-fold CV allocation (S4)",
                   fontsize=8.2, fontweight="bold", color=PALETTE["ink"], va="center")
    ax_cv = fig.add_axes([0.672, 0.736, 0.268, 0.178])
    style_axis(ax_cv, grid_axis="y")

    fx = np.arange(9)
    patches_arr = [r[2] for r in S4_FOLDS]
    gt_arr = [r[3] for r in S4_FOLDS]
    bar_cols = [PALETTE["test_gold"], PALETTE["val_grey"]] + [PALETTE["train_blue"]] * 7

    ax_cv.bar(fx, patches_arr, width=0.62, color=bar_cols, edgecolor="#1E293B", linewidth=0.55, zorder=3)
    ax_cv.text(1.0, 20, "13", fontsize=6.8, color=PALETTE["muted"], ha="center", va="bottom", zorder=4)
    ax_cv.set_ylim(0, 330)
    ax_cv.set_yticks([0, 100, 200, 300])
    ax_cv.tick_params(axis="y", labelsize=7.0)
    ax_cv.set_ylabel("Patches", fontsize=7.2, labelpad=3)
    ax_cv.set_xticks(fx)
    ax_cv.set_xticklabels([str(i) for i in fx], fontsize=7.0)
    ax_cv.set_xlabel("Fold ID (k = 0: Fold 0 test, Fold 1 val)", fontsize=7.0, labelpad=2)

    ax_cv_gt = ax_cv.twinx()
    ax_cv_gt.spines["top"].set_visible(False)
    ax_cv_gt.spines["left"].set_visible(False)
    ax_cv_gt.spines["right"].set_color(PALETTE["oracle"])
    ax_cv_gt.plot([1.75, 8.25], [19, 19], color=PALETTE["oracle"], linestyle=":",
                  linewidth=0.9, alpha=0.75, zorder=4)
    ax_cv_gt.scatter(fx, gt_arr, color=PALETTE["oracle"], marker="D", s=23,
                     edgecolor="white", linewidth=0.65, zorder=6)
    ax_cv_gt.set_ylim(0, 74)
    ax_cv_gt.set_yticks([0, 20, 40, 60])
    ax_cv_gt.tick_params(axis="y", colors=PALETTE["oracle"], labelsize=7.0)
    ax_cv_gt.set_ylabel("MIN4 GT", fontsize=7.2, color=PALETTE["oracle"], labelpad=3)
    ax_cv_gt.text(0.32, 60.5, "56 GT (1 sc)", fontsize=6.8, color=PALETTE["oracle"],
                  fontweight="bold", ha="left", va="center",
                  bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.85))
    ax_cv_gt.text(1.0, 26.5, "23 GT\n(1 sc)", fontsize=6.7, color=PALETTE["oracle"],
                  fontweight="bold", ha="center", va="bottom", linespacing=1.05)

    # =========================================================================
    # ROW (b): Real MARIDA Label-Derived Oracle Input [y: 0.47 .. 0.68]
    # =========================================================================
    banner_ax.text(0.018, 0.672, "(b) Label-derived auxiliary input (real MARIDA patch S2_18-9-20_16PCC_17)",
                   fontsize=8.4, fontweight="bold", color=PALETTE["ink"], va="center")

    # Tight 28x28 crop centered directly on the 33-pixel real debris polygon
    oy_pts, ox_pts = np.where(npz["oracle_mask_48"] > 0)
    ocy = int(np.clip(round(oy_pts.mean()) - 14, 0, 48 - 28))
    ocx = int(np.clip(round(ox_pts.mean()) - 14, 0, 48 - 28))
    ora_rgb = npz["oracle_rgb_48"][ocy : ocy + 28, ocx : ocx + 28]
    ora_msk = npz["oracle_mask_48"][ocy : ocy + 28, ocx : ocx + 28]
    ora_dst = npz["oracle_dist_48"][ocy : ocy + 28, ocx : ocx + 28]

    ax_b0 = fig.add_axes([0.022, 0.502, 0.094, 0.124])
    ax_b0.imshow(ora_rgb, interpolation="nearest")
    ax_b0.set_xticks([])
    ax_b0.set_yticks([])
    for sp in ax_b0.spines.values():
        sp.set_edgecolor("#334155")
        sp.set_linewidth(0.8)
    ax_b0.set_title("S2 RGB", fontsize=7.0, pad=2.5, color=PALETTE["ink"])
    banner_ax.text(0.069, 0.480, "280 \u00d7 280 m", fontsize=6.8, color=PALETTE["muted"], ha="center")

    banner_ax.annotate("", xy=(0.142, 0.564), xytext=(0.120, 0.564),
                       arrowprops=dict(arrowstyle="-|>", color=PALETTE["ink"], lw=0.95, mutation_scale=7))

    ax_b1 = fig.add_axes([0.146, 0.502, 0.094, 0.124])
    ax_b1.imshow(ora_msk, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_b1.set_xticks([])
    ax_b1.set_yticks([])
    for sp in ax_b1.spines.values():
        sp.set_edgecolor("#334155")
        sp.set_linewidth(0.8)
    ax_b1.set_title("Mask (cl > 0)", fontsize=7.0, pad=2.5, color=PALETTE["ink"])
    banner_ax.text(0.193, 0.480, "33 debris px", fontsize=6.8, color=PALETTE["muted"], ha="center")

    banner_ax.annotate("", xy=(0.266, 0.564), xytext=(0.244, 0.564),
                       arrowprops=dict(arrowstyle="-|>", color=PALETTE["ink"], lw=0.95, mutation_scale=7))

    ax_b2 = fig.add_axes([0.270, 0.502, 0.094, 0.124])
    im_dt = ax_b2.imshow(ora_dst, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax_b2.set_xticks([])
    ax_b2.set_yticks([])
    for sp in ax_b2.spines.values():
        sp.set_edgecolor(PALETTE["oracle"])
        sp.set_linewidth(0.95)
    ax_b2.set_title("Internal L2 dist.", fontsize=7.0, pad=2.5, color=PALETTE["oracle"], fontweight="bold")
    cax = fig.add_axes([0.368, 0.505, 0.008, 0.118])
    cb = fig.colorbar(im_dt, cax=cax, ticks=[0, 1])
    cb.ax.tick_params(labelsize=6.8, length=2)
    banner_ax.text(0.317, 0.480, "Scaled [0, 1]", fontsize=6.8, color=PALETTE["muted"], ha="center")

    banner_ax.annotate("", xy=(0.422, 0.564), xytext=(0.396, 0.564),
                       arrowprops=dict(arrowstyle="-|>", color=PALETTE["oracle"], lw=0.95, mutation_scale=7))

    ax_stk = fig.add_axes([0.424, 0.495, 0.136, 0.138], frameon=False)
    ax_stk.set_xlim(0, 1)
    ax_stk.set_ylim(0, 1)
    ax_stk.axis("off")
    for k in range(6):
        ox = 0.05 + k * 0.065
        oy = 0.16 + k * 0.052
        ax_stk.add_patch(mpatches.Rectangle(
            (ox, oy), 0.43, 0.48, facecolor="#E0F2FE", edgecolor="#0284C7", linewidth=0.7
        ))
    ax_stk.add_patch(mpatches.Rectangle(
        (0.05 + 6 * 0.065, 0.16 + 6 * 0.052), 0.43, 0.48,
        facecolor="#FFEDD5", edgecolor=PALETTE["oracle"], linewidth=1.2, linestyle="--"
    ))
    ax_stk.text(0.655, 0.71, "7th ch.\n(Oracle)", fontsize=6.8, color=PALETTE["oracle"],
                fontweight="bold", ha="center", va="center")
    ax_stk.text(0.48, 0.01, "6 bands + ch. 7", fontsize=6.8, color=PALETTE["muted"], ha="center")

    banner_ax.annotate(
        "", xy=(0.610, 0.564), xytext=(0.564, 0.564),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["oracle"], lw=1.1, ls="--", mutation_scale=9)
    )
    banner_ax.text(
        0.585, 0.564, "\u00d7", fontsize=11.0, fontweight="bold",
        color=PALETTE["oracle"], ha="center", va="center",
        bbox=dict(boxstyle="circle,pad=0.06", facecolor="white", edgecolor=PALETTE["oracle"], lw=0.8)
    )

    banner_ax.text(0.625, 0.642, "Excluded from evaluation inputs",
                   fontsize=8.2, fontweight="bold", color=PALETTE["oracle"], va="center")
    banner_ax.text(
        0.625, 0.608,
        "\u2022 Inference-feasible baseline: C6 (6 spectral bands only)\n"
        "\u2022 Physical index tested on equal footing: C7-NDWI\n"
        "\u2022 Ground-truth Oracle isolated strictly inside 2\u00d72 audit\n"
        "   to quantify label-leakage inflation (+0.079 AP50)",
        fontsize=7.0, color=PALETTE["ink"], va="top", linespacing=1.35
    )

    # =========================================================================
    # ROW (c): Real Shifted Footprint + 3x3 Matrix -> Deduplication Control
    # =========================================================================
    banner_ax.text(
        0.018, 0.440,
        "(c) Cross-dataset same-acquisition overlap: 1,013 / 1,381 (73.35%) MARIDA patches match MADOS",
        fontsize=8.4, fontweight="bold", color=PALETTE["ink"], va="center"
    )

    # Left-1: Real 3.54 x 2.84 km Sentinel-2 Composite Footprint (2020-12-22, 18QYF)
    ax_ov = fig.add_axes([0.020, 0.165, 0.212, 0.242], facecolor="#F8FAFC")
    comp_rgb = npz["overlap_composite_u8"]
    ch_h, ch_w, _ = comp_rgb.shape
    ax_ov.imshow(comp_rgb, extent=[0, ch_w, ch_h, 0], interpolation="nearest", zorder=1)

    # MADOS test frame (240x240 at x=44..284, y=0..240)
    ax_ov.add_patch(mpatches.Rectangle(
        (44, 0), 240, 240, facecolor="none", edgecolor=PALETTE["test_gold"],
        linewidth=1.9, zorder=4
    ))
    # MARIDA train frame (256x256 at x=0..256, y=98..354)
    ax_ov.add_patch(mpatches.Rectangle(
        (0, 98), 256, 256, facecolor="none", edgecolor=PALETTE["train_blue"],
        linewidth=1.9, zorder=5
    ))
    # Highlight shared overlap zone border subtly
    ax_ov.add_patch(mpatches.Rectangle(
        (44, 98), 212, 142, facecolor="none", edgecolor="white",
        linewidth=0.85, linestyle="--", alpha=0.85, zorder=6
    ))

    # Labels right on the two shifted frames
    ax_ov.text(278, 12, "MADOS test\nScene_143_1", fontsize=6.5, fontweight="bold",
               color="#FEF08A", ha="right", va="top", zorder=7,
               bbox=dict(boxstyle="round,pad=0.14", facecolor="#0F172A", edgecolor="none", alpha=0.78))
    ax_ov.text(7, 344, "MARIDA train\nS2_22-12-20_18QYF_0", fontsize=6.5, fontweight="bold",
               color="#BAE6FD", ha="left", va="bottom", zorder=7,
               bbox=dict(boxstyle="round,pad=0.14", facecolor="#0F172A", edgecolor="none", alpha=0.78))
    ax_ov.text(150, 114, "Shared pass (r = 1.000)\nSame slick (IoU = 1.0)", fontsize=6.3,
               fontweight="bold", color="white", ha="center", va="top", zorder=7,
               bbox=dict(boxstyle="round,pad=0.12", facecolor="#0F172A", edgecolor="none", alpha=0.68))

    # Crisp 1 km (= 100 px) white vector scale bar on dark ocean backdrop
    ax_ov.add_patch(mpatches.FancyBboxPatch(
        (146, 250), 104, 34, boxstyle="round,pad=1.5,rounding_size=4",
        facecolor="#0F172A", edgecolor="none", alpha=0.78, zorder=6
    ))
    ax_ov.plot([148, 248], [278, 278], color="white", linewidth=1.8, solid_capstyle="butt", zorder=7)
    ax_ov.text(198, 271, "1 km", fontsize=6.6, fontweight="bold", color="white",
               ha="center", va="bottom", zorder=7)

    ax_ov.set_xlim(-4, ch_w + 4)
    ax_ov.set_ylim(ch_h + 4, -4)
    ax_ov.set_xticks([])
    ax_ov.set_yticks([])
    for sp in ax_ov.spines.values():
        sp.set_edgecolor("#64748B")
        sp.set_linewidth(0.75)
    banner_ax.text(
        0.126, 0.142,
        "Same S2 pass (22 Dec 2020, 18QYF)\nshifted +440 m E, \u2212980 m N",
        fontsize=6.8, color=PALETTE["muted"], ha="center", va="center", linespacing=1.18
    )

    # Left-2: Compact 3x3 MARIDA x MADOS Cross-Split Matrix
    ax_mat = fig.add_axes([0.285, 0.174, 0.275, 0.214])
    ax_mat.set_xlim(0, 3)
    ax_mat.set_ylim(0, 3)
    ax_mat.invert_yaxis()
    ax_mat.xaxis.tick_top()

    cell_styles = {
        "same":        ("#F1F5F9", "#94A3B8", 0.8, PALETTE["ink"]),
        "val_cross":   ("#F8FAFC", "#CBD5E1", 0.8, PALETTE["muted"]),
        "shared_test": ("#E0F2FE", "#0284C7", 1.15, "#0369A1"),
        "rev_test":    ("#FEF3C7", "#D97706", 1.15, "#92400E"),
        "val_leak":    ("#FEF3C7", "#D97706", 1.25, "#9A3412"),
        "crit_leak":   ("#FFEDD5", PALETTE["oracle"], 1.7, "#9A3412"),
    }

    for r_i in range(3):
        for c_i in range(3):
            mados_n, pairs_n, marida_n, kind = S3_MATRIX[r_i][c_i]
            bg_c, ed_c, lw_c, txt_c = cell_styles[kind]
            ax_mat.add_patch(mpatches.Rectangle(
                (c_i + 0.03, r_i + 0.05), 0.94, 0.90,
                facecolor=bg_c, edgecolor=ed_c, linewidth=lw_c
            ))
            ax_mat.text(
                c_i + 0.50, r_i + 0.38, f"{mados_n}",
                ha="center", va="center", fontsize=8.6, fontweight="bold", color=txt_c
            )
            ax_mat.text(
                c_i + 0.50, r_i + 0.73, f"{pairs_n}p/{marida_n}m",
                ha="center", va="center", fontsize=6.5, color=txt_c
            )

    ax_mat.set_xticks([0.5, 1.5, 2.5])
    ax_mat.set_xticklabels(["MARIDA\ntrain", "MARIDA\nval", "MARIDA\ntest"], fontsize=6.9, fontweight="bold")
    ax_mat.set_yticks([0.5, 1.5, 2.5])
    ax_mat.set_yticklabels(["MADOS\ntrain", "MADOS\nval", "MADOS\ntest"], fontsize=6.9, fontweight="bold")
    ax_mat.tick_params(length=0, pad=2)
    for sp in ax_mat.spines.values():
        sp.set_visible(False)

    banner_ax.text(
        0.422, 0.142,
        "Warm = train/val\u2194test leak \u00b7 Blue (79) = shared test\n"
        "Cell: MADOS crops (pairs p / MARIDA m)",
        fontsize=6.7, color=PALETTE["muted"], ha="center", va="center", linespacing=1.18
    )

    banner_ax.annotate(
        "", xy=(0.608, 0.280), xytext=(0.570, 0.280),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["ink"], lw=1.1, mutation_scale=9)
    )

    # Right (c): Actual Audit Control — MADOS Official Test Deduplication (728 -> 525)
    banner_ax.text(0.625, 0.426, "Source check & MADOS test deduplication (Table 5)",
                   fontsize=8.1, fontweight="bold", color=PALETTE["ink"], va="center")
    banner_ax.text(0.625, 0.395, "MADOS official test set (728 crops, 42 scenes, 90 MIN4 GT):",
                   fontsize=6.9, color=PALETTE["muted"], va="center")

    # Stacked horizontal audit bar [0.625 .. 0.975]
    bx0, bw_tot, by0, bh = 0.625, 0.350, 0.312, 0.044
    w_tr = bw_tot * (155.0 / 728.0)
    w_va = bw_tot * (48.0 / 728.0)
    w_de = bw_tot * (525.0 / 728.0)

    banner_ax.add_patch(mpatches.Rectangle(
        (bx0, by0), w_tr, bh, facecolor="#FDBA74", edgecolor=PALETTE["oracle"], linewidth=1.1
    ))
    banner_ax.add_patch(mpatches.Rectangle(
        (bx0 + w_tr, by0), w_va, bh, facecolor="#FEF08A", edgecolor="#CA8A04", linewidth=1.0
    ))
    banner_ax.add_patch(mpatches.Rectangle(
        (bx0 + w_tr + w_va, by0), w_de, bh, facecolor="#E0F2FE", edgecolor="#0284C7", linewidth=1.1
    ))

    banner_ax.text(bx0 + w_tr / 2, by0 + bh / 2, "155\ntrain", fontsize=6.6, fontweight="bold",
                   color="#9A3412", ha="center", va="center", linespacing=1.0)
    banner_ax.text(bx0 + w_tr + w_va / 2, by0 + bh + 0.014, "48 val", fontsize=6.5, fontweight="bold",
                   color="#92400E", ha="center", va="center")
    banner_ax.text(bx0 + w_tr + w_va + w_de / 2, by0 + bh / 2, "525 deduplicated crops (72.1%)",
                   fontsize=6.9, fontweight="bold", color="#075985", ha="center", va="center")

    banner_ax.text(
        0.625, 0.282,
        "\u2022 Remove 203 leaked crops matching MARIDA train (155)\n"
        "   or val (48) before external claims\n"
        "\u2022 Evaluate frozen C6 / C7-NDWI weights on both full (728)\n"
        "   and deduplicated (525) sets without retuning thresholds\n"
        "\u2022 Report MADOS transparently as non-independent (79\n"
        "   crops also share acquisitions with MARIDA test)",
        fontsize=6.9, color=PALETTE["ink"], va="top", linespacing=1.28
    )

    # =========================================================================
    # ROW (d): Restored Validation-Locked Protocol Strip (Fig. 1d in Manuscript)
    # =========================================================================
    banner_ax.text(
        0.018, 0.098,
        "(d) Validation-locked evaluation workflow for inference-feasible inputs",
        fontsize=8.3, fontweight="bold", color=PALETTE["ink"], va="center"
    )

    steps_d = [
        ("1. S2 6-band input", "B4, B3, B2, B8, B11, B12\n(ACOLITE rhorc, 10 m)"),
        ("2. MIN4 proxy boxes", "4-connected cl = 1,\narea \u2265 4 px, +2 px pad"),
        ("3. Train-only scaling", "1\u201399% percentiles fit\nstrictly on train scenes"),
        ("4. Validation lock", "Checkpoint & operating\nthreshold locked on val"),
        ("5. Single test ledger", "9-fold OOF & deduplicated\nexternal evaluation"),
    ]
    sx0 = 0.018
    sw = 0.176
    sgap = 0.021
    for s_i, (s_ttl, s_sub) in enumerate(steps_d):
        cur_x = sx0 + s_i * (sw + sgap)
        is_last = s_i == len(steps_d) - 1
        fc = "#EFF6FF" if is_last else "#F8FAFC"
        ec = PALETTE["c6"] if is_last else "#94A3B8"
        banner_ax.add_patch(mpatches.FancyBboxPatch(
            (cur_x, 0.014), sw, 0.064, boxstyle="round,pad=0.004,rounding_size=0.008",
            facecolor=fc, edgecolor=ec, linewidth=1.0 if is_last else 0.8
        ))
        banner_ax.text(cur_x + sw / 2, 0.061, s_ttl, fontsize=7.0, fontweight="bold",
                       color=PALETTE["c6"] if is_last else PALETTE["ink"], ha="center", va="center")
        banner_ax.text(cur_x + sw / 2, 0.032, s_sub, fontsize=6.4, color=PALETTE["muted"],
                       ha="center", va="center", linespacing=1.15)
        if not is_last:
            banner_ax.annotate(
                "", xy=(cur_x + sw + sgap - 0.003, 0.046), xytext=(cur_x + sw + 0.003, 0.046),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=0.95, mutation_scale=7)
            )

    pdf_path = os.path.join(out_dir, "Fig1_Audit_Framework_Overlap_MPB.pdf")
    fig.savefig(pdf_path, format="pdf", dpi=600)
    if png_dpi > 0:
        fig.savefig(os.path.join(out_dir, "Fig1_Audit_Framework_Overlap_MPB.png"), dpi=png_dpi)
    plt.close(fig)
    return pdf_path


# =============================================================================
# 4. FIGURE 4: 190 mm Double-Column 3-Panel Detection & Error Diagnostics
# =============================================================================

def _wilson_ci(p: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (p, p)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def build_figure4(out_dir: str | Path, png_dpi: int = 0, *, evidence: Path | None = None) -> str:
    out_dir = Path(out_dir)
    if evidence is None:
        repo_root = Path(__file__).resolve().parents[2]
        cand = repo_root / "runs_v2" / "_tmp_eval_d4"
        evidence = cand if cand.is_dir() else (repo_root / "artifacts" / "results" / "cv9_d4")
    fig45 = evidence / "fig45"
    pr_path = fig45 / "fig4_pr_valcal.csv"
    if not pr_path.is_file():
        pr_path = fig45 / "fig4_pr.csv"
    pr_rows = read_rows(pr_path)
    operating_points = {row["system"]: row for row in read_rows(fig45 / "fig4_operating_points.csv")}
    strata_rows = [row for row in read_rows(fig45 / "miss_strata.csv")
                   if row["iou_threshold"] == "0.5" and row["recall_def"] == "max_recall"]
    strata = {(row["stratum"], row["system"]): row for row in strata_rows}
    err_csv = evidence / "error_analysis" / "fp_summary_over_seeds.csv"
    if not err_csv.is_file():
        err_csv = evidence / "errors" / "fp_summary_over_seeds.csv"
    pixel_shares = {row["category"]: float(row["c6_share_mean"])
                    for row in read_rows(err_csv)
                    if row["fp_def"] == "top_ngt" and row["level"] == "pixel"}
    assert abs(sum(pixel_shares.values()) - 1.0) < 1e-12
    debris_presence = sum(pixel_shares[key] for key in ("near_gt_iou_ge_0.1", "other_debris_pixels"))
    unannotated = pixel_shares["unlabeled_only"]
    other_annotated = 1.0 - debris_presence - unannotated

    fig = plt.figure(figsize=(190.0 * MM, 114.0 * MM), facecolor="white")
    gs_top = fig.add_gridspec(
        1, 2, width_ratios=[1.00, 1.15], wspace=0.26,
        left=0.084, right=0.985, top=0.890, bottom=0.455
    )
    gs_bot = fig.add_gridspec(
        1, 1,
        left=0.205, right=0.985, top=0.312, bottom=0.105
    )

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_bot[0, 0])

    # ---- Panel (a): 3-seed mean PR curve + min-max envelope & top-n_GT points ----
    style_axis(ax_a, grid_axis="both")
    system_colors = {"c6": PALETTE["c6"], "c7": PALETTE["ndwi"], "fdi": PALETTE["fdi"]}

    def _extract_curve(sys_name: str) -> tuple[np.ndarray, np.ndarray]:
        values = [row for row in pr_rows if row["system"] == sys_name]
        assert values and {row["n_gt"] for row in values} == {"212"}
        assert all(int(row["rank"]) == idx for idx, row in enumerate(values, start=1))
        r_raw = np.array([float(row["recall"]) for row in values])
        p_raw = np.array([float(row["precision_envelope"]) for row in values])
        assert (np.diff(r_raw) >= 0).all() and (np.diff(p_raw) <= 1e-12).all()
        recalls, first_pos = np.unique(r_raw, return_index=True)
        env = p_raw[first_pos]
        if recalls[0] > 0:
            recalls = np.r_[0.0, recalls]
            env = np.r_[env[0], env]
        return recalls, env

    def _eval_step_pre(recalls: np.ndarray, env: np.ndarray, grid: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(recalls, grid, side="left")
        out = np.zeros_like(grid, dtype=float)
        in_range = grid <= recalls[-1] + 1e-12
        out[in_range] = env[np.clip(idx[in_range], 0, len(env) - 1)]
        return out

    for ch in ("c6", "c7"):
        col = system_colors[ch]
        seed_curves = [_extract_curve(f"{ch}_s{s}") for s in range(3)]
        max_rs = [c[0][-1] for c in seed_curves]
        mean_max_r = float(np.mean(max_rs))
        r_grid = np.linspace(0.0, mean_max_r, 360)
        p_mat = np.vstack([_eval_step_pre(r_c, p_c, r_grid) for r_c, p_c in seed_curves])

        for r_c, p_c in seed_curves:
            ax_a.step(r_c, p_c, where="pre", color=col, linewidth=0.72, alpha=0.28, zorder=2)

        ax_a.fill_between(r_grid, p_mat.min(axis=0), p_mat.max(axis=0),
                          color=col, alpha=0.16, edgecolor="none", zorder=2.5)
        ax_a.plot(r_grid, p_mat.mean(axis=0), color=col, linewidth=1.85, zorder=4)

        for s in range(3):
            pt = operating_points[f"{ch}_s{s}"]
            ax_a.scatter(
                [float(pt["recall"])], [float(pt["precision"])],
                marker="o", s=24, facecolor="white", edgecolor=col, linewidth=1.2, zorder=6,
            )

    fdi_r, fdi_p = _extract_curve("fdi")
    ax_a.step(fdi_r, fdi_p, where="pre", color=system_colors["fdi"], linewidth=1.45, zorder=3)
    fdi_pt = operating_points["fdi"]
    ax_a.scatter(
        [float(fdi_pt["recall"])], [float(fdi_pt["precision"])],
        marker="D", s=26, facecolor="white", edgecolor=system_colors["fdi"], linewidth=1.2, zorder=6,
    )

    ax_a.set_xlim(0, 0.85)
    ax_a.set_ylim(-0.013, 1.025)
    ax_a.xaxis.set_major_locator(mticker.MultipleLocator(0.20))
    ax_a.yaxis.set_major_locator(mticker.MultipleLocator(0.20))
    ax_a.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_a.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_a.set_xlabel("Recall (IoU \u2265 0.50)", fontweight="bold", color=PALETTE["ink"])
    ax_a.set_ylabel("Precision", fontweight="bold", color=PALETTE["ink"])
    ax_a.text(-0.165, 1.045, "(a)", transform=ax_a.transAxes, fontsize=9, fontweight="bold", color=PALETTE["ink"])

    # ---- Panel (b): Stratified Max Recall with Wilson 95% CI & uniform seed circles ----
    style_axis(ax_b, grid_axis="y")
    x_area = np.array([0.0, 1.0, 2.0])
    x_conf = np.array([3.35, 4.35, 5.35])

    ax_b.axvline(2.68, color="#CBD5E1", linestyle="--", linewidth=0.85, zorder=1)
    ax_b.text(1.0, 1.055, "By MIN4 instance area", ha="center", va="top",
              fontsize=7.6, fontweight="bold", color=PALETTE["ink"])
    ax_b.text(4.35, 1.055, "By label confidence", ha="center", va="top",
              fontsize=7.6, fontweight="bold", color=PALETTE["ink"])

    all_xticks = np.concatenate([x_area, x_conf])
    all_xlabels = []
    for xb, (key, name) in zip(all_xticks, STRATA):
        fdi_row = strata[key, "fdi"]
        n_gt = int(fdi_row["n_gt"])
        all_xlabels.append(f"{name}\n(n = {n_gt})")
        for channel, offset in (("c6", -0.22), ("c7", 0.04)):
            rows = [strata[key, f"{channel}_s{seed}"] for seed in range(3)]
            assert all(int(row["n_gt"]) == n_gt for row in rows)
            recalls = [float(row["recall"]) for row in rows]
            avg = float(np.mean(recalls))
            ci_lo, ci_hi = _wilson_ci(avg, n_gt)
            ax_b.bar(
                xb + offset, avg, width=0.21, color=system_colors[channel], alpha=0.86,
                edgecolor="#1E293B", linewidth=0.60, zorder=2,
            )
            ax_b.errorbar(
                xb + offset, avg, yerr=[[avg - ci_lo], [ci_hi - avg]],
                fmt="none", ecolor="#1E293B", elinewidth=0.95, capsize=2.0, capthick=0.85, zorder=4,
            )
            for seed, recall in enumerate(recalls):
                ax_b.scatter(
                    [xb + offset + (seed - 1) * 0.052], [recall], marker="o",
                    s=13, facecolor="white", edgecolor="#1E293B", linewidth=0.80, zorder=5,
                )
        ax_b.scatter(
            [xb + 0.28], [float(fdi_row["recall"])], marker="D", s=22,
            facecolor="white", edgecolor=PALETTE["fdi"], linewidth=1.2, zorder=5,
        )

    ax_b.set_xticks(all_xticks)
    ax_b.set_xticklabels(all_xlabels, fontsize=7.0)
    ax_b.set_xlim(-0.52, 5.84)
    ax_b.set_ylim(0.00, 1.08)
    ax_b.yaxis.set_major_locator(mticker.MultipleLocator(0.20))
    ax_b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_b.set_ylabel("Maximum recall (IoU \u2265 0.50)", fontweight="bold", color=PALETTE["ink"])
    ax_b.text(-0.155, 1.045, "(b)", transform=ax_b.transAxes, fontsize=9, fontweight="bold", color=PALETTE["ink"])

    # ---- Panel (c): Full-Width Unmatched Prediction Composition ----
    style_axis(ax_c, grid_axis="x")
    yc = np.array([2.0, 1.0, 0.0])
    names_c = ["Debris-related", "Unannotated pixels only", "Other annotated"]
    pcts = 100 * np.array([debris_presence, unannotated, other_annotated])
    cols_c = ["#334155", "#7B8794", "#9E6385"]
    notes = [
        f'({100*pixel_shares["near_gt_iou_ge_0.1"]:.1f}% near GT + {100*pixel_shares["other_debris_pixels"]:.1f}% below MIN4)',
        "",
        "(algae, ships, foam, water, cloud)",
    ]
    ax_c.barh(yc, pcts, height=0.56, color=cols_c, alpha=0.92, edgecolor="none", zorder=3)
    for y_pos, pct, note in zip(yc, pcts, notes):
        ax_c.text(pct + 1.0, y_pos, f"{pct:.1f}%", va="center", ha="left",
                  fontsize=7.5, fontweight="bold", color=PALETTE["ink"],
                  bbox=dict(boxstyle="square,pad=0.06", facecolor="white", edgecolor="none", alpha=0.90),
                  zorder=4)
        if note:
            ax_c.text(pct + 6.8, y_pos, note, va="center", ha="left", fontsize=7.0,
                      color=PALETTE["muted"],
                      bbox=dict(boxstyle="square,pad=0.06", facecolor="white", edgecolor="none", alpha=0.90),
                      zorder=4)

    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(names_c, fontsize=7.3, fontweight="bold", color=PALETTE["ink"])
    ax_c.set_ylim(-0.62, 2.62)
    ax_c.set_xlim(0, 100)
    ax_c.set_xticks([0, 20, 40, 60, 80, 100])
    ax_c.set_xlabel("Share of C6 (6-band) unmatched boxes (%, per-fold top-N)", fontsize=8.0, fontweight="bold", color=PALETTE["ink"])
    ax_c.text(-0.237, 1.07, "(c)", transform=ax_c.transAxes, fontsize=9, fontweight="bold", color=PALETTE["ink"])

    legend_handles = [
        mlines.Line2D([], [], color=system_colors["c6"], lw=1.85, label="C6 (6-band)  (AP50 = 0.200)"),
        mlines.Line2D([], [], color=system_colors["c7"], lw=1.85, label="C7 (+NDWI)  (AP50 = 0.146)"),
        mlines.Line2D([], [], color=system_colors["fdi"], marker="D", markersize=4.5, lw=1.45,
                      markerfacecolor="white", markeredgewidth=1.15, label="FDI baseline  (AP50 = 0.004)"),
        mlines.Line2D([], [], color="none", marker="o", markersize=4.4,
                      markerfacecolor="white", markeredgecolor="#1E293B", markeredgewidth=0.9,
                      label="Individual seeds (n = 3)"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.515, 0.988),
        ncol=4, frameon=False, columnspacing=1.15, handletextpad=0.40,
        fontsize=6.9
    )

    pdf_path = out_dir / "Fig4_Detection_Error_Diagnostics_MPB.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=600)
    if png_dpi > 0:
        png_path = out_dir / "Fig4_Detection_Error_Diagnostics_MPB.png"
        fig.savefig(png_path, dpi=png_dpi)
        flatten_png_to_rgb(png_path, fallback_dpi=png_dpi)
    plt.close(fig)
    return str(pdf_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MPB-compliant vector PDFs for redesigned Figures 1 and 4."
    )
    parser.add_argument("--out-dir", default=".", help="Output directory for generated PDFs.")
    parser.add_argument("--png-dpi", type=int, default=0, help="Optional PNG export DPI (e.g. 600).")
    parser.add_argument("--figure", choices=("1", "4", "all"), default="all",
                        help="Render Figure 1, Figure 4, or both.")
    parser.add_argument("--evidence", type=Path, help="Figure 4 source tables directory.")
    args, _ = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configure_matplotlib_style()
    if args.figure in ("1", "all"):
        f1 = build_figure1(args.out_dir, png_dpi=args.png_dpi)
        print(f"Generated vector Figure 1: {f1}")
    if args.figure in ("4", "all"):
        configure_typography()
        f4 = build_figure4(args.out_dir, png_dpi=args.png_dpi, evidence=args.evidence)
        print(f"Generated vector Figure 4: {f4}")


if __name__ == "__main__":
    main()
