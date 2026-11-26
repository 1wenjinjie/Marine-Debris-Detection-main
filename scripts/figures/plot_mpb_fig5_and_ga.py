#!/usr/bin/env python3
"""Publication-ready vector plotting script for Figure 5 & Graphical Abstract.

Generates Elsevier / Marine Pollution Bulletin compliant vector PDFs from
lossless Sentinel-2 GeoTIFF reflectance, semantic class masks, and OOF boxes:
  - Figure 5 (190 mm x 182 mm, double-column): Six protocol-frozen C6 (Seed 0)
    Sentinel-2 detection cases (64x64 patches at 10 m) sharing a single linear
    reflectance stretch, with non-occluding side-docked zoomed RGB + MARIDA
    semantic class mask (_cl.tif) insets, pure vector GT/prediction boxes, and
    verified geographic sea names.
  - Graphical Abstract (225 mm x 90 mm, exact Elsevier 2.5:1 ratio matching
    1328 x 531 px): High-legibility synthesis using noise-free real Sentinel-2
    overlap imagery.

Usage:
    python3 plot_mpb_fig5_and_ga.py --out-dir ./figures_out [--export-ga-png]
"""

from __future__ import annotations

import argparse
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
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore

from PIL import Image
from mpb_typography import configure_typography, flatten_png_to_rgb


MM = 1.0 / 25.4

PALETTE = {
    "c6": "#2E5B88",
    "tp_cyan": "#00E5FF",
    "fp_magenta": "#E879F9",
    "ndwi": "#3C7A78",
    "oracle": "#B5442F",
    "fdi": "#7C8A99",
    "rf": "#64748B",
    "ink": "#1A1A1A",
}

# Unified semantic class palette (identical across Fig. 5 masks, Fig. 6 masks, and Fig. 6e spectra)
MASK_COLORS = {
    "unannotated": (10, 13, 18),      # #0A0D12 Pure black
    "debris":      (225, 29, 72),     # #E11D48 Vivid crimson-rose
    "algae":       (16, 185, 129),    # #10B981 Vivid emerald green
    "ship":        (234, 88, 12),     # #EA580C Burnt orange
    "foam":        (127, 140, 154),   # #7F8C9A Cool silver-grey (zero cyan collision)
    "water":       (43, 91, 132),     # #2B5B84 Medium ocean steel-blue
    "cloud":       (203, 213, 225),   # #CBD5E1 Pale cloud grey
}

CASE_SPECS = [
    {
        "tag": "(a) Matched: debris slick",
        "region": "Caribbean Sea, Haiti",
        "roi_size": 20.0,
        "roi_center": None,
        "mask_note": "Debris (6 px)",
    },
    {
        "tag": "(b) Missed: 4\u20137 px debris",
        "region": "Gulf of Honduras",
        "roi_size": 20.0,
        "roi_center": None,
        "mask_note": "Debris (4 px)",
    },
    {
        "tag": "(c) Unmatched: below-MIN4 debris",
        "region": "Caribbean Sea, Haiti",
        "roi_size": 20.0,
        "roi_center": None,
        "mask_note": "Debris (2 px, < MIN4)",
    },
    {
        "tag": "(d) Unmatched: natural organic material",
        "region": "Gulf of Honduras",
        "roi_size": 20.0,
        "roi_center": None,
        "mask_note": "Organic",
    },
    {
        "tag": "(e) Unmatched: vessel & wake",
        "region": "Caribbean Sea, Haiti",
        "roi_size": 20.0,
        "roi_center": None,
        "mask_note": "Ship & wake",
    },
    {
        "tag": "(f) Unmatched: breaking waves / foam",
        "region": "Bali\u2013Lombok Strait",
        "roi_size": 20.0,
        "roi_center": None,
        "mask_note": "Waves / foam",
    },
]


def configure_matplotlib_style() -> None:
    configure_typography()


def load_real_bundle() -> tuple[dict[str, np.ndarray], dict]:
    sdir = Path(__file__).resolve().parent / "assets"
    npz_path = sdir / "mpb_real_imagery_bundle.npz"
    meta_path = sdir / "mpb_real_imagery_meta.json"
    if not npz_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing real imagery bundle {npz_path}; run extract_real_assets.py first."
        )
    return dict(np.load(npz_path)), json.loads(meta_path.read_text(encoding="utf-8"))


def save_resized_png(src_png: str, dst_png: str, size_wh: tuple[int, int]) -> None:
    with Image.open(src_png) as pil_im:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        pil_im.convert("RGB").resize(size_wh, resample).save(dst_png, dpi=(300, 300))


def colorize_marida_mask(cl_arr: np.ndarray) -> np.ndarray:
    """Maps MARIDA 15-class integer array (H, W) to crisp RGB uint8."""
    out = np.empty((*cl_arr.shape, 3), dtype=np.uint8)
    out[:, :] = MASK_COLORS["unannotated"]
    out[np.isin(cl_arr, [7, 8, 10, 11, 15])] = MASK_COLORS["water"]
    out[np.isin(cl_arr, [6, 13])] = MASK_COLORS["cloud"]
    out[np.isin(cl_arr, [9, 12, 14])] = MASK_COLORS["foam"]
    out[cl_arr == 5] = MASK_COLORS["ship"]
    out[np.isin(cl_arr, [2, 3, 4])] = MASK_COLORS["algae"]
    out[cl_arr == 1] = MASK_COLORS["debris"]
    return out


def _draw_corner_brackets(
    ax: plt.Axes, rx0: float, ry0: float, rx1: float, ry1: float,
    arm: float = 4.2, color: str = "white", lw: float = 1.15, zorder: int = 8,
) -> None:
    corners = [
        ([rx0, rx0, rx0 + arm], [ry0 + arm, ry0, ry0]),
        ([rx1 - arm, rx1, rx1], [ry0, ry0, ry0 + arm]),
        ([rx0, rx0, rx0 + arm], [ry1 - arm, ry1, ry1]),
        ([rx1 - arm, rx1, rx1], [ry1, ry1, ry1 - arm]),
    ]
    for xs_c, ys_c in corners:
        ax.plot(xs_c, ys_c, color="#0A0D12", lw=lw + 1.1, solid_capstyle="butt", zorder=zorder)
        ax.plot(xs_c, ys_c, color=color, lw=lw, solid_capstyle="butt", zorder=zorder + 0.2)


def draw_vector_boxes_64(
    ax: plt.Axes,
    pmeta: dict,
    xlim_win: tuple[float, float],
    ylim_win: tuple[float, float],
    show_scores: bool = False,
    lw_scale: float = 1.0,
    complete_only: bool = False,
) -> None:
    wx0, wy0, _, _ = pmeta["window_xywh"]
    x_lo, x_hi = xlim_win
    y_lo, y_hi = ylim_win
    margin = 0.75

    def _in_view(bx0: float, by0: float, bx1: float, by1: float) -> bool:
        if complete_only:
            return bx0 >= x_lo + margin and bx1 <= x_hi - margin and by0 >= y_lo + margin and by1 <= y_hi - margin
        return not (bx1 < x_lo or bx0 > x_hi or by1 < y_lo or by0 > y_hi)

    for gb in pmeta["gt_boxes"]:
        bx0, by0, bx1, by1 = gb["box_256"]
        gx0, gy0, gx1, gy1 = bx0 - wx0, by0 - wy0, bx1 - wx0, by1 - wy0
        if not _in_view(gx0, gy0, gx1, gy1):
            continue
        ax.add_patch(mpatches.Rectangle(
            (gx0, gy0), gx1 - gx0, gy1 - gy0,
            facecolor="none", edgecolor="#0A0D12",
            linewidth=2.0 * lw_scale, zorder=4
        ))
        ax.add_patch(mpatches.Rectangle(
            (gx0, gy0), gx1 - gx0, gy1 - gy0,
            facecolor="none", edgecolor="white",
            linewidth=1.15 * lw_scale, linestyle=(0, (2.4, 1.6)), zorder=5
        ))

    placed_labels: list[tuple[float, float]] = []
    for pb in pmeta["selected_predictions"]:
        bx0, by0, bx1, by1 = pb["box_256"]
        px0, py0, px1, py1 = bx0 - wx0, by0 - wy0, bx1 - wx0, by1 - wy0
        if not _in_view(px0, py0, px1, py1):
            continue
        is_tp = bool(pb["tp"])
        ec = PALETTE["tp_cyan"] if is_tp else PALETTE["fp_magenta"]
        ax.add_patch(mpatches.Rectangle(
            (px0, py0), px1 - px0, py1 - py0,
            facecolor="none", edgecolor="#0A0D12",
            linewidth=2.2 * lw_scale, alpha=0.80, zorder=6
        ))
        ax.add_patch(mpatches.Rectangle(
            (px0, py0), px1 - px0, py1 - py0,
            facecolor="none", edgecolor=ec,
            linewidth=1.35 * lw_scale, zorder=7
        ))
        if show_scores and pb["score"] is not None:
            sc_txt = f"{pb['score']:.2f}"
            candidates = [
                (max(px0, x_lo + 0.5), py0 - 0.5, "left", "bottom"),
                (min(px1, x_hi - 0.5), py1 + 0.6, "right", "top"),
                (min(px1, x_hi - 0.5), py0 - 0.5, "right", "bottom"),
                (max(px0, x_lo + 0.5), py1 + 0.6, "left", "top"),
            ]
            chosen = candidates[0]
            for cx_t, cy_t, ha_t, va_t in candidates:
                # Keep clear of frame edges and bottom-left 50 m scale bar zone
                if cy_t < y_lo + 2.2 or cy_t > y_hi - 2.6:
                    continue
                if va_t == "top" and cy_t > y_hi - 5.2 and cx_t < x_lo + 9.2:
                    continue
                if all(abs(cx_t - px) > 4.6 or abs(cy_t - py) > 2.5 for px, py in placed_labels):
                    chosen = (cx_t, cy_t, ha_t, va_t)
                    break
            tx, ty, ha, va = chosen
            placed_labels.append((tx, ty))
            ax.text(
                tx, ty, sc_txt, fontsize=6.7, fontweight="bold",
                color="white", ha=ha, va=va, zorder=9, clip_on=True,
                bbox=dict(boxstyle="square,pad=0.10", facecolor="#0A0D12", edgecolor="none", alpha=0.84)
            )


def build_figure5(out_dir: str, png_dpi: int = 0) -> str:
    npz, meta = load_real_bundle()
    panels_meta = meta["fig5_panels"]

    all_refs = [npz[f"fig5_rgb_ref_{i}"] for i in range(6)]
    stacked_vals = np.concatenate([a.reshape(-1, 3) for a in all_refs], axis=0)
    sh_lo = np.percentile(stacked_vals, 0.5, axis=0)
    sh_hi = np.percentile(stacked_vals, 98.5, axis=0)

    fig_w_mm, fig_h_mm = 190.0, 224.0
    fig = plt.figure(figsize=(fig_w_mm * MM, fig_h_mm * MM), facecolor="white")

    # ---- Two-Row Unified Top Legend (hollow boxes matching image styling) ----
    leg_ax = fig.add_axes([0.020, 0.930, 0.960, 0.064], frameon=False)
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")

    # Row 1: Detection box legend
    leg_ax.add_patch(mpatches.Rectangle((0.015, 0.60), 0.028, 0.27, facecolor="none", edgecolor="#1E293B", lw=1.7))
    leg_ax.add_patch(mpatches.Rectangle((0.015, 0.60), 0.028, 0.27, facecolor="none",
                                        edgecolor="white", lw=1.05, linestyle=(0, (2.2, 1.5))))
    leg_ax.text(0.050, 0.735, "MIN4 ground truth", fontsize=7.1, color=PALETTE["ink"], va="center")

    leg_ax.add_patch(mpatches.Rectangle((0.255, 0.60), 0.026, 0.27, facecolor="none", edgecolor="#0A0D12", lw=1.9))
    leg_ax.add_patch(mpatches.Rectangle((0.255, 0.60), 0.026, 0.27, facecolor="none", edgecolor=PALETTE["tp_cyan"], lw=1.25))
    leg_ax.text(0.289, 0.735, "Matched (IoU \u2265 0.50)", fontsize=7.1, color=PALETTE["ink"], va="center")

    leg_ax.add_patch(mpatches.Rectangle((0.525, 0.60), 0.026, 0.27, facecolor="none", edgecolor="#0A0D12", lw=1.9))
    leg_ax.add_patch(mpatches.Rectangle((0.525, 0.60), 0.026, 0.27, facecolor="none", edgecolor=PALETTE["fp_magenta"], lw=1.25))
    leg_ax.text(0.559, 0.735, "Unmatched prediction", fontsize=7.1, color=PALETTE["ink"], va="center")

    # Hollow L-corner bracket icon for Zoom ROI
    zx0, zy0, zw, zh = 0.785, 0.60, 0.024, 0.27
    za_x, za_y = 0.007, 0.085
    for lx_c, ly_c in [
        ([zx0, zx0, zx0 + za_x], [zy0 + za_y, zy0, zy0]),
        ([zx0 + zw - za_x, zx0 + zw, zx0 + zw], [zy0, zy0, zy0 + za_y]),
        ([zx0, zx0, zx0 + za_x], [zy0 + zh - za_y, zy0 + zh, zy0 + zh]),
        ([zx0 + zw - za_x, zx0 + zw, zx0 + zw], [zy0 + zh, zy0 + zh, zy0 + zh - za_y]),
    ]:
        leg_ax.plot(lx_c, ly_c, color="#1E293B", lw=1.15, solid_capstyle="butt")
    leg_ax.text(0.818, 0.735, "200 m zoom ROI", fontsize=7.1, color=PALETTE["ink"], va="center")

    # Row 2: MARIDA Semantic Class Mask legend (100% shared with Fig. 6!)
    leg_ax.text(0.015, 0.22, "MARIDA pixel mask", fontsize=7.1, fontweight="bold", color=PALETTE["ink"], va="center")
    mask_leg_items = [
        ("Debris", MASK_COLORS["debris"]),
        ("Algae / organic", MASK_COLORS["algae"]),
        ("Ship", MASK_COLORS["ship"]),
        ("Waves / foam", MASK_COLORS["foam"]),
        ("Marine water", MASK_COLORS["water"]),
        ("Unannotated", MASK_COLORS["unannotated"]),
    ]
    mx_positions = [0.185, 0.305, 0.468, 0.562, 0.720, 0.870]
    for (m_lbl, m_rgb_t), mx0 in zip(mask_leg_items, mx_positions):
        col_norm = tuple(c / 255.0 for c in m_rgb_t)
        leg_ax.add_patch(mpatches.Rectangle((mx0, 0.10), 0.018, 0.24, facecolor=col_norm, edgecolor="#334155", lw=0.5))
        leg_ax.text(mx0 + 0.023, 0.22, m_lbl, fontsize=7.0, color=PALETTE["ink"], va="center")

    leg_ax.plot([0.005, 0.995], [0.01, 0.01], color="#E2E8F0", linewidth=0.75)

    # ---- Perfectly symmetric full-width 3x2 grid (4.5 .. 185.5 mm, middle gap 8.6 mm) ----
    main_mm = 56.4
    sub_mm = 27.2
    sub_gap_mm = 2.0  # 27.2 * 2 + 2.0 = 56.4 mm exact match
    col_x_mm = [4.5, 99.3]  # 4.5 + 86.2 = 90.7; 99.3 - 90.7 = 8.6 mm gap; 99.3 + 86.2 = 185.5 mm
    row_y_top_mm = [197.8, 129.4, 61.0]

    for idx in range(6):
        r_i, c_i = divmod(idx, 2)
        pmeta = panels_meta[idx]
        cspec = CASE_SPECS[idx]

        x_left_mm = col_x_mm[c_i]
        y_bot_mm = row_y_top_mm[r_i] - main_mm

        ax_main = fig.add_axes([
            x_left_mm / fig_w_mm,
            y_bot_mm / fig_h_mm,
            main_mm / fig_w_mm,
            main_mm / fig_h_mm,
        ])
        ax_zrgb = fig.add_axes([
            (x_left_mm + main_mm + 2.6) / fig_w_mm,
            (y_bot_mm + sub_mm + sub_gap_mm) / fig_h_mm,
            sub_mm / fig_w_mm,
            sub_mm / fig_h_mm,
        ])
        ax_zmsk = fig.add_axes([
            (x_left_mm + main_mm + 2.6) / fig_w_mm,
            y_bot_mm / fig_h_mm,
            sub_mm / fig_w_mm,
            sub_mm / fig_h_mm,
        ])

        rgb_ref = npz[f"fig5_rgb_ref_{idx}"]
        disp_rgb = np.clip((rgb_ref - sh_lo) / (sh_hi - sh_lo), 0.0, 1.0)
        cl_64 = npz[f"fig5_labels_{idx}"]
        mask_rgb = colorize_marida_mask(cl_64)

        ax_main.imshow(disp_rgb, extent=[0, 64, 64, 0], interpolation="nearest")
        draw_vector_boxes_64(ax_main, pmeta, (0, 64), (0, 64), show_scores=False, lw_scale=0.95, complete_only=False)

        ax_main.add_patch(mpatches.FancyBboxPatch(
            (2.8, 54.4), 22.4, 7.0, boxstyle="square,pad=0.2",
            facecolor="#0A0D12", edgecolor="none", alpha=0.80, zorder=8
        ))
        ax_main.plot([4.0, 24.0], [59.8, 59.8], color="white", linewidth=1.6, solid_capstyle="butt", zorder=9)
        ax_main.text(14.0, 58.3, "200 m", color="white", fontsize=6.9,
                     ha="center", va="bottom", zorder=9)

        if cspec["roi_center"] is None:
            wx0, wy0, _, _ = pmeta["window_xywh"]
            tb = pmeta["target_box_256"]
            rcx = (tb[0] + tb[2]) / 2.0 - wx0
            rcy = (tb[1] + tb[3]) / 2.0 - wy0
        else:
            rcx, rcy = cspec["roi_center"]
        r_half = cspec["roi_size"] / 2.0
        rx0 = float(np.clip(rcx - r_half, 0.0, 64.0 - cspec["roi_size"]))
        ry0 = float(np.clip(rcy - r_half, 0.0, 64.0 - cspec["roi_size"]))
        rx1, ry1 = rx0 + cspec["roi_size"], ry0 + cspec["roi_size"]
        _draw_corner_brackets(ax_main, rx0, ry0, rx1, ry1, arm=4.5, color="white", lw=1.15, zorder=8)

        ax_main.set_xlim(0, 64)
        ax_main.set_ylim(64, 0)
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        for sp in ax_main.spines.values():
            sp.set_edgecolor("#334155")
            sp.set_linewidth(0.8)

        panel_label, panel_title = cspec["tag"].split(" ", 1)
        ax_main.text(0.0, 1.095, panel_label, transform=ax_main.transAxes,
                     fontsize=9, fontweight="bold", color=PALETTE["ink"], va="bottom", ha="left")
        ax_main.text(0.105, 1.095, panel_title, transform=ax_main.transAxes,
                     fontsize=8.2, fontweight="bold", color=PALETTE["ink"], va="bottom", ha="left")
        ax_main.text(
            0.0, 1.018, f"{cspec['region']} ({pmeta['date']})",
            transform=ax_main.transAxes, fontsize=7.0, color="#52525B", va="bottom", ha="left"
        )

        # Top-Right: Uniform 200 m x 200 m zoomed RGB + complete boxes only + 50 m scale bar
        ax_zrgb.imshow(disp_rgb, extent=[0, 64, 64, 0], interpolation="nearest")
        draw_vector_boxes_64(ax_zrgb, pmeta, (rx0, rx1), (ry0, ry1), show_scores=True, lw_scale=1.05, complete_only=True)
        ax_zrgb.set_xlim(rx0, rx1)
        ax_zrgb.set_ylim(ry1, ry0)
        ax_zrgb.set_xticks([])
        ax_zrgb.set_yticks([])
        for sp in ax_zrgb.spines.values():
            sp.set_edgecolor("#334155")
            sp.set_linewidth(0.8)

        # Crisp 50 m (= 5 px) scale bar in bottom-left of zoomed ROI (clear of boxes in all 6 panels)
        sb_x0 = rx0 + 1.1
        sb_x1 = sb_x0 + 5.0
        sb_y = ry1 - 1.2
        ax_zrgb.add_patch(mpatches.Rectangle(
            (sb_x0 - 0.7, sb_y - 2.6), 6.4, 3.2, facecolor="#0A0D12", edgecolor="none", alpha=0.80, zorder=9
        ))
        ax_zrgb.plot([sb_x0, sb_x1], [sb_y, sb_y], color="white", lw=1.4, solid_capstyle="butt", zorder=10)
        ax_zrgb.text((sb_x0 + sb_x1) / 2.0, sb_y - 0.45, "50 m", color="white", fontsize=6.4,
                     ha="center", va="bottom", zorder=10)

        # Bottom-Right: Uniform 200 m x 200 m zoomed semantic class mask + complete boxes only
        ax_zmsk.imshow(mask_rgb, extent=[0, 64, 64, 0], interpolation="nearest")
        draw_vector_boxes_64(ax_zmsk, pmeta, (rx0, rx1), (ry0, ry1), show_scores=False, lw_scale=0.95, complete_only=True)
        ax_zmsk.set_xlim(rx0, rx1)
        ax_zmsk.set_ylim(ry1, ry0)
        ax_zmsk.set_xticks([])
        ax_zmsk.set_yticks([])
        for sp in ax_zmsk.spines.values():
            sp.set_edgecolor("#334155")
            sp.set_linewidth(0.8)

        note_x, note_ha = (0.04, "left") if idx == 5 else (0.96, "right")
        ax_zmsk.text(
            note_x, 0.05, cspec["mask_note"], transform=ax_zmsk.transAxes,
            fontsize=6.8, color="white", ha=note_ha, va="bottom", zorder=10,
            bbox=dict(boxstyle="square,pad=0.14", facecolor="#0A0D12", edgecolor="none", alpha=0.82)
        )

    pdf_path = os.path.join(out_dir, "Fig5_Detection_Examples_MPB.pdf")
    fig.savefig(pdf_path, format="pdf", dpi=600)
    if png_dpi > 0:
        png_path = os.path.join(out_dir, "Fig5_Detection_Examples_MPB.png")
        fig.savefig(png_path, dpi=png_dpi)
        flatten_png_to_rgb(png_path, fallback_dpi=png_dpi)
    plt.close(fig)
    return pdf_path


# =============================================================================
# 2. GRAPHICAL ABSTRACT: Elsevier 2.5:1 (225 mm x 90 mm = 1328 x 531 px ratio)
# =============================================================================

def _draw_mask_outline(ax: plt.Axes, mask: np.ndarray, color: str = "#FFF4AD", lw: float = 0.55) -> None:
    """Draws crisp pixel-boundary vector contour around a binary mask."""
    m = mask.astype(bool)
    padded = np.pad(m, 1)
    segs = []
    for y, x in np.argwhere(m):
        if not padded[y, x + 1]:
            segs.append([(x, y), (x + 1, y)])
        if not padded[y + 2, x + 1]:
            segs.append([(x, y + 1), (x + 1, y + 1)])
        if not padded[y + 1, x]:
            segs.append([(x, y), (x, y + 1)])
        if not padded[y + 1, x + 2]:
            segs.append([(x + 1, y), (x + 1, y + 1)])
    if segs:
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=5))


def build_graphical_abstract(out_dir: str, export_ga_png: bool = True) -> str:
    """Builds an Elsevier 2.5:1 Graphical Abstract (225 x 90 mm = 1328 x 531 px)
    optimized for legibility at 13 x 5 cm screen thumbnail scale (>= 9.2-14 pt typography).
    """
    npz, _ = load_real_bundle()

    W, H = 225.0, 90.0
    fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    C_INK = "#1A1A1A"
    C_MUTED = "#52525B"
    C_RULE = "#D4D8DD"
    C_BLUE = "#2E5B88"
    C_BRICK = "#B5442F"
    C_OCHRE = "#D29A3A"
    C_NEUT = "#DCE2E8"

    # ---- Column Headers (11.5 pt bold) & Clean Hairline Dividers ----
    headers = [
        (3.5,   74.0, "1. Sentinel-2 leakage"),
        (80.5, 154.5, "2. Controlled audit"),
        (161.0, 221.5, "3. QA/QC workflow"),
    ]
    for x0, x1, htxt in headers:
        ax.text(x0, 84.8, htxt, fontsize=11.5, fontweight="bold", color=C_INK, va="center", ha="left")
        ax.plot([x0, x1], [80.4, 80.4], color=C_INK, lw=1.2)

    for vx in (77.2, 157.8):
        ax.plot([vx, vx], [4.5, 80.4], color=C_RULE, lw=0.75)

    # =========================================================================
    # COLUMN 1: Three Real Sentinel-2 Leakage Pathways
    # =========================================================================
    # Row A: Cross-dataset overlap (NO equivalence symbol; large 73.4% readout)
    p_mar = npz["overlap_marida_zoom64_u8"]
    p_mad = npz["overlap_mados_zoom64_u8"]
    m_mar = npz["overlap_marida_deb64"]
    m_mad = npz["overlap_mados_deb64"]

    for x_mm, arr_p, m_arr, b_ls, t_str in [
        (3.5,  p_mar, m_mar, "-",              "MARIDA"),
        (23.0, p_mad, m_mad, (0, (3.4, 2.0)), "MADOS"),
    ]:
        sub_a = fig.add_axes([x_mm / W, 54.6 / H, 17.5 / W, 17.5 / H])
        sub_a.imshow(arr_p, extent=(0, 64, 64, 0), interpolation="nearest")
        _draw_mask_outline(sub_a, m_arr, color="#FFF4AD", lw=0.6)
        sub_a.set_xlim(0, 64)
        sub_a.set_ylim(64, 0)
        sub_a.set_xticks([])
        sub_a.set_yticks([])
        for sp in sub_a.spines.values():
            sp.set_edgecolor(C_OCHRE)
            sp.set_linewidth(1.5)
            sp.set_linestyle(b_ls)
        ax.text(x_mm + 8.75, 75.2, t_str, fontsize=9.4, fontweight="bold", color=C_INK, ha="center", va="center")

    ax.text(58.5, 72.0, "73.4%", fontsize=14.2, fontweight="bold", color=C_INK, ha="center", va="center")
    bar_x, bar_w = 44.5, 28.0
    ax.add_patch(mpatches.Rectangle((bar_x, 65.4), bar_w, 3.0, facecolor=C_NEUT, edgecolor="none"))
    ax.add_patch(mpatches.Rectangle((bar_x, 65.4), bar_w * 0.7335, 3.0, facecolor=C_OCHRE, edgecolor="none"))
    ax.text(58.5, 59.2, "Same pass\nIoU = 1.00", fontsize=9.4, fontweight="bold",
            color=C_INK, ha="center", va="center", linespacing=1.18)

    ax.plot([3.5, 74.0], [51.6, 51.6], color="#E5E7EB", lw=0.7)

    # Row B: Label-derived shortcut (enlarged crisp arrow)
    or_mask_full = npz["oracle_mask_48"]
    or_dist_full = npz["oracle_dist_48"]
    ys_m, xs_m = np.where(or_mask_full > 0)
    cy_m = int(np.clip(round(float(ys_m.mean())), 14, 48 - 14))
    cx_m = int(np.clip(round(float(xs_m.mean())), 14, 48 - 14))
    or_mask = or_mask_full[cy_m - 14:cy_m + 14, cx_m - 14:cx_m + 14]
    or_dist = or_dist_full[cy_m - 14:cy_m + 14, cx_m - 14:cx_m + 14]

    for x_mm, img_o, cmap_o, t_o, edge_o, lw_o in [
        (3.5,  or_mask, "gray",    "Mask",   "#475569", 0.9),
        (23.5, or_dist, "inferno", "Oracle", C_BRICK,   1.5),
    ]:
        sub_o = fig.add_axes([x_mm / W, 29.8 / H, 14.5 / W, 14.5 / H])
        sub_o.imshow(img_o, cmap=cmap_o, vmin=0, vmax=1, interpolation="nearest")
        sub_o.set_xticks([])
        sub_o.set_yticks([])
        for sp in sub_o.spines.values():
            sp.set_edgecolor(edge_o)
            sp.set_linewidth(lw_o)
        ax.text(x_mm + 7.25, 47.4, t_o, fontsize=9.4, fontweight="bold", color=C_INK, ha="center", va="center")

    ax.annotate(
        "", xy=(22.8, 37.0), xytext=(18.6, 37.0),
        arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.7, mutation_scale=11),
        zorder=4
    )
    ax.text(41.2, 40.4, "Label-derived", fontsize=10.2, fontweight="bold", color=C_INK, va="center")
    ax.text(41.2, 33.4, "input (mask)", fontsize=9.4, color=C_MUTED, va="center")

    ax.plot([3.5, 74.0], [26.8, 26.8], color="#E5E7EB", lw=0.7)

    # Row C: Same-scene split
    cross_u8 = npz["pair_20_24_cross_u8"]
    cw_mm = 34.5
    ch_mm = cw_mm * (cross_u8.shape[0] / cross_u8.shape[1])
    sub_c = fig.add_axes([3.5 / W, 3.6 / H, cw_mm / W, ch_mm / H])
    sub_c.imshow(cross_u8, extent=(0, 192, 104, 0), interpolation="nearest")
    sub_c.set_xlim(0, 192)
    sub_c.set_ylim(104, 0)
    sub_c.set_xticks([])
    sub_c.set_yticks([])
    for sp in sub_c.spines.values():
        sp.set_visible(False)
    sub_c.plot([0, 96, 96, 0, 0], [0, 0, 104, 104, 0], color=C_OCHRE, lw=1.5, ls="-", clip_on=False, zorder=5)
    sub_c.plot([96, 192, 192, 96], [0, 0, 104, 104], color=C_OCHRE, lw=1.5, ls=(0, (3.4, 2.0)), clip_on=False, zorder=6)
    sub_c.plot([96, 96], [0, 104], color="white", lw=1.1, ls="--", zorder=7)
    ax.text(3.5 + cw_mm * 0.25, 24.5, "Train", fontsize=9.2, fontweight="bold", color=C_INK, ha="center", va="center")
    ax.text(3.5 + cw_mm * 0.75, 24.5, "Test", fontsize=9.2, fontweight="bold", color=C_INK, ha="center", va="center")

    ax.text(41.2, 16.0, "Same-scene", fontsize=10.6, fontweight="bold", color=C_INK, va="center")
    ax.text(41.2, 9.0, "Shared slick", fontsize=9.4, color=C_MUTED, va="center")

    # =========================================================================
    # COLUMN 2: Significant Leakage Inflation + Clean Baseline Headline
    # =========================================================================
    ax.text(80.5, 75.2, "Leakage inflation (\u0394AP50, 95% CI)", fontsize=10.2,
            fontweight="bold", color=C_INK, va="center")

    ax_f = fig.add_axes([109.2 / W, 35.8 / H, 29.0 / W, 33.5 / H], facecolor="white")
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["right"].set_visible(False)
    ax_f.spines["left"].set_color("#334155")
    ax_f.spines["bottom"].set_color("#334155")
    ax_f.spines["left"].set_linewidth(0.85)
    ax_f.spines["bottom"].set_linewidth(0.85)
    ax_f.axvline(0.0, color=C_INK, linestyle="--", lw=1.0, zorder=1)

    eff_rows = [
        (1, "Label-derived\ninput", 0.187, 0.089, 0.283, C_BRICK, "+0.187"),
        (0, "Same-scene\nexposure", 0.099, 0.023, 0.159, C_OCHRE, "+0.099"),
    ]
    for yi, _, m_v, l_v, h_v, c_v, num_s in eff_rows:
        ax_f.plot([l_v, h_v], [yi, yi], color=c_v, lw=2.6, zorder=3)
        ax_f.plot([l_v, l_v], [yi - 0.13, yi + 0.13], color=c_v, lw=1.4, zorder=3)
        ax_f.plot([h_v, h_v], [yi - 0.13, yi + 0.13], color=c_v, lw=1.4, zorder=3)
        ax_f.scatter([m_v], [yi], marker="D", facecolor=c_v, edgecolor="#1E293B",
                     linewidth=0.75, s=54, zorder=4)
        ax_f.text(
            1.06, yi, num_s, transform=ax_f.get_yaxis_transform(),
            fontsize=11.0, fontweight="bold",
            color=C_BRICK if yi == 1 else C_INK,
            ha="left", va="center", clip_on=False,
        )

    ax_f.set_yticks([1, 0])
    ax_f.set_yticklabels([r[1] for r in eff_rows], fontsize=9.2, fontweight="bold", color=C_INK)
    ax_f.set_ylim(-0.55, 1.55)
    ax_f.set_xlim(-0.02, 0.31)
    ax_f.set_xticks([0.00, 0.10, 0.20, 0.30])
    ax_f.set_xticklabels(["0.00", "+0.10", "+0.20", "+0.30"], fontsize=9.0)
    ax_f.tick_params(axis="x", length=2.8, width=0.8, pad=2.2)
    ax_f.tick_params(axis="y", length=2.8, width=0.8, pad=3.0)

    ax.plot([80.5, 154.5], [25.6, 25.6], color="#E5E7EB", lw=0.75)

    # Clean, high-impact single-sentence baseline conclusion (no non-significant C7 contrast!)
    ax.add_patch(mpatches.Rectangle((80.5, 6.2), 2.4, 15.6, facecolor=C_BLUE, edgecolor="none"))
    ax.text(85.8, 17.6, "Clean C6 (6-band): AP50 = 0.200", fontsize=10.8,
            fontweight="bold", color=C_BLUE, va="center", ha="left")
    ax.text(85.8, 9.8, "2.4\u20133.1\u00d7 RF (0.064\u20130.084) \u00b7 FDI 0.004", fontsize=9.2,
            color=C_INK, va="center", ha="left")

    # =========================================================================
    # Column 3: vertical numbered audit workflow
    # =========================================================================
    steps = [
        ("1", "Scene-grouped splits",      "No shared satellite pass"),
        ("2", "Archive deduplication",     "Remove overlap patches"),
        ("3", "Inference-time inputs only", "No label-derived inputs"),
        ("4", "Validation-locked test",    "Calibrate & audit errors"),
    ]
    cx_dot = 165.8
    ys_steps = [69.0, 50.2, 31.4, 12.6]
    ax.plot([cx_dot, cx_dot], [ys_steps[-1], ys_steps[0]], color="#CBD5E1", lw=1.6, zorder=1)
    for cy_s, (snum, stitle, ssub) in zip(ys_steps, steps):
        ax.add_patch(mpatches.Circle((cx_dot, cy_s), radius=4.1, facecolor=C_BLUE, edgecolor="none", zorder=3))
        ax.text(cx_dot, cy_s, snum, fontsize=10.8, fontweight="bold",
                color="white", ha="center", va="center", zorder=4)
        ax.text(cx_dot + 6.4, cy_s + 2.8, stitle, fontsize=9.8, fontweight="bold",
                color=C_INK, ha="left", va="center", zorder=4)
        ax.text(cx_dot + 6.4, cy_s - 3.0, ssub, fontsize=9.1,
                color=C_MUTED, ha="left", va="center", zorder=4)

    pdf_path = os.path.join(out_dir, "Graphical_Abstract_MPB.pdf")
    fig.savefig(pdf_path, format="pdf", dpi=600)

    if export_ga_png:
        tmp_png = os.path.join(out_dir, "_tmp_ga_hi.png")
        fig.savefig(tmp_png, dpi=600, facecolor="white")
        save_resized_png(
            tmp_png,
            os.path.join(out_dir, "Graphical_Abstract_MPB_1328x531.png"),
            (1328, 531),
        )
        if os.path.exists(tmp_png):
            os.remove(tmp_png)

    plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MPB-compliant vector Figure 5 and Graphical Abstract."
    )
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument("--png-dpi", type=int, default=0, help="Optional PNG DPI for Figure 5.")
    parser.add_argument("--export-ga-png", action="store_true", help="Also export 1328x531 px PNG for GA.")
    args, _ = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configure_matplotlib_style()

    f5 = build_figure5(args.out_dir, png_dpi=args.png_dpi)
    ga = build_graphical_abstract(args.out_dir, export_ga_png=args.export_ga_png)
    print(f"Generated vector Figure 5: {f5}")
    print(f"Generated vector Graphical Abstract: {ga}")


if __name__ == "__main__":
    main()
