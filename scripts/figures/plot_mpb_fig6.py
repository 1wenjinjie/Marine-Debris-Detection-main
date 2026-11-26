#!/usr/bin/env python3
"""Publication-ready vector plotting script for Figure 6 (Elsevier MPB).

Generates 190 mm double-column Figure 6 (190 mm x 204 mm):
  - (a-d) Four independent marine surface regimes (non-overlapping with Fig. 5)
    across RGB, NIR-SWIR-Red, C7 (+NDWI) channel, FDI + locked q=0.90 OOF boxes,
    and MARIDA labels + C6 (6-band) OOF detections.
  - (e) Empirical 11-band Sentinel-2 reflectance spectra (median & IQR error bars)
    on a true-wavelength broken x-axis (VNIR 420-900 nm // SWIR 1535-2280 nm),
    using the exact same class colors as the semantic masks in Figs. 5 and 6.
"""

from __future__ import annotations

import argparse
import json
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
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from mpb_typography import configure_typography, flatten_png_to_rgb


MM = 1.0 / 25.4

# Exact same class hex codes across Fig. 5 masks, Fig. 6 masks, and Fig. 6e spectra
PALETTE = {
    "ink": "#1A1A1A",
    "muted": "#52525B",
    "grid": "#E5E7EB",
    "debris": "#E11D48",
    "algae": "#10B981",
    "ship": "#EA580C",
    "foam": "#7F8C9A",
    "water": "#2B5B84",
    "fdi_box": "#4ADE80",
    "tp_cyan": "#00E5FF",
    "fp_magenta": "#E879F9",
}

MASK_COLORS = {
    "unannotated": (10, 13, 18),      # #0A0D12
    "debris":      (225, 29, 72),     # #E11D48
    "algae":       (16, 185, 129),    # #10B981
    "ship":        (234, 88, 12),     # #EA580C
    "foam":        (127, 140, 154),   # #7F8C9A
    "water":       (43, 91, 132),     # #2B5B84
    "cloud":       (203, 213, 225),   # #CBD5E1
}

TEXT_HALO = [patheffects.withStroke(linewidth=2.0, foreground="#0A0D12")]


def configure_matplotlib_style() -> None:
    plt.rcParams["axes.linewidth"] = 0.75
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


def colorize_marida_mask(cl_arr: np.ndarray) -> np.ndarray:
    out = np.empty((*cl_arr.shape, 3), dtype=np.uint8)
    out[:, :] = MASK_COLORS["unannotated"]
    out[np.isin(cl_arr, [7, 8, 10, 11, 15])] = MASK_COLORS["water"]
    out[np.isin(cl_arr, [6, 13])] = MASK_COLORS["cloud"]
    out[np.isin(cl_arr, [9, 12, 14])] = MASK_COLORS["foam"]
    out[cl_arr == 5] = MASK_COLORS["ship"]
    out[np.isin(cl_arr, [2, 3, 4])] = MASK_COLORS["algae"]
    out[cl_arr == 1] = MASK_COLORS["debris"]
    return out


def build_figure6(out_dir: str, png_dpi: int = 0) -> str:
    npz, meta = load_real_bundle()
    rows_meta = meta["fig6_rows"]
    stretch = meta["fig6_stretch"]
    spectra = meta["spectra_11band"]

    rgb_lo = np.array(stretch["rgb_lo"], dtype=np.float32)
    rgb_hi = np.array(stretch["rgb_hi"], dtype=np.float32)
    ns_lo = np.array(stretch["nirswir_lo"], dtype=np.float32)
    ns_hi = np.array(stretch["nirswir_hi"], dtype=np.float32)

    fig_w_mm, fig_h_mm = 190.0, 204.0
    fig = plt.figure(figsize=(fig_w_mm * MM, fig_h_mm * MM), facecolor="white")

    canvas_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    canvas_ax.set_xlim(0, fig_w_mm)
    canvas_ax.set_ylim(0, fig_h_mm)
    canvas_ax.axis("off")

    # ---- Top Legend Bar (hollow boxes matching Fig. 5 + unified mask colors) ----
    ly = 198.8
    canvas_ax.add_patch(mpatches.Rectangle((3.8, ly - 1.3), 4.2, 2.6, facecolor="none", edgecolor="#1E293B", lw=1.6))
    canvas_ax.add_patch(mpatches.Rectangle((3.8, ly - 1.3), 4.2, 2.6, facecolor="none", edgecolor="white",
                                           lw=1.0, linestyle=(0, (2.0, 1.4))))
    canvas_ax.text(9.0, ly, "MIN4 GT", fontsize=6.9, color=PALETTE["ink"], va="center")

    canvas_ax.add_patch(mpatches.Rectangle((22.5, ly - 1.3), 3.8, 2.6, facecolor="none", edgecolor="#0A0D12", lw=1.6))
    canvas_ax.add_patch(mpatches.Rectangle((22.5, ly - 1.3), 3.8, 2.6, facecolor="none", edgecolor=PALETTE["fdi_box"], lw=1.15))
    canvas_ax.text(27.3, ly, "FDI box (q = 0.90)", fontsize=6.9, color=PALETTE["ink"], va="center")

    canvas_ax.add_patch(mpatches.Rectangle((53.5, ly - 1.3), 3.8, 2.6, facecolor="none", edgecolor="#0A0D12", lw=1.6))
    canvas_ax.add_patch(mpatches.Rectangle((53.5, ly - 1.3), 3.8, 2.6, facecolor="none", edgecolor=PALETTE["tp_cyan"], lw=1.15))
    canvas_ax.text(58.3, ly, "C6 matched", fontsize=6.9, color=PALETTE["ink"], va="center")

    canvas_ax.add_patch(mpatches.Rectangle((76.0, ly - 1.3), 3.8, 2.6, facecolor="none", edgecolor="#0A0D12", lw=1.6))
    canvas_ax.add_patch(mpatches.Rectangle((76.0, ly - 1.3), 3.8, 2.6, facecolor="none", edgecolor=PALETTE["fp_magenta"], lw=1.15))
    canvas_ax.text(80.8, ly, "C6 unmatched", fontsize=6.9, color=PALETTE["ink"], va="center")

    mask_items = [
        ("Debris", MASK_COLORS["debris"], 103.5),
        ("Algae / organic", MASK_COLORS["algae"], 118.2),
        ("Ship", MASK_COLORS["ship"], 141.5),
        ("Foam / waves", MASK_COLORS["foam"], 153.2),
        ("Water", MASK_COLORS["water"], 175.0),
    ]
    for lbl_m, rgb_m, mx_mm in mask_items:
        c_norm = tuple(v / 255.0 for v in rgb_m)
        canvas_ax.add_patch(mpatches.Rectangle((mx_mm, ly - 1.3), 3.0, 2.6, facecolor=c_norm, edgecolor="#334155", lw=0.45))
        canvas_ax.text(mx_mm + 3.8, ly, lbl_m, fontsize=6.8, color=PALETTE["ink"], va="center")

    canvas_ax.plot([3.5, 186.5], [195.0, 195.0], color="#E2E8F0", linewidth=0.75)

    # ---- Compact Left Header Column + Enlarged 5-Column Imagery Grid ----
    tile_mm = 29.2
    col_gap_mm = 2.6
    col_x0_mm = 29.5  # 29.5 + 5*29.2 + 4*2.6 = 185.9 mm
    col_headers = [
        "True-color RGB\n(B4, B3, B2)",
        "NIR\u2013SWIR\u2013Red\n(B8, B11, B4)",
        "C7 (+NDWI) channel\n(1 \u2212 NDWI) / 2",
        "FDI & OOF boxes\n(locked q = 0.90)",
        "Labels &\nC6 (6-band) boxes",
    ]
    for c_i, chdr in enumerate(col_headers):
        cx_center = col_x0_mm + c_i * (tile_mm + col_gap_mm) + tile_mm / 2.0
        line1, line2 = chdr.split("\n", 1)
        canvas_ax.text(cx_center, 191.8, line1, fontsize=7.6, fontweight="bold",
                       color=PALETTE["ink"], ha="center", va="center")
        canvas_ax.text(cx_center, 188.3, line2, fontsize=6.9,
                       color=PALETTE["muted"], ha="center", va="center")

    # ---- 4 Independent Rows x 5 Columns of 64x64 Patches ----
    row_top_y_mm = [184.4, 153.2, 122.0, 90.8]  # bottom of row 3 = 61.6 mm
    im_ndwi_ref = None
    im_fdi_ref = None

    row_labels = [
        ("(a)", "Floating\ndebris", "Caribbean Sea,\nHaiti"),
        ("(b)", "Dense\nSargassum", "Gulf of\nHonduras"),
        ("(c)", "Vessel &\nwake", "Gulf of\nHonduras"),
        ("(d)", "Breaking\nwaves / foam", "Bali\u2013Lombok\nStrait"),
    ]

    c6_badges = []
    for r_i, rm in enumerate(rows_meta):
        cbs = rm["c6_boxes_256"]
        if not cbs:
            c6_badges.append("Patch: 0 boxes")
        else:
            n_tp = sum(1 for x in cbs if x.get("tp", False))
            n_unm = len(cbs) - n_tp
            top_sc = max(float(x["score"]) for x in cbs)
            if n_tp and not n_unm:
                c6_badges.append(f"Patch: {n_tp} matched\n(max {top_sc:.2f})")
            elif n_unm and not n_tp:
                c6_badges.append(f"Patch: {n_unm} unm.\n(max {top_sc:.2f})")
            else:
                c6_badges.append(f"Patch: {n_tp} matched,\n{n_unm} unm. (max {top_sc:.2f})")

    for r_i, rmeta in enumerate(rows_meta):
        y_bot = row_top_y_mm[r_i] - tile_mm
        wx0, wy0, _, _ = rmeta["win_xywh"]

        p_lbl, p_ttl, p_reg = row_labels[r_i]
        y_mid = y_bot + tile_mm * 0.54
        canvas_ax.text(3.5, y_mid + 5.0, p_lbl,
                       fontsize=8.8, fontweight="bold", color=PALETTE["ink"], va="top", ha="left")
        canvas_ax.text(8.6, y_mid + 5.0, p_ttl,
                       fontsize=7.9, fontweight="bold", color=PALETTE["ink"], va="top", ha="left", linespacing=1.14)
        canvas_ax.text(8.6, y_mid - 4.8, p_reg,
                       fontsize=6.7, color=PALETTE["muted"], va="top", ha="left", linespacing=1.14)

        rgb_arr = np.clip((npz[f"fig6_rgb_ref_{r_i}"] - rgb_lo) / (rgb_hi - rgb_lo), 0.0, 1.0)
        ns_arr = np.clip((npz[f"fig6_nirswir_ref_{r_i}"] - ns_lo) / (ns_hi - ns_lo), 0.0, 1.0)
        ndwi_arr = npz[f"fig6_c7ndwi_{r_i}"]
        fdi_arr = npz[f"fig6_fdi_{r_i}"]
        cl_rgb = colorize_marida_mask(npz[f"fig6_cl_{r_i}"])

        for c_i in range(5):
            x_left = col_x0_mm + c_i * (tile_mm + col_gap_mm)
            ax_t = fig.add_axes([
                x_left / fig_w_mm,
                y_bot / fig_h_mm,
                tile_mm / fig_w_mm,
                tile_mm / fig_h_mm,
            ])

            if c_i == 0:
                ax_t.imshow(rgb_arr, extent=[0, 64, 64, 0], interpolation="nearest")
                if r_i == 0:
                    ax_t.plot([4.0, 24.0], [59.5, 59.5], color="#0A0D12", lw=2.8, solid_capstyle="butt", zorder=5)
                    ax_t.plot([4.0, 24.0], [59.5, 59.5], color="white", lw=1.6, solid_capstyle="butt", zorder=6)
                    ax_t.text(14.0, 57.6, "200 m", color="white", fontsize=6.9, fontweight="bold",
                              ha="center", va="bottom", zorder=7, path_effects=TEXT_HALO)
            elif c_i == 1:
                ax_t.imshow(ns_arr, extent=[0, 64, 64, 0], interpolation="nearest")
            elif c_i == 2:
                im_ndwi_ref = ax_t.imshow(ndwi_arr, extent=[0, 64, 64, 0], cmap="viridis",
                                          vmin=0.15, vmax=0.72, interpolation="nearest")
            elif c_i == 3:
                im_fdi_ref = ax_t.imshow(fdi_arr, extent=[0, 64, 64, 0], cmap="magma",
                                         vmin=-0.008, vmax=0.048, interpolation="nearest")
                for fb in rmeta["fdi_boxes_256"]:
                    fx0, fy0, fx1, fy1 = fb[0] - wx0, fb[1] - wy0, fb[2] - wx0, fb[3] - wy0
                    if fx1 < 0 or fx0 > 64 or fy1 < 0 or fy0 > 64:
                        continue
                    ax_t.add_patch(mpatches.Rectangle(
                        (fx0, fy0), fx1 - fx0, fy1 - fy0,
                        facecolor="none", edgecolor=PALETTE["fdi_box"], linewidth=1.1, zorder=5
                    ))
                n_f = int(rmeta["n_fdi_patch"])
                ax_t.text(
                    0.96, 0.04, f"Patch: {n_f} {'box' if n_f == 1 else 'boxes'}", transform=ax_t.transAxes,
                    fontsize=6.8, fontweight="bold", color="white", ha="right", va="bottom",
                    zorder=7, path_effects=TEXT_HALO,
                )
            elif c_i == 4:
                ax_t.imshow(cl_rgb, extent=[0, 64, 64, 0], interpolation="nearest")
                for gb in rmeta["gt_boxes_256"]:
                    gx0, gy0, gx1, gy1 = gb[0] - wx0, gb[1] - wy0, gb[2] - wx0, gb[3] - wy0
                    if gx1 < 0 or gx0 > 64 or gy1 < 0 or gy0 > 64:
                        continue
                    ax_t.add_patch(mpatches.Rectangle(
                        (gx0, gy0), gx1 - gx0, gy1 - gy0,
                        facecolor="none", edgecolor="#0A0D12", lw=1.8, zorder=4
                    ))
                    ax_t.add_patch(mpatches.Rectangle(
                        (gx0, gy0), gx1 - gx0, gy1 - gy0,
                        facecolor="none", edgecolor="white", lw=1.05, linestyle=(0, (2.2, 1.4)), zorder=5
                    ))
                for cb_item in rmeta["c6_boxes_256"]:
                    bx = cb_item["box_256"]
                    px0, py0, px1, py1 = bx[0] - wx0, bx[1] - wy0, bx[2] - wx0, bx[3] - wy0
                    if px1 < 0 or px0 > 64 or py1 < 0 or py0 > 64:
                        continue
                    ec_c6 = PALETTE["tp_cyan"] if cb_item.get("tp", False) else PALETTE["fp_magenta"]
                    ax_t.add_patch(mpatches.Rectangle(
                        (px0, py0), px1 - px0, py1 - py0,
                        facecolor="none", edgecolor="#0A0D12", lw=2.0, zorder=5.5
                    ))
                    ax_t.add_patch(mpatches.Rectangle(
                        (px0, py0), px1 - px0, py1 - py0,
                        facecolor="none", edgecolor=ec_c6, lw=1.25, zorder=6
                    ))
                ax_t.text(
                    0.96, 0.04, c6_badges[r_i], transform=ax_t.transAxes,
                    fontsize=6.6, fontweight="bold", color="white", ha="right", va="bottom",
                    linespacing=1.15, zorder=7, path_effects=TEXT_HALO,
                )

            ax_t.set_xlim(0, 64)
            ax_t.set_ylim(64, 0)
            ax_t.set_xticks([])
            ax_t.set_yticks([])
            for sp in ax_t.spines.values():
                sp.set_edgecolor("#334155")
                sp.set_linewidth(0.75)

    # Compact horizontal colorbars (continuous + overflow triangles where clipped)
    if im_ndwi_ref is not None:
        c3_x0 = col_x0_mm + 2 * (tile_mm + col_gap_mm)
        cax_n = fig.add_axes([
            c3_x0 / fig_w_mm,
            57.2 / fig_h_mm,
            tile_mm / fig_w_mm,
            2.0 / fig_h_mm,
        ])
        cb_n = fig.colorbar(im_ndwi_ref, cax=cax_n, orientation="horizontal",
                            extend="both", ticks=[0.20, 0.45, 0.70])
        cb_n.ax.tick_params(labelsize=6.6, length=1.8, pad=1.0)
        canvas_ax.text(c3_x0 + tile_mm / 2.0, 51.6, "C7 (+NDWI)", fontsize=7.0, fontweight="bold",
                       color=PALETTE["ink"], ha="center", va="center")

    if im_fdi_ref is not None:
        c4_x0 = col_x0_mm + 3 * (tile_mm + col_gap_mm)
        cax_f = fig.add_axes([
            c4_x0 / fig_w_mm,
            57.2 / fig_h_mm,
            tile_mm / fig_w_mm,
            2.0 / fig_h_mm,
        ])
        cb_f = fig.colorbar(im_fdi_ref, cax=cax_f, orientation="horizontal",
                            extend="both", ticks=[0.00, 0.025, 0.045])
        cb_f.ax.set_xticklabels(["0.00", r"$t\approx$0.025", "0.045"])
        cb_f.ax.tick_params(labelsize=6.6, length=1.8, pad=1.0)
        canvas_ax.text(c4_x0 + tile_mm / 2.0, 51.6, "FDI", fontsize=7.0, fontweight="bold",
                       color=PALETTE["ink"], ha="center", va="center")

    canvas_ax.plot([3.5, 186.5], [49.0, 49.0], color="#D4D8DD", linewidth=0.75)

    # ---- Panel (e): True-Wavelength Broken-Axis 11-Band Reflectance Spectra ----
    canvas_ax.text(3.5, 46.6, "(e)", fontsize=9, fontweight="bold", color=PALETTE["ink"], va="center")
    canvas_ax.text(
        9.2, 46.6,
        "MARIDA 11-band reflectance (median & IQR, true wavelength with 900\u20131540 nm break)",
        fontsize=8.2, fontweight="bold", color=PALETTE["ink"], va="center"
    )

    ax_v = fig.add_axes([15.5 / fig_w_mm, 12.2 / fig_h_mm, 117.5 / fig_w_mm, 27.6 / fig_h_mm])
    ax_s = fig.add_axes([136.8 / fig_w_mm, 12.2 / fig_h_mm, 48.8 / fig_w_mm, 27.6 / fig_h_mm], sharey=ax_v)

    for ax_sub in (ax_v, ax_s):
        ax_sub.spines["top"].set_visible(False)
        ax_sub.spines["bottom"].set_color("#475569")
        ax_sub.grid(axis="both", color=PALETTE["grid"], linewidth=0.55, zorder=0)
        ax_sub.set_axisbelow(True)
    ax_v.spines["right"].set_visible(False)
    ax_v.spines["left"].set_color("#475569")
    ax_s.spines["left"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.tick_params(axis="y", left=False, labelleft=False)

    wls = np.array(spectra["wavelengths_nm"], dtype=float)
    bnames = spectra["band_names"]
    vnir_idx = np.arange(0, 9)   # B1..B8A (443..865 nm)
    swir_idx = np.arange(9, 11)  # B11..B12 (1614..2202 nm)

    # Vertical dashed band markers for NDWI and FDI
    for wl_m, tag_m in [(560, "NDWI"), (740, "FDI"), (833, "NDWI+FDI")]:
        ax_v.axvline(wl_m, color="#CBD5E1", linestyle="--", linewidth=0.8, zorder=1)
        ax_v.text(wl_m, 0.194, tag_m, fontsize=6.7, fontweight="bold", color=PALETTE["muted"],
                  ha="center", va="top",
                  bbox=dict(boxstyle="square,pad=0.06", facecolor="white", edgecolor="none", alpha=0.90))
    ax_s.axvline(1614, color="#CBD5E1", linestyle="--", linewidth=0.8, zorder=1)
    ax_s.text(1614, 0.194, "FDI", fontsize=6.7, fontweight="bold", color=PALETTE["muted"],
              ha="center", va="top",
              bbox=dict(boxstyle="square,pad=0.06", facecolor="white", edgecolor="none", alpha=0.90))

    spec_styles = [
        ("debris", "Debris (n = 3,399)",            PALETTE["debris"], "o", "-",  -4.4, -14.0),
        ("algae",  "Sargassum / organic (n = 5,760)", PALETTE["algae"], "s", "-",  -2.2,  -7.0),
        ("ship",   "Ships (n = 5,803)",             PALETTE["ship"],   "^", "--",  0.0,   0.0),
        ("foam",   "Foam / waves (n = 13,938)",     PALETTE["foam"],   "D", "-.",  2.2,   7.0),
        ("water",  "Water (n = 8,387)",             PALETTE["water"],  "v", ":",   4.4,  14.0),
    ]

    leg_handles = []
    for s_key, s_lbl, s_col, s_mk, s_ls, jit_v, jit_s in spec_styles:
        cls_d = spectra["classes"][s_key]
        med = np.array(cls_d["median"], dtype=float)
        p25 = np.array(cls_d["p25"], dtype=float)
        p75 = np.array(cls_d["p75"], dtype=float)

        # VNIR segment (exact wavelengths, thin IQR error bars with tiny horizontal offset)
        xv = wls[vnir_idx]
        mv = med[vnir_idx]
        ax_v.errorbar(
            xv + jit_v, mv, yerr=[mv - p25[vnir_idx], p75[vnir_idx] - mv],
            fmt="none", ecolor=s_col, elinewidth=0.85, alpha=0.62, capsize=0, zorder=2,
        )
        ln, = ax_v.plot(
            xv, mv, color=s_col, marker=s_mk, markersize=3.7,
            linewidth=1.55, linestyle=s_ls, label=s_lbl, zorder=4,
        )
        leg_handles.append(ln)

        # SWIR segment (disconnected across 900-1540 nm break!)
        xs_w = wls[swir_idx]
        ms = med[swir_idx]
        ax_s.errorbar(
            xs_w + jit_s, ms, yerr=[ms - p25[swir_idx], p75[swir_idx] - ms],
            fmt="none", ecolor=s_col, elinewidth=0.85, alpha=0.62, capsize=0, zorder=2,
        )
        ax_s.plot(
            xs_w, ms, color=s_col, marker=s_mk, markersize=3.7,
            linewidth=1.55, linestyle=s_ls, zorder=4,
        )

    ax_v.set_xlim(420, 898)
    ax_s.set_xlim(1535, 2280)
    ax_v.set_ylim(0.00, 0.205)
    ax_v.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax_v.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax_v.set_xticks(wls[vnir_idx])
    ax_v.set_xticklabels([f"{bnames[i]}\n{int(wls[i])}" for i in vnir_idx], fontsize=6.5)
    ax_s.set_xticks(wls[swir_idx])
    ax_s.set_xticklabels([f"{bnames[i]}\n{int(wls[i])}" for i in swir_idx], fontsize=6.5)
    ax_v.tick_params(axis="x", pad=2.0)
    ax_s.tick_params(axis="x", pad=2.0)

    # Diagonal break ticks on x-axis at 900 // 1535 nm
    d_y = 0.032
    d_xv = 0.011
    d_xs = d_xv * (117.5 / 48.8)
    ax_v.plot((1 - d_xv, 1 + d_xv), (-d_y, +d_y), transform=ax_v.transAxes,
              color="#475569", lw=0.95, clip_on=False)
    ax_s.plot((-d_xs, +d_xs), (-d_y, +d_y), transform=ax_s.transAxes,
              color="#475569", lw=0.95, clip_on=False)

    ax_v.set_ylabel(r"Reflectance ($\rho_{\mathrm{rc}}$)", fontweight="bold", color=PALETTE["ink"])
    canvas_ax.text(100.5, 3.6, "Sentinel-2 MSI band & central wavelength (nm)",
                   fontsize=7.6, fontweight="bold", color=PALETTE["ink"], ha="center", va="center")

    fig.legend(
        handles=leg_handles, loc="upper center", bbox_to_anchor=(0.525, 0.214),
        ncol=5, frameon=False, columnspacing=0.72, handletextpad=0.35, handlelength=1.6, fontsize=6.7,
    )

    pdf_path = os.path.join(out_dir, "Fig6_Multispectral_Index_Diagnostics_MPB.pdf")
    fig.savefig(pdf_path, format="pdf", dpi=600)
    if png_dpi > 0:
        png_path = os.path.join(out_dir, "Fig6_Multispectral_Index_Diagnostics_MPB.png")
        fig.savefig(png_path, dpi=png_dpi)
        flatten_png_to_rgb(png_path, fallback_dpi=png_dpi)
    plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MPB-compliant vector Figure 6 (Multi-spectral & Index Diagnostics)."
    )
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument("--png-dpi", type=int, default=0, help="Optional PNG DPI (e.g. 600).")
    args, _ = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configure_matplotlib_style()

    f6 = build_figure6(args.out_dir, png_dpi=args.png_dpi)
    print(f"Generated vector Figure 6: {f6}")


if __name__ == "__main__":
    main()
