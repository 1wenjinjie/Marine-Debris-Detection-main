#!/usr/bin/env python3
"""Publication-ready vector plotting script for MPB Supplementary Figures (S1–S3).

Generates three 190 mm Elsevier / Marine Pollution Bulletin compliant vector PDFs
and high-DPI PNGs:
  - FigS1_MIN4_Instance_Construction_MPB: Step-by-step visualization of sparse
    MARIDA polygon masks -> 4-connected component filtering (>=4 px retained vs
    1-3 px excluded) -> +-2 px padded MIN4 boxes + full-archive size breakdown.
  - FigS2_PLP2021_External_Target_Audit_MPB: Multi-date Gulf of Gera (Lesvos)
    Sentinel-2 L2A audit of the 28 m circular open-mesh HDPE target vs solid
    wooden-plank target, pairing spatial crops, Sen2Cor vs ACOLITE spectral domain shift,
    and validation-locked (0/21) vs diagnostic (conf>=0.001) recovery rates.
  - FigS3_Sensitivity_and_Fold_Diagnostics_MPB: Consolidated diagnostic & sensitivity
    panels moved from Main Text Figs. 3–4 — (a–b) 10-scene locked split & paired CIs,
    (c) per-fold AP50 heterogeneity across all 9 scene-grouped folds, and
    (d) localization tolerance sensitivity across IoU and center-distance criteria (Table S8).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
import shutil
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
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from scipy.ndimage import label

from mpb_typography import configure_typography, flatten_png_to_rgb
from plot_mpb_fig1_fig4 import FIG4_EVIDENCE, MATCHING_CRITERIA, read_rows
from plot_mpb_fig2_fig3 import (
    FIG3_SPLIT10,
    FIG3_SPLIT10_DIFFS,
    FOLD_AP50_C6,
    FOLD_AP50_C7,
    FOLD_AP50_FDI,
    FOLD_META,
    PALETTE as F3_PALETTE,
    fmt_half_up,
    fmt_signed,
    style_axis,
)


MM = 1.0 / 25.4
REPO_ROOT = Path(__file__).resolve().parents[2]
SUPP_OUT = REPO_ROOT / "figures_out" / "supplementary"

PALETTE = {
    "ink": "#1A1A1A",
    "muted": "#52525B",
    "grid": "#E5E7EB",
    "c6": "#2E5B88",
    "ndwi": "#3C7A78",
    "wood": "#C88A2E",
    "oracle": "#B5442F",
    "fdi": "#7C8A99",
    "debris": "#B5442F",
    "sub_min4": "#EAB308",
    "tp_cyan": "#38BDF8",
    "water": "#1E3A5F",
}


def configure_style() -> None:
    plt.rcParams.update({"axes.linewidth": 0.75, "axes.unicode_minus": True})
    configure_typography()


def panel_title(ax, letter: str, text: str, pad: int = 5) -> None:
    ax.annotate(f"({letter})", (0, 1), xycoords="axes fraction", xytext=(0, pad),
                textcoords="offset points", va="bottom", fontsize=9, fontweight="bold",
                color=PALETTE["ink"], annotation_clip=False)
    ax.annotate(text, (0, 1), xycoords="axes fraction", xytext=(15.5, pad),
                textcoords="offset points", va="bottom", fontsize=8.2, fontweight="bold",
                color=PALETTE["ink"], annotation_clip=False)


def stretch_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, p_lo: float = 2.0, p_hi: float = 99.2, gamma: float = 0.78) -> np.ndarray:
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    lo = np.nanpercentile(rgb, p_lo)
    hi = np.nanpercentile(rgb, p_hi)
    if hi <= lo:
        hi = lo + 1e-4
    norm = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0) ** gamma
    return norm


def build_figs1_min4(repo_root: Path, out_dir: Path, png_dpi: int) -> Path:
    patch_stem = repo_root / "data" / "raw" / "MARIDA" / "patches" / "S2_4-9-16_16PCC" / "S2_4-9-16_16PCC_20"
    with rasterio.open(str(patch_stem) + ".tif") as src:
        arr = src.read().astype(np.float32)
    with rasterio.open(str(patch_stem) + "_cl.tif") as src:
        cl = src.read(1)

    m_deb = (cl == 1).astype(np.uint8)
    lbl_full, _ = label(m_deb)
    # Deterministic 54x54 window where every retained MIN4 padded box sits >=3 px inside all edges
    x0, x1 = 185, 239
    y0, y1 = 168, 222

    rgb_full = stretch_rgb(arr[3], arr[2], arr[1], p_lo=3.0, p_hi=99.0, gamma=0.75)
    rgb_win = rgb_full[y0:y1, x0:x1]
    cl_win = cl[y0:y1, x0:x1]
    lbl_win = lbl_full[y0:y1, x0:x1]
    comp_sizes = np.bincount(lbl_full.ravel())

    ledger = repo_root / "data/processed/marida_mpb_v1/audit/marine_debris_components.csv"
    with ledger.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    areas_np = np.array([int(row["area_pixels"]) for row in rows if row["original_class"] == "1"])
    assert areas_np.size == 1425 and np.count_nonzero(areas_np >= 4) == 212
    assert areas_np.sum() == 3399

    fig = plt.figure(figsize=(190 * MM, 108 * MM), dpi=300, facecolor="white")
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        height_ratios=[1.08, 0.92],
        hspace=0.26, wspace=0.14,
        left=0.072, right=0.985, top=0.935, bottom=0.100,
    )

    H, W = cl_win.shape
    c_ret_light = (96, 165, 250)    # #60A5FA (same blue family as #2E5B88)
    c_exc_ochre = (210, 154, 58)    # #D29A3A (unified ochre)

    # Panel (a): RGB
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(rgb_win, interpolation="nearest")
    panel_title(ax_a, "a", "Sentinel-2 RGB (10 m)")
    ax_a.add_patch(mpatches.Rectangle((2.5, H - 8.2), 23.0, 5.8, facecolor="#0A0D12", alpha=0.82, edgecolor="none"))
    ax_a.plot([4.0, 24.0], [H - 4.2, H - 4.2], color="white", lw=1.8, solid_capstyle="butt")
    ax_a.text(14.0, H - 5.5, "200 m", color="white", fontsize=6.9, ha="center", va="bottom")

    # Panel (b): Pure binary sparse debris mask (white on black — zero unexplained dots or red clash)
    ax_b = fig.add_subplot(gs[0, 1])
    raw_vis = np.full((H, W, 3), (10, 13, 18), dtype=np.uint8)
    raw_vis[cl_win == 1] = (248, 250, 252)
    ax_b.imshow(raw_vis, interpolation="nearest")
    panel_title(ax_b, "b", "Sparse debris mask")
    ax_b.text(
        2.5, H - 2.8, f"Debris pixels in ROI: {int((cl_win == 1).sum())} px",
        color="white", fontsize=6.7,
        bbox=dict(boxstyle="square,pad=0.16", facecolor="#0A0D12", alpha=0.84, edgecolor="none"),
    )

    # Panel (c): 4-connected area filter (blue retained >=4 px, ochre excluded <4 px)
    ax_c = fig.add_subplot(gs[0, 2])
    flt_vis = np.full((H, W, 3), (10, 13, 18), dtype=np.uint8)
    n_ge4, n_lt4 = 0, 0
    for cid in np.unique(lbl_win):
        if cid == 0:
            continue
        if comp_sizes[cid] >= 4:
            flt_vis[lbl_win == cid] = c_ret_light
            n_ge4 += 1
        else:
            flt_vis[lbl_win == cid] = c_exc_ochre
            n_lt4 += 1
    ax_c.imshow(flt_vis, interpolation="nearest")
    panel_title(ax_c, "c", "4-neighbour area filter")
    ax_c.text(
        2.5, H - 2.8, f"Retained \u22654 px: {n_ge4}   <4 px: {n_lt4}",
        color="white", fontsize=6.7,
        bbox=dict(boxstyle="square,pad=0.16", facecolor="#0A0D12", alpha=0.84, edgecolor="none"),
    )

    # Panel (d): Final MIN4 padded boxes (+2 px) in blue + ochre excluded circles (zero edge clipping)
    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.imshow(rgb_win, interpolation="nearest")
    for cid in np.unique(lbl_win):
        if cid == 0:
            continue
        ys_c, xs_c = np.where(lbl_full == cid)
        if len(ys_c) >= 4:
            bx0 = max(0, int(xs_c.min()) - 2) - x0 - 0.5
            by0 = max(0, int(ys_c.min()) - 2) - y0 - 0.5
            bx1 = min(255, int(xs_c.max()) + 2) - x0 + 0.5
            by1 = min(255, int(ys_c.max()) + 2) - y0 + 0.5
            ax_d.add_patch(mpatches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, facecolor="none", edgecolor="#0A0D12", lw=2.0, zorder=4))
            ax_d.add_patch(mpatches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, facecolor="none", edgecolor="#60A5FA", lw=1.25, zorder=5))
        else:
            mx, my = float(xs_c.mean()) - x0, float(ys_c.mean()) - y0
            if 1 <= mx <= W - 2 and 1 <= my <= H - 2:
                ax_d.add_patch(mpatches.Circle((mx, my), radius=2.1, facecolor="none", edgecolor="#0A0D12", lw=1.9, zorder=5.5))
                ax_d.add_patch(mpatches.Circle((mx, my), radius=2.1, facecolor="none", edgecolor="#D29A3A", lw=1.15, zorder=6))
    panel_title(ax_d, "d", "MIN4 boxes (+2 px pad)")
    ax_d.text(
        2.5, H - 2.8, "Blue: MIN4   Ochre: <4 px",
        color="white", fontsize=6.7,
        bbox=dict(boxstyle="square,pad=0.16", facecolor="#0A0D12", alpha=0.84, edgecolor="none"),
    )

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#94A3B8")
            sp.set_linewidth(0.75)

    # Panel (e): Archive-wide component & pixel breakdown (uniform Ochre vs Deep Navy)
    ax_e = fig.add_subplot(gs[1, :])
    cats = [
        "1 px\n(excluded)",
        "2 px\n(excluded)",
        "3 px\n(excluded)",
        "4\u20137 px\n(MIN4 small)",
        "8\u201315 px\n(MIN4 medium)",
        "\u226516 px\n(MIN4 large)",
    ]
    counts = [
        int(np.sum(areas_np == 1)),
        int(np.sum(areas_np == 2)),
        int(np.sum(areas_np == 3)),
        int(np.sum((areas_np >= 4) & (areas_np <= 7))),
        int(np.sum((areas_np >= 8) & (areas_np <= 15))),
        int(np.sum(areas_np >= 16)),
    ]
    exc_px = int(areas_np[areas_np < 4].sum())
    ret_px = int(areas_np[areas_np >= 4].sum())
    tot_px = int(areas_np.sum())

    colors = ["#D29A3A"] * 3 + ["#2E5B88"] * 3
    xs = np.arange(len(cats))
    ax_e.set_axisbelow(True)
    ax_e.grid(axis="y", color=PALETTE["grid"], lw=0.6, zorder=0)
    bars = ax_e.bar(xs, counts, width=0.54, color=colors, edgecolor="#1E293B", linewidth=0.65, zorder=3)
    ax_e.axvline(2.5, color="#64748B", linestyle="--", lw=1.0, zorder=4)
    ax_e.set_xticks(xs)
    ax_e.set_xticklabels(cats, color=PALETTE["ink"], fontsize=7.2)
    ax_e.set_ylabel("Connected components\n(4-neighbour)", fontsize=7.6, fontweight="bold", color=PALETTE["ink"])
    ax_e.set_ylim(0, 860)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    total_c = len(areas_np)
    for b, cnt in zip(bars, counts):
        pct = 100.0 * cnt / total_c
        ax_e.text(
            b.get_x() + b.get_width() / 2.0, cnt + 15,
            f"{cnt} ({pct:.1f}%)",
            ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=PALETTE["ink"],
        )

    ax_e.text(
        1.15, 835,
        f"Excluded (<4 px): 85.1% of components ({sum(counts[:3]):,} / {total_c:,})\n"
        f"{100.0 * exc_px / tot_px:.1f}% of debris pixels ({exc_px:,} / {tot_px:,})",
        ha="center", va="top", fontsize=7.1, fontweight="bold", color=PALETTE["ink"], linespacing=1.22,
    )
    ax_e.text(
        4.10, 835,
        f"Retained MIN4 (\u22654 px): 14.9% ({sum(counts[3:]):,} instances)\n"
        f"{100.0 * ret_px / tot_px:.1f}% of debris pixels ({ret_px:,} / {tot_px:,})",
        ha="center", va="top", fontsize=7.1, fontweight="bold", color=PALETTE["ink"], linespacing=1.22,
    )
    panel_title(ax_e, "e", "MARIDA four-neighbour debris components and pixel share (1,381 patches)")

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "FigS1_MIN4_Instance_Construction_MPB.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=600)
    if png_dpi > 0:
        png_path = out_dir / "FigS1_MIN4_Instance_Construction_MPB.png"
        fig.savefig(png_path, dpi=png_dpi)
        flatten_png_to_rgb(png_path, fallback_dpi=png_dpi)
    plt.close(fig)
    return pdf_path


def build_figs2_plp2021(repo_root: Path, out_dir: Path, png_dpi: int) -> Path:
    plp_dir = repo_root / "data" / "PLP" / "PLP2021" / "Sentinel-2"
    targets_csv = repo_root / "runs_mpb" / "external_plp2021" / "targets.csv"
    preds_json = repo_root / "runs_mpb_d4" / "external_plp2021" / "full_scene_6ch_s0_conf001.json"

    targets_by_date: dict[str, list[dict]] = {}
    with open(targets_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            targets_by_date.setdefault(row["date"], []).append(row)

    preds_data = json.loads(preds_json.read_text(encoding="utf-8"))
    preds_by_date: dict[str, list[dict]] = {}
    for pr in preds_data["predictions"]:
        preds_by_date.setdefault(str(pr["date"]), []).append(pr)

    all_tifs = {Path(p).name[:8]: Path(p) for p in sorted(glob.glob(str(plp_dir / "*.tif")))}

    band_idx_11 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    hdpe_specs, wood_specs, water_specs = [], [], []
    for dt, trows in sorted(targets_by_date.items()):
        if dt not in all_tifs:
            continue
        with rasterio.open(all_tifs[dt]) as src:
            cube = src.read().astype(np.float32) / 10000.0
            for tr in trows:
                e, n = float(tr["easting"]), float(tr["northing"])
                rr, cc = src.index(e, n)
                patch3 = cube[:, max(0, rr - 1):rr + 2, max(0, cc - 1):cc + 2]
                b8_sub = patch3[7]
                mr, mc = np.unravel_index(np.argmax(b8_sub), b8_sub.shape)
                spec11 = patch3[band_idx_11, mr, mc]
                if tr["material"] == "HDPE" and not tr["note"]:
                    hdpe_specs.append(spec11)
                elif tr["material"] == "wood":
                    wood_specs.append(spec11)
            rr0, cc0 = src.index(float(trows[0]["easting"]) - 140.0, float(trows[0]["northing"]) + 60.0)
            water_specs.append(cube[band_idx_11, rr0, cc0])

    hdpe_arr = np.array(hdpe_specs)
    wood_arr = np.array(wood_specs)
    water_arr = np.array(water_specs)

    meta_json = Path(__file__).resolve().parent / "assets" / "mpb_real_imagery_meta.json"
    spec_meta = json.loads(meta_json.read_text(encoding="utf-8"))["spectra_11band"]
    spectra = spec_meta["classes"]["debris"]
    assert spectra["n_pixels"] == 3399
    marida_med, marida_q25, marida_q75 = (np.array(spectra[k]) for k in ("median", "p25", "p75"))
    wls = np.array(spec_meta["wavelengths_nm"], dtype=float)
    bnames = spec_meta["band_names"]

    fig = plt.figure(figsize=(190 * MM, 132 * MM), dpi=300, facecolor="white")
    gs_top = gridspec.GridSpec(1, 4, figure=fig, left=0.04, right=0.985, top=0.895, bottom=0.535, wspace=0.14)

    # Top legend strip for (a)-(d) imagery annotations
    top_leg_handles = [
        mlines.Line2D([], [], color="#60A5FA", marker="o", linestyle="none", markersize=6.2,
                      markerfacecolor="none", markeredgewidth=1.4, label="HDPE mesh (\u00d8 28 m)"),
        mlines.Line2D([], [], color="#E5A93C", marker="o", linestyle="none", markersize=6.2,
                      markerfacecolor="none", markeredgewidth=1.4, label="Wood planks"),
        mlines.Line2D([], [], color="#FB923C", marker="x", linestyle="none", markersize=5.5,
                      markeredgewidth=1.5, label="C6 (6-band) diagnostic hit (score \u2265 0.001)"),
    ]
    fig.legend(
        handles=top_leg_handles, loc="upper center", bbox_to_anchor=(0.51, 0.992),
        ncol=3, frameon=False, columnspacing=1.5, handletextpad=0.35, fontsize=7.0,
    )

    show_dates = [
        ("20210621", "21 Jun RGB (nominal)"),
        ("20210711", "11 Jul RGB (biofouled)"),
        ("20210820", "20 Aug RGB (submerged)"),
    ]
    hdpe_ring_col = "#60A5FA"
    wood_ring_col = "#E5A93C"

    half_w = 14
    for idx, (dt, title_txt) in enumerate(show_dates):
        ax = fig.add_subplot(gs_top[0, idx])
        with rasterio.open(all_tifs[dt]) as src:
            cube = src.read().astype(np.float32) / 10000.0
            trows = targets_by_date[dt]
            hdpe_tr = next(t for t in trows if t["material"] == "HDPE")
            r_c, c_c = src.index(float(hdpe_tr["easting"]), float(hdpe_tr["northing"]))
            r_mid, c_mid = r_c + 2, c_c
            y0, y1 = r_mid - half_w, r_mid + half_w + 1
            x0, x1 = c_mid - half_w, c_mid + half_w + 1

            rgb = stretch_rgb(cube[3, y0:y1, x0:x1], cube[2, y0:y1, x0:x1], cube[1, y0:y1, x0:x1],
                              p_lo=2.0, p_hi=99.4, gamma=0.78)
            ax.imshow(rgb, interpolation="nearest")

            for tr in trows:
                tr_r, tr_c = src.index(float(tr["easting"]), float(tr["northing"]))
                ly, lx = tr_r - y0, tr_c - x0
                if tr["material"] == "HDPE":
                    ax.add_patch(mpatches.Circle((lx, ly), radius=2.4, facecolor="none",
                                                 edgecolor=hdpe_ring_col, lw=1.4, linestyle="-", zorder=5))
                    ax.text(lx + 2.8, ly - 0.8, "HDPE mesh\n(\u00d8 28 m)", color="white", fontsize=6.4,
                            linespacing=1.08,
                            bbox=dict(boxstyle="square,pad=0.12", facecolor="#0A0D12", alpha=0.80, edgecolor="none"))
                else:
                    ax.add_patch(mpatches.Circle((lx, ly), radius=2.4, facecolor="none",
                                                 edgecolor=wood_ring_col, lw=1.35, linestyle="--", zorder=5))
                    ax.text(lx + 2.8, ly + 1.4, "Wood planks", color="white", fontsize=6.4,
                            bbox=dict(boxstyle="square,pad=0.12", facecolor="#0A0D12", alpha=0.80, edgecolor="none"))

            for pr in preds_by_date.get(dt, []):
                pr_r, pr_c = src.index(float(pr["center_easting"]), float(pr["center_northing"]))
                py, px = pr_r - y0, pr_c - x0
                if 0 <= px < (x1 - x0) and 0 <= py < (y1 - y0):
                    sc = float(pr["confidence"])
                    ax.scatter([px], [py], marker="x", s=22, color="#FB923C", linewidths=1.45, zorder=6)
                    ax.text(px, py + 4.2, f"C6 score {sc:.2f}", color="white", fontsize=6.4, ha="center", va="top",
                            bbox=dict(boxstyle="square,pad=0.12", facecolor="#0A0D12", alpha=0.86, edgecolor="none"))

            if idx == 0:
                H_w = y1 - y0
                ax.add_patch(mpatches.Rectangle((1.4, H_w - 5.2), 11.6, 4.0, facecolor="#0A0D12", alpha=0.84, edgecolor="none"))
                ax.plot([2.2, 12.2], [H_w - 2.3, H_w - 2.3], color="white", lw=1.7, solid_capstyle="butt")
                ax.text(7.2, H_w - 2.9, "100 m", color="white", fontsize=6.5, ha="center", va="bottom")

        panel_title(ax, "abc"[idx], title_txt)
        ax.set_xticks([])
        ax.set_yticks([])

    # Panel (d): FDI response on 2021-06-21 with equal-interval colorbar & shoreline annotation
    ax_fdi = fig.add_subplot(gs_top[0, 3])
    with rasterio.open(all_tifs["20210621"]) as src:
        cube = src.read().astype(np.float32) / 10000.0
        trows = targets_by_date["20210621"]
        hdpe_tr = next(t for t in trows if t["material"] == "HDPE")
        r_c, c_c = src.index(float(hdpe_tr["easting"]), float(hdpe_tr["northing"]))
        r_mid, c_mid = r_c + 2, c_c
        y0, y1 = r_mid - half_w, r_mid + half_w + 1
        x0, x1 = c_mid - half_w, c_mid + half_w + 1
        assert (src.descriptions[5], src.descriptions[7], src.descriptions[10]) == ("B6", "B8", "B11")
        b6, b8, b11 = cube[[5, 7, 10], y0:y1, x0:x1]
        fdi = b8 - (b6 + (b11 - b6) * (832.8 - 664.6) / (1613.7 - 664.6) * 10)
        im_f = ax_fdi.imshow(fdi, cmap="magma", vmin=-0.10, vmax=0.50, interpolation="nearest")
        for tr in trows:
            tr_r, tr_c = src.index(float(tr["easting"]), float(tr["northing"]))
            ly, lx = tr_r - y0, tr_c - x0
            ec = hdpe_ring_col if tr["material"] == "HDPE" else wood_ring_col
            ls = "-" if tr["material"] == "HDPE" else "--"
            ax_fdi.add_patch(mpatches.Circle((lx, ly), radius=2.4, facecolor="none", edgecolor=ec, lw=1.35, linestyle=ls))
        ax_fdi.text(
            1.5, (y1 - y0) - 2.0, "Shoreline (FDI < 0)", color="#E2E8F0", fontsize=6.2,
            ha="left", va="bottom",
        )
    panel_title(ax_fdi, "d", "21 Jun FDI")
    ax_fdi.set_xticks([])
    ax_fdi.set_yticks([])
    fdi_bbox = ax_fdi.get_position()
    cax = fig.add_axes([fdi_bbox.x0 + 0.008, fdi_bbox.y0 - 0.022, fdi_bbox.width - 0.016, 0.013])
    cb = fig.colorbar(im_f, cax=cax, orientation="horizontal", extend="both", ticks=[-0.10, 0.10, 0.30, 0.50])
    cb.ax.set_xticklabels(["\u22120.10", "0.10", "0.30", "0.50"])
    cax.tick_params(labelsize=6.6, length=1.6, pad=1.0)

    # Panel (e): True-Wavelength Broken-Axis Reflectance Spectra (VNIR 420-898 // SWIR 1535-2280 nm)
    sp_bot, sp_h = 0.095, 0.270
    ax_sp_v = fig.add_axes([0.088, sp_bot, 0.345, sp_h])
    ax_sp_s = fig.add_axes([0.448, sp_bot, 0.132, sp_h], sharey=ax_sp_v)
    for ax_s in (ax_sp_v, ax_sp_s):
        ax_s.spines["top"].set_visible(False)
        ax_s.spines["bottom"].set_color("#475569")
        ax_s.grid(True, color=PALETTE["grid"], lw=0.55, zorder=0)
        ax_s.set_axisbelow(True)
    ax_sp_v.spines["right"].set_visible(False)
    ax_sp_v.spines["left"].set_color("#475569")
    ax_sp_s.spines["left"].set_visible(False)
    ax_sp_s.spines["right"].set_visible(False)
    ax_sp_s.tick_params(axis="y", left=False, labelleft=False)

    vnir_idx = np.arange(0, 9)
    swir_idx = np.arange(9, 11)
    w_med = np.median(wood_arr, axis=0)
    w_q25, w_q75 = np.percentile(wood_arr, [25, 75], axis=0)
    h_med = np.median(hdpe_arr, axis=0)
    h_q25, h_q75 = np.percentile(hdpe_arr, [25, 75], axis=0)
    wt_med = np.median(water_arr, axis=0)

    series_s2 = [
        ("Wood planks (Sen2Cor L2A)",          PALETTE["wood"], "s", "-",  w_med,      w_q25,      w_q75,      0.16),
        ("HDPE mesh (Sen2Cor L2A)",            PALETTE["c6"],   "^", "-",  h_med,      h_q25,      h_q75,      0.16),
        (r"MARIDA debris (ACOLITE $\rho_{\mathrm{rc}}$)", "#475569", "o", "--", marida_med, marida_q25, marida_q75, 0.11),
        ("PLP2021 water (Sen2Cor L2A)",        "#64748B",       None, ":", wt_med,     None,       None,       0.0),
    ]
    sp_handles = []
    for s_lbl, s_col, s_mk, s_ls, s_med, s_q25, s_q75, s_alp in series_s2:
        if s_q25 is not None:
            ax_sp_v.fill_between(wls[vnir_idx], s_q25[vnir_idx], s_q75[vnir_idx], color=s_col, alpha=s_alp, zorder=2)
            ax_sp_s.fill_between(wls[swir_idx], s_q25[swir_idx], s_q75[swir_idx], color=s_col, alpha=s_alp, zorder=2)
        ln, = ax_sp_v.plot(wls[vnir_idx], s_med[vnir_idx], color=s_col, lw=1.6, linestyle=s_ls,
                           marker=s_mk, ms=3.4 if s_mk else 0, label=s_lbl, zorder=4)
        ax_sp_s.plot(wls[swir_idx], s_med[swir_idx], color=s_col, lw=1.6, linestyle=s_ls,
                     marker=s_mk, ms=3.4 if s_mk else 0, zorder=4)
        sp_handles.append(ln)

    ax_sp_v.set_xlim(420, 898)
    ax_sp_s.set_xlim(1535, 2280)
    ax_sp_v.set_ylim(-0.005, 0.305)
    ax_sp_v.set_xticks([443, 492, 560, 665, 740, 833])
    ax_sp_v.set_xticklabels(["B1\n443", "B2\n492", "B3\n560", "B4\n665", "B6\n740", "B8\n833"], fontsize=6.4)
    ax_sp_s.set_xticks([1614, 2202])
    ax_sp_s.set_xticklabels(["B11\n1614", "B12\n2202"], fontsize=6.4)
    ax_sp_v.set_ylabel("Reflectance\n(product-specific)", fontsize=7.5, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.335, 0.020, "Sentinel-2 MSI central wavelength (nm, 900\u20131540 nm break)",
             fontsize=7.2, fontweight="bold", color=PALETTE["ink"], ha="center", va="center")

    d_y = 0.032
    ax_sp_v.plot((1 - 0.016, 1 + 0.016), (-d_y, +d_y), transform=ax_sp_v.transAxes,
                 color="#475569", lw=0.9, clip_on=False)
    ax_sp_s.plot((-0.042, +0.042), (-d_y, +d_y), transform=ax_sp_s.transAxes,
                 color="#475569", lw=0.9, clip_on=False)

    ax_sp_v.legend(
        handles=sp_handles, loc="lower left", bbox_to_anchor=(0.0, 1.03), borderaxespad=0,
        frameon=False, ncol=2, columnspacing=0.85, handlelength=1.45, handletextpad=0.35, fontsize=6.5,
    )
    panel_title(ax_sp_v, "e", "Spectral shapes in PLP2021 and MARIDA samples", pad=28)

    # Panel (f): Target recovery bar chart (unified 3-seed means + gridlines behind bars!)
    ax_bar = fig.add_axes([0.665, sp_bot, 0.320, sp_h])
    plp_d4 = json.loads((repo_root / "runs_mpb_d4/external_plp2021/summary.json").read_text(encoding="utf-8"))["cells"]["full_scene_6ch"]
    locked = plp_d4["validation_locked_per_seed"]
    diagnostic = plp_d4["conf001_diagnostic_per_seed"]
    hdpe_k, wood_k = "hdpe_target_recovery_primary_24m", "wood_target_marine_debris_assignments_24m"
    groups = ["Locked validation\nthreshold", "Diagnostic floor\n(score \u2265 0.001)"]
    hdpe_locked = [100 * row[hdpe_k]["rate"] for row in locked]
    wood_locked = [100 * row[wood_k]["rate"] for row in locked]
    hdpe_diag = [100 * row[hdpe_k]["rate"] for row in diagnostic]
    wood_diag = [100 * row[wood_k]["rate"] for row in diagnostic]
    hdpe_rates = [float(np.mean(hdpe_locked)), float(np.mean(hdpe_diag))]
    wood_rates = [float(np.mean(wood_locked)), float(np.mean(wood_diag))]
    wood_locked_det = float(np.mean([r[wood_k]["detected"] for r in locked]))
    hdpe_diag_det = float(np.mean([r[hdpe_k]["detected"] for r in diagnostic]))
    wood_diag_det = float(np.mean([r[wood_k]["detected"] for r in diagnostic]))

    xg = np.arange(len(groups))
    bw = 0.31
    ax_bar.set_axisbelow(True)
    ax_bar.grid(axis="y", color=PALETTE["grid"], lw=0.6, zorder=0)
    b1 = ax_bar.bar(xg - bw / 2 - 0.02, hdpe_rates, width=bw, color=PALETTE["c6"], edgecolor="#1E293B", lw=0.7,
                    zorder=3, label="HDPE mesh (n = 21)")
    b2 = ax_bar.bar(xg + bw / 2 + 0.02, wood_rates, width=bw, color=PALETTE["wood"], edgecolor="#1E293B", lw=0.7,
                    zorder=3, label="Wood planks (n = 15)")

    for gi, (vals_h, vals_w) in enumerate([(hdpe_locked, wood_locked), (hdpe_diag, wood_diag)]):
        xh = xg[gi] - bw / 2 - 0.02
        xw = xg[gi] + bw / 2 + 0.02
        for si, (vh, vw) in enumerate(zip(vals_h, vals_w)):
            ax_bar.scatter([xh + (si - 1) * 0.06], [vh], marker="o", s=14, facecolor="white",
                           edgecolor="#1E293B", lw=0.85, zorder=5)
            ax_bar.scatter([xw + (si - 1) * 0.06], [vw], marker="o", s=14, facecolor="white",
                           edgecolor="#1E293B", lw=0.85, zorder=5)

    ax_bar.set_xticks(xg)
    ax_bar.set_xticklabels(groups, fontsize=7.0)
    ax_bar.set_ylabel("24 m match rate (%)", fontsize=7.6, fontweight="bold", color=PALETTE["ink"])
    ax_bar.set_ylim(0, 118)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.legend(loc="upper left", frameon=False, fontsize=6.7, handlelength=1.2, handletextpad=0.38)

    labels_h = ["0.0 / 21\n(0.0%)", f"{hdpe_diag_det:.1f} / 21\n({hdpe_rates[1]:.1f}%)"]
    labels_w = [f"{wood_locked_det:.1f} / 15\n({wood_rates[0]:.1f}%)", f"{wood_diag_det:.1f} / 15\n({wood_rates[1]:.1f}%)"]
    top_h = [max(hdpe_locked), max(hdpe_diag)]
    top_w = [max(wood_locked), max(wood_diag)]
    for idx_b, (bar, txt, th) in enumerate(zip(b1, labels_h, top_h)):
        dx = -0.03 if idx_b == 0 else 0.0
        ax_bar.text(bar.get_x() + bar.get_width() / 2 + dx, max(bar.get_height(), th) + 3.2, txt,
                    ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=PALETTE["ink"])
    for idx_b, (bar, txt, tw) in enumerate(zip(b2, labels_w, top_w)):
        dx = 0.04 if idx_b == 0 else 0.0
        ax_bar.text(bar.get_x() + bar.get_width() / 2 + dx, max(bar.get_height(), tw) + 3.2, txt,
                    ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=PALETTE["ink"])

    panel_title(ax_bar, "f", "C6 (6-band) target recovery (3-seed mean)", pad=28)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "FigS2_PLP2021_External_Target_Audit_MPB.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=600)
    if png_dpi > 0:
        png_path = out_dir / "FigS2_PLP2021_External_Target_Audit_MPB.png"
        fig.savefig(png_path, dpi=png_dpi)
        flatten_png_to_rgb(png_path, fallback_dpi=png_dpi)
    plt.close(fig)
    return pdf_path


def build_figs3_sensitivity_diagnostics(repo_root: Path, out_dir: Path, png_dpi: int) -> Path:
    """Builds consolidated 190 mm x 164 mm Supplementary Fig. S3 (D4 diagnostic panels)."""
    val_cal_json = repo_root / "runs_v2" / "_tmp_eval_d4" / "val_calibration" / "summary.json"
    cal_data = json.loads(val_cal_json.read_text(encoding="utf-8"))["cv9_relaxed_val_calibrated"]

    fig = plt.figure(figsize=(190.0 * MM, 164.0 * MM), facecolor="white")

    # Strict vertical alignment of left column (a, c, d) at left=0.145; panel letters at fig x=0.022
    ax_a = fig.add_axes([0.145, 0.720, 0.255, 0.195])
    ax_b = fig.add_axes([0.560, 0.720, 0.242, 0.195])
    ax_c = fig.add_axes([0.145, 0.398, 0.695, 0.215])
    ax_d = fig.add_axes([0.145, 0.068, 0.695, 0.215])

    # Strictly left-aligned panel letters
    fig.text(0.022, 0.930, "(a)", fontsize=9.2, fontweight="bold", color=PALETTE["ink"], va="bottom")
    fig.text(0.435, 0.930, "(b)", fontsize=9.2, fontweight="bold", color=PALETTE["ink"], va="bottom")
    fig.text(0.022, 0.625, "(c)", fontsize=9.2, fontweight="bold", color=PALETTE["ink"], va="bottom")
    fig.text(0.022, 0.296, "(d)", fontsize=9.2, fontweight="bold", color=PALETTE["ink"], va="bottom")

    c6_col = F3_PALETTE["c6"]
    c7_col = F3_PALETTE["ndwi"]
    shuf_face = "#CFE4E3"
    fdi_col = F3_PALETTE["fdi"]

    # ---- Panel (a): 10-Scene Locked Split AP50 (clean dots + mean diamonds, no tangled lines) ----
    style_axis(ax_a, grid_axis="x")
    s10_rows = [
        ("C6 (6-band)", c6_col,    c6_col, "#1E293B", 0.75),
        ("C7-shuffled", shuf_face, c7_col, c7_col,    1.25),
        ("C7 (+NDWI)",  c7_col,    c7_col, "#1E293B", 0.75),
    ]
    y_s10 = [2.0, 1.0, 0.0]

    for y_pos, (key, fcol, lcol, ecol, elw) in zip(y_s10, s10_rows):
        seeds = FIG3_SPLIT10[key]["seeds"]
        mval = FIG3_SPLIT10[key]["mean"]
        ax_a.plot([min(seeds), max(seeds)], [y_pos, y_pos], color=lcol, linewidth=1.4, alpha=0.55, zorder=2)
        for s_idx, sv in enumerate(seeds):
            ax_a.scatter(
                [sv], [y_pos - 0.22 + (s_idx - 1) * 0.05], marker="o", s=15,
                facecolor="white", edgecolor="#1E293B", linewidth=0.85, zorder=4,
            )
        ax_a.scatter(
            [mval], [y_pos], marker="D", s=38,
            facecolor=fcol, edgecolor=ecol, linewidth=elw, zorder=5,
        )
        ax_a.text(
            mval, y_pos + 0.20, f"{mval:.3f}", ha="center", va="bottom",
            fontsize=7.0, fontweight="bold", color=PALETTE["ink"],
        )

    ax_a.set_yticks(y_s10)
    ax_a.set_yticklabels([r[0] for r in s10_rows], fontsize=7.2, fontweight="bold", color=PALETTE["ink"])
    ax_a.set_ylim(-0.52, 2.68)
    ax_a.set_xlim(0.00, 0.36)
    ax_a.xaxis.set_major_locator(mticker.MultipleLocator(0.10))
    ax_a.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_a.set_xlabel("Test AP50 (10-scene split, 32 GT)", fontweight="bold", color=PALETTE["ink"])

    # ---- Panel (b): 10-Scene Paired Differences (neutral grey, no +-0.05 band, external numbers) ----
    style_axis(ax_b, grid_axis="x")
    ax_b.axvline(0.0, color=PALETTE["ink"], linestyle="--", linewidth=0.9, zorder=1)

    y_b = [2.0, 1.0, 0.0]
    trans_b = ax_b.get_yaxis_transform()
    for yp, d in zip(y_b, FIG3_SPLIT10_DIFFS):
        lo, hi = d["ci"]
        is_sig = bool(d.get("sig", False))
        line_c = "#1E293B" if is_sig else "#64748B"
        face_c = "#1E293B" if is_sig else "white"

        ax_b.plot([lo, hi], [yp, yp], color=line_c, linewidth=1.95 if is_sig else 1.45, zorder=3)
        ax_b.plot([lo, lo], [yp - 0.11, yp + 0.11], color=line_c, linewidth=1.15, zorder=3)
        ax_b.plot([hi, hi], [yp - 0.11, yp + 0.11], color=line_c, linewidth=1.15, zorder=3)
        ax_b.scatter(
            [d["mean"]], [yp], marker="D", s=32 if is_sig else 28,
            facecolor=face_c, edgecolor=line_c, linewidth=1.15, zorder=5,
        )
        for s_idx, sv in enumerate(d["seeds"]):
            ax_b.scatter(
                [sv], [yp - 0.25 + (s_idx - 1) * 0.04], marker="o", s=13,
                facecolor="white", edgecolor="#1E293B", linewidth=0.82, zorder=4,
            )

        txt = f"{fmt_signed(d['mean'])} [{fmt_signed(lo)}, {fmt_signed(hi)}]"
        ax_b.text(
            1.05, yp, txt, transform=trans_b, clip_on=False,
            va="center", ha="left", fontsize=6.9,
            fontweight="bold" if is_sig else "normal",
            color=PALETTE["ink"] if is_sig else PALETTE["muted"],
        )

    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels([d["label"] for d in FIG3_SPLIT10_DIFFS], fontsize=7.1, fontweight="bold", color=PALETTE["ink"])
    ax_b.set_ylim(-0.52, 2.52)
    ax_b.set_xlim(-0.205, 0.105)
    ax_b.set_xticks([-0.20, -0.10, 0.00, 0.10])
    ax_b.set_xticklabels(["\u22120.20", "\u22120.10", "0.00", "+0.10"])
    ax_b.set_xlabel("Difference in AP50 (95% scene CI)", fontweight="bold", color=PALETTE["ink"])

    # ---- Panel (c): Per-Fold AP50 Across All 9 Scene Folds (right-margin mean readout) ----
    style_axis(ax_c, grid_axis="y")
    folds_x = np.arange(9)

    ax_c.axvspan(-0.48, 1.48, color="#F1F5F9", alpha=0.9, zorder=0)
    ax_c.axvline(1.5, color="#CBD5E1", linestyle="--", linewidth=0.8, zorder=1)
    ax_c.text(-0.36, 0.725, "Single-scene folds\n(56 & 23 GT)",
              ha="left", va="top", fontsize=6.9, fontweight="bold", color=PALETTE["muted"],
              linespacing=1.18)

    c6_fold_means = [float(np.mean(v)) for v in FOLD_AP50_C6]
    c7_fold_means = [float(np.mean(v)) for v in FOLD_AP50_C7]

    for f_i in range(9):
        v6 = FOLD_AP50_C6[f_i]
        v7 = FOLD_AP50_C7[f_i]
        x6, x7, xf = f_i - 0.15, f_i + 0.13, f_i + 0.34

        ax_c.plot([x6, x6], [min(v6), max(v6)], color=c6_col, linewidth=1.2, alpha=0.55, zorder=2)
        ax_c.plot([x7, x7], [min(v7), max(v7)], color=c7_col, linewidth=1.2, alpha=0.55, zorder=2)
        for s_idx in range(3):
            ax_c.scatter(
                [x6 + (s_idx - 1) * 0.045], [v6[s_idx]], marker="o", s=13,
                facecolor="white", edgecolor="#1E293B", linewidth=0.80, zorder=3,
            )
            ax_c.scatter(
                [x7 + (s_idx - 1) * 0.045], [v7[s_idx]], marker="o", s=13,
                facecolor="white", edgecolor="#1E293B", linewidth=0.80, zorder=3,
            )
        ax_c.scatter([x6], [c6_fold_means[f_i]], marker="D", s=30,
                     facecolor=c6_col, edgecolor="#1E293B", linewidth=0.7, zorder=5)
        ax_c.scatter([x7], [c7_fold_means[f_i]], marker="D", s=30,
                     facecolor=c7_col, edgecolor="#1E293B", linewidth=0.7, zorder=5)
        ax_c.scatter([xf], [FOLD_AP50_FDI[f_i]], marker="D", s=22,
                     facecolor="white", edgecolor=fdi_col, linewidth=1.15, zorder=4)

    c6_gmean = float(np.mean(c6_fold_means))
    c7_gmean = float(np.mean(c7_fold_means))
    fdi_gmean = float(np.mean(FOLD_AP50_FDI))

    # Clean right-edge tick stubs & right-margin summary (zero clutter across the plot)
    trans_c = ax_c.get_yaxis_transform()
    ax_c.text(1.03, 0.56, "Mean of\nper-fold AP50", transform=trans_c, clip_on=False,
              fontsize=6.9, fontweight="bold", color=PALETTE["muted"], va="bottom", ha="left", linespacing=1.18)
    for y_m, y_txt, lbl_m, col_m, mk_face, mk_edge in [
        (c6_gmean,  0.34,  f"C6: {c6_gmean:.3f}",   c6_col,  c6_col,  "#1E293B"),
        (c7_gmean,  0.22,  f"C7: {c7_gmean:.3f}",   c7_col,  c7_col,  "#1E293B"),
        (fdi_gmean, 0.025, f"FDI: {fdi_gmean:.3f}", fdi_col, "white", fdi_col),
    ]:
        ax_c.plot([0.992, 1.015], [y_m, y_m], transform=trans_c, clip_on=False,
                  color=col_m, lw=1.4, solid_capstyle="butt")
        ax_c.text(1.03, y_txt, lbl_m, transform=trans_c, clip_on=False,
                  fontsize=7.0, fontweight="bold" if col_m != fdi_col else "normal",
                  color=col_m if col_m != fdi_col else PALETTE["muted"], va="center", ha="left")

    fold_tick_labels = [
        f"Fold {f}\n{ns} {'scene' if ns == 1 else 'scenes'}\n{nb} GT" for (f, ns, nb) in FOLD_META
    ]
    ax_c.set_xticks(folds_x)
    ax_c.set_xticklabels(fold_tick_labels, fontsize=6.9)
    ax_c.set_xlim(-0.52, 8.62)
    ax_c.set_ylim(-0.03, 0.76)
    ax_c.yaxis.set_major_locator(mticker.MultipleLocator(0.20))
    ax_c.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_c.set_ylabel("Per-fold AP50", fontweight="bold", color=PALETTE["ink"])

    # ---- Panel (d): Horizontal 6-Row Dumbbell Plot with Individual Seed Spread ----
    style_axis(ax_d, grid_axis="x")
    crit_rows = [
        ("iou_0.50",   "Box IoU \u2265 0.50",       5.35),
        ("iou_0.30",   "Box IoU \u2265 0.30",       4.35),
        ("iou_0.10",   "Box IoU \u2265 0.10",       3.35),
        ("center_1px", "Center \u2264 1 px (10 m)", 2.00),
        ("center_2px", "Center \u2264 2 px (20 m)", 1.00),
        ("center_3px", "Center \u2264 3 px (30 m)", 0.00),
    ]
    ax_d.axhline(2.68, color="#CBD5E1", linestyle="--", linewidth=0.8, zorder=1)
    trans_d = ax_d.get_yaxis_transform()
    ax_d.text(1.03, 5.92, "C7  \u2192  C6", transform=trans_d, clip_on=False,
              fontsize=6.9, fontweight="bold", color=PALETTE["muted"], va="bottom", ha="left")

    for crit_k, _, y_pos in crit_rows:
        m6 = float(cal_data[crit_k]["c6"]["mean_AP"])
        m7 = float(cal_data[crit_k]["c7"]["mean_AP"])
        s6 = [float(x) for x in cal_data[crit_k]["c6"]["per_seed_AP"]]
        s7 = [float(x) for x in cal_data[crit_k]["c7"]["per_seed_AP"]]

        # Seed ranges (subtle offset above/below the main dumbbell connector)
        ax_d.plot([min(s6), max(s6)], [y_pos + 0.18, y_pos + 0.18], color=c6_col, lw=1.0, alpha=0.45, zorder=2)
        ax_d.plot([min(s7), max(s7)], [y_pos - 0.18, y_pos - 0.18], color=c7_col, lw=1.0, alpha=0.45, zorder=2)
        for sv in s6:
            ax_d.scatter([sv], [y_pos + 0.18], marker="o", s=10, facecolor="white", edgecolor="#1E293B", lw=0.72, zorder=3)
        for sv in s7:
            ax_d.scatter([sv], [y_pos - 0.18], marker="o", s=10, facecolor="white", edgecolor="#1E293B", lw=0.72, zorder=3)

        # Dumbbell connector & mean diamonds
        ax_d.plot([m7, m6], [y_pos, y_pos], color="#64748B", lw=2.0, zorder=3.5)
        ax_d.scatter([m7], [y_pos], marker="D", s=36, facecolor=c7_col, edgecolor="#1E293B", lw=0.75, zorder=5)
        ax_d.scatter([m6], [y_pos], marker="D", s=36, facecolor=c6_col, edgecolor="#1E293B", lw=0.75, zorder=5)

        ax_d.text(
            1.03, y_pos, f"{m7:.3f} \u2192 {m6:.3f}", transform=trans_d, clip_on=False,
            va="center", ha="left", fontsize=7.0, fontweight="normal", color=PALETTE["ink"],
        )

    ax_d.set_yticks([r[2] for r in crit_rows])
    ax_d.set_yticklabels([r[1] for r in crit_rows], fontsize=7.1, fontweight="bold", color=PALETTE["ink"])
    ax_d.set_ylim(-0.55, 5.95)
    ax_d.set_xlim(0.00, 0.28)
    ax_d.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax_d.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_d.set_xlabel("Pooled out-of-fold AP", fontweight="bold", color=PALETTE["ink"])

    # ---- Streamlined Single-Row Unified Top Legend (5 items) ----
    legend_handles = [
        mlines.Line2D([], [], color=c6_col, marker="D", linestyle="none",
                      markersize=5.2, markeredgecolor="#1E293B", markeredgewidth=0.75, label="C6 (6-band) mean"),
        mlines.Line2D([], [], color=c7_col, marker="D", linestyle="none",
                      markersize=5.2, markeredgecolor="#1E293B", markeredgewidth=0.75, label="C7 (+NDWI) mean"),
        mlines.Line2D([], [], color=shuf_face, marker="D", linestyle="none",
                      markersize=5.2, markeredgecolor=c7_col, markeredgewidth=1.2, label="C7-shuffled mean"),
        mlines.Line2D([], [], color="white", marker="D", linestyle="none",
                      markersize=5.0, markeredgecolor=fdi_col, markeredgewidth=1.15, label="FDI baseline"),
        mlines.Line2D([], [], color="none", marker="o", markersize=4.4,
                      markerfacecolor="white", markeredgecolor="#1E293B", markeredgewidth=0.85,
                      label="Individual seeds (n = 3)"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.505, 0.992),
        ncol=5, frameon=False, columnspacing=0.95, handletextpad=0.35, fontsize=6.9,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "FigS3_Sensitivity_and_Fold_Diagnostics_MPB.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=600)
    if png_dpi > 0:
        png_path = out_dir / "FigS3_Sensitivity_and_Fold_Diagnostics_MPB.png"
        fig.savefig(png_path, dpi=png_dpi)
        flatten_png_to_rgb(png_path, fallback_dpi=png_dpi)
    plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, help="Destination for supplementary figures.")
    parser.add_argument("--png-dpi", type=int, default=350, help="PNG DPI, or zero for PDFs only.")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir is not None else SUPP_OUT

    configure_style()
    s1 = build_figs1_min4(REPO_ROOT, out_dir, args.png_dpi)
    s2 = build_figs2_plp2021(REPO_ROOT, out_dir, args.png_dpi)
    s3 = build_figs3_sensitivity_diagnostics(REPO_ROOT, out_dir, args.png_dpi)
    print(s1)
    print(s2)
    print(s3)

    if args.out_dir is None and args.png_dpi > 0:
        PPT_ASSETS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s1.with_suffix(".png"), PPT_ASSETS / "mpb_figs1_min4.png")
        shutil.copy2(s2.with_suffix(".png"), PPT_ASSETS / "mpb_figs2_plp2021.png")
        shutil.copy2(s3.with_suffix(".png"), PPT_ASSETS / "mpb_figs3_sensitivity.png")


if __name__ == "__main__":
    main()
