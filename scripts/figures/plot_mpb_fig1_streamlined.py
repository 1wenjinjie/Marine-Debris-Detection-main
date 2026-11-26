"""Draw the publication-grade MPB Figure 1 from the original Sentinel-2 rasters.

Classic Nature / RSE / MPB editorial styling (190 mm x 126 mm, strict 4.5-185.5 mm grid):
  - Pure white canvas, charcoal typography, muted scientific palette (steel blue #2B5C8F,
    muted brick #B5493E), zero rounded dashboard cards or bright orange KPI callouts.
  (a) Same-scene exposure: Gulf of Honduras coastline + 48 cloud-excluded Sentinel-2 patches
      (Fold 0: 24 siblings vs 24 fixed test panel) + real debris filament crossing (_20 | _24).
  (b) Label-derived input: 48x48 clean ocean RGB -> Label mask (cl > 0) -> Oracle distance.
  (c) Cross-dataset same-acquisition overlap: framed footprints + twin detail tiles
      (MARIDA train = MADOS test, IoU = 1.000) + scientific stacked bar plot of MARIDA (73.4%)
      and MADOS test deduplication (728 -> 525).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt

from mpb_typography import configure_typography, flatten_png_to_rgb


HERE = Path(__file__).resolve().parent
DESIGN_FILE = Path("data/processed/marida_cv_factorial_v1/design.json")
ARCHIVED_DESIGN = (
    HERE.parent.parent / "Zenodo_upload_20261016"
    / "marine_debris_research_artifacts_20261016" / DESIGN_FILE
)

W_MM, H_MM = 190.0, 126.0
INK = "#1A1A1A"
MUTED = "#52525B"
NAVY = "#2E5B88"
BRICK = "#B5442F"
OCHRE = "#D29A3A"
OCHRE_BRIGHT = "#E8B44C"
KEPT_GREY = "#64748B"
NEUTRAL_BAR = "#DCE2E8"


def stretch(rgb: np.ndarray, low: np.ndarray, high: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    norm = np.clip((rgb - low) / np.maximum(high - low, 1e-5), 0.0, 1.0)
    return norm ** gamma if gamma != 1.0 else norm


def read_marida_rgb(root: Path, scene: str, stem: str) -> np.ndarray:
    return tifffile.imread(root / scene / f"{stem}.tif")[..., [3, 2, 1]]


def axes_mm(fig: plt.Figure, x: float, y: float, width: float, height: float,
            facecolor: str = "white", edge: str | None = None, lw: float = 0.65) -> plt.Axes:
    ax = fig.add_axes([x / W_MM, y / H_MM, width / W_MM, height / H_MM], facecolor=facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        if edge is None:
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_edgecolor(edge)
            spine.set_linewidth(lw)
    return ax


def label(fig: plt.Figure, x: float, y: float, text: str, **kwargs):
    return fig.text(x / W_MM, y / H_MM, text, va="center", **kwargs)


def panel_header(fig: plt.Figure, x: float, y: float, letter: str, heading: str,
                 right_x: float | None = None, right_text: str | None = None) -> None:
    label(fig, x, y, f"({letter})", fontsize=9, weight="bold", color=INK)
    label(fig, x + 5.8, y, heading, fontsize=8.4, weight="bold", color=INK)
    if right_x is not None and right_text:
        label(fig, right_x, y, right_text, fontsize=7.1, color=MUTED, ha="right")


def pill_scale_bar(ax: plt.Axes, x0: float, y0: float, length: float, text: str,
                   pill_w: float, pill_h: float, y_bar: float, y_txt: float,
                   pad_r: float = 0.6) -> None:
    ax.add_patch(FancyBboxPatch(
        (x0, y0), pill_w, pill_h, boxstyle=f"round,pad=0.15,rounding_size={pad_r}",
        facecolor="#111827", edgecolor="none", alpha=0.78, zorder=8
    ))
    cx = x0 + pill_w / 2.0
    ax.plot([cx - length / 2.0, cx + length / 2.0], [y_bar, y_bar],
            color="white", lw=1.35, solid_capstyle="butt", zorder=9)
    ax.text(cx, y_txt, text, ha="center", va="bottom", fontsize=7, color="white", zorder=9)


def mask_outline(ax: plt.Axes, mask: np.ndarray, color: str = "#FFF4AD", lw: float = 0.50) -> None:
    padded = np.pad(mask.astype(bool), 1)
    segments = []
    for y, x in np.argwhere(mask):
        if not padded[y, x + 1]:
            segments.append([(x, y), (x + 1, y)])
        if not padded[y + 2, x + 1]:
            segments.append([(x, y + 1), (x + 1, y + 1)])
        if not padded[y + 1, x]:
            segments.append([(x, y), (x, y + 1)])
        if not padded[y + 1, x + 2]:
            segments.append([(x + 1, y), (x + 1, y + 1)])
    if segments:
        ax.add_collection(LineCollection(segments, colors=color, linewidths=lw, zorder=5))


def draw_same_scene(fig: plt.Figure, marida_root: Path, scene_meta: dict, design: dict) -> None:
    panel_header(fig, 4.5, 121.0, "a", "Same-scene exposure")

    patches = scene_meta["patches"]
    bounds = np.asarray([patch["bounds"] for patch in patches])
    origin = np.array([bounds[:, 0].min(), bounds[:, 1].min()])
    placements = (bounds - np.tile(origin, 2)) / 1000.0

    ax = axes_mm(fig, 4.5, 74.5, 45.0, 40.5, facecolor="#E8F0F6", edge="#8C96A2", lw=0.65)
    ax.set_aspect("equal", adjustable="box")

    for ring in scene_meta.get("coast_utm16", []):
        arr = np.asarray(ring, dtype=float)
        rx = (arr[:, 0] - origin[0]) / 1000.0
        ry = (arr[:, 1] - origin[1]) / 1000.0
        ax.add_patch(Polygon(
            np.column_stack([rx, ry]), closed=True,
            facecolor="#ECEAE4", edgecolor="#94A3B8", lw=0.6, zorder=1
        ))

    styles = {"panel": (0, (2.8, 1.8)), "siblings": "-"}
    widths = {"panel": 0.90, "siblings": 0.80}
    groups = {key: set(design[key]) for key in styles}
    if groups["panel"] & groups["siblings"] or (
        groups["panel"] | groups["siblings"]
    ) != {patch["stem"] for patch in patches}:
        raise ValueError("Fold 0 does not partition the illustrated scene")
    patch_roles = {stem: key for key, stems in groups.items() for stem in stems}

    rgbs = [read_marida_rgb(marida_root, scene_meta["scene"], patch["stem"]) for patch in patches]
    samples = np.concatenate([rgb[::8, ::8].reshape(-1, 3) for rgb in rgbs], axis=0)
    ocean_mask = np.all(np.isfinite(samples) & (samples > 0.0) & (samples < 0.11), axis=1)
    ocean_samples = samples[ocean_mask] if ocean_mask.sum() > 200 else samples
    low, high = np.nanpercentile(ocean_samples, [2.0, 98.5], axis=0)

    inset = 0.09
    for patch, rgb, (x0, y0, x1, y1) in zip(patches, rgbs, placements):
        ax.imshow(stretch(rgb, low, high, gamma=0.84), extent=(x0, x1, y0, y1),
                  interpolation="nearest", origin="upper", zorder=2)
        split = patch_roles[patch["stem"]]
        ax.add_patch(Rectangle(
            (x0 + inset, y0 + inset), (x1 - x0) - 2 * inset, (y1 - y0) - 2 * inset,
            fill=False, edgecolor=OCHRE_BRIGHT, lw=widths[split], ls=styles[split], zorder=3
        ))

    max_x = float(placements[:, 2].max())
    max_y = float(placements[:, 3].max())
    ax.set_xlim(-2.5, max_x + 2.5)
    ax.set_ylim(-5.5, max_y + 2.2)

    ax.add_patch(Rectangle((-1.2, -4.4), 13.5, 4.3, facecolor="white", edgecolor="none", alpha=0.86, zorder=6))
    ax.plot([0.5, 10.5], [-3.4, -3.4], color=INK, lw=1.25, solid_capstyle="butt", zorder=7)
    ax.text(5.5, -2.8, "10 km", ha="center", va="bottom", fontsize=7, color=INK, zorder=7)

    i_20 = next(i for i, p in enumerate(patches) if p["stem"].endswith("_20"))
    i_24 = next(i for i, p in enumerate(patches) if p["stem"].endswith("_24"))
    left, right = placements[i_20], placements[i_24]
    if not (np.isclose(left[2], right[0]) and np.allclose(left[[1, 3]], right[[1, 3]])):
        raise ValueError("Hero pair _20/_24 does not share an east-west edge")

    ax.add_patch(Rectangle(
        (left[0] - 0.55, left[1] - 0.55),
        right[2] - left[0] + 1.1, left[3] - left[1] + 1.1,
        fill=False, edgecolor=INK, lw=1.0, zorder=8
    ))

    stitched = np.concatenate([rgbs[i_20], rgbs[i_24]], axis=1)
    lo_s = np.nanpercentile(stitched, 1.5, axis=(0, 1))
    hi_s = np.nanpercentile(stitched, 99.6, axis=(0, 1))
    crop = stretch(stitched[152:256, 160:352], lo_s, hi_s, gamma=0.82)

    ax_pair = axes_mm(fig, 53.0, 84.5, 38.5, 20.85)
    ax_pair.imshow(crop, extent=(0, 192, 104, 0), interpolation="nearest")
    ax_pair.set_xlim(0, 192)
    ax_pair.set_ylim(104, 0)
    ax_pair.plot([0, 96, 96, 0, 0], [0, 0, 104, 104, 0], color=OCHRE, lw=1.35, ls="-", clip_on=False, zorder=5)
    ax_pair.plot([96, 192, 192, 96], [0, 0, 104, 104], color=OCHRE, lw=1.35, ls=(0, (3.4, 2.0)), clip_on=False, zorder=6)
    ax_pair.plot([96, 96], [0, 104], color="white", lw=1.05, ls="--", zorder=7)
    pill_scale_bar(ax_pair, 6.0, 81.0, 50.0, "500 m", 58.0, 18.0, 95.5, 92.5, pad_r=1.5)

    label(fig, 53.0 + 38.5 * 0.25, 107.8, "Train patch", fontsize=7.2, weight="bold", color=INK, ha="center")
    label(fig, 53.0 + 38.5 * 0.75, 107.8, "Test patch", fontsize=7.2, weight="bold", color=INK, ha="center")
    label(fig, 53.0 + 38.5 * 0.50, 81.2, "Debris slick crossing split", fontsize=7, color=MUTED, ha="center")

    conn = ConnectionPatch(
        xyA=(right[2] + 0.55, (left[1] + left[3]) / 2.0), coordsA="data", axesA=ax,
        xyB=(0.0, 0.55), coordsB="axes fraction", axesB=ax_pair,
        color="#52525B", lw=0.72, ls="--", zorder=9
    )
    fig.add_artist(conn)

    fig.legend(
        handles=[
            Rectangle((0, 0), 1, 1, fill=False, edgecolor=OCHRE, ls="-", lw=1.05,
                      label=f"Same-scene train patches ({len(groups['siblings'])})"),
            Rectangle((0, 0), 1, 1, fill=False, edgecolor=OCHRE, ls=(0, (2.8, 1.8)), lw=1.05,
                      label=f"Test patches ({len(groups['panel'])})"),
        ],
        loc="center left", bbox_to_anchor=(4.5 / W_MM, 69.4 / H_MM),
        ncols=2, frameon=False, borderaxespad=0, handlelength=1.45,
        handletextpad=0.45, columnspacing=1.1, fontsize=7,
    )


def draw_oracle(fig: plt.Figure, marida_root: Path, meta: dict) -> None:
    panel_header(fig, 98.0, 121.0, "b", "Label-derived input")

    scene, stem = meta["scene"], meta["stem"]
    rgb = read_marida_rgb(marida_root, scene, stem)
    mask = tifffile.imread(marida_root / scene / f"{stem}_cl.tif") > 0
    dist = distance_transform_edt(mask)
    dist /= dist.max()

    y0_m, x0_m, h_m, w_m = meta["window_yxwh"]
    sub_mask = mask[y0_m:y0_m + h_m, x0_m:x0_m + w_m]
    row_c, col_c = np.argwhere(sub_mask).mean(axis=0)
    side = 48
    r0 = int(np.clip(round(y0_m + row_c) - side // 2, 0, rgb.shape[0] - side))
    c0 = int(np.clip(round(x0_m + col_c) - side // 2, 0, rgb.shape[1] - side))
    view = np.s_[r0:r0 + side, c0:c0 + side]

    low, high = np.nanpercentile(rgb, [1.0, 99.85], axis=(0, 1))
    rgb_vis = stretch(rgb[view], low, high, gamma=0.85)

    images = (
        (rgb_vis, None, "Sentinel-2 RGB", "#334155", 0.6),
        (mask[view], "gray", "Annotated mask", "#334155", 0.6),
        (dist[view], "inferno", "Oracle distance", BRICK, 1.2),
    )
    tile = 26.4
    gap = 2.0
    xs = [98.0 + k * (tile + gap) for k in range(3)]
    y_bot = 81.5
    image_artist = None
    for x, (image, cmap, name, edge_c, edge_lw) in zip(xs, images):
        ax = axes_mm(fig, x, y_bot, tile, tile, edge=edge_c, lw=edge_lw)
        image_artist = ax.imshow(image, cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                                 extent=(0, side, side, 0))
        label(fig, x + tile / 2.0, y_bot + tile + 2.8, name, fontsize=7.4, weight="bold", color=INK, ha="center")
        if name == "Sentinel-2 RGB":
            pill_scale_bar(ax, 2.0, 37.0, 10.0, "100 m", 15.0, 9.0, 44.2, 42.6, pad_r=1.0)

    cax = axes_mm(fig, 182.2, y_bot, 1.5, tile)
    bar = fig.colorbar(image_artist, cax=cax, ticks=[0, 1])
    bar.ax.tick_params(labelsize=7, length=1.8, width=0.5, pad=1.0)
    bar.outline.set_edgecolor(BRICK)
    bar.outline.set_linewidth(0.55)

    label(fig, 98.0, 76.2, "480 \u00d7 480 m  \u00b7  Audit control only", fontsize=7, color=MUTED)


def draw_overlap(fig: plt.Figure, marida_root: Path, mados_root: Path, meta: dict) -> None:
    panel_header(fig, 4.5, 59.8, "c", "Cross-dataset same-acquisition overlap")

    stem = meta["marida_patch"]
    scene = stem.rsplit("_", 1)[0]
    mados_scene, crop_id = meta["mados_crop"].rsplit("_", 1)
    mados_path = mados_root / mados_scene / "10"
    prefix = f"{mados_scene}_L2R"

    marida_rgb = read_marida_rgb(marida_root, scene, stem)
    mados_rgb = np.stack([
        tifffile.imread(mados_path / f"{prefix}_rhorc_{band}_{crop_id}.tif")
        for band in (665, 559, 492)
    ], axis=-1)
    marida_mask = tifffile.imread(marida_root / scene / f"{stem}_cl.tif") == 1
    mados_mask = tifffile.imread(mados_path / f"{prefix}_cl_{crop_id}.tif") == 1

    low = np.array([0.0068, 0.0112, 0.0245])
    high = np.array([0.0440, 0.0470, 0.0560])
    height, width = meta["canvas_hw"]
    canvas = np.full((height, width, 3), np.array([0.88, 0.91, 0.94]), dtype=float)
    aligned_masks = []
    for key, rgb, mask in (("mados", mados_rgb, mados_mask), ("marida", marida_rgb, marida_mask)):
        x, y, w, h = meta[f"{key}_rect_xywh"]
        canvas[y:y + h, x:x + w] = stretch(rgb, low, high)
        aligned = np.zeros((height, width), dtype=bool)
        aligned[y:y + h, x:x + w] = mask
        aligned_masks.append(aligned)

    x, y, w, h = meta["shared_rect_xywh"]
    shared_mados, shared_marida = (mask[y:y + h, x:x + w] for mask in aligned_masks)
    shared_count = np.count_nonzero(shared_mados & shared_marida)
    shared_union = np.count_nonzero(shared_mados | shared_marida)
    if (shared_count, shared_count / shared_union) != (meta["overlap_debris_px"], meta["debris_iou"]):
        raise ValueError("Overlap mask no longer matches the source metadata")

    ax = axes_mm(fig, 4.5, 6.5, 49.0, 48.0, facecolor="#E0E7EE", edge="#8C96A2", lw=0.65)
    ax.imshow(canvas, interpolation="nearest", extent=(0, width, height, 0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-18, width + 18)
    ax.set_ylim(height + 16, -16)

    for key, linestyle in (("marida", "-"), ("mados", (0, (4.0, 2.4)))):
        rx, ry, rw, rh = meta[f"{key}_rect_xywh"]
        ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, edgecolor=OCHRE_BRIGHT, lw=1.15, ls=linestyle, zorder=4))
    xz, yz, wz, hz = meta["zoom64_in_canvas_xywh"]
    ax.add_patch(Rectangle((xz, yz), wz, hz, fill=False, edgecolor="white", lw=0.85, zorder=5))
    ax.text(xz + 2, yz - 6, "Detail", color="white", fontsize=7, zorder=6)
    pill_scale_bar(ax, 128.0, 298.0, 100.0, "1 km", 118.0, 44.0, 333.0, 326.0, pad_r=5.0)

    det_s = 39.5
    det_y = 10.5
    for x_mm, key, name, rgb, mask, style in (
        (58.5,  "marida", "MARIDA (10-sc. train)", marida_rgb, aligned_masks[1], "-"),
        (103.5, "mados",  "MADOS test",            mados_rgb,  aligned_masks[0], (0, (3.8, 2.2))),
    ):
        dx, dy, _, _ = meta[f"{key}_rect_xywh"]
        zoom = stretch(rgb[yz - dy:yz - dy + hz, xz - dx:xz - dx + wz], low, high)
        detail = axes_mm(fig, x_mm, det_y, det_s, det_s)
        detail.imshow(zoom, interpolation="nearest", extent=(0, wz, hz, 0))
        mask_outline(detail, mask[yz:yz + hz, xz:xz + wz], "#FFF4AD", lw=0.52)
        pill_scale_bar(detail, 3.0, 52.5, 20.0, "200 m", 24.0, 9.0, 59.8, 58.2, pad_r=1.0)
        label(fig, x_mm + det_s / 2.0, det_y + det_s + 2.5, name, ha="center", fontsize=7.8, weight="bold", color=INK)
        detail.add_patch(Rectangle((0, 0), wz, hz, fill=False, edgecolor=OCHRE,
                                   lw=1.25, ls=style, clip_on=False, zorder=6))

    label(fig, 58.5 + (det_s * 2 + 5.5) / 2.0, 6.8,
          "Same acquisition; IoU = 1.00  \u00b7  640 \u00d7 640 m",
          ha="center", fontsize=7.2, weight="bold", color=INK)

    ax_bar = fig.add_axes([154.5 / W_MM, 12.8 / H_MM, 31.0 / W_MM, 35.5 / H_MM], facecolor="white")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color("#334155")
    ax_bar.spines["bottom"].set_color("#334155")
    ax_bar.spines["left"].set_linewidth(0.65)
    ax_bar.spines["bottom"].set_linewidth(0.65)
    ax_bar.grid(axis="y", color="#E5E7EB", linewidth=0.55, zorder=0)
    ax_bar.set_axisbelow(True)

    xs_b = np.array([0.0, 1.0])
    bw = 0.54
    ax_bar.bar([0.0], [73.35], width=bw, color=OCHRE, edgecolor="none", zorder=3, label="Overlap")
    ax_bar.bar([0.0], [26.65], bottom=[73.35], width=bw, color=NEUTRAL_BAR, edgecolor="none", zorder=3, label="Unique")
    ax_bar.bar([1.0], [27.88], width=bw, color=OCHRE, edgecolor="none", zorder=3)
    ax_bar.bar([1.0], [72.12], bottom=[27.88], width=bw, color=KEPT_GREY, edgecolor="none", zorder=3, label="Kept (525)")

    ax_bar.text(0.0, 73.35 / 2.0, "73.4%", ha="center", va="center", fontsize=7.0, weight="bold", color=INK, zorder=4)
    ax_bar.text(1.0, 27.88 / 2.0, "27.9%", ha="center", va="center", fontsize=6.8, weight="bold", color=INK, zorder=4)
    ax_bar.text(1.0, 27.88 + 72.12 / 2.0, "72.1%", ha="center", va="center", fontsize=7.0, weight="bold", color="white", zorder=4)

    ax_bar.set_xlim(-0.52, 1.52)
    ax_bar.set_ylim(0, 108)
    ax_bar.set_yticks([0, 50, 100])
    ax_bar.set_yticklabels(["0", "50", "100"], fontsize=6.8)
    ax_bar.set_ylabel("Share of patches (%)", fontsize=7.2, weight="bold", color=INK, labelpad=1.8)
    ax_bar.set_xticks(xs_b)
    ax_bar.set_xticklabels(["MARIDA\n(1,381)", "MADOS test\n(728; 10-sc.)"], fontsize=6.6)
    ax_bar.tick_params(length=2.2, width=0.6, pad=1.8)
    ax_bar.legend(
        loc="upper center", bbox_to_anchor=(0.48, 1.24), ncol=2, frameon=False,
        fontsize=6.4, handlelength=0.9, handletextpad=0.35, columnspacing=0.65, labelspacing=0.25
    )

    ov = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    ov.set_xlim(0, W_MM)
    ov.set_ylim(0, H_MM)
    ov.axis("off")
    ov.plot([94.8, 94.8], [67.8, 122.2], color="#D4D8DD", lw=0.65)
    ov.plot([4.5, 185.5], [64.4, 64.4], color="#D4D8DD", lw=0.65)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--design", type=Path, help="Path to the published factorial audit design.json")
    parser.add_argument("--out-dir", type=Path, default=HERE.parent)
    parser.add_argument("--png-dpi", type=int, default=600, help="PNG resolution; use 0 to skip PNG export")
    args = parser.parse_args()

    meta = json.loads((HERE / "assets" / "mpb_real_imagery_meta.json").read_text(encoding="utf-8"))
    local_design = args.project_root / DESIGN_FILE
    design_path = args.design or (local_design if local_design.is_file() else ARCHIVED_DESIGN)
    design = next(fold for fold in json.loads(design_path.read_text(encoding="utf-8")) if fold["fold"] == 0)

    configure_typography()
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4), facecolor="white")
    marida_root = args.project_root / "data" / "raw" / "MARIDA" / "patches"

    draw_same_scene(fig, marida_root, meta["tile_16pcc"], design)
    draw_oracle(fig, marida_root, meta["oracle_real"])
    draw_overlap(fig, marida_root, args.project_root / "data" / "MADOS", meta["overlap_pair"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf = args.out_dir / "Fig1StreamlinedSentinel2MPB.pdf"
    fig.savefig(pdf, dpi=600)
    print(pdf)
    if args.png_dpi > 0:
        png = pdf.with_suffix(".png")
        fig.savefig(png, dpi=args.png_dpi)
        flatten_png_to_rgb(png, fallback_dpi=args.png_dpi)
        print(png)
    plt.close(fig)


if __name__ == "__main__":
    main()
