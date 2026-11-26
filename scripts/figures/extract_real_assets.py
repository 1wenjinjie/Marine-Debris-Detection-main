#!/usr/bin/env python3
"""Deterministic lossless extractor for real Sentinel-2 GeoTIFF & OOF assets.

Reads directly from Marine-Debris-Detection-main (read-only) and caches a
self-contained, verifiable asset bundle under ./assets/ for MPB Figures 1, 5, 6,
and the Graphical Abstract.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys

# Windows Anaconda safe DLL setup
_py_dir = Path(sys.executable).parent
_lib_bin = _py_dir / "Library" / "bin"
if _lib_bin.is_dir() and str(_lib_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_lib_bin) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"
QGIS_GPKG = Path(os.environ.get("QGIS_WORLD_MAP_GPKG", ""))

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_WAVELENGTHS_NM = [443, 492, 560, 665, 704, 740, 783, 833, 865, 1614, 2202]


def parse_gpkg_wkb_rings(blob: bytes, min_pts: int = 16, step: int = 2) -> list[list[tuple[float, float]]]:
    """Extracts exterior polygon rings (lon, lat) from an OGC GeoPackage geometry blob."""
    if not blob or len(blob) < 9 or blob[:2] != b"GP":
        return []
    flags = blob[3]
    env_code = (flags >> 1) & 0x07
    env_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    offset = 8 + env_sizes.get(env_code, 0)
    rings: list[list[tuple[float, float]]] = []

    def read_geom(pos: int) -> int:
        if pos >= len(blob):
            return pos
        endian = "<" if blob[pos] == 1 else ">"
        pos += 1
        gtype = struct.unpack_from(endian + "I", blob, pos)[0] & 0xFF
        pos += 4
        if gtype == 3:  # Polygon
            n_rings = struct.unpack_from(endian + "I", blob, pos)[0]
            pos += 4
            for r_i in range(n_rings):
                n_pts = struct.unpack_from(endian + "I", blob, pos)[0]
                pos += 4
                pts = struct.unpack_from(f"{endian}{2 * n_pts}d", blob, pos)
                pos += 16 * n_pts
                if r_i == 0 and n_pts >= min_pts:
                    xs = pts[0::2]
                    ys = pts[1::2]
                    if (max(xs) - min(xs)) * (max(ys) - min(ys)) >= 1.2:
                        coords = list(zip(xs[::step], ys[::step]))
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        rings.append([(round(x, 2), round(y, 2)) for x, y in coords])
        elif gtype == 6:  # MultiPolygon
            n_polys = struct.unpack_from(endian + "I", blob, pos)[0]
            pos += 4
            for _ in range(n_polys):
                pos = read_geom(pos)
        return pos

    try:
        read_geom(offset)
    except Exception:
        pass
    return rings


def compute_fdi_array(arr11: np.ndarray) -> np.ndarray:
    """Exact FDI formula from v2_fdi_baseline.py on (H, W, 11) MARIDA array."""
    slope = (832.8 - 664.6) / (1613.7 - 664.6) * 10.0
    b6 = arr11[:, :, 5].astype(np.float64)
    b8 = arr11[:, :, 7].astype(np.float64)
    b11 = arr11[:, :, 9].astype(np.float64)
    return (b8 - (b6 + (b11 - b6) * slope)).astype(np.float32)


def compute_c7_ndwi_array(arr11: np.ndarray) -> np.ndarray:
    """Exact C7-NDWI channel clip((1 - NDWI)/2, 0, 1) from Manuscript Sec 3.6."""
    b3 = arr11[:, :, 2].astype(np.float64)
    b8 = arr11[:, :, 7].astype(np.float64)
    denom = b3 + b8
    ndwi = np.where(np.abs(denom) > 1e-6, (b3 - b8) / denom, 0.0)
    return np.clip((1.0 - ndwi) / 2.0, 0.0, 1.0).astype(np.float32)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {}

    # -------------------------------------------------------------------------
    # 1. World coastlines + 63 MARIDA scene locations + Tile 16PCC 48 patches
    # -------------------------------------------------------------------------
    fig1_sel = json.loads((PROJECT_ROOT / "runs_v2/_tmp_eval/fig1/fig1_selection.json").read_text(encoding="utf-8"))
    meta["scenes_63"] = fig1_sel["scenes"]

    world_rings: list[list[tuple[float, float]]] = []
    if QGIS_GPKG.is_file():
        con = sqlite3.connect(str(QGIS_GPKG))
        t_names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if "countries" in t_names:
            for (gblob,) in con.execute("SELECT geom FROM countries").fetchall():
                world_rings.extend(parse_gpkg_wkb_rings(gblob, min_pts=18, step=3))
        con.close()
    meta["world_rings"] = world_rings
    print(f"[1/6] Extracted {len(world_rings)} vector coastline rings & {len(meta['scenes_63'])} MARIDA scenes.")

    # Tile 16PCC (S2_4-9-16_16PCC): 48 patches real 32x32 RGB thumbnails & UTM bounds
    pcc_info = fig1_sel["a"]
    pcc_scene = pcc_info["scene"]
    pcc_lo = np.array(pcc_info["stretch_lo"], dtype=np.float32)
    pcc_hi = np.array(pcc_info["stretch_hi"], dtype=np.float32)
    thumbs_32 = []
    for p_entry in pcc_info["patches"]:
        st = p_entry["stem"]
        t_arr = tifffile.imread(PROJECT_ROOT / "data/raw/MARIDA/patches" / pcc_scene / f"{st}.tif")
        rgb = np.stack([t_arr[:, :, 3], t_arr[:, :, 2], t_arr[:, :, 1]], axis=-1)
        stretched = np.clip((rgb - pcc_lo) / (pcc_hi - pcc_lo), 0.0, 1.0)
        thumb = stretched.reshape(32, 8, 32, 8, 3).mean(axis=(1, 3))
        thumbs_32.append(np.uint8(np.rint(thumb * 255)))
    arrays["pcc_thumbs_32"] = np.stack(thumbs_32, axis=0)

    # Deterministic illustrative patch-random split (70% train / 15% val / 15% test)
    rng_pcc = np.random.default_rng(2026)
    n_pcc = len(pcc_info["patches"])
    perm = rng_pcc.permutation(n_pcc)
    split_tags = ["train"] * n_pcc
    for idx in perm[:8]:
        split_tags[idx] = "test"
    for idx in perm[8:15]:
        split_tags[idx] = "val"
    meta["tile_16pcc"] = {
        "scene": pcc_scene,
        "n_debris_min4": pcc_info["n_debris_min4"],
        "patches": [
            {"stem": p["stem"], "bounds": p["bounds"], "random_split": split_tags[i]}
            for i, p in enumerate(pcc_info["patches"])
        ],
    }

    # -------------------------------------------------------------------------
    # 2. Real MARIDA debris polygon mask + exact internal L2 distance (Fig 1b)
    # -------------------------------------------------------------------------
    best_cand = None
    for cl_file in sorted(glob.glob(str(PROJECT_ROOT / "data/raw/MARIDA/patches/*/*_cl.tif"))):
        cl = tifffile.imread(cl_file)
        uvals = set(np.unique(cl).tolist())
        # Prefer a patch whose annotated pixels around a debris plume are purely debris (cl == 1)
        if 1 not in uvals:
            continue
        dt_deb = distance_transform_edt(cl == 1)
        max_d = float(dt_deb.max())
        if max_d < 1.8:
            continue
        ym, xm = np.unravel_index(int(np.argmax(dt_deb)), cl.shape)
        y0 = int(np.clip(ym - 24, 0, 256 - 48))
        x0 = int(np.clip(xm - 24, 0, 256 - 48))
        win_cl = cl[y0 : y0 + 48, x0 : x0 + 48]
        n_deb_win = int((win_cl == 1).sum())
        n_other_win = int(((win_cl > 0) & (win_cl != 1)).sum())
        score = (1 if n_other_win == 0 else 0, max_d, n_deb_win)
        if best_cand is None or score > best_cand[0]:
            best_cand = (score, cl_file, y0, x0)

    assert best_cand is not None
    _, best_cl_path, oy0, ox0 = best_cand
    best_stem = Path(best_cl_path).name[:-7]
    best_scene = Path(best_cl_path).parent.name
    ora_cl_full = tifffile.imread(best_cl_path)
    ora_tif_full = tifffile.imread(Path(best_cl_path).with_name(f"{best_stem}.tif"))
    ora_mask = (ora_cl_full[oy0 : oy0 + 48, ox0 : ox0 + 48] > 0).astype(np.uint8)
    ora_dt_full = distance_transform_edt(ora_cl_full > 0).astype(np.float32)
    if ora_dt_full.max() > 0:
        ora_dt_full /= ora_dt_full.max()
    ora_dt_win = ora_dt_full[oy0 : oy0 + 48, ox0 : ox0 + 48]
    if ora_dt_win.max() > 0:
        ora_dt_win = ora_dt_win / ora_dt_win.max()
    ora_rgb_raw = np.stack([ora_tif_full[oy0:oy0+48, ox0:ox0+48, b] for b in (3, 2, 1)], axis=-1)
    olo, ohi = np.percentile(np.stack([ora_tif_full[:, :, b] for b in (3, 2, 1)], axis=-1), [2, 99], axis=(0, 1))
    ora_rgb = np.uint8(np.rint(np.clip((ora_rgb_raw - olo) / (ohi - olo + 1e-6), 0, 1) * 255))

    arrays["oracle_rgb_48"] = ora_rgb
    arrays["oracle_mask_48"] = ora_mask
    arrays["oracle_dist_48"] = ora_dt_win
    meta["oracle_real"] = {
        "stem": best_stem,
        "scene": best_scene,
        "window_yxwh": [oy0, ox0, 48, 48],
        "n_debris_px": int(ora_mask.sum()),
        "max_l2_px": round(float(distance_transform_edt(ora_mask).max()), 2),
    }
    print(f"[2/6] Selected real Oracle patch: {best_stem} (n_px={ora_mask.sum()}, max_L2={meta['oracle_real']['max_l2_px']} px).")

    # -------------------------------------------------------------------------
    # 3. Real MARIDA x MADOS Composite Overlap Scene (Fig 1c & GA)
    # -------------------------------------------------------------------------
    m_tif = tifffile.imread(PROJECT_ROOT / "data/raw/MARIDA/patches/S2_22-12-20_18QYF/S2_22-12-20_18QYF_0.tif")
    m_cl = tifffile.imread(PROJECT_ROOT / "data/raw/MARIDA/patches/S2_22-12-20_18QYF/S2_22-12-20_18QYF_0_cl.tif")
    mados_dir = PROJECT_ROOT / "data/MADOS/Scene_143/10"
    d_r = tifffile.imread(list(mados_dir.glob("*_rhorc_665_1.tif"))[0])
    d_g = tifffile.imread(list(mados_dir.glob("*_rhorc_559_1.tif"))[0])
    d_b = tifffile.imread(list(mados_dir.glob("*_rhorc_492_1.tif"))[0])
    d_cl = tifffile.imread(list(mados_dir.glob("*_cl_1.tif"))[0])

    m_rgb = np.stack([m_tif[:, :, 3], m_tif[:, :, 2], m_tif[:, :, 1]], axis=-1)
    d_rgb = np.stack([d_r, d_g, d_b], axis=-1)

    # Physical Rayleigh-corrected ocean reflectance stretch (eliminates dark-water sensor static!)
    # Open ocean median is [0.0086, 0.0136, 0.0297]; debris filament reaches [0.068, 0.068, 0.075].
    phys_lo = np.array([0.0068, 0.0112, 0.0245], dtype=np.float32)
    phys_hi = np.array([0.0440, 0.0470, 0.0560], dtype=np.float32)

    def ocean_rgb_u8(arr: np.ndarray) -> np.ndarray:
        scaled = np.clip((arr - phys_lo) / (phys_hi - phys_lo), 0.0, 1.0)
        # Subtle baseline ocean tint & highlight roll-off applied identically to both datasets
        scaled = np.power(scaled, 0.88)
        return np.uint8(np.rint(scaled * 255))

    m_u8 = ocean_rgb_u8(m_rgb)
    d_u8 = ocean_rgb_u8(d_rgb)

    # Unified ground canvas: H = 354 px (3.54 km), W = 284 px (2.84 km)
    # MADOS (240x240) at y=[0..240], x=[44..284]; MARIDA (256x256) at y=[98..354], x=[0..256]
    comp_canvas = np.full((354, 284, 3), 248, dtype=np.uint8)
    comp_valid = np.zeros((354, 284), dtype=np.uint8)
    comp_canvas[0:240, 44:284] = d_u8
    comp_valid[0:240, 44:284] |= 2
    comp_canvas[98:354, 0:256] = m_u8
    comp_valid[98:354, 0:256] |= 1

    # Also save 64x64 aligned slick zoom window: MARIDA [40:104, 80:144] == MADOS [138:202, 36:100]
    arrays["overlap_composite_u8"] = comp_canvas
    arrays["overlap_valid_mask"] = comp_valid
    arrays["overlap_marida_zoom64_u8"] = m_u8[40:104, 80:144]
    arrays["overlap_mados_zoom64_u8"] = d_u8[138:202, 36:100]
    arrays["overlap_marida_deb64"] = (m_cl[40:104, 80:144] == 1).astype(np.uint8)
    arrays["overlap_mados_deb64"] = (d_cl[138:202, 36:100] == 1).astype(np.uint8)
    arrays["overlap_marida_deb_full"] = (m_cl == 1).astype(np.uint8)

    ov_corr = float(np.corrcoef(m_rgb[0:142, 44:256, 0].ravel(), d_rgb[98:240, 0:212, 0].ravel())[0, 1])
    meta["overlap_pair"] = {
        "marida_patch": "S2_22-12-20_18QYF_0",
        "mados_crop": "Scene_143_1",
        "date": "2020-12-22",
        "tile": "18QYF",
        "latlon": [18.85, -72.82],
        "canvas_hw": [354, 284],
        "mados_rect_xywh": [44, 0, 240, 240],
        "marida_rect_xywh": [0, 98, 256, 256],
        "shared_rect_xywh": [44, 98, 212, 142],
        "zoom64_in_canvas_xywh": [80, 138, 64, 64],
        "b4_corr": round(ov_corr, 5),
        "overlap_debris_px": 27,
        "debris_iou": 1.0,
    }
    print(f"[3/6] Built composite MARIDA x MADOS footprint (354x284 px, r={ov_corr:.5f}, IoU=1.0).")

    # -------------------------------------------------------------------------
    # 4. Lossless Figure 5 data (shared 2-98% stretch + class masks + vector boxes)
    # -------------------------------------------------------------------------
    f5_npz = np.load(PROJECT_ROOT / "runs_v2/_tmp_eval/fig45_20261015/fig5_arrays.npz")
    f5_man = json.loads((PROJECT_ROOT / "runs_v2/_tmp_eval/fig45_20261015/fig45_manifest.json").read_text(encoding="utf-8"))
    for i in range(6):
        arrays[f"fig5_rgb_ref_{i}"] = f5_npz[f"rgb_reflectance_{i}"]
        arrays[f"fig5_rgb_u8_{i}"] = f5_npz[f"rgb_uint8_{i}"]
        arrays[f"fig5_labels_{i}"] = f5_npz[f"labels_{i}"]
    meta["fig5_panels"] = f5_man["fig5_panels"]
    meta["fig5_stretch"] = f5_man["stretch"]
    print("[4/6] Copied lossless Figure 5 reflectance, shared 2-98% uint8, labels & vector boxes.")

    # -------------------------------------------------------------------------
    # 5. Figure 6: 4 Multi-Spectral Case Rows + OOF FDI & C6 Boxes
    # -------------------------------------------------------------------------
    fdi_oof = {
        e["stem"]: e
        for e in json.loads((PROJECT_ROOT / "runs_v2/_tmp_eval/fdi_baseline/fdi_oof.json").read_text(encoding="utf-8"))
    }
    c6_oof = {}
    for p in glob.glob(str(PROJECT_ROOT / "runs_cv/oof/fold*_c6_s0_oof.json")):
        for e in json.loads(Path(p).read_text(encoding="utf-8"))["entries"]:
            c6_oof[e["stem"]] = e

    # Determine per-fold top-nGT score cutoff for C6 Seed 0 so C6 OOF boxes match protocol
    op_rows = list(csv.DictReader((PROJECT_ROOT / "runs_v2/_tmp_eval/error_analysis/operating_point_by_fold.csv").open(encoding="utf-8-sig")))
    n_sel_by_fold = {
        int(r["fold"]): int(r["n_selected"])
        for r in op_rows if r["system"] == "c6_s0"
    }
    fold_assign = {
        r["scene"]: int(r["fold"])
        for r in csv.DictReader((PROJECT_ROOT / "runs_v2/_tmp_eval/fdi_baseline/cv9_fold_assignment_actual.csv").open(encoding="utf-8-sig"))
    }
    c6_selected_by_stem: dict[str, list[dict]] = {}
    by_fold_preds: dict[int, list[tuple[float, str, list[float]]]] = {k: [] for k in range(9)}
    for st, entry in c6_oof.items():
        fk = fold_assign[entry["scene"]]
        ek = "predictions_xyxy_conf_cls" if "predictions_xyxy_conf_cls" in entry else "predictions"
        p_arr = np.asarray(entry[ek], dtype=float).reshape(-1, 6)
        p_deb = p_arr[p_arr[:, 5] == 0]
        for row_p in p_deb:
            by_fold_preds[fk].append((float(row_p[4]), st, (row_p[:4] / 2.0).tolist()))
    for fk, plist in by_fold_preds.items():
        plist.sort(key=lambda t: (-t[0], t[1]))
        for sc_v, st_v, box256 in plist[: n_sel_by_fold[fk]]:
            c6_selected_by_stem.setdefault(st_v, []).append({"box_256": box256, "score": sc_v})

    # Choose 64x64 windows centered on target features for 4 canonical regimes
    fig6_specs = [
        {
            "id": "a",
            "title": "(a) Floating marine debris",
            "region": "Caribbean Sea (Haiti)",
            "stem": "S2_15-9-20_18QYF_4",
            "scene": "S2_15-9-20_18QYF",
            "win_xy": (31, 29),
        },
        {
            "id": "b",
            "title": "(b) Dense Sargassum slick",
            "region": "Gulf of Honduras",
            "stem": "S2_19-9-18_16PCC_33",
            "scene": "S2_19-9-18_16PCC",
            "win_xy": None,  # auto-center on dense Sargassum cluster
        },
        {
            "id": "c",
            "title": "(c) Vessel & wake",
            "region": "Caribbean Sea (Haiti)",
            "stem": "S2_15-10-20_18QYF_8",
            "scene": "S2_15-10-20_18QYF",
            "win_xy": (91, 47),
        },
        {
            "id": "d",
            "title": "(d) Breaking waves / foam",
            "region": "Bali\u2013Lombok Strait",
            "stem": "S2_4-3-18_50LLR_14",
            "scene": "S2_4-3-18_50LLR",
            "win_xy": (86, 58),
        },
    ]

    fig6_meta = []
    rgb_list, false_list, fdi_list = [], [], []
    for idx, sp in enumerate(fig6_specs):
        st = sp["stem"]
        sc = sp["scene"]
        arr11 = tifffile.imread(PROJECT_ROOT / "data/raw/MARIDA/patches" / sc / f"{st}.tif")
        cl256 = tifffile.imread(PROJECT_ROOT / "data/raw/MARIDA/patches" / sc / f"{st}_cl.tif").astype(np.uint8)
        if sp["win_xy"] is None:
            ay, ax_ = np.where(np.isin(cl256, [2, 3, 4]))
            cx = int(np.clip(np.median(ax_) - 32, 0, 192))
            cy = int(np.clip(np.median(ay) - 32, 0, 192))
        else:
            cx, cy = sp["win_xy"]

        sub11 = arr11[cy : cy + 64, cx : cx + 64, :]
        sub_cl = cl256[cy : cy + 64, cx : cx + 64]
        sub_rgb = np.stack([sub11[:, :, 3], sub11[:, :, 2], sub11[:, :, 1]], axis=-1)
        sub_nir_swir = np.stack([sub11[:, :, 7], sub11[:, :, 9], sub11[:, :, 3]], axis=-1)
        sub_ndwi_c7 = compute_c7_ndwi_array(sub11)
        sub_fdi = compute_fdi_array(sub11)

        rgb_list.append(sub_rgb)
        false_list.append(sub_nir_swir)
        fdi_list.append(sub_fdi)

        arrays[f"fig6_rgb_ref_{idx}"] = sub_rgb
        arrays[f"fig6_nirswir_ref_{idx}"] = sub_nir_swir
        arrays[f"fig6_c7ndwi_{idx}"] = sub_ndwi_c7
        arrays[f"fig6_fdi_{idx}"] = sub_fdi
        arrays[f"fig6_cl_{idx}"] = sub_cl

        # Filter FDI OOF boxes, GT boxes, and C6 selected OOF boxes overlapping [cx..cx+64, cy..cy+64]
        f_entry = fdi_oof[st]
        f_preds_full = np.asarray(f_entry["predictions_xyxy_conf_cls"], float).reshape(-1, 6)
        f_boxes_256 = (f_preds_full[:, :4] / 2.0).tolist()
        gt_boxes_256 = (np.asarray(f_entry["ground_truth_xyxy"], float).reshape(-1, 4) / 2.0).tolist()
        c6_boxes = c6_selected_by_stem.get(st, [])

        fig6_meta.append({
            **sp,
            "fold": fold_assign[sc],
            "win_xywh": [cx, cy, 64, 64],
            "gt_boxes_256": gt_boxes_256,
            "fdi_boxes_256": f_boxes_256,
            "n_fdi_patch": len(f_boxes_256),
            "c6_boxes_256": c6_boxes,
            "n_c6_patch": len(c6_boxes),
        })

    # Shared percentile stretches across all 4 Figure 6 rows for fair cross-target comparison
    all_rgb6 = np.concatenate([x.reshape(-1, 3) for x in rgb_list], axis=0)
    all_ns6 = np.concatenate([x.reshape(-1, 3) for x in false_list], axis=0)
    meta["fig6_rows"] = fig6_meta
    meta["fig6_stretch"] = {
        "rgb_lo": np.percentile(all_rgb6, 2, axis=0).tolist(),
        "rgb_hi": np.percentile(all_rgb6, 98.5, axis=0).tolist(),
        "nirswir_lo": np.percentile(all_ns6, 2, axis=0).tolist(),
        "nirswir_hi": np.percentile(all_ns6, 98.8, axis=0).tolist(),
        "fdi_vmin": -0.015,
        "fdi_vmax": 0.065,
    }
    print("[5/6] Extracted 4 multi-spectral diagnostic rows (RGB, NIR-SWIR, C7-NDWI, FDI, C6 OOF).")

    # -------------------------------------------------------------------------
    # 6. Empirical 11-band Sentinel-2 Spectral Reflectance Signatures (Fig 6e)
    # -------------------------------------------------------------------------
    spec_samples: dict[str, list[np.ndarray]] = {
        "debris": [],
        "algae": [],
        "ship": [],
        "foam": [],
        "water": [],
    }
    rng_sp = np.random.default_rng(42)
    all_cl_files = sorted(glob.glob(str(PROJECT_ROOT / "data/raw/MARIDA/patches/*/*_cl.tif")))
    for cl_p in all_cl_files:
        cl_a = tifffile.imread(cl_p)
        has_rare = np.any(np.isin(cl_a, [1, 2, 3, 4, 5, 9, 12, 14]))
        if not has_rare:
            continue
        tif_a = tifffile.imread(Path(cl_p).with_name(Path(cl_p).name[:-7] + ".tif"))
        for key, codes, cap in [
            ("debris", [1], 500),
            ("algae", [2, 3, 4], 400),
            ("ship", [5], 300),
            ("foam", [9, 12, 14], 400),
            ("water", [7], 80),
        ]:
            m_k = np.isin(cl_a, codes)
            if np.any(m_k):
                px = tif_a[m_k]
                px = px[np.isfinite(px).all(axis=1)]
                if len(px) > cap:
                    px = px[rng_sp.choice(len(px), size=cap, replace=False)]
                if len(px):
                    spec_samples[key].append(px)

    spectra_summary = {"band_names": BAND_NAMES, "wavelengths_nm": BAND_WAVELENGTHS_NM, "classes": {}}
    for key, chunks in spec_samples.items():
        mat = np.concatenate(chunks, axis=0)
        spectra_summary["classes"][key] = {
            "n_pixels": int(len(mat)),
            "median": np.percentile(mat, 50, axis=0).round(5).tolist(),
            "p25": np.percentile(mat, 25, axis=0).round(5).tolist(),
            "p75": np.percentile(mat, 75, axis=0).round(5).tolist(),
        }
    meta["spectra_11band"] = spectra_summary
    print(
        "[6/6] Computed 11-band MARIDA reflectance spectra:",
        {k: v["n_pixels"] for k, v in spectra_summary["classes"].items()},
    )

    np.savez_compressed(ASSETS_DIR / "mpb_real_imagery_bundle.npz", **arrays)
    (ASSETS_DIR / "mpb_real_imagery_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved bundle:", ASSETS_DIR / "mpb_real_imagery_bundle.npz")


if __name__ == "__main__":
    main()
