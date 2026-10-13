"""
v2.1 Phase 0 — MIN=1 sanity check.

Question this experiment answers:
    Are the 1-pixel Marine_Debris annotations in MARIDA real signal,
    or noise?

Method:
    1. Walk every raw _cl.tif in data/raw/MARIDA/patches/
    2. Find every connected component of cls=1 (Marine_Debris) of size
       exactly 1 pixel
    3. For each such pixel, also read the co-located _conf.tif value
    4. Stratified sample 10 per confidence level (1=High, 2=Mod, 3=Low),
       30 total
    5. For each sample, generate a side-by-side panel:
         (A) RGB false color (B4/B3/B2) of the 41×41 px neighbourhood
         (B) FDI heatmap of the same neighbourhood
         (C) The 1-px GT location marked with a green crosshair
       Plus a metadata header (tile name, position, conf level) with
       the conf level **blacked out** for blind judgement.
    6. Output panels into runs_v2_1/sanity/min1_panels/ as numbered
       PNGs + a master CSV with hidden ground truth.

Human reviewer flow:
    - Open each PNG, decide: "real" / "fake" / "unclear"
    - Fill the corresponding row in the CSV "judgement" column
    - Reveal hidden confidence via:
        python sanity_min1_judge.py --reveal

Outputs:
    runs_v2_1/sanity/min1_panels/<NN>_<tile>_p<row>_<col>.png
    runs_v2_1/sanity/min1_panels/_judgement.csv   (blind, no conf shown)
    runs_v2_1/sanity/min1_panels/_keymap.json     (NN -> true conf level)

Usage:
    python scripts/v2_1/preprocessing/sanity_min1_judge.py
    python scripts/v2_1/preprocessing/sanity_min1_judge.py --reveal
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[3]
RAW_PATCHES = ROOT / "data" / "raw" / "MARIDA" / "patches"
OUT_DIR = ROOT / "runs_v2_1" / "sanity" / "min1_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sentinel-2 band wavelengths (nm) - used for FDI interpolation.
# MARIDA ACOLITE 11-band order (1-indexed in rasterio):
#   1=B1  2=B2  3=B3  4=B4  5=B5  6=B6  7=B7  8=B8  9=B8A  10=B11  11=B12
WL_B6 = 740.0
WL_B8 = 842.0
WL_B11 = 1610.0

# Patch / context settings
PATCH_SIZE = 41          # 41×41 neighbourhood for visualisation
SAMPLES_PER_CONF = 10
RANDOM_SEED = 42

# Marine_Debris in original MARIDA mask is class id 1
MARIDA_DEBRIS_CLS = 1


def find_all_1px_debris():
    """Walk every patch, find every 1-px connected component of cls=1.

    Returns:
        List of dicts: {tile, scene, row, col, conf_level, cl_path,
                        tif_path, conf_path}
    """
    items = []
    print(f"Scanning {RAW_PATCHES} for 1-px Marine_Debris components...")
    n_scenes = 0
    n_patches_scanned = 0

    for scene_dir in sorted(RAW_PATCHES.iterdir()):
        if not scene_dir.is_dir():
            continue
        n_scenes += 1
        for cl_path in sorted(scene_dir.glob("S2_*_cl.tif")):
            n_patches_scanned += 1
            stem = cl_path.stem  # 'S2_..._N_cl'
            base_stem = stem[:-3] if stem.endswith("_cl") else stem
            tif_path = scene_dir / f"{base_stem}.tif"
            conf_path = scene_dir / f"{base_stem}_conf.tif"
            if not (tif_path.exists() and conf_path.exists()):
                continue

            with rasterio.open(cl_path) as src:
                cl = src.read(1)
            debris_mask = (cl == MARIDA_DEBRIS_CLS)
            if not debris_mask.any():
                continue

            labels, n = ndi.label(debris_mask, structure=np.ones((3, 3), dtype=int))
            if n == 0:
                continue
            sizes = ndi.sum_labels(debris_mask, labels, range(1, n + 1))
            one_px_label_ids = [i + 1 for i, s in enumerate(sizes) if s == 1]
            if not one_px_label_ids:
                continue

            with rasterio.open(conf_path) as src:
                conf = src.read(1)
            for lid in one_px_label_ids:
                rr, cc = np.where(labels == lid)
                # exactly 1 pixel by construction
                r, c = int(rr[0]), int(cc[0])
                conf_level = int(conf[r, c])
                items.append({
                    "tile": base_stem,
                    "scene": scene_dir.name,
                    "row": r,
                    "col": c,
                    "conf_level": conf_level,
                    "cl_path": str(cl_path),
                    "tif_path": str(tif_path),
                    "conf_path": str(conf_path),
                })

    print(f"  Scanned {n_patches_scanned} patches across {n_scenes} scenes")
    print(f"  Found {len(items)} 1-px Marine_Debris instances")
    by_conf = defaultdict(int)
    for it in items:
        by_conf[it["conf_level"]] += 1
    for lvl in sorted(by_conf.keys()):
        name = {1: "High", 2: "Mod", 3: "Low", 0: "Unlabeled (anomaly)"}[lvl]
        print(f"    conf={lvl} ({name}): {by_conf[lvl]}")
    return items


def stratified_sample(items, k_per_conf=SAMPLES_PER_CONF, seed=RANDOM_SEED):
    """Pick k samples per confidence level (1=High, 2=Mod, 3=Low)."""
    by_conf = defaultdict(list)
    for it in items:
        by_conf[it["conf_level"]].append(it)
    rng = random.Random(seed)
    picked = []
    for lvl in (1, 2, 3):
        pool = by_conf.get(lvl, [])
        if not pool:
            print(f"  [WARN] no 1-px samples at conf={lvl}")
            continue
        k = min(k_per_conf, len(pool))
        chosen = rng.sample(pool, k)
        picked.extend(chosen)
    rng.shuffle(picked)   # shuffle so reviewer can't infer conf from order
    return picked


def build_neighbourhood(tif_path: Path, r: int, c: int, half: int = PATCH_SIZE // 2):
    """Crop a (2*half+1)×(2*half+1) neighbourhood around (r,c).

    Pads with NaN if near edge.
    Returns:
        crop_11band:  (2h+1, 2h+1, 11) float32 with NaN at edges
        (r_local, c_local): coordinates within the crop
    """
    with rasterio.open(tif_path) as src:
        H, W = src.shape
        n_bands = src.count
        arr = src.read()      # (n_bands, H, W)
    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)  # (H, W, n_bands)

    r0, r1 = r - half, r + half + 1
    c0, c1 = c - half, c + half + 1
    out = np.full((2 * half + 1, 2 * half + 1, n_bands), np.nan, dtype=np.float32)
    # clip to valid range
    rr0 = max(0, r0)
    rr1 = min(H, r1)
    cc0 = max(0, c0)
    cc1 = min(W, c1)
    out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0, :] = arr[rr0:rr1, cc0:cc1, :]
    r_local = r - r0
    c_local = c - c0
    return out, r_local, c_local


def compute_fdi(crop_11: np.ndarray) -> np.ndarray:
    """FDI = R_B8 - R'_B8 where R'_B8 = R_B6 + (R_B11 - R_B6) *
    (lambda_B8 - lambda_B6) / (lambda_B11 - lambda_B6) * 10.

    crop_11: (H, W, 11) — MARIDA ACOLITE band order.
    Returns (H, W) float32.
    """
    # 0-indexed: B6 is index 5, B8 is index 7, B11 is index 9
    b6 = crop_11[..., 5]
    b8 = crop_11[..., 7]
    b11 = crop_11[..., 9]
    r_prime_b8 = b6 + (b11 - b6) * (WL_B8 - WL_B6) / (WL_B11 - WL_B6) * 10.0
    return b8 - r_prime_b8


def stretch_rgb(crop_11: np.ndarray) -> np.ndarray:
    """B4/B3/B2 (0-indexed: 3, 2, 1) -> stretched RGB uint8."""
    rgb = np.stack([crop_11[..., 3], crop_11[..., 2], crop_11[..., 1]], axis=-1)
    out = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        ch = rgb[..., c]
        valid = ch[~np.isnan(ch)]
        if valid.size == 0:
            continue
        lo = np.percentile(valid, 1)
        hi = np.percentile(valid, 99)
        scaled = np.clip((ch - lo) / (hi - lo + 1e-6) * 255, 0, 255)
        out[..., c] = np.nan_to_num(scaled, nan=0).astype(np.uint8)
    return out


def make_panel(item, idx, total, output_dir: Path) -> str:
    """Build one PNG. Returns the saved filename."""
    tif_path = Path(item["tif_path"])
    r, c = item["row"], item["col"]
    crop, r_local, c_local = build_neighbourhood(tif_path, r, c)

    rgb = stretch_rgb(crop)
    fdi = compute_fdi(crop)
    fdi_valid = fdi[~np.isnan(fdi)]
    fdi_lo, fdi_hi = (np.percentile(fdi_valid, [2, 98])
                       if fdi_valid.size > 0 else (-0.01, 0.05))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)

    # Panel A: RGB false color (B4/B3/B2)
    axes[0].imshow(rgb)
    axes[0].plot(c_local, r_local, marker="o", markersize=14,
                 markerfacecolor="none", markeredgecolor="#00ff66",
                 markeredgewidth=2)
    axes[0].plot(c_local, r_local, marker="+", markersize=22,
                 color="#00ff66", markeredgewidth=2)
    axes[0].set_title("(A) RGB false color (B4/B3/B2)\nGreen marker = 1-px GT",
                       fontsize=10)
    axes[0].axis("off")

    # Panel B: FDI heatmap
    im = axes[1].imshow(fdi, cmap="RdBu_r", vmin=fdi_lo, vmax=fdi_hi)
    axes[1].plot(c_local, r_local, marker="o", markersize=14,
                 markerfacecolor="none", markeredgecolor="#00ff66",
                 markeredgewidth=2)
    axes[1].set_title(f"(B) FDI heatmap\nrange [{fdi_lo:+.4f}, {fdi_hi:+.4f}]",
                       fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], shrink=0.7)

    # Panel C: GT location closeup (5×5 around the pixel)
    half_zoom = 5
    rz0 = max(0, r_local - half_zoom)
    rz1 = min(rgb.shape[0], r_local + half_zoom + 1)
    cz0 = max(0, c_local - half_zoom)
    cz1 = min(rgb.shape[1], c_local + half_zoom + 1)
    rgb_zoom = rgb[rz0:rz1, cz0:cz1]
    axes[2].imshow(rgb_zoom)
    axes[2].plot(c_local - cz0, r_local - rz0, marker="o", markersize=22,
                 markerfacecolor="none", markeredgecolor="#00ff66",
                 markeredgewidth=2)
    axes[2].plot(c_local - cz0, r_local - rz0, marker="+", markersize=30,
                 color="#00ff66", markeredgewidth=2)
    axes[2].set_title(f"(C) 11×11 px zoom (10 m/px → 110 m × 110 m)\n"
                       f"GT pixel at (row={r}, col={c})", fontsize=10)
    axes[2].axis("off")

    fig.suptitle(f"#{idx + 1:02d} / {total}     tile = {item['tile']}     "
                 f"⚠ confidence = HIDDEN for blind judgement",
                 fontsize=11, fontweight="bold")

    fname = f"{idx + 1:02d}_{item['tile']}_p{r}_{c}.png"
    fig.savefig(output_dir / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return fname


def write_csv_and_keymap(picked, fnames, output_dir: Path) -> None:
    """Blind CSV (no conf shown) + hidden keymap (true conf)."""
    csv_path = output_dir / "_judgement.csv"
    key_path = output_dir / "_keymap.json"

    rows = []
    keymap = {}
    for idx, (item, fname) in enumerate(zip(picked, fnames)):
        n = idx + 1
        rows.append({
            "id": f"{n:02d}",
            "filename": fname,
            "tile": item["tile"],
            "row": item["row"],
            "col": item["col"],
            "judgement": "",      # ← reviewer fills: real / fake / unclear
            "comment": "",
        })
        keymap[f"{n:02d}"] = {
            "tile": item["tile"],
            "row": item["row"],
            "col": item["col"],
            "true_conf_level": item["conf_level"],
            "true_conf_name": {1: "High", 2: "Moderate", 3: "Low"}.get(
                item["conf_level"], "?"),
        }

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "filename", "tile", "row", "col",
                           "judgement", "comment"])
        writer.writeheader()
        writer.writerows(rows)
    key_path.write_text(json.dumps(keymap, indent=2))
    print(f"\n  blind judgement CSV : {csv_path}")
    print(f"  hidden keymap        : {key_path}")
    print()
    print("  Instructions for reviewer:")
    print("    1. Open each PNG in runs_v2_1/sanity/min1_panels/")
    print("    2. Decide 'real' / 'fake' / 'unclear' based on RGB + FDI")
    print("    3. Fill _judgement.csv 'judgement' column")
    print("    4. When done:  python sanity_min1_judge.py --reveal")


def reveal(output_dir: Path) -> None:
    csv_path = output_dir / "_judgement.csv"
    key_path = output_dir / "_keymap.json"
    if not csv_path.exists() or not key_path.exists():
        print(f"Missing CSV or keymap in {output_dir}")
        return
    keymap = json.loads(key_path.read_text())
    by_conf = defaultdict(lambda: {"real": 0, "fake": 0, "unclear": 0, "blank": 0})

    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            true_conf = keymap.get(row["id"], {}).get("true_conf_name", "?")
            j = (row.get("judgement") or "").strip().lower()
            if j not in ("real", "fake", "unclear"):
                by_conf[true_conf]["blank"] += 1
            else:
                by_conf[true_conf][j] += 1

    print("\n=== Reveal: judgement breakdown by true confidence ===")
    for conf in ("High", "Moderate", "Low"):
        stats = by_conf[conf]
        n = sum(stats.values())
        if n == 0:
            continue
        real_pct = stats["real"] / n * 100 if n > 0 else 0
        print(f"  {conf:8s}  total={n:2d}  "
              f"real={stats['real']:2d} ({real_pct:.0f}%)  "
              f"fake={stats['fake']:2d}  "
              f"unclear={stats['unclear']:2d}  "
              f"blank={stats['blank']:2d}")

    print()
    print("=== Decision guideline (see v2_1_DATASET_SPEC.md §7) ===")
    print("  High-conf 'real' rate ≥ 70%   → keep MIN_COMPONENT_PIXELS = 1")
    print("  High-conf 'real' rate 30-70%  → MIN_COMPONENT_PIXELS = 2")
    print("  High-conf 'real' rate < 30%   → keep MIN_COMPONENT_PIXELS = 4")


def main() -> int:
    ap = argparse.ArgumentParser(description="v2.1 Phase 0 MIN=1 sanity check")
    ap.add_argument("--reveal", action="store_true",
                    help="Reveal hidden confidence levels and tally judgements")
    args = ap.parse_args()

    if args.reveal:
        reveal(OUT_DIR)
        return 0

    items = find_all_1px_debris()
    if not items:
        print("No 1-px Marine_Debris components found. Cannot proceed.")
        return 1

    picked = stratified_sample(items)
    print(f"\nGenerating {len(picked)} panel PNGs...")
    fnames = []
    for i, it in enumerate(picked):
        fname = make_panel(it, i, len(picked), OUT_DIR)
        print(f"  [{i + 1:2d}/{len(picked)}] {fname}")
        fnames.append(fname)

    write_csv_and_keymap(picked, fnames, OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
