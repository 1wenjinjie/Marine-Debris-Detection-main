"""
v2.1 dataset builder — single-entry preprocessing pipeline.

Conforms to docs/paper/v2_1/v2_1_DATASET_SPEC.md (LOCKED).

Stages:
  1. Scan raw MARIDA patches; build inventory.
  2. Scene-group 2D-stratified split (debris_count_bin × mean_size_bin)
     + reproducible seed (SPLIT_SEED=42).
  3. Compute 9 channels per tile: B4/B3/B2/B6/B8/B11/B12 + FDI + NDVI.
     Save .npy to images/.
  4. Compute valid_mask (NaN region) per tile. Save to valid_masks/.
  5. Compute train-only water-pixel p01/p99 stats.
     Save to channel_stats_9ch_train_water.json.
  6. Generate three label sets (highonly / highmod / all):
       - instance_masks_<set>/ : int16 raster (0=bg, k=instance)
       - labels_<set>/         : YOLO polygon .txt
       - instance_metadata_<set>/ : per-tile json
  7. Verify split outcome against targets (test debris ≥ 25).
  8. Emit build_report.json + sha256 of every output.

Usage:
  python scripts/v2_1/preprocessing/build_v2_1.py
  python scripts/v2_1/preprocessing/build_v2_1.py --dry-run
  python scripts/v2_1/preprocessing/build_v2_1.py --resume-from <stage>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[3]
RAW_PATCHES = ROOT / "data" / "raw" / "MARIDA" / "patches"
OUT_DIR = ROOT / "data" / "processed" / "marida_v2_1"

SPLIT_SEED = 42

# ============================================================================
# Per-SPEC constants
# ============================================================================

# Wavelengths (nm) for FDI
WL_B6 = 740.0
WL_B8 = 842.0
WL_B11 = 1610.0

# MARIDA ACOLITE 11-band order (1-indexed rasterio bands):
#   1=B1  2=B2  3=B3  4=B4  5=B5  6=B6  7=B7  8=B8  9=B8A  10=B11  11=B12
# We need: B4(4), B3(3), B2(2), B6(6), B8(8), B11(10), B12(11)
RASTERIO_BAND_IDX = [4, 3, 2, 6, 8, 10, 11]   # 1-indexed
CHANNEL_NAMES = ["B4_Red", "B3_Green", "B2_Blue", "B6_RedEdge2",
                 "B8_NIR", "B11_SWIR1", "B12_SWIR2", "FDI", "NDVI"]
N_CHANNELS = 9

# Class-id mappings (MARIDA original ID → v2.1 instance class)
MARIDA_DEBRIS = 1
# v2.1 main = binary; ext = 3-class
V2_1_MAIN_MAP = {
    1: 0,   # Marine Debris -> MD
}
V2_1_EXT_MAP = {
    1: 0,   # Marine Debris
    2: 1,   # Dense Sargassum -> Algae_Organic
    3: 1,   # Sparse Sargassum
    4: 1,   # Natural Organic Material
    5: 2,   # Ship
}
# Water-related classes for normalization-pixel pool
WATER_RELATED_CLS = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 15}
# Excluded from normalization (clouds, cloud shadow)
EXCLUDED_FROM_NORM = {6, 13}

# Confidence filter sets
CONF_FILTERS = {
    "highonly": {1},
    "highmod":  {1, 2},
    "all":      {1, 2, 3},
}

MIN_COMPONENT_PIXELS = 1   # confirmed by Phase 0 sanity check (High 90% real)

# 2D strata bin edges
# count_bin: 0=background, 1=1-5 box, 2=6-15, 3=16-50, 4=51+ (MEGA)
COUNT_BINS = [0, 0, 5, 15, 50, np.inf]
SIZE_BINS = [0, 4, 16, np.inf]       # bin labels 0,1,2 based on median area

# Split target
SPLIT_RATIO = (5, 1, 1)   # train, val, test (sums to 7)

# Mega-scene rule: scenes with > MEGA_THRESHOLD debris boxes are forced
# into train. v2.1.1 used 50; sanity check still showed val 34% / test
# 52% domination because MARIDA box-per-scene distribution is
# power-law right-skewed. v2.1.2 keeps the mega rule but adds an
# iterative top-1 constraint (see ENFORCE_TOP1_MAX_SHARE).
MEGA_THRESHOLD = 50
MEGA_COUNT_BIN = 4

# After round-robin + mega forcing, iteratively move val/test's
# largest-debris scene to train until top-1 share ≤ ENFORCE_TOP1_MAX_SHARE
# for both val and test. 35% is a pragmatic choice for MARIDA's
# power-law distribution; 30% causes the enforcer to drain val/test
# entirely on small samples.
ENFORCE_TOP1_MAX_SHARE = 0.35
# Safety bounds — never move a scene out of val/test if doing so
# would leave the split with fewer than these.
ENFORCE_MIN_BOXES_PER_SPLIT = 25
ENFORCE_MIN_DEBRIS_SCENES = 2

# Mid-range explicit allocation (v2.1.3): the round-robin + enforcer
# strategy left test at n=3 scenes / 29 boxes (top-1 52%). To get a
# more balanced test/val composition, we now explicitly allocate the
# "mid" scenes (15 < boxes <= MEGA_THRESHOLD) by alternating to val
# and test in box-count-descending order, so both splits get a mix of
# medium and large scenes.
MID_LOWER = 15   # boxes
MID_UPPER = MEGA_THRESHOLD

# v2.1 build version
BUILD_VERSION = "v2.1.3"


# ============================================================================
# Helpers
# ============================================================================

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=str(ROOT), timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def bin_of(value, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


# ============================================================================
# Stage 1: Inventory
# ============================================================================

def stage1_inventory():
    print("\n[1/8] Building patch inventory from raw MARIDA...")
    items = []
    n_scenes = 0
    for scene_dir in sorted(RAW_PATCHES.iterdir()):
        if not scene_dir.is_dir():
            continue
        n_scenes += 1
        for tif_path in sorted(scene_dir.glob("S2_*[0-9].tif")):
            stem = tif_path.stem
            cl_path = scene_dir / f"{stem}_cl.tif"
            conf_path = scene_dir / f"{stem}_conf.tif"
            if not (cl_path.exists() and conf_path.exists()):
                continue
            items.append({
                "stem": stem,
                "scene": scene_dir.name,
                "tif_path": str(tif_path),
                "cl_path": str(cl_path),
                "conf_path": str(conf_path),
            })
    print(f"  scenes={n_scenes}  patches={len(items)}")
    return items


# ============================================================================
# Stage 2: Scene-group 2D-stratified split
# ============================================================================

def compute_scene_strata(items):
    """For each scene, compute (debris_count, mean_size) over its tiles.
    Then bin into 2D strata.
    Uses the 'highmod' label set for stratification (main experiment).
    """
    scene_stats = defaultdict(lambda: {"debris_count": 0, "sizes": []})
    print("  Reading cl.tif + conf.tif to estimate debris counts per scene...")
    for it in items:
        with rasterio.open(it["cl_path"]) as src:
            cl = src.read(1)
        with rasterio.open(it["conf_path"]) as src:
            conf = src.read(1)
        # highmod filter
        mask = (cl == MARIDA_DEBRIS) & (np.isin(conf, [1, 2]))
        if not mask.any():
            continue
        labels, n = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
        for k in range(1, n + 1):
            comp = (labels == k)
            n_px = int(comp.sum())
            if n_px < MIN_COMPONENT_PIXELS:
                continue
            scene_stats[it["scene"]]["debris_count"] += 1
            scene_stats[it["scene"]]["sizes"].append(n_px)

    strata = {}
    for scene, st in scene_stats.items():
        n_box = st["debris_count"]
        median_size = float(np.median(st["sizes"])) if st["sizes"] else 0
        cb = bin_of(n_box, COUNT_BINS)
        sb = bin_of(median_size, SIZE_BINS)
        strata[scene] = {
            "debris_count": n_box,
            "median_size": median_size,
            "count_bin": cb,
            "size_bin": sb,
        }
    return strata


def stage2_split(items, strata):
    print("\n[2/8] Scene-group 2D-stratified split...")
    all_scenes = sorted(set(it["scene"] for it in items))
    debris_scenes = sorted(strata.keys())
    no_debris_scenes = sorted(set(all_scenes) - set(debris_scenes))
    print(f"  total scenes: {len(all_scenes)} "
          f"({len(debris_scenes)} debris-bearing, "
          f"{len(no_debris_scenes)} background-only)")

    # 2D stratification of debris scenes
    by_stratum = defaultdict(list)
    for scene in debris_scenes:
        key = (strata[scene]["count_bin"], strata[scene]["size_bin"])
        by_stratum[key].append(scene)

    scene_to_split = {}
    rng = random.Random(SPLIT_SEED)
    # Within each stratum: shuffle, round-robin to splits in pattern
    # [train, train, train, train, train, val, test]
    pattern = (["train"] * SPLIT_RATIO[0]
               + ["val"] * SPLIT_RATIO[1]
               + ["test"] * SPLIT_RATIO[2])

    # ----- Three-tier allocation -----
    # Tier 1: MEGA  scenes (> MEGA_THRESHOLD boxes)  -> all train
    # Tier 2: MID   scenes (MID_LOWER < boxes <= MID_UPPER)
    #         -> alternate val/test in box-count-descending order, then
    #         the remaining ones go to train. This gives val/test a
    #         meaningful sample of medium-density scenes.
    # Tier 3: small scenes (boxes <= MID_LOWER) -> normal round-robin.
    n_mega = 0
    mid_scenes = []
    small_scenes = []
    for key, scenes in sorted(by_stratum.items()):
        count_bin = key[0]
        if count_bin == MEGA_COUNT_BIN:
            for sc in scenes:
                scene_to_split[sc] = "train"
                n_mega += 1
        else:
            for sc in scenes:
                n_box = strata[sc]["debris_count"]
                if MID_LOWER < n_box <= MID_UPPER:
                    mid_scenes.append(sc)
                else:
                    small_scenes.append(sc)
    print(f"  tier 1 (mega > {MEGA_THRESHOLD}):  {n_mega} scenes -> all train")
    print(f"  tier 2 (mid {MID_LOWER + 1}-{MID_UPPER}): "
          f"{len(mid_scenes)} scenes -> alternate val/test")
    print(f"  tier 3 (small <= {MID_LOWER}): "
          f"{len(small_scenes)} scenes -> round-robin 5:1:1")

    # Mid scenes: sort by box count descending, then ALTERNATE val/test
    # so both splits get a mix of medium and larger scenes.
    mid_sorted = sorted(mid_scenes,
                         key=lambda s: -strata[s]["debris_count"])
    for i, sc in enumerate(mid_sorted):
        scene_to_split[sc] = "val" if i % 2 == 0 else "test"

    # Small scenes: round-robin (with shuffled order for fairness)
    rng.shuffle(small_scenes)
    for i, sc in enumerate(small_scenes):
        scene_to_split[sc] = pattern[i % len(pattern)]

    # Background-only scenes: distribute also via round-robin
    for i, sc in enumerate(no_debris_scenes):
        scene_to_split[sc] = pattern[i % len(pattern)]

    # Iterative top-1 enforcement: move dominant scenes out of val/test
    # until top-1 share <= ENFORCE_TOP1_MAX_SHARE, BUT respect safety
    # bounds (split must retain ENFORCE_MIN_BOXES_PER_SPLIT boxes and
    # ENFORCE_MIN_DEBRIS_SCENES scenes).
    n_moved = 0
    while True:
        improved = False
        for split in ("val", "test"):
            scenes_here = [(s, strata.get(s, {}).get("debris_count", 0))
                            for s, sp in scene_to_split.items()
                            if sp == split and s in strata]
            if not scenes_here:
                continue
            scenes_here.sort(key=lambda x: -x[1])
            total = sum(n for _, n in scenes_here)
            if total == 0:
                continue
            top_scene, top_count = scenes_here[0]
            if top_count / total <= ENFORCE_TOP1_MAX_SHARE:
                continue
            # Safety: would removing leave too little?
            remaining_boxes = total - top_count
            remaining_scenes = len(scenes_here) - 1
            if (remaining_boxes < ENFORCE_MIN_BOXES_PER_SPLIT
                    or remaining_scenes < ENFORCE_MIN_DEBRIS_SCENES):
                # Cannot move this one without violating safety bounds.
                # Leave it in place; sanity will flag the split as
                # naturally dominated by data structure.
                continue
            scene_to_split[top_scene] = "train"
            n_moved += 1
            improved = True
            break  # restart loop to re-check val first
        if not improved:
            break
    if n_moved:
        print(f"  top-1 enforcer (<= {int(ENFORCE_TOP1_MAX_SHARE * 100)}%, "
              f"min_boxes={ENFORCE_MIN_BOXES_PER_SPLIT}, "
              f"min_scenes={ENFORCE_MIN_DEBRIS_SCENES}): "
              f"{n_moved} scenes moved from val/test to train")

    # Summary
    split_counts = defaultdict(lambda: {"scenes": 0, "patches": 0,
                                          "debris_boxes": 0})
    for it in items:
        split = scene_to_split.get(it["scene"], "train")
        split_counts[split]["patches"] += 1
    seen_scene = set()
    for sc, sp in scene_to_split.items():
        if sc in seen_scene:
            continue
        seen_scene.add(sc)
        split_counts[sp]["scenes"] += 1
        if sc in strata:
            split_counts[sp]["debris_boxes"] += strata[sc]["debris_count"]

    print(f"  {'split':6s} {'scenes':>6s} {'patches':>8s} {'debris_box':>10s}")
    for sp in ("train", "val", "test"):
        s = split_counts[sp]
        print(f"  {sp:6s} {s['scenes']:6d} {s['patches']:8d} {s['debris_boxes']:10d}")

    return scene_to_split, dict(split_counts)


# ============================================================================
# Stage 3: 9-channel npy
# ============================================================================

def compute_9ch(tif_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read 7 raw bands, compute FDI + NDVI, return (H,W,9) float32 + valid_mask."""
    with rasterio.open(tif_path) as src:
        # read in 1-indexed band order, keep this order
        bands = src.read(RASTERIO_BAND_IDX).astype(np.float32)  # (7, H, W)
    bands = np.transpose(bands, (1, 2, 0))                       # (H, W, 7)
    valid = ~np.isnan(bands).any(axis=-1)                        # (H, W) bool

    # bands: [B4, B3, B2, B6, B8, B11, B12]
    b4 = bands[..., 0]
    b3 = bands[..., 1]
    b6 = bands[..., 3]
    b8 = bands[..., 4]
    b11 = bands[..., 5]

    eps = 1e-6
    r_prime_b8 = b6 + (b11 - b6) * (WL_B8 - WL_B6) / (WL_B11 - WL_B6) * 10.0
    fdi = b8 - r_prime_b8
    ndvi = (b8 - b4) / (b8 + b4 + eps)

    out = np.concatenate([bands, fdi[..., None], ndvi[..., None]], axis=-1).astype(np.float32)
    return out, valid


def stage3_4_npys_and_masks(items, scene_to_split, dry_run: bool):
    print("\n[3-4/8] Writing per-tile 9ch .npy + valid_mask .npy...")
    for split in ("train", "val", "test"):
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "valid_masks").mkdir(parents=True, exist_ok=True)

    n_done = 0
    t0 = time.time()
    for it in items:
        split = scene_to_split.get(it["scene"], "train")
        img_dir = OUT_DIR / split / "images"
        vm_dir = OUT_DIR / split / "valid_masks"
        out_npy = img_dir / f"{it['stem']}.npy"
        out_vm = vm_dir / f"{it['stem']}.npy"
        if out_npy.exists() and out_vm.exists():
            n_done += 1
            continue
        arr9, valid = compute_9ch(it["tif_path"])
        if not dry_run:
            np.save(out_npy, arr9)
            np.save(out_vm, valid)
        n_done += 1
        if n_done % 100 == 0:
            print(f"  ...{n_done}/{len(items)} ({time.time()-t0:.1f}s)")
    print(f"  done {n_done} tiles in {time.time()-t0:.1f}s")


# ============================================================================
# Stage 5: water-only train stats
# ============================================================================

def stage5_stats(items, scene_to_split, dry_run: bool):
    print("\n[5/8] Computing water-only train stats (channel p01/p99)...")
    train_items = [it for it in items if scene_to_split.get(it["scene"]) == "train"]
    print(f"  train tiles: {len(train_items)}")

    # Collect water-related pixel values across train set
    # We sub-sample to avoid memory blowup: at most 200 px per tile randomly.
    rng = np.random.default_rng(SPLIT_SEED)
    per_tile_cap = 200
    pool = []   # list of (N_kept, 9) arrays
    for it in train_items:
        npy_path = OUT_DIR / "train" / "images" / f"{it['stem']}.npy"
        if not npy_path.exists():
            continue
        arr9 = np.load(npy_path)
        with rasterio.open(it["cl_path"]) as src:
            cl = src.read(1)
        water_mask = np.isin(cl, list(WATER_RELATED_CLS))
        # Also exclude NaN pixels
        not_nan = ~np.isnan(arr9).any(axis=-1)
        keep_mask = water_mask & not_nan
        if not keep_mask.any():
            continue
        rr, cc = np.where(keep_mask)
        if len(rr) > per_tile_cap:
            idx = rng.choice(len(rr), size=per_tile_cap, replace=False)
            rr = rr[idx]
            cc = cc[idx]
        pool.append(arr9[rr, cc, :])
    if not pool:
        raise RuntimeError("No water-related pixels found across train set; aborting.")
    big = np.concatenate(pool, axis=0)
    p01 = np.percentile(big, 1, axis=0).tolist()
    p99 = np.percentile(big, 99, axis=0).tolist()
    mean = big.mean(axis=0).tolist()
    std = big.std(axis=0).tolist()
    stats = {
        "version": BUILD_VERSION,
        "channel_names": CHANNEL_NAMES,
        "n_channels": N_CHANNELS,
        "pixel_pool": "train split, water-related MARIDA classes "
                       f"{sorted(WATER_RELATED_CLS)}, max {per_tile_cap} px / tile",
        "n_pool_pixels": int(big.shape[0]),
        "percentiles": {"p01": p01, "p99": p99},
        "mean": mean,
        "std": std,
    }
    stats_path = OUT_DIR / "channel_stats_9ch_train_water.json"
    if not dry_run:
        stats_path.write_text(json.dumps(stats, indent=2))
    print(f"  pool size: {big.shape[0]} px from {len(pool)} tiles")
    print(f"  p01: {[round(x, 5) for x in p01]}")
    print(f"  p99: {[round(x, 5) for x in p99]}")
    print(f"  saved: {stats_path}")
    return stats


# ============================================================================
# Stage 6: three label sets + instance masks + metadata
# ============================================================================

def polygon_from_mask(mask: np.ndarray, H: int, W: int) -> list[tuple[float, float]]:
    """Return normalized (x, y) polygon vertices from boolean mask via OpenCV-style
    contour. Use a simple bbox-derived 4-point polygon if mask is 1-2 px (cv2 would
    return degenerate contour)."""
    rr, cc = np.where(mask)
    if len(rr) == 0:
        return []
    y_min, y_max = rr.min(), rr.max()
    x_min, x_max = cc.min(), cc.max()
    # For now use the axis-aligned bbox 4 corners as polygon (simple, robust)
    # — TODO Phase 2: replace with cv2.findContours for true polygon mask training
    poly_px = [(x_min, y_min), (x_max + 1, y_min),
               (x_max + 1, y_max + 1), (x_min, y_max + 1)]
    return [(x / W, y / H) for x, y in poly_px]


def stage6_labels(items, scene_to_split, dry_run: bool):
    print("\n[6/8] Generating three label sets (highonly / highmod / all)...")
    # Pre-create dirs
    for split in ("train", "val", "test"):
        for filt in CONF_FILTERS:
            (OUT_DIR / split / f"instance_masks_{filt}").mkdir(parents=True, exist_ok=True)
            (OUT_DIR / split / f"labels_{filt}").mkdir(parents=True, exist_ok=True)
        # only emit instance_metadata for the main (highmod) set
        (OUT_DIR / split / "instance_metadata_highmod").mkdir(parents=True, exist_ok=True)
        # Also create a 'labels/' alias for the main (highmod) set so
        # stock ultralytics dataset loaders find them at the default path.
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    totals = {filt: defaultdict(lambda: {"tiles": 0, "tiles_with_gt": 0,
                                          "boxes": 0})
              for filt in CONF_FILTERS}

    for idx, it in enumerate(items):
        split = scene_to_split.get(it["scene"], "train")
        with rasterio.open(it["cl_path"]) as src:
            cl = src.read(1)
        with rasterio.open(it["conf_path"]) as src:
            conf = src.read(1)
        H, W = cl.shape

        for filt_name, conf_set in CONF_FILTERS.items():
            totals[filt_name][split]["tiles"] += 1
            mask = (cl == MARIDA_DEBRIS) & np.isin(conf, list(conf_set))
            if not mask.any():
                # still write empty files to keep parallel structure
                if not dry_run:
                    (OUT_DIR / split / f"labels_{filt_name}" /
                     f"{it['stem']}.txt").write_text("")
                    if filt_name == "highmod":
                        (OUT_DIR / split / "labels" /
                         f"{it['stem']}.txt").write_text("")
                continue

            labels, n = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
            inst_raster = np.zeros((H, W), dtype=np.int16)
            label_lines = []
            metadata = {}
            inst_id = 0
            for k in range(1, n + 1):
                comp = (labels == k)
                n_px = int(comp.sum())
                if n_px < MIN_COMPONENT_PIXELS:
                    continue
                inst_id += 1
                inst_raster[comp] = inst_id

                cls_v2 = V2_1_MAIN_MAP[MARIDA_DEBRIS]  # always 0 for main
                poly = polygon_from_mask(comp, H, W)
                if len(poly) < 3:
                    continue
                line = f"{cls_v2} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                label_lines.append(line)
                metadata[str(inst_id)] = {
                    "cls": cls_v2,
                    "cls_name": "Marine_Debris",
                    "n_pixels": n_px,
                    "source_cl_class": int(MARIDA_DEBRIS),
                }
                totals[filt_name][split]["boxes"] += 1

            if inst_id > 0:
                totals[filt_name][split]["tiles_with_gt"] += 1

            if not dry_run:
                np.save(OUT_DIR / split / f"instance_masks_{filt_name}" /
                        f"{it['stem']}.npy", inst_raster)
                label_text = "\n".join(label_lines)
                (OUT_DIR / split / f"labels_{filt_name}" /
                 f"{it['stem']}.txt").write_text(label_text)
                if filt_name == "highmod":
                    (OUT_DIR / split / "instance_metadata_highmod" /
                     f"{it['stem']}.json").write_text(json.dumps(metadata, indent=2))
                    # Mirror to labels/ alias so ultralytics default
                    # img2label_paths() finds them at the standard path.
                    (OUT_DIR / split / "labels" /
                     f"{it['stem']}.txt").write_text(label_text)

        if (idx + 1) % 200 == 0:
            print(f"  ...{idx + 1}/{len(items)} tiles processed")

    print("\n  Per-set GT counts:")
    print(f"  {'set':10s} {'split':6s} {'tiles':>6s} {'w/ GT':>6s} {'boxes':>8s}")
    for filt in CONF_FILTERS:
        for split in ("train", "val", "test"):
            s = totals[filt][split]
            print(f"  {filt:10s} {split:6s} {s['tiles']:6d} "
                  f"{s['tiles_with_gt']:6d} {s['boxes']:8d}")

    return {filt: dict(d) for filt, d in totals.items()}


# ============================================================================
# Stage 7: verify split outcome
# ============================================================================

def stage7_verify(label_totals):
    print("\n[7/8] Verifying split outcome against targets...")
    highmod = label_totals["highmod"]
    test_boxes = highmod["test"]["boxes"]
    val_boxes = highmod["val"]["boxes"]
    train_boxes = highmod["train"]["boxes"]
    total = train_boxes + val_boxes + test_boxes

    warnings = []
    if test_boxes < 25:
        warnings.append(f"test debris box count {test_boxes} < 25 target")
    if val_boxes < 25:
        warnings.append(f"val debris box count {val_boxes} < 25 target")
    if total == 0:
        warnings.append("ZERO debris boxes overall — aborting")
        return warnings

    train_pct = 100 * train_boxes / total
    val_pct = 100 * val_boxes / total
    test_pct = 100 * test_boxes / total
    print(f"  M.Debris box distribution (highmod):")
    print(f"    train: {train_boxes:4d} ({train_pct:5.1f}%)")
    print(f"    val:   {val_boxes:4d} ({val_pct:5.1f}%)")
    print(f"    test:  {test_boxes:4d} ({test_pct:5.1f}%)")
    print(f"  TARGET: ~70/15/15  ACTUAL: {train_pct:.0f}/{val_pct:.0f}/{test_pct:.0f}")
    return warnings


# ============================================================================
# Stage 8: build_report.json
# ============================================================================

def stage8_report(items, scene_to_split, strata, split_counts, label_totals,
                  stats, warnings, dry_run: bool, t_start: float):
    print("\n[8/8] Writing build_report.json + tile geo index...")
    # Tile geo index (copy from v2.0 if exists, else build fresh)
    v2_index = ROOT / "data" / "processed" / "marida_v2" / "tile_geo_index.json"
    v2_1_index = OUT_DIR / "tile_geo_index.json"
    if v2_index.exists() and not v2_1_index.exists():
        v2_1_index.write_text(v2_index.read_text())
        print(f"  copied tile_geo_index.json from v2.0 ({v2_index})")

    # Split manifest
    manifest = {
        "version": BUILD_VERSION,
        "split_seed": SPLIT_SEED,
        "scene_to_split": scene_to_split,
        "scene_strata": strata,
        "split_summary": split_counts,
    }
    if not dry_run:
        (OUT_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  split_manifest.json")

    # Build report
    report = {
        "version": BUILD_VERSION,
        "build_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "build_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - t_start,
        "git_commit": get_git_commit(),
        "n_total_tiles": len(items),
        "n_total_scenes": len(set(it["scene"] for it in items)),
        "channels": CHANNEL_NAMES,
        "min_component_pixels": MIN_COMPONENT_PIXELS,
        "stats_pixel_pool": stats["pixel_pool"],
        "split_counts": split_counts,
        "label_set_counts": label_totals,
        "warnings": warnings,
        "spec_version": "v2_1_DATASET_SPEC.md",
    }
    if not dry_run:
        (OUT_DIR / "build_report.json").write_text(json.dumps(report, indent=2,
                                                                default=str))
    print(f"  build_report.json")

    if warnings:
        print("\n  [WARN] WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("\n  [OK] no warnings -- build clean")

    print(f"\nElapsed: {time.time() - t_start:.1f}s")


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="v2.1 dataset builder")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute everything but skip writing files")
    args = ap.parse_args()

    t_start = time.time()
    print(f"v2.1 build starting at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"  output dir: {OUT_DIR}")
    print(f"  dry-run: {args.dry_run}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items = stage1_inventory()
    strata = compute_scene_strata(items)
    scene_to_split, split_counts = stage2_split(items, strata)

    if args.dry_run:
        print("\n[dry-run] stopping after split planning")
        return 0

    stage3_4_npys_and_masks(items, scene_to_split, dry_run=args.dry_run)
    stats = stage5_stats(items, scene_to_split, dry_run=args.dry_run)
    label_totals = stage6_labels(items, scene_to_split, dry_run=args.dry_run)
    warnings = stage7_verify(label_totals)
    stage8_report(items, scene_to_split, strata, split_counts,
                  label_totals, stats, warnings, args.dry_run, t_start)

    return 0


if __name__ == "__main__":
    sys.exit(main())
