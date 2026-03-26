"""
Build MARIDA v2 dataset from raw MARIDA patches.

Plan A: 3-class detection
  0: Marine_Debris     (MARIDA class 1)
  1: Algae_Organic     (MARIDA classes 2=Dense Sargassum, 3=Sparse Sargassum, 4=Natural Organic)
  2: Ship              (MARIDA class 5)

Key improvements over v1:
  - Scene-level split → no same-image leakage between train/val/test
  - All 1,381 patches used (patches without target classes → pure background negatives)
  - Connected component analysis → YOLO bboxes with pixel-count filter + padding
  - Stratified split: Marine_Debris scenes distributed across all splits
  - Reproducible, single-script pipeline from raw MARIDA to YOLO-ready dataset

Usage:
  python scripts/preprocessing/build_marida_v2.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy import ndimage
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "MARIDA" / "patches"
SPLIT_DIR = ROOT / "data" / "raw" / "MARIDA" / "splits"
OUT_DIR = ROOT / "data" / "processed" / "marida_v2"

# 15 -> 3 class mapping
CLASS_MAP = {
    1: 0,   # Marine Debris       -> Marine_Debris
    2: 1,   # Dense Sargassum     -> Algae_Organic
    3: 1,   # Sparse Sargassum    -> Algae_Organic
    4: 1,   # Natural Organic Mat -> Algae_Organic
    5: 2,   # Ship                -> Ship
}

CLASS_NAMES = ["Marine_Debris", "Algae_Organic", "Ship"]
NC = len(CLASS_NAMES)

# Band selection: 6 spectral channels from 11-band Sentinel-2 (rasterio 1-indexed)
BAND_INDICES = [4, 3, 2, 8, 10, 11]  # B4, B3, B2, B8, B11, B12
BAND_NAMES = ["B4_Red", "B3_Green", "B2_Blue", "B8_NIR", "B11_SWIR1", "B12_SWIR2"]

# Minimum connected-component pixel count (filters single-pixel noise)
MIN_COMPONENT_PIXELS = 4
# Padding around bbox in pixels (prevents cutting object edges)
BBOX_PADDING = 2

# ============================================================================
# HELPERS
# ============================================================================

def parse_patch_name(name: str):
    """Parse 'S2_1-12-19_48MYU_0' -> (scene='1-12-19_48MYU', tile='48MYU', idx=0)."""
    if name.startswith("S2_"):
        name = name[3:]
    name = name.split(".")[0]
    for suffix in ("_cl", "_conf"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    m = re.match(r"^(.+?)_([A-Z0-9]{5})_(\d+)$", name)
    if not m:
        return None, None, None
    return f"{m.group(1)}_{m.group(2)}", m.group(2), int(m.group(3))


def scan_raw_patches(raw_dir: Path):
    """Return list of patch dicts from raw MARIDA."""
    patches = []
    for scene_dir in sorted(raw_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for tif_path in sorted(scene_dir.glob("S2_*[0-9].tif")):
            stem = tif_path.stem
            if stem.endswith("_cl") or stem.endswith("_conf"):
                continue
            scene, tile, idx = parse_patch_name(stem)
            if scene is None:
                continue
            patches.append({
                "stem": stem,
                "scene": scene,
                "tile": tile,
                "idx": idx,
                "tif_path": tif_path,
                "cl_path": scene_dir / f"{stem}_cl.tif",
            })
    return patches


def prescan_scene_classes(patches):
    """Quick scan: which v2 target classes exist in each scene (by reading _cl.tif)."""
    scene_classes = defaultdict(set)
    for p in tqdm(patches, desc="Pre-scanning masks"):
        if not p["cl_path"].exists():
            continue
        with rasterio.open(p["cl_path"]) as src:
            mask = src.read(1)
        for orig_cls, yolo_cls in CLASS_MAP.items():
            if (mask == orig_cls).any():
                scene_classes[p["scene"]].add(yolo_cls)
    return scene_classes


def load_official_splits(split_dir: Path):
    """Load MARIDA official splits -> {patch_short_name: split}."""
    mapping = {}
    for split_name, fname in [("train", "train_X.txt"), ("val", "val_X.txt"), ("test", "test_X.txt")]:
        fpath = split_dir / fname
        if not fpath.exists():
            continue
        for line in fpath.read_text().strip().splitlines():
            line = line.strip()
            if line:
                mapping[line] = split_name
    return mapping


def assign_scene_splits(patches, official_splits, scene_classes):
    """Scene-level split with stratified Marine_Debris distribution.

    1. Group patches by scene.
    2. Initial assignment from MARIDA official split (majority vote per scene).
    3. Stratified rebalance: ensure Marine_Debris scenes appear in all splits.
    """
    scene_patches = defaultdict(list)
    for p in patches:
        scene_patches[p["scene"]].append(p)

    # Initial assignment from official splits
    scene_split_votes = defaultdict(lambda: defaultdict(int))
    for p in patches:
        short = p["stem"][3:] if p["stem"].startswith("S2_") else p["stem"]
        if short in official_splits:
            scene_split_votes[p["scene"]][official_splits[short]] += 1

    scene_to_split = {}
    for scene, votes in scene_split_votes.items():
        scene_to_split[scene] = max(votes, key=votes.get)
    for scene in scene_patches:
        if scene not in scene_to_split:
            scene_to_split[scene] = "train"

    # Separate scenes by whether they contain Marine_Debris (class 0)
    debris_scenes = sorted([s for s in scene_patches if 0 in scene_classes.get(s, set())])
    non_debris_scenes = sorted([s for s in scene_patches if s not in debris_scenes])

    # Redistribute debris scenes: aim for ~70/15/15
    # Sort debris scenes by patch count for balanced distribution
    debris_scenes.sort(key=lambda s: len(scene_patches[s]))

    n_debris = len(debris_scenes)
    # At least 2 scenes in val, 2 in test, rest in train
    n_test = max(2, round(n_debris * 0.15))
    n_val = max(2, round(n_debris * 0.15))
    n_train = n_debris - n_test - n_val

    # Interleave assignment to balance patch counts
    # Assign every ~7th debris scene to test, every ~7th to val
    for i, scene in enumerate(debris_scenes):
        if i % 7 == 1 and n_val > 0:
            scene_to_split[scene] = "val"
            n_val -= 1
        elif i % 7 == 3 and n_test > 0:
            scene_to_split[scene] = "test"
            n_test -= 1
        else:
            scene_to_split[scene] = "train"

    # Handle remaining quota
    for scene in debris_scenes:
        if n_val > 0 and scene_to_split[scene] == "train":
            scene_to_split[scene] = "val"
            n_val -= 1
        elif n_test > 0 and scene_to_split[scene] == "train":
            scene_to_split[scene] = "test"
            n_test -= 1

    # Non-debris scenes: keep official split assignment (already set above)
    # But check if val/test need more non-debris scenes for background diversity
    total_patches = sum(len(scene_patches[s]) for s in scene_patches)
    for split in ["val", "test"]:
        split_patches = sum(len(scene_patches[s]) for s in scene_patches
                            if scene_to_split[s] == split)
        target = total_patches * 0.15
        if split_patches < target * 0.5:
            # Move some non-debris scenes from train to this split
            for scene in non_debris_scenes:
                if scene_to_split[scene] == "train":
                    scene_to_split[scene] = split
                    split_patches += len(scene_patches[scene])
                    if split_patches >= target:
                        break

    return scene_to_split, scene_patches


def mask_to_bboxes(cl_mask: np.ndarray, img_h: int, img_w: int):
    """Extract YOLO bboxes from class mask via connected components."""
    bboxes = []
    for orig_cls, yolo_cls in CLASS_MAP.items():
        binary = (cl_mask == orig_cls).astype(np.uint8)
        if binary.sum() == 0:
            continue
        labeled, n_components = ndimage.label(binary)
        for comp_id in range(1, n_components + 1):
            comp_mask = (labeled == comp_id)
            if comp_mask.sum() < MIN_COMPONENT_PIXELS:
                continue
            ys, xs = np.where(comp_mask)
            x_min = max(0, int(xs.min()) - BBOX_PADDING)
            y_min = max(0, int(ys.min()) - BBOX_PADDING)
            x_max = min(img_w - 1, int(xs.max()) + BBOX_PADDING)
            y_max = min(img_h - 1, int(ys.max()) + BBOX_PADDING)
            bw = x_max - x_min + 1
            bh = y_max - y_min + 1
            cx = (x_min + x_max + 1) / 2.0 / img_w
            cy = (y_min + y_max + 1) / 2.0 / img_h
            bboxes.append((yolo_cls, cx, cy, bw / img_w, bh / img_h))
    return bboxes


def mask_to_heatmap(cl_mask: np.ndarray) -> np.ndarray:
    """Distance-transform heatmap from class mask (7th knowledge channel).

    IMPORTANT: Only includes pixels from connected components that pass
    MIN_COMPONENT_PIXELS filter — consistent with mask_to_bboxes().
    Patches with empty labels will have an all-zero heatmap.
    """
    # Build filtered binary: only keep components >= MIN_COMPONENT_PIXELS
    filtered = np.zeros_like(cl_mask, dtype=np.uint8)
    for orig_cls in CLASS_MAP:
        binary = (cl_mask == orig_cls).astype(np.uint8)
        if binary.sum() == 0:
            continue
        labeled, n_comp = ndimage.label(binary)
        for comp_id in range(1, n_comp + 1):
            comp_mask = (labeled == comp_id)
            if comp_mask.sum() >= MIN_COMPONENT_PIXELS:
                filtered[comp_mask] = 1
    if filtered.sum() == 0:
        return np.zeros(cl_mask.shape, dtype=np.float32)
    dist = cv2.distanceTransform(filtered, cv2.DIST_L2, 5)
    m = dist.max()
    if m > 1e-6:
        return (dist / m).astype(np.float32)
    return dist.astype(np.float32)


# ============================================================================
# MAIN
# ============================================================================

def build_v2():
    print("=" * 70)
    print("MARIDA v2 Dataset Builder -- Plan A (3 classes)")
    print("=" * 70)
    print(f"  Raw:     {RAW_DIR}")
    print(f"  Output:  {OUT_DIR}")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Bands:   {BAND_NAMES}")
    print(f"  Filter:  min_pixels={MIN_COMPONENT_PIXELS}, padding={BBOX_PADDING}")
    print()

    # Step 1: Scan
    print("[1/6] Scanning raw patches...")
    patches = scan_raw_patches(RAW_DIR)
    n_tiles = len(set(p["tile"] for p in patches))
    n_scenes = len(set(p["scene"] for p in patches))
    print(f"  {len(patches)} patches, {n_tiles} tiles, {n_scenes} scenes")
    if not patches:
        print("ERROR: No patches found.")
        sys.exit(1)

    # Step 2: Pre-scan class masks + assign splits
    print("[2/6] Pre-scanning class masks and assigning scene-level splits...")
    scene_classes = prescan_scene_classes(patches)
    official_splits = load_official_splits(SPLIT_DIR)
    scene_to_split, scene_patches = assign_scene_splits(patches, official_splits, scene_classes)

    # Print split summary
    split_scenes = defaultdict(list)
    split_patch_counts = defaultdict(int)
    for scene, split in sorted(scene_to_split.items()):
        split_scenes[split].append(scene)
        split_patch_counts[split] += len(scene_patches[scene])

    print("  Scene split assignment:")
    for split in ["train", "val", "test"]:
        scenes = split_scenes[split]
        n_p = split_patch_counts[split]
        # Count scenes with each class
        cls_counts = defaultdict(int)
        for s in scenes:
            for c in scene_classes.get(s, set()):
                cls_counts[c] += 1
        cls_str = ", ".join(f"{CLASS_NAMES[c]}={cls_counts.get(c,0)}scenes" for c in range(NC))
        print(f"    {split:5s}: {len(scenes):2d} scenes, {n_p:4d} patches  [{cls_str}]")

    # Verify zero scene overlap
    for s1 in ["train", "val", "test"]:
        for s2 in ["train", "val", "test"]:
            if s1 >= s2:
                continue
            overlap = set(split_scenes[s1]) & set(split_scenes[s2])
            assert not overlap, f"Scene overlap {s1}/{s2}: {overlap}"
    print("  [OK] Zero scene overlap verified")

    # Step 3: Create dirs
    print("[3/6] Creating output directories...")
    for split in ["train", "val", "test"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "images_7ch").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Step 4: Process
    print("[4/6] Processing patches...")
    stats = {
        "total": 0, "with_objects": 0, "pure_background": 0,
        "per_split": defaultdict(int),
        "class_instances": defaultdict(int),
        "class_patches": defaultdict(int),
        "per_split_class_instances": defaultdict(lambda: defaultdict(int)),
        "errors": [],
    }
    manifests = {"train": [], "val": [], "test": []}
    manifests_7ch = {"train": [], "val": [], "test": []}

    for p in tqdm(patches, desc="Processing"):
        split = scene_to_split[p["scene"]]
        stem = p["stem"]
        try:
            # Read 6 spectral bands
            with rasterio.open(p["tif_path"]) as src:
                bands = src.read(BAND_INDICES)
            img = bands.transpose(1, 2, 0).astype(np.float32)
            h, w = img.shape[:2]

            # Read class mask -> bboxes + heatmap
            bboxes = []
            heatmap = np.zeros((h, w), dtype=np.float32)
            if p["cl_path"].exists():
                with rasterio.open(p["cl_path"]) as src:
                    cl_mask = src.read(1)
                if cl_mask.shape != (h, w):
                    cl_mask = cv2.resize(cl_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                bboxes = mask_to_bboxes(cl_mask, h, w)
                heatmap = mask_to_heatmap(cl_mask)

            # Save 6ch
            np.save(OUT_DIR / split / "images" / f"{stem}.npy", img)
            # Save 7ch
            img7 = np.concatenate([img, heatmap[..., None]], axis=2)
            np.save(OUT_DIR / split / "images_7ch" / f"{stem}.npy", img7)

            # Save label
            label_path = OUT_DIR / split / "labels" / f"{stem}.txt"
            if bboxes:
                lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                         for c, cx, cy, bw, bh in bboxes]
                label_path.write_text("\n".join(lines) + "\n")
            else:
                label_path.write_text("")

            # Stats
            stats["total"] += 1
            stats["per_split"][split] += 1
            if bboxes:
                stats["with_objects"] += 1
                seen = set()
                for c, *_ in bboxes:
                    stats["class_instances"][c] += 1
                    stats["per_split_class_instances"][split][c] += 1
                    seen.add(c)
                for c in seen:
                    stats["class_patches"][c] += 1
            else:
                stats["pure_background"] += 1
            manifests[split].append(f"{split}/images/{stem}.npy")
            manifests_7ch[split].append(f"{split}/images_7ch/{stem}.npy")

        except Exception as e:
            stats["errors"].append({"stem": stem, "error": str(e)})
            tqdm.write(f"  ERROR: {stem}: {e}")

    # Step 5: Write manifests
    print("[5/6] Writing split files...")
    for split in ["train", "val", "test"]:
        entries = sorted(manifests[split])
        (OUT_DIR / f"{split}.txt").write_text("\n".join(entries) + "\n")
        # 7ch manifests
        entries_7ch = sorted(manifests_7ch[split])
        (OUT_DIR / f"{split}_7ch.txt").write_text("\n".join(entries_7ch) + "\n")
        print(f"  {split}.txt: {len(entries)} entries (6ch + 7ch)")
    (OUT_DIR / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n")

    # Step 6: Statistics
    print("[6/6] Statistics")
    print()
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"  Total:       {stats['total']}")
    print(f"  With objects: {stats['with_objects']}")
    print(f"  Background:  {stats['pure_background']} "
          f"({100*stats['pure_background']/max(stats['total'],1):.1f}%)")
    print(f"  Errors:      {len(stats['errors'])}")
    print()
    print("  Split distribution:")
    for split in ["train", "val", "test"]:
        n = stats["per_split"][split]
        print(f"    {split:5s}: {n:4d} ({100*n/max(stats['total'],1):.1f}%)")
    print()
    print("  Class instances:")
    total_inst = 0
    for c in range(NC):
        n = stats["class_instances"].get(c, 0)
        total_inst += n
        print(f"    {c} {CLASS_NAMES[c]:20s}: {n:5d} inst, "
              f"{stats['class_patches'].get(c,0)} patches")
    print(f"    {'TOTAL':22s}: {total_inst:5d}")
    print()
    print("  Per-split class instances:")
    for split in ["train", "val", "test"]:
        parts = [f"{CLASS_NAMES[c]}={stats['per_split_class_instances'][split].get(c,0)}"
                 for c in range(NC)]
        print(f"    {split:5s}: {', '.join(parts)}")
    print()

    # Verify
    for split in ["val", "test"]:
        missing = [CLASS_NAMES[c] for c in range(NC)
                   if stats["per_split_class_instances"][split].get(c, 0) == 0]
        if missing:
            print(f"  [WARN] {split} missing: {missing}")
        else:
            print(f"  [OK] {split} has all {NC} classes")

    # Audit log
    audit = {
        "version": "v2", "plan": "A",
        "class_map": {str(k): v for k, v in CLASS_MAP.items()},
        "class_names": CLASS_NAMES,
        "band_indices_1indexed": BAND_INDICES,
        "band_names": BAND_NAMES,
        "min_component_pixels": MIN_COMPONENT_PIXELS,
        "bbox_padding": BBOX_PADDING,
        "scene_to_split": dict(scene_to_split),
        "total_patches": stats["total"],
        "with_objects": stats["with_objects"],
        "pure_background": stats["pure_background"],
        "errors": stats["errors"],
        "per_split": dict(stats["per_split"]),
        "class_instances": {CLASS_NAMES[k]: v for k, v in stats["class_instances"].items()},
        "class_patches": {CLASS_NAMES[k]: v for k, v in stats["class_patches"].items()},
        "per_split_class_instances": {
            split: {CLASS_NAMES[k]: v for k, v in cls_map.items()}
            for split, cls_map in stats["per_split_class_instances"].items()
        },
    }
    audit_path = OUT_DIR / "build_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"\n  Audit: {audit_path}")
    print(f"\nDone! {OUT_DIR}")


if __name__ == "__main__":
    build_v2()
