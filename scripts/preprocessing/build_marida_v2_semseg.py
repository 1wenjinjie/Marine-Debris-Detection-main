"""Build MARIDA v2-semseg dataset for semantic segmentation.

Reuses the EXACT same scene-level split as marida_v2 (loaded from
data/processed/marida_v2/build_audit.json). Difference from
marida_v2_seg/marida_v2:
  - Output ground truth is a (H, W) class-id mask (no bbox, no polygon)
  - **No MIN_COMPONENT_PIXELS filter** -- 1/2/3 px ground truth is kept.
    This is the entire point of B.2: detection drops 85% of debris
    instances at MIN=4; semantic segmentation can train on every pixel.

Layout:
  data/processed/marida_v2_semseg/
    train.txt / val.txt / test.txt
    train/images/*.npy            # float32 (H, W, 6) bands
    train/masks/*.npy             # uint8  (H, W)    class id (0=bg, 1-3)
    val/   test/                  # same structure
    channel_stats_6ch.json
    build_audit.json

Class mapping (mask values):
  0 = background (water, clouds, etc -- all non-target MARIDA classes)
  1 = Marine_Debris (orig 1)
  2 = Algae_Organic (orig 2/3/4)
  3 = Ship           (orig 5)
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "MARIDA" / "patches"
V2_DIR = ROOT / "data" / "processed" / "marida_v2"
OUT_DIR = ROOT / "data" / "processed" / "marida_v2_semseg"

CLASS_MAP = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3}
CLASS_NAMES = ["background", "Marine_Debris", "Algae_Organic", "Ship"]
NC = len(CLASS_NAMES)

BAND_INDICES = [4, 3, 2, 8, 10, 11]
BAND_NAMES = ["B4_Red", "B3_Green", "B2_Blue", "B8_NIR", "B11_SWIR1", "B12_SWIR2"]


def load_v2_scene_split() -> dict[str, str]:
    audit = json.loads((V2_DIR / "build_audit.json").read_text(encoding="utf-8"))
    return audit["scene_to_split"]


def parse_scene(name: str):
    if name.startswith("S2_"):
        name = name[3:]
    name = name.split(".")[0]
    for suf in ("_cl", "_conf"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    m = re.match(r"^(.+?)_([A-Z0-9]{5})_(\d+)$", name)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def scan_patches() -> list[dict]:
    out = []
    for sd in sorted(RAW_DIR.iterdir()):
        if not sd.is_dir():
            continue
        for tif in sd.glob("*.tif"):
            if tif.name.endswith(("_cl.tif", "_conf.tif")):
                continue
            stem = tif.stem
            scene = parse_scene(stem)
            if scene is None:
                continue
            out.append({
                "scene": scene,
                "stem": stem,
                "tif_path": tif,
                "cl_path": tif.parent / f"{stem}_cl.tif",
            })
    return out


def cl_to_classmask(cl_mask: np.ndarray) -> np.ndarray:
    """Map 15-class MARIDA segmentation mask to 4-class (bg, debris, algae, ship)."""
    out = np.zeros_like(cl_mask, dtype=np.uint8)
    for orig, mapped in CLASS_MAP.items():
        out[cl_mask == orig] = mapped
    return out


def build():
    print("=" * 70)
    print("MARIDA v2-semseg builder (semantic segmentation, no MIN_PIXELS filter)")
    print("=" * 70)
    scene_to_split = load_v2_scene_split()
    print(f"  Output: {OUT_DIR}")
    print(f"  Classes: {CLASS_NAMES} (no filter -- 1/2/3 px GT kept)")
    print()

    patches = scan_patches()
    print(f"[scan] {len(patches)} patches")

    for split in ["train", "val", "test"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "masks").mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "per_split": defaultdict(int),
        "per_split_class_pixels": defaultdict(lambda: defaultdict(int)),
        "per_split_tiles_with_class": defaultdict(lambda: defaultdict(int)),
        "errors": [],
    }
    manifests = defaultdict(list)

    for p in tqdm(patches, desc="processing"):
        scene = p["scene"]
        if scene not in scene_to_split:
            continue
        split = scene_to_split[scene]
        stem = p["stem"]
        try:
            with rasterio.open(p["tif_path"]) as src:
                bands = src.read(BAND_INDICES)
            img = bands.transpose(1, 2, 0).astype(np.float32)
            h, w = img.shape[:2]

            mask = np.zeros((h, w), dtype=np.uint8)
            if p["cl_path"].exists():
                with rasterio.open(p["cl_path"]) as src:
                    cl = src.read(1)
                if cl.shape != (h, w):
                    cl = cv2.resize(cl, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = cl_to_classmask(cl)

            np.save(OUT_DIR / split / "images" / f"{stem}.npy", img)
            np.save(OUT_DIR / split / "masks" / f"{stem}.npy", mask)

            stats["total"] += 1
            stats["per_split"][split] += 1
            for c in range(NC):
                n = int((mask == c).sum())
                stats["per_split_class_pixels"][split][c] += n
                if c > 0 and n > 0:
                    stats["per_split_tiles_with_class"][split][c] += 1
            manifests[split].append(f"{split}/images/{stem}.npy")

        except Exception as e:
            stats["errors"].append({"stem": stem, "error": str(e)})
            tqdm.write(f"  ERROR {stem}: {e}")

    for split in ["train", "val", "test"]:
        entries = sorted(manifests[split])
        (OUT_DIR / f"{split}.txt").write_text("\n".join(entries) + "\n")
        print(f"  {split}.txt: {len(entries)} entries")

    (OUT_DIR / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n")

    src_stats = V2_DIR / "channel_stats_6ch.json"
    if src_stats.exists():
        shutil.copy2(src_stats, OUT_DIR / "channel_stats_6ch.json")
        print(f"  copied channel_stats_6ch.json from v2")

    print()
    print("=" * 70)
    print("STATS")
    print("=" * 70)
    print(f"  total: {stats['total']}, errors: {len(stats['errors'])}")
    print()
    print("  Per-split class PIXEL counts (incl. 1-2 px components, no filter):")
    for split in ["train", "val", "test"]:
        parts = []
        for c in range(NC):
            n = stats["per_split_class_pixels"][split].get(c, 0)
            parts.append(f"{CLASS_NAMES[c]}={n}")
        print(f"    {split:5s}: n_tiles={stats['per_split'][split]:4d}  {', '.join(parts)}")
    print()
    print("  Per-split tiles WITH each target class:")
    for split in ["train", "val", "test"]:
        parts = []
        for c in range(1, NC):
            parts.append(
                f"{CLASS_NAMES[c]}={stats['per_split_tiles_with_class'][split].get(c, 0)}"
            )
        print(f"    {split:5s}: {', '.join(parts)}")

    audit = {
        "version": "v2_semseg",
        "task": "semantic_segmentation",
        "source_split": "marida_v2 scene-level (reused)",
        "class_map": {str(k): v for k, v in CLASS_MAP.items()},
        "class_names": CLASS_NAMES,
        "band_indices_1indexed": BAND_INDICES,
        "band_names": BAND_NAMES,
        "min_component_pixels": "NONE (1/2/3 px GT kept)",
        "scene_to_split": dict(scene_to_split),
        "total_patches": stats["total"],
        "per_split": dict(stats["per_split"]),
        "per_split_class_pixels": {
            s: {CLASS_NAMES[c]: n for c, n in m.items()}
            for s, m in stats["per_split_class_pixels"].items()
        },
        "per_split_tiles_with_class": {
            s: {CLASS_NAMES[c]: n for c, n in m.items()}
            for s, m in stats["per_split_tiles_with_class"].items()
        },
        "errors": stats["errors"],
    }
    (OUT_DIR / "build_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDone! {OUT_DIR}")


if __name__ == "__main__":
    build()
