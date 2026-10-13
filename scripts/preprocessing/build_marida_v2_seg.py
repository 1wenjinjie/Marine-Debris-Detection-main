"""Build MARIDA v2-seg dataset for instance segmentation.

Reuses the EXACT same scene-level split as marida_v2 (loaded from
data/processed/marida_v2/build_audit.json) so detection vs segmentation
numbers are directly comparable.

Output layout (mirrors marida_v2):
  data/processed/marida_v2_seg/
    train.txt / val.txt / test.txt
    train/images/*.npy            # 6-channel float32 (same content as v2)
    train/labels/*.txt            # YOLO seg polygon format
    val/   test/                  # same structure
    channel_stats_6ch.json        # copied from v2
    build_audit.json

Label format (YOLO seg): each non-empty line is
    <cls> x1 y1 x2 y2 ... xn yn
with all xi, yi normalized to [0, 1]. Polygon must have >= 3 vertices.

Filter parameters MATCH v2:
  MIN_COMPONENT_PIXELS = 4   (same)
  No bbox padding (polygon is exact contour from cv2.findContours)

Note: components with < 3 contour vertices are dropped (YOLO seg requirement).
4-pixel L-shaped components typically yield 4 contour points so survive.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy import ndimage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "MARIDA" / "patches"
V2_DIR = ROOT / "data" / "processed" / "marida_v2"
OUT_DIR = ROOT / "data" / "processed" / "marida_v2_seg"

CLASS_MAP = {1: 0, 2: 1, 3: 1, 4: 1, 5: 2}
CLASS_NAMES = ["Marine_Debris", "Algae_Organic", "Ship"]
NC = len(CLASS_NAMES)

BAND_INDICES = [4, 3, 2, 8, 10, 11]
BAND_NAMES = ["B4_Red", "B3_Green", "B2_Blue", "B8_NIR", "B11_SWIR1", "B12_SWIR2"]

MIN_COMPONENT_PIXELS = 4
MIN_POLYGON_VERTICES = 3


def load_v2_scene_split() -> dict[str, str]:
    audit_path = V2_DIR / "build_audit.json"
    if not audit_path.exists():
        sys.exit(f"v2 audit not found: {audit_path}. Build v2 first.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    s2s = audit.get("scene_to_split")
    if not s2s:
        sys.exit("v2 audit has no scene_to_split")
    return s2s


def parse_patch_name(name: str):
    import re
    if name.startswith("S2_"):
        name = name[3:]
    name = name.split(".")[0]
    for suffix in ("_cl", "_conf"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    m = re.match(r"^(.+?)_([A-Z0-9]{5})_(\d+)$", name)
    if not m:
        return None, None
    return f"{m.group(1)}_{m.group(2)}", int(m.group(3))


def scan_patches() -> list[dict]:
    out = []
    for sd in sorted(RAW_DIR.iterdir()):
        if not sd.is_dir():
            continue
        for tif in sd.glob("*.tif"):
            if tif.name.endswith("_cl.tif") or tif.name.endswith("_conf.tif"):
                continue
            stem = tif.stem
            cl = tif.parent / f"{stem}_cl.tif"
            scene, idx = parse_patch_name(stem)
            if scene is None:
                continue
            out.append({"scene": scene, "stem": stem, "tif_path": tif, "cl_path": cl})
    return out


def component_to_polygon(comp_mask: np.ndarray) -> list[tuple[float, float]] | None:
    """Return contour polygon (largest external) as list of (x, y) pixel coords,
    or None if < MIN_POLYGON_VERTICES."""
    bin_u8 = comp_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2)
    if len(pts) < MIN_POLYGON_VERTICES:
        return None
    return [(float(x), float(y)) for x, y in pts]


def mask_to_polygons(cl_mask: np.ndarray, img_h: int, img_w: int):
    """Return list of (yolo_cls, [(x1,y1), ..., (xn,yn)]) normalized to [0,1]."""
    out = []
    for orig_cls, yolo_cls in CLASS_MAP.items():
        binary = (cl_mask == orig_cls)
        if not binary.any():
            continue
        labeled, n = ndimage.label(binary)
        for cid in range(1, n + 1):
            comp = (labeled == cid)
            if comp.sum() < MIN_COMPONENT_PIXELS:
                continue
            poly = component_to_polygon(comp)
            if poly is None:
                continue
            norm = [(x / img_w, y / img_h) for x, y in poly]
            out.append((yolo_cls, norm))
    return out


def build():
    print("=" * 70)
    print("MARIDA v2-seg builder (instance segmentation, reuses v2 scene split)")
    print("=" * 70)
    scene_to_split = load_v2_scene_split()
    print(f"  Reusing v2 scene split: {len(scene_to_split)} scenes")
    print(f"  Output: {OUT_DIR}")
    print(f"  Filter: MIN_COMPONENT_PIXELS={MIN_COMPONENT_PIXELS}, MIN_POLYGON_VERTICES={MIN_POLYGON_VERTICES}")
    print()

    patches = scan_patches()
    if not patches:
        sys.exit("no patches found")
    print(f"[scan] {len(patches)} patches")

    for split in ["train", "val", "test"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0, "with_objects": 0, "pure_background": 0,
        "per_split": defaultdict(int),
        "class_instances": defaultdict(int),
        "per_split_class_instances": defaultdict(lambda: defaultdict(int)),
        "dropped_low_pixels": 0,
        "dropped_short_polygon": 0,
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

            polys = []
            if p["cl_path"].exists():
                with rasterio.open(p["cl_path"]) as src:
                    cl_mask = src.read(1)
                if cl_mask.shape != (h, w):
                    cl_mask = cv2.resize(cl_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                polys = mask_to_polygons(cl_mask, h, w)

                for orig_cls in CLASS_MAP:
                    binary = (cl_mask == orig_cls)
                    if not binary.any():
                        continue
                    labeled, n = ndimage.label(binary)
                    for cid in range(1, n + 1):
                        comp = (labeled == cid)
                        sz = int(comp.sum())
                        if sz < MIN_COMPONENT_PIXELS:
                            stats["dropped_low_pixels"] += 1
                        else:
                            poly = component_to_polygon(comp)
                            if poly is None:
                                stats["dropped_short_polygon"] += 1

            np.save(OUT_DIR / split / "images" / f"{stem}.npy", img)

            label_path = OUT_DIR / split / "labels" / f"{stem}.txt"
            if polys:
                lines = []
                for cls, pts in polys:
                    parts = [str(cls)]
                    for x, y in pts:
                        parts.append(f"{x:.6f}")
                        parts.append(f"{y:.6f}")
                    lines.append(" ".join(parts))
                label_path.write_text("\n".join(lines) + "\n")
            else:
                label_path.write_text("")

            stats["total"] += 1
            stats["per_split"][split] += 1
            if polys:
                stats["with_objects"] += 1
                for cls, _ in polys:
                    stats["class_instances"][cls] += 1
                    stats["per_split_class_instances"][split][cls] += 1
            else:
                stats["pure_background"] += 1
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
    else:
        print(f"  [WARN] {src_stats} not found; need to copy/regenerate manually")

    print()
    print("=" * 70)
    print("STATS")
    print("=" * 70)
    print(f"  total: {stats['total']}, with_objects: {stats['with_objects']}, "
          f"background: {stats['pure_background']}, errors: {len(stats['errors'])}")
    print(f"  dropped_low_pixels (< {MIN_COMPONENT_PIXELS}): {stats['dropped_low_pixels']}")
    print(f"  dropped_short_polygon (< {MIN_POLYGON_VERTICES} verts): {stats['dropped_short_polygon']}")
    print()
    print("  Per-split class instances:")
    for split in ["train", "val", "test"]:
        parts = [f"{CLASS_NAMES[c]}={stats['per_split_class_instances'][split].get(c, 0)}"
                 for c in range(NC)]
        print(f"    {split:5s}: n={stats['per_split'][split]:4d}  {', '.join(parts)}")

    audit = {
        "version": "v2_seg",
        "task": "segment",
        "source_split": "marida_v2 scene-level (reused)",
        "class_map": {str(k): v for k, v in CLASS_MAP.items()},
        "class_names": CLASS_NAMES,
        "band_indices_1indexed": BAND_INDICES,
        "band_names": BAND_NAMES,
        "min_component_pixels": MIN_COMPONENT_PIXELS,
        "min_polygon_vertices": MIN_POLYGON_VERTICES,
        "scene_to_split": dict(scene_to_split),
        "total_patches": stats["total"],
        "with_objects": stats["with_objects"],
        "pure_background": stats["pure_background"],
        "dropped_low_pixels": stats["dropped_low_pixels"],
        "dropped_short_polygon": stats["dropped_short_polygon"],
        "errors": stats["errors"],
        "per_split": dict(stats["per_split"]),
        "class_instances": {CLASS_NAMES[k]: v for k, v in stats["class_instances"].items()},
        "per_split_class_instances": {
            split: {CLASS_NAMES[k]: v for k, v in m.items()}
            for split, m in stats["per_split_class_instances"].items()
        },
    }
    (OUT_DIR / "build_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"\nDone! {OUT_DIR}")


if __name__ == "__main__":
    build()
