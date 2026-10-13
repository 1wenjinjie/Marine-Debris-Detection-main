"""
Phase 1 closing sanity checks. Two non-optional verifications before
moving to Phase 2.

CHECK A — polygon label integrity
  For each tile in train/labels_highmod/:
    - every line has class_id + ≥6 floats (3+ xy pairs)
    - all coords in [0, 1]
    - polygon count == instance_mask_highmod unique value count (excl. 0)
    - for instances with n_pixels ≤ 2: how many polygon vertices?
      (current build_v2_1.polygon_from_mask uses bbox-4-corner, so this
      will always be 4, but bbox area for 1-px is 1×1 which after
      letterbox can vanish. We need to know.)

CHECK B — split balance: is any single scene dominating its split?
  Reads split_manifest.json + per-tile labels_highmod count.
  For each split, computes:
    - scene-level debris count distribution
    - top-1 scene share of total debris in that split
  Flags any split where top-1 share > 30%.

Outputs:
  runs_v2_1/sanity/phase1_sanity_report.json
  console summary

Usage:
  python scripts/v2_1/preprocessing/sanity_phase1.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V21 = ROOT / "data" / "processed" / "marida_v2_1"
OUT = ROOT / "runs_v2_1" / "sanity"
OUT.mkdir(parents=True, exist_ok=True)

SCENE_RE = re.compile(r"^(S2_\d+-\d+-\d+_[A-Z0-9]+)_\d+$")


def scene_of(tile_stem):
    m = SCENE_RE.match(tile_stem)
    return m.group(1) if m else tile_stem


def check_a_polygon_integrity(label_set: str = "highmod", sample_size: int = 30):
    """CHECK A — polygon label integrity across all tiles."""
    report = {
        "label_set": label_set,
        "n_files_checked": 0,
        "n_files_with_gt": 0,
        "n_polygons_total": 0,
        "n_degenerate_polygons": 0,           # < 3 vertices
        "n_oor_polygons": 0,                  # coord out of [0,1]
        "n_mismatch_with_instmask": 0,
        "vertex_count_histogram": defaultdict(int),
        "polygon_size_histogram": {           # for the *normalized* xy span
            "<=2px (≤0.008)": 0,              # 2/256 = 0.008
            "3-4px": 0,
            "5-10px": 0,
            ">10px": 0,
        },
        "tiny_gt_examples": [],               # show some 1-2 px GT cases
    }

    for split in ("train", "val", "test"):
        label_dir = V21 / split / f"labels_{label_set}"
        instmask_dir = V21 / split / f"instance_masks_{label_set}"
        meta_dir = V21 / split / "instance_metadata_highmod"

        for label_path in sorted(label_dir.glob("*.txt")):
            text = label_path.read_text().strip()
            stem = label_path.stem
            report["n_files_checked"] += 1
            if not text:
                continue
            report["n_files_with_gt"] += 1
            polygons = []
            for line in text.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(parts[0])
                    xy = list(map(float, parts[1:]))
                except ValueError:
                    report["n_degenerate_polygons"] += 1
                    continue
                if len(xy) < 6 or len(xy) % 2 != 0:
                    report["n_degenerate_polygons"] += 1
                    continue
                n_vert = len(xy) // 2
                report["vertex_count_histogram"][n_vert] += 1

                # range check
                xs = xy[0::2]
                ys = xy[1::2]
                if min(xs) < -1e-6 or max(xs) > 1 + 1e-6 or min(ys) < -1e-6 or max(ys) > 1 + 1e-6:
                    report["n_oor_polygons"] += 1

                # size
                span = max(max(xs) - min(xs), max(ys) - min(ys))
                if span <= 0.008:
                    report["polygon_size_histogram"]["<=2px (≤0.008)"] += 1
                elif span <= 0.016:
                    report["polygon_size_histogram"]["3-4px"] += 1
                elif span <= 0.04:
                    report["polygon_size_histogram"]["5-10px"] += 1
                else:
                    report["polygon_size_histogram"][">10px"] += 1
                polygons.append({"cls": cls, "span": span, "n_vert": n_vert})
                report["n_polygons_total"] += 1

            # cross-check vs instance mask
            inst_npy = instmask_dir / f"{stem}.npy"
            if inst_npy.exists():
                inst = np.load(inst_npy)
                n_inst = int(inst.max())
                if n_inst != len(polygons):
                    report["n_mismatch_with_instmask"] += 1

            # collect 1-2 px examples
            meta_json = meta_dir / f"{stem}.json"
            if meta_json.exists() and len(report["tiny_gt_examples"]) < 10:
                meta = json.loads(meta_json.read_text())
                for inst_id, info in meta.items():
                    if info["n_pixels"] <= 2:
                        # find the corresponding polygon in the line list
                        idx = int(inst_id) - 1
                        if 0 <= idx < len(polygons):
                            report["tiny_gt_examples"].append({
                                "tile": stem,
                                "inst_id": int(inst_id),
                                "n_pixels": info["n_pixels"],
                                "polygon_n_vert": polygons[idx]["n_vert"],
                                "polygon_span_normalized": round(polygons[idx]["span"], 5),
                            })
    report["vertex_count_histogram"] = dict(sorted(report["vertex_count_histogram"].items()))
    return report


def check_b_split_balance():
    """CHECK B — single-scene domination per split."""
    manifest = json.loads((V21 / "split_manifest.json").read_text())
    scene_to_split = manifest["scene_to_split"]

    # Count actual highmod boxes per tile (re-read labels)
    per_scene_boxes = defaultdict(int)
    per_scene_tiles = defaultdict(int)
    for split in ("train", "val", "test"):
        for label_path in (V21 / split / "labels_highmod").glob("*.txt"):
            stem = label_path.stem
            sc = scene_of(stem)
            per_scene_tiles[sc] += 1
            text = label_path.read_text().strip()
            if not text:
                continue
            per_scene_boxes[sc] += len(text.strip().splitlines())

    by_split = defaultdict(list)   # split -> list of (scene, n_boxes)
    for sc, n in per_scene_boxes.items():
        split = scene_to_split.get(sc, "unknown")
        by_split[split].append((sc, n))

    report = {}
    for split, scene_list in by_split.items():
        scene_list.sort(key=lambda x: -x[1])
        total = sum(n for _, n in scene_list)
        top1 = scene_list[0] if scene_list else ("", 0)
        report[split] = {
            "total_boxes": total,
            "n_debris_scenes": len([s for s, n in scene_list if n > 0]),
            "top1_scene": top1[0],
            "top1_boxes": top1[1],
            "top1_share_pct": round(100 * top1[1] / total, 1) if total else 0,
            "scenes_breakdown": [{"scene": s, "n_boxes": n} for s, n in scene_list[:5]],
            "domination_warning": top1[1] / total > 0.30 if total else False,
        }
    return report


def main() -> int:
    print("=" * 70)
    print("PHASE 1 CLOSING SANITY CHECKS")
    print("=" * 70)

    print("\n[CHECK A] polygon label integrity (label_set=highmod)")
    print("-" * 70)
    a = check_a_polygon_integrity("highmod")
    print(f"  n_files_checked:              {a['n_files_checked']}")
    print(f"  n_files_with_gt:              {a['n_files_with_gt']}")
    print(f"  n_polygons_total:             {a['n_polygons_total']}")
    print(f"  n_degenerate (<3 vert):       {a['n_degenerate_polygons']}")
    print(f"  n_out_of_range coord:         {a['n_oor_polygons']}")
    print(f"  n_mismatch w/ instance_mask:  {a['n_mismatch_with_instmask']}")
    print(f"  vertex count histogram:       {a['vertex_count_histogram']}")
    print(f"  polygon size histogram:")
    for bucket, count in a["polygon_size_histogram"].items():
        print(f"      {bucket:25s}: {count}")
    print(f"  tiny GT (n_pixels ≤ 2) examples:")
    for ex in a["tiny_gt_examples"][:5]:
        print(f"      {ex['tile']:30s} inst#{ex['inst_id']}  "
              f"n_px={ex['n_pixels']} -> polygon n_vert={ex['polygon_n_vert']}, "
              f"span={ex['polygon_span_normalized']}")

    # Decision rules
    print()
    if a["n_degenerate_polygons"] > 0:
        print(f"  [FAIL] {a['n_degenerate_polygons']} degenerate polygons (< 3 vertices)")
    if a["n_oor_polygons"] > 0:
        print(f"  [FAIL] {a['n_oor_polygons']} polygons with coordinates out of [0,1]")
    if a["n_mismatch_with_instmask"] > 0:
        print(f"  [FAIL] {a['n_mismatch_with_instmask']} tiles where polygon count != instance mask count")

    tiny_share = a["polygon_size_histogram"]["<=2px (≤0.008)"]
    if tiny_share > 0:
        print(f"  [INFO] {tiny_share} polygons are ≤2 px in normalized span.")
        print(f"         Current build_v2_1 uses bbox-4-corner polygons.")
        print(f"         No dilation applied at build time.")
        print(f"         These tiny polygons MUST be handled in Phase 2 dataloader")
        print(f"         (dilate to 3x3 at train time per SPEC §7).")

    print()
    print("[CHECK B] split balance — is any single scene dominating?")
    print("-" * 70)
    b = check_b_split_balance()
    for split in ("train", "val", "test"):
        s = b.get(split, {})
        if not s:
            continue
        print(f"  {split:5s}  total={s['total_boxes']:4d}  "
              f"n_debris_scenes={s['n_debris_scenes']}  "
              f"top-1 scene={s['top1_scene']} ({s['top1_boxes']} boxes, "
              f"{s['top1_share_pct']}%)  "
              f"{'[!] DOMINATION' if s['domination_warning'] else 'OK'}")
        print(f"         top 5 scenes:")
        for entry in s["scenes_breakdown"]:
            print(f"           {entry['scene']:30s} {entry['n_boxes']:4d}")

    # Write JSON report
    full_report = {"check_a_polygon": a, "check_b_split_balance": b}
    out_path = OUT / "phase1_sanity_report.json"
    out_path.write_text(json.dumps(full_report, indent=2, default=str))
    print(f"\n  full report -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
