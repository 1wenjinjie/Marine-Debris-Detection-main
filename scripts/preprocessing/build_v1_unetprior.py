"""Build v1 7-channel dataset with U-Net Marine_Debris probability as ch7.

Input:
  data/processed/mask_bbox_rescue_split_physidx/{train,val,test}/images/*.npy  (H, W, 7)
    ch0-5: B4 B3 B2 B8 B11 B12
    ch6:   NDWI-inverse physical prior (we DISCARD this and replace with U-Net prob)
  data/processed/mask_bbox_rescue_split_physidx_v2split/{train,val,test}.txt
    scene-clean split manifests

Plus:
  U-Net Marine_Debris probability masks (one .npy per tile, same stem, in --prob-dir)
    train tiles: from K-fold cross-prediction (leak-free, this script's preferred input)
    val/test:    from full-train U-Net (D seed=0)

Output:
  data/processed/mask_bbox_rescue_split_unetprior_v2split/
    train/images/*.npy            (H, W, 7) -- spectral 6ch + U-Net Marine_Debris prob
    train/labels/                 symlink/copy of v1 labels
    train.txt val.txt test.txt    (copied from v2split manifests)
    channel_stats_7ch.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
V1_PHYSIDX = ROOT / "data" / "processed" / "mask_bbox_rescue_split_physidx"
V1_PHYSIDX_V2SPLIT = ROOT / "data" / "processed" / "mask_bbox_rescue_split_physidx_v2split"
OUT_DIR = ROOT / "data" / "processed" / "mask_bbox_rescue_split_unetprior_v2split"


def build(prob_dir: Path):
    if not prob_dir.exists():
        sys.exit(f"prob_dir does not exist: {prob_dir}")
    print("=" * 70)
    print("Build v1 7ch dataset with U-Net Marine_Debris prior")
    print("=" * 70)
    print(f"  v1 source: {V1_PHYSIDX}")
    print(f"  v2split:   {V1_PHYSIDX_V2SPLIT}")
    print(f"  prob_dir:  {prob_dir}")
    print(f"  output:    {OUT_DIR}")

    for split in ["train", "val", "test"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    ch7_min, ch7_max = float("inf"), float("-inf")
    ch7_sum, ch7_count = 0.0, 0
    n_done, n_missing_prob = 0, 0

    for split in ["train", "val", "test"]:
        manifest = V1_PHYSIDX_V2SPLIT / f"{split}.txt"
        if not manifest.exists():
            print(f"  [WARN] missing {manifest}")
            continue
        rels = [l.strip() for l in manifest.read_text().splitlines() if l.strip()]
        print(f"\n[{split}] {len(rels)} tiles")

        for rel in tqdm(rels, desc=split):
            src_img = V1_PHYSIDX / rel
            if not src_img.exists():
                continue
            stem = Path(rel).stem
            prob_path = prob_dir / f"{stem}.npy"
            if not prob_path.exists():
                n_missing_prob += 1
                tqdm.write(f"  missing prob: {stem}")
                continue

            arr = np.load(src_img)  # (H, W, 7)
            if arr.ndim != 3 or arr.shape[-1] < 6:
                tqdm.write(f"  bad shape {arr.shape} in {stem}")
                continue
            spec6 = arr[..., :6].astype(np.float32)
            prob = np.load(prob_path).astype(np.float32)
            if prob.shape != spec6.shape[:2]:
                tqdm.write(f"  prob shape mismatch {prob.shape} vs {spec6.shape[:2]} for {stem}")
                continue

            new_tile = np.concatenate([spec6, prob[..., None]], axis=2)
            out_img = OUT_DIR / rel
            out_img.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_img, new_tile)

            # Copy label
            label_rel = rel.replace("/images/", "/labels/").replace(".npy", ".txt")
            src_lb = V1_PHYSIDX / label_rel
            dst_lb = OUT_DIR / label_rel
            dst_lb.parent.mkdir(parents=True, exist_ok=True)
            if src_lb.exists():
                shutil.copy2(src_lb, dst_lb)
            else:
                dst_lb.write_text("")

            ch7_min = min(ch7_min, float(prob.min()))
            ch7_max = max(ch7_max, float(prob.max()))
            ch7_sum += float(prob.sum())
            ch7_count += prob.size
            n_done += 1

    for split in ["train", "val", "test"]:
        src = V1_PHYSIDX_V2SPLIT / f"{split}.txt"
        if src.exists():
            shutil.copy2(src, OUT_DIR / f"{split}.txt")

    src_stats = V1_PHYSIDX / "channel_stats_7ch.json"
    if src_stats.exists():
        stats = json.loads(src_stats.read_text())
        if ch7_count > 0:
            stats["percentiles"].setdefault("p01", [0.0] * 7)
            stats["percentiles"].setdefault("p99", [1.0] * 7)
            stats["percentiles"]["p01"][6] = max(0.0, ch7_min)
            stats["percentiles"]["p99"][6] = min(1.0, max(ch7_max, ch7_min + 1e-3))
        if "channel_names" in stats and len(stats["channel_names"]) >= 7:
            stats["channel_names"][6] = "UNet_Marine_Debris_probability"
        stats["note"] = "ch7 replaced with U-Net Marine_Debris probability; p01/p99 recomputed from prob_dir"
        (OUT_DIR / "channel_stats_7ch.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False)
        )

    audit = {
        "version": "unetprior_v2split",
        "ch7_source": str(prob_dir),
        "ch7_replaces": "NDWI-inverse physical prior",
        "n_tiles_written": n_done,
        "n_missing_prob": n_missing_prob,
        "ch7_stats": {
            "min": ch7_min if ch7_count > 0 else None,
            "max": ch7_max if ch7_count > 0 else None,
            "mean": ch7_sum / ch7_count if ch7_count > 0 else None,
        },
    }
    (OUT_DIR / "build_audit.json").write_text(json.dumps(audit, indent=2))
    print()
    print(f"done: {n_done} tiles, {n_missing_prob} missing prob masks")
    print(f"ch7 stats: min={ch7_min:.4f} max={ch7_max:.4f} mean={audit['ch7_stats']['mean']:.4f}")
    print(f"output: {OUT_DIR}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prob-dir", type=Path, required=True,
                   help="Directory containing U-Net Marine_Debris prob mask npy (one per tile, same stem)")
    args = p.parse_args()
    build(args.prob_dir)


if __name__ == "__main__":
    main()
