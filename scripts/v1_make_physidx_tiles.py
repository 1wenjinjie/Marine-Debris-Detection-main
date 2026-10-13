"""Replace v1 7ch tiles' GT-derived 7th channel with a physical NDWI-inverse index.

Channel layout (preserved): B4, B3, B2, B8, B11, B12, prior
- v1 original: prior = GT distance-transform heatmap (oracle, leaks GT at test)
- this script:  prior = (1 - NDWI) / 2 = (1 - (B3-B8)/(B3+B8)) / 2
  where NDWI = (Green - NIR) / (Green + NIR). Floating objects (debris, algae)
  have NIR > Green => NDWI < 0 => prior > 0.5. Pure water => prior ~0.

This makes the 7th channel a deterministic function of spectral bands, with
no dependence on GT labels. Trains/evaluates identically to v1 7ch pipeline.

Output: data/processed/mask_bbox_rescue_split_physidx/ with the same
{train,val,test}/{images,labels}/ layout. Labels are copied unchanged.
Also writes channel_stats_7ch.json with the new ch7 statistics.

Skips aug_mosaic_* tiles (consistent with v2-split experiments).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

SRC = Path("data/processed/mask_bbox_rescue_split")
DST = Path("data/processed/mask_bbox_rescue_split_physidx")
EPS = 1e-6


def ndwi_inverse(b3: np.ndarray, b8: np.ndarray) -> np.ndarray:
    ndwi = (b3 - b8) / (b3 + b8 + EPS)
    out = (1.0 - ndwi) / 2.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def main():
    ch7_samples = []  # for computing new channel stats
    total = 0
    for split in ("train", "val", "test"):
        img_dst = DST / split / "images"
        lbl_dst = DST / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        src_imgs = sorted((SRC / split / "images").glob("*.npy"))
        src_imgs = [p for p in src_imgs if not p.name.startswith("aug_")]
        for p in tqdm(src_imgs, desc=f"{split:5s}", unit="tile"):
            a = np.load(p)
            if a.ndim != 3 or a.shape[-1] != 7:
                raise ValueError(f"unexpected shape {a.shape} at {p}")
            # channel layout: 0=B4, 1=B3, 2=B2, 3=B8, 4=B11, 5=B12, 6=prior
            new_prior = ndwi_inverse(a[..., 1], a[..., 3])
            out = np.concatenate([a[..., :6], new_prior[..., None]], axis=-1)
            np.save(img_dst / p.name, out.astype(np.float32))
            ch7_samples.append(new_prior.ravel()[::64])  # subsample for stats
            total += 1
            # mirror the label
            lbl_src = SRC / split / "labels" / (p.stem + ".txt")
            if lbl_src.exists():
                shutil.copy(lbl_src, lbl_dst / lbl_src.name)

    print(f"wrote {total} tiles to {DST}")

    # compute new ch7 percentile stats; keep ch0..5 stats identical to source
    src_stats = json.loads((SRC / "channel_stats_7ch.json").read_text())
    pool = np.concatenate(ch7_samples)
    new_ch7_p01 = float(np.percentile(pool, 1))
    new_ch7_p99 = float(np.percentile(pool, 99))
    print(f"new ch7 stats: p01={new_ch7_p01:.4f} p99={new_ch7_p99:.4f} mean={pool.mean():.4f}")

    new_stats = json.loads(json.dumps(src_stats))  # deep copy
    # The loader reads keys p01 and p99 as lists of length=channels. Replace last entry.
    if "p01" in new_stats and "p99" in new_stats and len(new_stats["p01"]) == 7:
        new_stats["p01"][-1] = new_ch7_p01
        new_stats["p99"][-1] = new_ch7_p99
    else:
        print(f"WARNING: source stats has unexpected format; keys={list(new_stats.keys())}")
    (DST / "channel_stats_7ch.json").write_text(json.dumps(new_stats, indent=2))

    # mirror class_balance_weights and knowledge_summary if present (loader expects them)
    for name in ("class_balance_weights.json", "knowledge_generation_summary.json"):
        src_f = SRC / name
        if src_f.exists():
            shutil.copy(src_f, DST / name)
            print(f"copied {name}")


if __name__ == "__main__":
    main()
