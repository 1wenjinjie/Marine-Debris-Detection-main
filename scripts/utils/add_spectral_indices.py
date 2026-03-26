#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

def compute_fdi(npy_7ch):
    red, nir, swir1 = npy_7ch[..., 3], npy_7ch[..., 4], npy_7ch[..., 5]
    fdi = nir - (red + (swir1 - red) * (865 - 665) / (1610 - 665) * 10)
    return fdi

def compute_ndvi(npy_7ch):
    red, nir = npy_7ch[..., 3], npy_7ch[..., 4]
    return (nir - red) / (nir + red + 1e-8)

def compute_red_edge(npy_7ch):
    nir, swir1 = npy_7ch[..., 4], npy_7ch[..., 5]
    return (swir1 - nir) / (swir1 + nir + 1e-8)

def augment_with_spectral_indices(npy_7ch):
    fdi = compute_fdi(npy_7ch)[..., None]
    ndvi = compute_ndvi(npy_7ch)[..., None]
    red_edge = compute_red_edge(npy_7ch)[..., None]
    
    fdi = (fdi - fdi.min()) / (fdi.max() - fdi.min() + 1e-8)
    ndvi = (ndvi + 1) / 2
    red_edge = (red_edge + 1) / 2
    
    return np.concatenate([npy_7ch, fdi, ndvi, red_edge], axis=-1)

if __name__ == "__main__":
    input_dir = Path("mask_bbox_rescue_split")
    output_dir = Path("mask_bbox_rescue_10ch")
    
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    stats = {"total": 0, "success": 0}
    for split in ["train", "val", "test"]:
        npy_files = list((input_dir / split / "images").glob("*.npy"))
        print(f"Processing {split}: {len(npy_files)} files")
        for npy_file in tqdm(npy_files):
            data_7ch = np.load(npy_file)
            data_10ch = augment_with_spectral_indices(data_7ch)
            np.save(output_dir / split / "images" / npy_file.name, data_10ch.astype(np.float32))
            
            label_src = input_dir / split / "labels" / npy_file.with_suffix(".txt").name
            label_dst = output_dir / split / "labels" / npy_file.with_suffix(".txt").name
            if label_src.exists():
                label_dst.write_text(label_src.read_text())
            stats["success"] += 1
            stats["total"] += 1
    
    print(f"Complete: {stats}")
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
