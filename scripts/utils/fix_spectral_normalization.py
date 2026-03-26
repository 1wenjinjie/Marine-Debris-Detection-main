import numpy as np
from pathlib import Path
from tqdm import tqdm

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

def augment_with_spectral_indices_FIXED(npy_7ch):
    """Fixed version with proper normalization"""
    fdi = compute_fdi(npy_7ch)[..., None]
    ndvi = compute_ndvi(npy_7ch)[..., None]
    red_edge = compute_red_edge(npy_7ch)[..., None]
    
    # CRITICAL FIX: Normalize each index to [0, 1]
    fdi_norm = (fdi - fdi.min()) / (fdi.max() - fdi.min() + 1e-8)
    ndvi_norm = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-8)  # Fixed!
    red_edge_norm = (red_edge - red_edge.min()) / (red_edge.max() - red_edge.min() + 1e-8)  # Fixed!
    
    return np.concatenate([npy_7ch, fdi_norm, ndvi_norm, red_edge_norm], axis=-1)

# Re-generate 10ch dataset with fixed normalization
input_dir = Path("mask_bbox_rescue_split")
output_dir = Path("mask_bbox_rescue_10ch_fixed")

for split in ["train", "val", "test"]:
    (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

print("Regenerating 10ch dataset with FIXED normalization...")
for split in ["train", "val", "test"]:
    npy_files = list((input_dir / split / "images").glob("*.npy"))
    print(f"\n{split}: {len(npy_files)} files")
    for npy_file in tqdm(npy_files):
        data_7ch = np.load(npy_file)
        data_10ch = augment_with_spectral_indices_FIXED(data_7ch)
        np.save(output_dir / split / "images" / npy_file.name, data_10ch.astype(np.float32))
        
        label_src = input_dir / split / "labels" / npy_file.with_suffix(".txt").name
        label_dst = output_dir / split / "labels" / npy_file.with_suffix(".txt").name
        if label_src.exists():
            label_dst.write_text(label_src.read_text())

# Verify fix
sample = np.load(list((output_dir / "train" / "images").glob("*.npy"))[0])
print(f"\n=== Normalization Verification ===")
for i in range(10):
    ch = sample[:,:,i]
    print(f"Channel {i}: [{ch.min():.4f}, {ch.max():.4f}]", end="")
    if i >= 7 and (ch.min() < -0.01 or ch.max() > 1.01):
        print("   STILL WRONG!")
    elif i >= 7:
        print(" ")
    else:
        print()

print("\n Fixed dataset saved to: mask_bbox_rescue_10ch_fixed/")
