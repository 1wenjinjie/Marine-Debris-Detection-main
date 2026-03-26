"""
Build the "Perfect" Dataset from MARIDA raw sources.
Features:
1. Uses original 256x256 patches (No Mosaic/Resize artifacts).
2. Extracts 6 bands (RGB+NIR+SWIR) for maximum spectral information.
3. Uses 7-Class mapping to explicitly separate Wakes from Debris.
4. Respects original MARIDA splits (Train/Val/Test) to prevent leakage.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml
import numpy as np

# Import processing functions
from preprocess_extract_bands import extract_bands
from preprocess_mask_to_yolo import convert_mask_to_yolo, YOLO_CLASS_NAMES

# Configuration
BASE_DIR = Path('D:/OceanEdge_Group/Marine-Debris-Detection-main')
RAW_PATCHES_DIR = BASE_DIR / 'MARIDA/patches'
SPLITS_DIR = BASE_DIR / 'MARIDA/splits'
OUTPUT_DIR = BASE_DIR / 'MARIDA_YOLO_Perfect'

# Classes of interest (Must match preprocess_mask_to_yolo.py)
MARINE_DEBRIS_CLASSES = [1, 2, 3, 4, 5, 9, 14]

def load_split_files(split_name):
    """Load list of filenames from split file."""
    split_file = SPLITS_DIR / f"{split_name}_X.txt"
    if not split_file.exists():
        print(f"Warning: Split file {split_file} not found.")
        return []
    
    with open(split_file, 'r') as f:
        # Read lines and strip whitespace
        files = [line.strip() for line in f.readlines() if line.strip()]
    return files

def find_source_files(filename):
    """
    Find the raw .tif and _cl.tif files for a given filename.
    The filename in splits might be '1-12-19_48MYU_0' (missing S2_ prefix).
    The file is located in 'MARIDA/patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0.tif'.
    """
    # Handle missing S2_ prefix
    if not filename.startswith('S2_'):
        real_filename = f"S2_{filename}"
    else:
        real_filename = filename

    # Extract scene name (everything before the last underscore)
    # Example: S2_1-12-19_48MYU_0 -> S2_1-12-19_48MYU
    scene_name = "_".join(real_filename.split('_')[:-1])
    
    scene_dir = RAW_PATCHES_DIR / scene_name
    
    img_path = scene_dir / f"{real_filename}.tif"
    mask_path = scene_dir / f"{real_filename}_cl.tif"
    
    if not img_path.exists():
        # Try searching recursively if simple path construction fails
        # This is a fallback
        found_imgs = list(RAW_PATCHES_DIR.rglob(f"{real_filename}.tif"))
        if found_imgs:
            img_path = found_imgs[0]
            mask_path = img_path.parent / f"{real_filename}_cl.tif"
        else:
            # print(f"  Not found: {img_path}")
            return None, None

    if not mask_path.exists():
        return None, None
        
    return img_path, mask_path

def process_dataset():
    print(f"Building Perfect Dataset at: {OUTPUT_DIR}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    stats = {
        'train': {'total': 0, 'valid': 0},
        'val': {'total': 0, 'valid': 0},
        'test': {'total': 0, 'valid': 0}
    }

    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} set...")
        filenames = load_split_files(split)
        stats[split]['total'] = len(filenames)
        
        for filename in tqdm(filenames, desc=f"Building {split}"):
            img_src, mask_src = find_source_files(filename)
            
            if not img_src or not mask_src:
                # print(f"Missing source for {filename}")
                continue
                
            # Define output paths
            img_dst = OUTPUT_DIR / split / 'images' / f"{filename}.tif"
            label_dst = OUTPUT_DIR / split / 'labels' / f"{filename}.txt"
            
            # 1. Extract 6 bands
            # We extract directly to the destination
            success = extract_bands(
                input_path=img_src,
                output_path=img_dst,
                bands=6,
                normalize=True,
                nodata=0
            )
            
            if not success:
                continue
                
            # 2. Generate YOLO Labels
            # We use the 7-class mapping defined in preprocess_mask_to_yolo.py
            num_objects, _ = convert_mask_to_yolo(
                mask_path=mask_src,
                output_txt_path=label_dst,
                classes_of_interest=MARINE_DEBRIS_CLASSES,
                min_area=None, # Auto-calculate
                bbox_padding=5
            )
            
            # If no objects, we still keep the image as background (negative sample)
            # unless we want to filter empty images. 
            # For "Perfect" dataset, let's keep them to reduce False Positives.
            stats[split]['valid'] += 1

    # Generate YAML config
    yaml_content = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(YOLO_CLASS_NAMES)}
    }
    
    yaml_path = OUTPUT_DIR / 'marida_perfect.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"\nDataset generation complete!")
    print(f"Config saved to: {yaml_path}")
    print("Stats:", stats)

if __name__ == "__main__":
    process_dataset()
