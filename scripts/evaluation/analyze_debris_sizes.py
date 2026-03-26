import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def analyze_bbox_sizes():
    label_dir = Path("MARIDA_YOLO_Enriched/train/labels")
    img_size = 256
    
    widths = []
    heights = []
    areas = []
    
    debris_class_id = 0 # In our map, Debris is 0
    
    print(f"Analyzing labels in {label_dir}...")
    
    txt_files = list(label_dir.glob("*.txt"))
    
    debris_count = 0
    
    for txt in tqdm(txt_files):
        with open(txt, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            
            if cls_id == debris_class_id:
                # YOLO format: cls x y w h (normalized)
                w_norm = float(parts[3])
                h_norm = float(parts[4])
                
                w_pix = w_norm * img_size
                h_pix = h_norm * img_size
                area_pix = w_pix * h_pix
                
                widths.append(w_pix)
                heights.append(h_pix)
                areas.append(area_pix)
                debris_count += 1
                
    print(f"\nFound {debris_count} Debris instances.")
    
    if debris_count == 0:
        print("No debris found.")
        return

    widths = np.array(widths)
    heights = np.array(heights)
    areas = np.array(areas)
    
    print("-" * 30)
    print("Debris Size Statistics (Pixels):")
    print(f"Width:  Min={widths.min():.2f}, Max={widths.max():.2f}, Mean={widths.mean():.2f}, Median={np.median(widths):.2f}")
    print(f"Height: Min={heights.min():.2f}, Max={heights.max():.2f}, Mean={heights.mean():.2f}, Median={np.median(heights):.2f}")
    print(f"Area:   Min={areas.min():.2f}, Max={areas.max():.2f}, Mean={areas.mean():.2f}, Median={np.median(areas):.2f}")
    
    print("-" * 30)
    print("Distribution:")
    print(f"< 1 pixel area: {np.sum(areas < 1)} ({np.sum(areas < 1)/debris_count:.1%})")
    print(f"< 4 pixel area (2x2): {np.sum(areas < 4)} ({np.sum(areas < 4)/debris_count:.1%})")
    print(f"< 9 pixel area (3x3): {np.sum(areas < 9)} ({np.sum(areas < 9)/debris_count:.1%})")
    print(f"> 25 pixel area (5x5): {np.sum(areas > 25)} ({np.sum(areas > 25)/debris_count:.1%})")

if __name__ == "__main__":
    analyze_bbox_sizes()
