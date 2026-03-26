import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

def analyze_bbox_sizes():
    label_dir = Path("MARIDA_YOLO_Enriched_Filtered/train/labels")
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
    
    print(f"Width (px): Min={widths.min():.2f}, Max={widths.max():.2f}, Mean={widths.mean():.2f}, Median={np.median(widths):.2f}")
    print(f"Height (px): Min={heights.min():.2f}, Max={heights.max():.2f}, Mean={heights.mean():.2f}, Median={np.median(heights):.2f}")
    print(f"Area (px^2): Min={areas.min():.2f}, Max={areas.max():.2f}, Mean={areas.mean():.2f}, Median={np.median(areas):.2f}")
    
    print(f"Count < 1 px area: {np.sum(areas < 1)}")
    print(f"Count < 3 px area: {np.sum(areas < 3)}")
    print(f"Count >= 5 px area: {np.sum(areas >= 5)}")

if __name__ == "__main__":
    analyze_bbox_sizes()
