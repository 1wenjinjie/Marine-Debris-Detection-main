import os
# Fix OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import cv2

# Configuration
MODEL_PATH = 'marida_enriched_runs/yolo11n_6ch_enriched_dilated/weights/best.pt'
DATA_DIR = 'MARIDA_YOLO_Enriched_Dilated/train'
OUTPUT_DIR = 'spectral_analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Band Names (Based on preprocess_extract_bands.py)
# The extraction script selects: Blue, Green, Red, NIR, SWIR1, SWIR2
# But usually they are stacked in order.
# Let's assume the output order in the 6-channel TIF is:
# 0: Blue
# 1: Green
# 2: Red
# 3: NIR
# 4: SWIR1
# 5: SWIR2
BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
BAND_WAVELENGTHS = [490, 560, 665, 842, 1610, 2190] # Approx center wavelengths in nm

def load_image_for_model(tif_path, channels=6):
    with rasterio.open(tif_path) as src:
        arr = src.read() # (C, H, W)
        arr = np.transpose(arr, (1, 2, 0)) # (H, W, C)
        
    if arr.shape[-1] > channels:
        arr = arr[..., :channels]
        
    # Normalize to 0-1 for analysis (raw values)
    # Note: The model training used 0-255 scaling.
    # For spectral analysis, we want the relative shape of the curve, 
    # so 0-1 or raw values are fine as long as consistent.
    # Let's use 0-1 normalized by the image max to keep it comparable across images.
    if arr.max() > arr.min():
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr_norm = arr
        
    # For Model Prediction, we need the 0-255 scaled version
    arr_model = arr_norm * 255.0
    
    return arr_norm, arr_model.astype(np.float32)

def analyze_spectra():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    
    label_dir = Path(DATA_DIR) / 'labels'
    image_dir = Path(DATA_DIR) / 'images'
    
    # Storage for spectral signatures
    signatures = {
        'TP_Debris': [], # True Positive Debris
        'FP_Debris': [], # False Positive Debris (Model said Debris, was not)
        'FN_Debris': [], # False Negative Debris (Was Debris, Model missed)
        'Background': [] # Random water/other
    }
    
    print("Analyzing images...")
    # Find images with Debris
    debris_files = []
    for lbl_file in label_dir.glob("*.txt"):
        with open(lbl_file, 'r') as f:
            if '0 ' in f.read(): # Class 0 is Debris
                debris_files.append(lbl_file.stem)
    
    # Limit to a subset for speed, but enough for stats
    sample_files = debris_files[:100] 
    
    for file_stem in tqdm(sample_files):
        img_path = image_dir / (file_stem + ".tif")
        lbl_path = label_dir / (file_stem + ".txt")
        
        if not img_path.exists(): continue
        
        # Load
        img_norm, img_model = load_image_for_model(img_path)
        h, w, c = img_norm.shape
        
        # Predict
        results = model.predict(img_model, conf=0.1, verbose=False)
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0: # Debris
                    pred_boxes.append(box.xywh[0].cpu().numpy()) # x_c, y_c, w, h
        
        # Load GT
        gt_boxes = []
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if int(parts[0]) == 0:
                    # Convert normalized xywh to pixels
                    nx, ny, nw, nh = map(float, parts[1:5])
                    gt_boxes.append([nx*w, ny*h, nw*w, nh*h])
        
        # Match Predictions to GT
        # Simple center containment or proximity check
        matched_gt_indices = set()
        
        # 1. Analyze Predictions (TP vs FP)
        for pbox in pred_boxes:
            px, py, pw, ph = pbox
            
            # Extract spectrum at center pixel
            ix, iy = int(px), int(py)
            ix = np.clip(ix, 0, w-1)
            iy = np.clip(iy, 0, h-1)
            spectrum = img_norm[iy, ix, :]
            
            is_match = False
            for i, gbox in enumerate(gt_boxes):
                gx, gy, gw, gh = gbox
                # Check if prediction center is inside GT box (or close)
                # Since we dilated, GT might be small, but let's check distance
                dist = np.sqrt((px-gx)**2 + (py-gy)**2)
                if dist < 5.0: # 5 pixels tolerance
                    is_match = True
                    matched_gt_indices.add(i)
                    break
            
            if is_match:
                signatures['TP_Debris'].append(spectrum)
            else:
                signatures['FP_Debris'].append(spectrum)
                
        # 2. Analyze Missed GT (FN)
        for i, gbox in enumerate(gt_boxes):
            if i not in matched_gt_indices:
                gx, gy, gw, gh = gbox
                ix, iy = int(gx), int(gy)
                ix = np.clip(ix, 0, w-1)
                iy = np.clip(iy, 0, h-1)
                spectrum = img_norm[iy, ix, :]
                signatures['FN_Debris'].append(spectrum)
                
        # 3. Background Samples (Random points away from debris)
        # Simple approach: take 5 random points
        for _ in range(5):
            rx, ry = np.random.randint(0, w), np.random.randint(0, h)
            # Ensure far from any GT
            min_dist = 9999
            for gbox in gt_boxes:
                gx, gy, _, _ = gbox
                dist = np.sqrt((rx-gx)**2 + (ry-gy)**2)
                min_dist = min(min_dist, dist)
            
            if min_dist > 10:
                spectrum = img_norm[ry, rx, :]
                signatures['Background'].append(spectrum)

    # Plotting
    plt.figure(figsize=(12, 8))
    
    colors = {'TP_Debris': 'red', 'FP_Debris': 'orange', 'FN_Debris': 'purple', 'Background': 'blue'}
    labels = {'TP_Debris': 'Correctly Detected Debris', 
              'FP_Debris': 'False Positives', 
              'FN_Debris': 'Missed Debris', 
              'Background': 'Water/Background'}
    
    for key, data in signatures.items():
        if len(data) == 0: continue
        data = np.array(data)
        mean_spec = np.mean(data, axis=0)
        std_spec = np.std(data, axis=0)
        
        # Plot mean
        plt.plot(BAND_NAMES, mean_spec, label=f"{labels[key]} (n={len(data)})", color=colors[key], linewidth=2)
        # Plot std dev area
        plt.fill_between(BAND_NAMES, mean_spec - std_spec*0.5, mean_spec + std_spec*0.5, color=colors[key], alpha=0.1)
        
    plt.title("Spectral Signatures of Detected vs Missed Debris")
    plt.xlabel("Sentinel-2 Bands")
    plt.ylabel("Normalized Reflectance (0-1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = Path(OUTPUT_DIR) / "spectral_analysis_plot.png"
    plt.savefig(out_file)
    print(f"Saved spectral plot to {out_file}")
    
    # Save stats
    with open(Path(OUTPUT_DIR) / "spectral_stats.txt", "w") as f:
        for key, data in signatures.items():
            if len(data) == 0: continue
            data = np.array(data)
            mean_spec = np.mean(data, axis=0)
            f.write(f"{key} Mean: {mean_spec}\n")

if __name__ == "__main__":
    analyze_spectra()
