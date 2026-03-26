import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import random
import rasterio

# Configuration
MODEL_PATH = 'marida_enriched_runs/yolo11n_6ch_enriched_dilated/weights/best.pt'
DATA_DIR = 'MARIDA_YOLO_Enriched_Dilated/train' # We used train for val
OUTPUT_DIR = 'visualizations_dilated_results'
NUM_SAMPLES = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class Names
class_names = {
  0: 'Marine_Debris',
  1: 'Dense_Sargassum',
  2: 'Sparse_Sargassum',
  3: 'Natural_Organic',
  4: 'Ship',
  5: 'Foam',
  6: 'Wake'
}

# Load Model
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

def normalize_band(band):
    min_val = band.min()
    max_val = band.max()
    if max_val > min_val:
        return (band - min_val) / (max_val - min_val)
    return band

def read_tif_as_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        # Read RGB bands (usually 1, 2, 3 or 3, 2, 1 depending on order)
        # Assuming standard S2 order: B2(Blue), B3(Green), B4(Red), B8(NIR), ...
        # But our preprocess_extract_bands might have changed order.
        # Let's assume standard RGB visualization from first 3 channels for now.
        # If it looks weird, we can adjust.
        # Actually, let's read all 6 and try to make a True Color (4,3,2) or similar.
        # If the input is 6 channels, we need to know what they are.
        # Usually: R, G, B, NIR, RE1, RE2 or similar.
        # Let's just take first 3 and treat as RGB.
        
        img = src.read([1, 2, 3]) # Read first 3 bands
        img = np.moveaxis(img, 0, -1) # (C, H, W) -> (H, W, C)
        
        # Normalize for display
        img_norm = np.zeros_like(img, dtype=np.float32)
        for i in range(3):
            img_norm[..., i] = normalize_band(img[..., i])
            
        return (img_norm * 255).astype(np.uint8)

def load_image_for_model(tif_path, channels=6):
    with rasterio.open(tif_path) as src:
        arr = src.read() # (C, H, W)
        arr = np.transpose(arr, (1, 2, 0)) # (H, W, C)
        
    # Slice channels
    if arr.shape[-1] > channels:
        arr = arr[..., :channels]
        
    # Normalize to 0-1
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        
    # Scale to 0-255 (Model expects this range based on training dataset logic)
    arr = arr * 255.0
    
    return arr.astype(np.float32)

def visualize():
    # Find images with Debris (Class 0)
    label_dir = Path(DATA_DIR) / 'labels'
    image_dir = Path(DATA_DIR) / 'images'
    
    debris_files = []
    for lbl_file in label_dir.glob("*.txt"):
        with open(lbl_file, 'r') as f:
            content = f.read()
            # Check for class 0
            if content.strip().startswith('0 ') or '\n0 ' in content:
                debris_files.append(lbl_file.stem)
                
    print(f"Found {len(debris_files)} images containing Debris.")
    
    # Select samples
    selected_files = random.sample(debris_files, min(NUM_SAMPLES, len(debris_files)))
    
    for file_stem in selected_files:
        img_path = image_dir / (file_stem + ".tif")
        if not img_path.exists():
            continue
            
        # Load and Preprocess for Model
        model_input = load_image_for_model(img_path)
        
        # Run Inference
        # Pass numpy array directly
        results = model.predict(model_input, conf=0.1, verbose=False) 
        
        # Prepare Visualization
        # Load RGB for background
        bg_img = read_tif_as_rgb(img_path)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR)
        
        # Draw Ground Truth
        lbl_path = label_dir / (file_stem + ".txt")
        h, w = bg_img.shape[:2]
        
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            if cls != 0: continue # Only draw Debris GT
            
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            cv2.rectangle(bg_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for GT
            cv2.putText(bg_img, "GT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw Predictions
        pred_count = 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Only draw Debris predictions or high conf others
                if cls == 0:
                    color = (0, 0, 255) # Red for Pred Debris
                    label = f"Pred {conf:.2f}"
                    pred_count += 1
                else:
                    continue # Skip others for clarity
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(bg_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(bg_img, label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        print(f"Image {file_stem}: {pred_count} Debris predictions.")
                
        # Save
        out_path = Path(OUTPUT_DIR) / f"{file_stem}_vis.png"
        cv2.imwrite(str(out_path), bg_img)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    visualize()
