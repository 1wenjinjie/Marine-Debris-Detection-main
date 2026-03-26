import os
import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import shutil

# Configuration
MODEL_PATH = 'marida_enriched_runs/yolo11n_6ch_enriched_dilated/weights/best.pt'
DATA_DIR = 'MARIDA_YOLO_Enriched_Dilated/train'  # Use training set as val might be empty or small
OUTPUT_DIR = 'visualizations_false_color'
NUM_SAMPLES = 20
CONF_THRESHOLD = 0.15

# Indices in the 11-channel Sentinel-2 file
# 1: Blue, 2: Green, 3: Red, 7: NIR, 9: SWIR1, 10: SWIR2
SOURCE_BAND_INDICES = [1, 2, 3, 7, 9, 10]

# Map for the extracted 6-channel array
BAND_MAP = {
    'Blue': 0,
    'Green': 1,
    'Red': 2,
    'NIR': 3,
    'SWIR1': 4,
    'SWIR2': 5
}

def normalize_band(band):
    """Normalize a band to 0-255 range for visualization."""
    min_val = band.min()
    max_val = band.max()
    if max_val > min_val:
        return ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return band.astype(np.uint8)

def create_false_color_composite(image_data):
    """NIR, Red, Green"""
    nir = image_data[:, :, BAND_MAP['NIR']]
    red = image_data[:, :, BAND_MAP['Red']]
    green = image_data[:, :, BAND_MAP['Green']]
    
    composite = np.dstack([
        normalize_band(green), # Blue channel
        normalize_band(red),   # Green channel
        normalize_band(nir)    # Red channel
    ])
    return composite

def create_swir_composite(image_data):
    """SWIR2, NIR, Red"""
    swir2 = image_data[:, :, BAND_MAP['SWIR2']]
    nir = image_data[:, :, BAND_MAP['NIR']]
    red = image_data[:, :, BAND_MAP['Red']]
    
    composite = np.dstack([
        normalize_band(red),   # Blue channel
        normalize_band(nir),   # Green channel
        normalize_band(swir2)  # Red channel
    ])
    return composite

def run_visualization():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    image_dir = Path(DATA_DIR) / 'images'
    label_dir = Path(DATA_DIR) / 'labels'
    image_files = list(image_dir.glob('*.tif'))
    
    if not image_files:
        print(f"No .tif files found in {image_dir}")
        return

    print(f"Processing {min(NUM_SAMPLES, len(image_files))} samples...")
    
    count = 0
    for img_path in image_files:
        if count >= NUM_SAMPLES:
            break
            
        label_path = label_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
            
        # Load Image
        with rasterio.open(img_path) as src:
            raw_data = src.read() # (C, H, W)
            raw_data = np.transpose(raw_data, (1, 2, 0)) # (H, W, C)
            
        # Extract 6 channels
        if raw_data.shape[2] >= 11:
            img_6ch = raw_data[:, :, SOURCE_BAND_INDICES]
        elif raw_data.shape[2] == 6:
            img_6ch = raw_data
        else:
            print(f"Skipping {img_path.name}: Unexpected channel count {raw_data.shape[2]}")
            continue
            
        # Create Visualizations
        fc_img = create_false_color_composite(img_6ch)
        swir_img = create_swir_composite(img_6ch)
        
        # Run Inference
        img_for_model = img_6ch.copy()
        if img_for_model.dtype != np.uint8:
             for i in range(img_for_model.shape[2]):
                 band = img_for_model[:, :, i]
                 min_v, max_v = band.min(), band.max()
                 if max_v > min_v:
                     img_for_model[:, :, i] = ((band - min_v) / (max_v - min_v) * 255)
                 else:
                     img_for_model[:, :, i] = 0
             img_for_model = img_for_model.astype(np.uint8)

        results = model.predict(img_for_model, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]
        
        # Draw Predictions on False Color Image
        vis_fc = fc_img.copy()
        vis_swir = swir_img.copy()
        
        # Draw Ground Truth (Green)
        if label_path.exists():
            h, w = fc_img.shape[:2]
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    
                    cv2.rectangle(vis_fc, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.rectangle(vis_swir, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Draw Predictions (Red)
        if result.boxes:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                
                cv2.rectangle(vis_fc, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 1)
                cv2.rectangle(vis_swir, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 1)
                
                label = f"{conf:.2f}"
                cv2.putText(vis_fc, label, (coords[0], coords[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                cv2.putText(vis_swir, label, (coords[0], coords[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        # Combine
        vis_fc = cv2.copyMakeBorder(vis_fc, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        cv2.putText(vis_fc, "False Color (NIR-R-G)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        vis_swir = cv2.copyMakeBorder(vis_swir, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        cv2.putText(vis_swir, "SWIR Composite (SWIR2-NIR-R)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        combined = np.hstack([vis_fc, vis_swir])
        
        out_path = os.path.join(OUTPUT_DIR, f"vis_{img_path.stem}.png")
        cv2.imwrite(out_path, combined)
        count += 1
        
    print(f"Saved {count} visualizations to {OUTPUT_DIR}")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    run_visualization()