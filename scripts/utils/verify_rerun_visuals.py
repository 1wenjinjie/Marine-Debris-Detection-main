import os
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import yaml

# Constants
DATASET_ROOT = Path('D:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split')
VAL_LIST = DATASET_ROOT / 'val.txt'
OUTPUT_DIR = Path('visualizations_rerun_verification')
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = {
    '3ch': {
        'weights': 'marida_runs/yolo11n_3ch_rgb_rerun/weights/best.pt',
        'channels': 3
    },
    '4ch': {
        'weights': 'marida_runs/yolo11n_4ch_nir_rerun/weights/best.pt',
        'channels': 4
    },
    '5ch': {
        'weights': 'marida_runs/yolo11n_5ch_swir1_rerun/weights/best.pt',
        'channels': 5
    },
    '6ch': {
        'weights': 'marida_runs/yolo11n_6ch_full_rerun/weights/best.pt',
        'channels': 6
    }
}

# Load class names from one of the YAMLs
with open('marida_rescue_split_3ch.yaml', 'r') as f:
    CLASS_NAMES = yaml.safe_load(f)['names']

def load_npy_image(path, channels):
    """
    Loads and preprocesses .npy image matching MaskRescueNPYDataset logic.
    """
    arr = np.load(path, allow_pickle=False)
    
    # Transpose (C, H, W) -> (H, W, C) if needed
    if arr.shape[0] in (6, 7) and arr.shape[0] <= arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)
        
    # Slice channels (assuming < 11 channels, so just take first N)
    arr = arr[..., :channels]
    
    # Scaling (x1500)
    arr = arr * 1500.0
    arr = np.clip(arr, 0.0, 255.0)
    
    return arr.astype(np.float32)

def get_ground_truth(npy_path, width, height):
    label_path = str(npy_path).replace('images', 'labels').replace('.npy', '.txt')
    boxes = []
    classes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:])
                
                x1 = int((x_c - w/2) * width)
                y1 = int((y_c - h/2) * height)
                x2 = int((x_c + w/2) * width)
                y2 = int((y_c + h/2) * height)
                
                boxes.append([x1, y1, x2, y2])
                classes.append(cls_id)
    return boxes, classes

def draw_boxes(img_rgb, boxes, classes, color=(0, 255, 0), title=None):
    """
    Draws boxes on a copy of the RGB image.
    img_rgb: (H, W, 3) uint8 numpy array
    """
    img = img_rgb.copy()
    # Convert RGB to BGR for OpenCV drawing
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        # label = CLASS_NAMES[int(cls)]
        # cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    if title:
        cv2.putText(img_bgr, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    # Convert back to RGB
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def main():
    # 1. Load Models
    loaded_models = {}
    for name, cfg in MODELS.items():
        print(f"Loading {name} model from {cfg['weights']}...")
        if os.path.exists(cfg['weights']):
            loaded_models[name] = YOLO(cfg['weights'])
        else:
            print(f"Warning: Weights for {name} not found.")
            
    # 2. Select Samples
    with open(VAL_LIST, 'r') as f:
        val_files = [DATASET_ROOT / line.strip() for line in f.readlines()]
    
    # Pick 5 random samples
    random.seed(42)
    samples = random.sample(val_files, 5)
    
    # 3. Process each sample
    for i, npy_path in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}: {npy_path.name}")
        
        # Load base RGB for visualization (channels=3)
        img_base = load_npy_image(npy_path, 7) # Load all to extract RGB
        img_rgb = img_base[..., :3].astype(np.uint8)
        h, w = img_rgb.shape[:2]
        
        # Get GT
        gt_boxes, gt_classes = get_ground_truth(npy_path, w, h)
        img_gt = draw_boxes(img_rgb, gt_boxes, gt_classes, color=(0, 255, 0), title="Ground Truth")
        
        images_row = [img_gt]
        
        # Run inference for each model
        for name in ['3ch', '4ch', '5ch', '6ch']:
            if name not in loaded_models:
                # Placeholder if model missing
                images_row.append(np.zeros_like(img_rgb))
                continue
                
            model = loaded_models[name]
            channels = MODELS[name]['channels']
            
            # Load input with correct channels
            img_input = load_npy_image(npy_path, channels)
            
            # Inference
            results = model(img_input, verbose=False)
            
            pred_boxes = []
            pred_classes = []
            for r in results:
                for box in r.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    if conf > 0.25:
                        pred_boxes.append(coords)
                        pred_classes.append(cls)
            
            # Draw predictions on RGB image
            img_pred = draw_boxes(img_rgb, pred_boxes, pred_classes, color=(255, 0, 0), title=f"{name} Pred")
            images_row.append(img_pred)
            
        # Concatenate row
        combined = np.hstack(images_row)
        
        # Save
        save_path = OUTPUT_DIR / f"verify_{npy_path.stem}.png"
        # Convert RGB to BGR for imwrite
        cv2.imwrite(str(save_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
