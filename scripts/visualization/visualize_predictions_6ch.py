import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import random

# Configuration
MODEL_PATH = 'runs_marida/yolov11n_6ch_gpu_spectral_aug/weights/best.pt'
DATA_YAML = 'data_npy_6ch.yaml'
VAL_LIST = 'D:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split/val.txt'
OUTPUT_DIR = 'visualizations_6ch_spectral_aug'
NUM_SAMPLES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Class Names
with open(DATA_YAML, 'r') as f:
    data_config = yaml.safe_load(f)
class_names = data_config['names']

# Load Model
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Load Validation List
DATASET_ROOT = 'D:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split'
with open(VAL_LIST, 'r') as f:
    val_files = [os.path.join(DATASET_ROOT, line.strip()) for line in f.readlines()]

# Select Random Samples
samples = random.sample(val_files, min(NUM_SAMPLES, len(val_files)))

def load_and_preprocess(path, channels=6):
    """Replicates the logic from MaskRescueNPYDataset"""
    arr = np.load(path, allow_pickle=False)
    
    # Transpose if needed (C, H, W) -> (H, W, C)
    if arr.shape[0] in (6, 7) and arr.shape[0] <= arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)
    
    # Slice channels
    if arr.shape[-1] > channels:
        arr = arr[..., :channels]
    
    # Min-Max Normalization
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    
    # Scale to 0-255
    arr = arr * 255.0
    
    # Handle NaNs
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    np.clip(arr, 0.0, 255.0, out=arr)
    
    return arr.astype(np.float32)

def get_ground_truth(npy_path, width, height):
    label_path = npy_path.replace('images', 'labels').replace('.npy', '.txt')
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

def draw_boxes(img, boxes, classes, color=(0, 255, 0), label_prefix=""):
    img_copy = img.copy()
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        label = f"{label_prefix}{class_names[int(cls)]}"
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy

print(f"Processing {len(samples)} samples...")

for i, npy_path in enumerate(samples):
    print(f"Processing {npy_path}...")
    
    # Load Image
    img_6ch = load_and_preprocess(npy_path)
    h, w = img_6ch.shape[:2]
    
    # Run Inference
    # Model expects list of numpy arrays or single array. 
    # Since we scaled to 0-255, we can pass it directly.
    # Note: YOLOv8/11 usually expects (H,W,C) BGR if passing numpy.
    # Our data is RGB+... so we should probably pass it as is.
    # However, the model was trained on this data, so it expects this channel order.
    
    results = model(img_6ch, verbose=False)
    
    # Extract Predictions
    pred_boxes = []
    pred_classes = []
    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            if conf > 0.25: # Confidence threshold
                pred_boxes.append(coords)
                pred_classes.append(cls)
    
    # Get Ground Truth
    gt_boxes, gt_classes = get_ground_truth(npy_path, w, h)
    
    # Prepare RGB for visualization (Channels 0,1,2)
    # Convert to uint8 for display
    img_rgb = img_6ch[..., :3].astype(np.uint8)
    # Convert RGB to BGR for OpenCV if we were using cv2.imshow, but for matplotlib RGB is fine.
    # Actually, let's use OpenCV drawing which expects BGR, then convert back to RGB for saving/plotting if needed.
    # Or just stick to RGB and pass correct color tuples.
    
    # Draw Ground Truth (Green)
    img_gt = draw_boxes(img_rgb, gt_boxes, gt_classes, color=(0, 255, 0), label_prefix="GT: ")
    
    # Draw Predictions (Red)
    img_pred = draw_boxes(img_rgb, pred_boxes, pred_classes, color=(255, 0, 0), label_prefix="Pred: ")
    
    # Combine side by side
    combined = np.hstack((img_gt, img_pred))
    
    # Save
    filename = os.path.basename(npy_path).replace('.npy', '.png')
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"Saved {save_path}")

print("Visualization complete.")
