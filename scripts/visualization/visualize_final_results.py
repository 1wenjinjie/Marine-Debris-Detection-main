import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import yaml

# Configuration
MODEL_PATH = r"runs_marida/yolov11n_6ch_7class_best_100e/weights/best.pt"
DATA_YAML = "data_npy_6ch.yaml"
RESULTS_CSV = r"runs_marida/yolov11n_6ch_7class_best_100e/results.csv"
OUTPUT_DIR = "final_visualizations"
NUM_SAMPLES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Class Names
with open(DATA_YAML, 'r') as f:
    data_config = yaml.safe_load(f)
CLASS_NAMES = data_config['names']
DATA_ROOT = Path(data_config['path'])

def load_npy_image(path, channels=6):
    """Loads and preprocesses .npy image similar to training pipeline."""
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

    if arr.ndim != 3:
        # Try to fix dimensions if possible or raise error
        if arr.ndim == 2:
             arr = arr[..., None]
        else:
             print(f"Skipping {path}: Array must be 3-D, got {arr.shape}")
             return None

    if arr.shape[0] in (6, 7) and arr.shape[0] <= arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)
    
    # Slice channels
    if arr.shape[-1] > channels:
        arr = arr[..., :channels]
    
    # Min-Max Normalization to 0-1
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    
    # Scale to 0-255 for Ultralytics
    arr = arr * 255.0
    
    # Handle NaNs
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    
    return arr.astype(np.float32) # Keep as float32 as per training

def get_ground_truth(npy_path):
    """Reads ground truth labels for a given npy file."""
    # Construct label path: replace 'images' with 'labels' and .npy with .txt
    # Assuming structure: .../images/name.npy -> .../labels/name.txt
    # Or based on dataset_root structure in data_npy_6ch.yaml
    
    # The data_npy_6ch.yaml points to D:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split
    # The file paths in val.txt are likely relative or absolute.
    
    # Heuristic to find label file
    path_obj = Path(npy_path)
    
    # Try standard YOLO structure
    # parent/images/file.npy -> parent/labels/file.txt
    if path_obj.parent.name == 'images':
        label_path = path_obj.parent.parent / 'labels' / path_obj.with_suffix('.txt').name
    else:
        # Try adjacent folder
        label_path = path_obj.parent / 'labels' / path_obj.with_suffix('.txt').name
        if not label_path.exists():
             # Try same folder
             label_path = path_obj.with_suffix('.txt')

    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    boxes.append((cls, x, y, w, h))
    return boxes

def draw_boxes(img_rgb, boxes, color=(0, 255, 0), label_prefix=""):
    """Draws boxes on the image."""
    h_img, w_img = img_rgb.shape[:2]
    img_out = img_rgb.copy()
    
    for cls, x, y, w, h in boxes:
        x1 = int((x - w / 2) * w_img)
        y1 = int((y - h / 2) * h_img)
        x2 = int((x + w / 2) * w_img)
        y2 = int((y + h / 2) * h_img)
        
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
        label = f"{label_prefix}{CLASS_NAMES[int(cls)]}"
        cv2.putText(img_out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_out

def plot_metrics():
    """Plots training metrics from results.csv."""
    if not os.path.exists(RESULTS_CSV):
        print("results.csv not found.")
        return

    df = pd.read_csv(RESULTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
    plt.title('Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot mAP
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
    plt.title('Mean Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_metrics.png'))
    print(f"Saved metrics plot to {os.path.join(OUTPUT_DIR, 'training_metrics.png')}")

def main():
    # 1. Plot Metrics
    plot_metrics()
    
    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # 3. Get Validation Files
    val_txt_path = os.path.join(DATA_ROOT, "val.txt")
    if not os.path.exists(val_txt_path):
        # Fallback if DATA_ROOT is not absolute or correct relative to CWD
        val_txt_path = "mask_bbox_rescue_split/val.txt" # Try relative
        
    if not os.path.exists(val_txt_path):
        print(f"Could not find val.txt at {val_txt_path}")
        return

    with open(val_txt_path, 'r') as f:
        val_files = [line.strip() for line in f.readlines()]
    
    # Filter for files that exist
    valid_files = []
    for f in val_files:
        p = Path(f)
        if not p.is_absolute():
             p = DATA_ROOT / p
        if p.exists():
            valid_files.append(str(p))
    
    print(f"Found {len(valid_files)} validation files.")
    
    # 4. Select Samples (Prioritize those with Marine Debris - Class 0)
    selected_files = []
    debris_files = []
    other_files = []
    
    for f in valid_files:
        gt = get_ground_truth(f)
        has_debris = any(cls == 0 for cls, _, _, _, _ in gt)
        if has_debris:
            debris_files.append(f)
        else:
            other_files.append(f)
            
    # Select mix
    import random
    random.shuffle(debris_files)
    random.shuffle(other_files)
    
    selected_files.extend(debris_files[:min(len(debris_files), NUM_SAMPLES // 2)])
    selected_files.extend(other_files[:NUM_SAMPLES - len(selected_files)])
    
    print(f"Visualizing {len(selected_files)} samples...")
    
    for i, file_path in enumerate(selected_files):
        print(f"Processing {i+1}/{len(selected_files)}: {os.path.basename(file_path)}")
        
        # Load Image
        img_array = load_npy_image(file_path)
        if img_array is None:
            continue
            
        # Run Prediction
        # Note: Ultralytics predict() handles numpy arrays. 
        # It expects HWC, BGR (if 3 channel). Our data is 6 channel.
        # The model was trained on 6 channels.
        # We pass the 6-channel array directly.
        results = model.predict(img_array, verbose=False)
        
        # Prepare Visualization Image (RGB only)
        # Normalize to 0-255 uint8 for display
        img_disp = img_array[..., :3].copy()
        # img_array is 0-255 float.
        img_disp = img_disp.astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)
        
        # Draw Ground Truth
        gt_boxes = get_ground_truth(file_path)
        img_gt = draw_boxes(img_disp.copy(), gt_boxes, color=(0, 255, 0), label_prefix="GT: ")
        
        # Draw Predictions
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # box.xywhn gives normalized xywh
                x, y, w, h = box.xywhn[0].tolist()
                if conf > 0.25: # Confidence threshold
                    pred_boxes.append((cls, x, y, w, h))
        
        img_pred = draw_boxes(img_disp.copy(), pred_boxes, color=(0, 0, 255), label_prefix="Pred: ")
        
        # Combine side-by-side
        combined = np.hstack((img_gt, img_pred))
        
        # Save
        out_name = f"sample_{i}_{os.path.basename(file_path).replace('.npy', '.png')}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), combined)

    print(f"Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
