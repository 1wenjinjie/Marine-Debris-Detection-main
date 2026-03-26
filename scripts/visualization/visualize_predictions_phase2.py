import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt
import rasterio

def visualize_predictions(model_path, val_images_dir, val_labels_dir, output_dir, target_classes=[2, 6]):
    """
    Visualizes predictions for specific classes.
    target_classes: List of class IDs to focus on (e.g., [2, 6] for Sparse Sargassum and Wakes).
    """
    # Load model
    model = YOLO(model_path)
    
    # Class names from the yaml (hardcoded for now based on data_npy_6ch.yaml)
    class_names = [
        "Marine Debris",      # 0
        "Dense Sargassum",    # 1
        "Sparse Sargassum",   # 2
        "Natural Organic",    # 3
        "Ship",               # 4
        "Foam",               # 5
        "Wakes"               # 6
    ]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all validation images
    # We need to read val.txt to know which images are in validation set, 
    # but since we don't have easy access to the absolute paths in val.txt right now (it might be relative),
    # we'll try to find them in the directory provided.
    # Actually, the user provided `MARIDA_YOLO_6band_split/val/images` in the prompt context, 
    # but the yaml points to `val.txt`. 
    # Let's assume the images are in `MARIDA_YOLO_6band_split/val/images` for now, or read `val.txt`.
    
    # Force using the directory search for .tif files
    val_txt_path = "d:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split/val.txt" 
    image_paths = []
    if False: # os.path.exists(val_txt_path):
        with open(val_txt_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
    else:
        # Fallback to searching in a likely directory
        print(f"Warning: {val_txt_path} not found or disabled. Searching in mask_bbox_rescue_split/val/images")
        search_dir = "d:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split/val/images"
        image_paths = glob.glob(os.path.join(search_dir, "*.tif"))

    print(f"Found {len(image_paths)} validation images.")
    if len(image_paths) > 0:
        print(f"First image path: {image_paths[0]}")

    count = 0
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
            
        # Check if this image contains target classes in Ground Truth
        # We need to find the corresponding label file.
        # Assuming standard YOLO structure: images/ -> labels/
        label_path = img_path.replace("images", "labels").replace(".tif", ".txt")
        
        has_target_class = False
        gt_boxes = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    gt_boxes.append(parts)
                    if cls_id in target_classes:
                        has_target_class = True
        
        if not has_target_class:
            if "S2_11-1-19_19QDA_0" in img_path:
                print(f"DEBUG: Found S2_11-1-19_19QDA_0 but has_target_class is False.")
                print(f"DEBUG: Label path: {label_path}")
                print(f"DEBUG: Exists: {os.path.exists(label_path)}")
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        print(f"DEBUG: Content: {f.read()}")
            continue

        print(f"Processing {os.path.basename(img_path)}...")

        # Read image using rasterio
        try:
            with rasterio.open(img_path) as src:
                img = src.read() # (C, H, W)
                img = np.transpose(img, (1, 2, 0)) # (H, W, C)
                print(f"DEBUG: {os.path.basename(img_path)} dtype={img.dtype}, min={img.min()}, max={img.max()}")
        except Exception as e:
            print(f"Failed to read {img_path}: {e}")
            continue
            
        # Handle 6 channels: take RGB (bands 1,2,3 usually, or 3,2,1 depending on order)
        # Assuming RGB are the first 3 channels for visualization
        if len(img.shape) == 3 and img.shape[2] >= 3:
            img_vis = img[:, :, :3].copy()
            # Normalize for display if needed (if 16-bit or float)
            if img_vis.dtype == np.uint16:
                img_vis = (img_vis / 256).astype(np.uint8)
            elif img_vis.dtype == np.float32:
                img_vis = (img_vis * 255).astype(np.uint8)
        else:
            img_vis = img.copy()
            
        # Ensure img_vis is contiguous for cv2
        img_vis = np.ascontiguousarray(img_vis)


        h, w = img_vis.shape[:2]

        # Run Inference
        # Pass the numpy array directly to avoid OpenCV loading issues in Ultralytics
        # img is (H, W, 6)
        # The model expects 0-255 input (which it will normalize to 0-1 internally)
        # Our images are 0-1 float32, so we must multiply by 255
        img_input = img * 255.0
        results = model(img_input, verbose=False)
        
        # Draw Ground Truth (Green)
        for parts in gt_boxes:
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            color = (0, 255, 0) # Green
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            label = f"GT: {class_names[cls_id]}"
            cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw Predictions (Red)
        pred_summary = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Only draw if confidence is reasonable, e.g., > 0.1 to see weak detections
                if conf > 0.1:
                    color = (0, 0, 255) # Red
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                    label = f"Pred: {class_names[cls_id]} {conf:.2f}"
                    cv2.putText(img_vis, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    pred_summary.append(f"{class_names[cls_id]}:{conf:.2f}")
        
        print(f"  GT: {[class_names[int(p[0])] for p in gt_boxes]}")
        print(f"  Pred: {pred_summary}")

        # Save
        out_file = os.path.join(output_dir, f"vis_{os.path.basename(img_path)}.png")
        cv2.imwrite(out_file, img_vis)
        count += 1
        
    print(f"Saved {count} visualizations to {output_dir}")

if __name__ == "__main__":
    model_path = "runs_marida/yolov11n_6ch_rebalance_v2_50e/weights/best.pt"
    # We need to locate the validation images. 
    # Based on previous context, they might be in MARIDA_YOLO_6band_split/val/images
    # But let's try to use the paths from the yaml if possible.
    # For now, I'll point to the directory I know exists.
    val_images_dir = "d:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split/val/images"
    val_labels_dir = "d:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split/val/labels"
    output_dir = "visualizations_phase2_v2"
    
    visualize_predictions(model_path, val_images_dir, val_labels_dir, output_dir, target_classes=[2, 6])
