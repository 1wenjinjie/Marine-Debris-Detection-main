import cv2
import numpy as np
import os

run_dir = r"d:\OceanEdge_Group\Marine-Debris-Detection-main\runs_marida\yolov11n_6ch_gpu"

files = ["train_batch0.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]

print(f"Analyzing images in {run_dir}")

# Label info
lx, ly, lw, lh = 0.550781, 0.601562, 0.046875, 0.046875

for f in files:
    path = os.path.join(run_dir, f)
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is None:
            print(f"{f}: Failed to load")
            continue
        
        print(f"\nFile: {f}")
        print(f"Shape: {img.shape}")
        print(f"Min: {img.min()}, Max: {img.max()}")
        print(f"Mean: {img.mean():.2f}, Std: {img.std():.2f}")
        
        # Check if it's mostly black
        if img.mean() < 5:
            print("WARNING: Image is very dark/black!")
        
        # Check for unique colors (to detect labels)
        unique_colors = np.unique(img.reshape(-1, 3), axis=0)
        print(f"Unique colors count: {len(unique_colors)}")

        # Analyze the bounding box area in train_batch0.jpg
        if f == "train_batch0.jpg":
            h, w, _ = img.shape
            # Convert normalized xywh to pixel xyxy
            x_center = lx * w
            y_center = ly * h
            width = lw * w
            height = lh * h
            
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Clip to image
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            print(f"Checking Box: [{x1}, {y1}, {x2}, {y2}]")
            
            if x2 > x1 and y2 > y1:
                roi = img[y1:y2, x1:x2]
                print(f"ROI Mean: {roi.mean():.2f}, ROI Std: {roi.std():.2f}")
                
                # Compare with whole image
                print(f"Global Mean: {img.mean():.2f}")
                
                if roi.mean() > img.mean() * 1.2:
                    print("SUCCESS: ROI is significantly brighter than background.")
                else:
                    print("WARNING: ROI is not distinct from background.")
            else:
                print("Invalid box dimensions")

    else:
        print(f"{f}: Not found")
