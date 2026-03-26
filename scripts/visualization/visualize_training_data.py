import torch
import cv2
import numpy as np
import os
from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset
from pathlib import Path
import yaml

def visualize_training_batch():
    # Load config to get path
    with open('marida_enriched_rgb.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_path = config['train']
    
    # Class names
    names = {0: 'Marine_Debris', 1: 'Dense_Sargassum', 2: 'Sparse_Sargassum', 
             3: 'Natural_Wood', 4: 'Ship', 5: 'Wake', 6: 'Cloud', 7: 'Cloud_Shadow', 
             8: 'Water', 9: 'Wavetop', 10: 'Land'}

    # Initialize dataset with the EXACT same parameters as training
    dataset = MaskRescueNPYDataset(
        img_path=train_path,
        imgsz=256,
        batch_size=16,
        augment=False, # Turn off augmentation to see raw data first
        hyp=None,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix='',
        channels=3, # RGB mode
        data={'names': names, 'nc': len(names)} # Pass data dict to satisfy YOLODataset
    )
    
    print(f"Dataset loaded. Length: {len(dataset)}")
    
    output_dir = 'debug_visualizations_rgb'
    os.makedirs(output_dir, exist_ok=True)

    # Visualize first 20 images
    for i in range(min(20, len(dataset))):
        # Get item
        batch = dataset[i]
        
        # Image is normalized 0-1 (or 0-255 depending on transforms) and tensor
        # The dataset __getitem__ returns:
        # 'img': tensor, 'cls': tensor, 'bboxes': tensor, ...
        
        img_tensor = batch['img'] # (3, 256, 256)
        
        # Convert back to numpy uint8 for display
        # The dataset class does: arr = arr.astype(np.float32) -> transforms -> ToTensor
        # And we know we applied * 1500.0 and clipped to 255 in the dataset class BEFORE transforms
        
        # If it's already a tensor, it's likely 0-1 float if ToTensor scaled it, 
        # or 0-255 byte if not. Let's check range.
        
        img_np = img_tensor.numpy().transpose(1, 2, 0) # (256, 256, 3)
        
        print(f"Image {i} range: min={img_np.min():.4f}, max={img_np.max():.4f}")
        
        # If range is 0-1, scale to 255
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
            
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Draw boxes
        # bboxes are in normalized xywh or xyxy? 
        # Dataset returns 'bboxes' in normalized xywh usually if it's YOLO format
        # But MaskRescueNPYDataset might return them differently.
        # Let's check the 'instances' key if available, or 'bboxes'
        
        # The dataset returns a dict. Let's look at 'bboxes' and 'cls'
        # bboxes are usually normalized xywh
        
        bboxes = batch['bboxes']
        cls = batch['cls']
        
        h, w = img_bgr.shape[:2]
        
        if len(bboxes) > 0:
            for box, c in zip(bboxes, cls):
                # box is normalized xywh
                cx, cy, bw, bh = box
                
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                class_id = int(c.item())
                class_name = names.get(class_id, str(class_id))
                
                # Color based on class
                color = (0, 255, 0) # Green for debris
                if class_id != 0:
                    color = (0, 0, 255) # Red for others
                
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 1)
                cv2.putText(img_bgr, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Print box stats
                print(f"  Box: Class={class_name}, Size={x2-x1}x{y2-y1} pixels")
        
        cv2.imwrite(f"{output_dir}/train_sample_{i}.png", img_bgr)
        
    print(f"Saved visualizations to {output_dir}")

if __name__ == "__main__":
    visualize_training_batch()
