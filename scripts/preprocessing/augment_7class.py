import os
import numpy as np
import rasterio
import albumentations as A
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil

# Configuration
INPUT_DIR = Path("d:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split")
OUTPUT_DIR = Path("d:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_augmented")
TARGET_CLASSES = {
    6: 40,  # Wakes: 40x augmentation
    2: 80,  # Sparse Sargassum: 80x augmentation
}
DEFAULT_AUGMENTATION = 1 # Others: 1x augmentation (plus original)

def get_augmentation_pipeline():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.Perspective(p=0.1),
        A.Affine(shear=(-10, 10), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def sanitize_bbox(bbox):
    cx, cy, w, h = bbox
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    
    w_new = x2 - x1
    h_new = y2 - y1
    cx_new = x1 + w_new/2
    cy_new = y1 + h_new/2
    
    if w_new <= 0 or h_new <= 0:
        return None
        
    return [cx_new, cy_new, w_new, h_new]

def load_image(path):
    with rasterio.open(path) as src:
        img = src.read() # (C, H, W)
        img = np.transpose(img, (1, 2, 0)) # (H, W, C)
        return img, src.profile

def save_image(path, img, profile):
    # img is (H, W, C)
    # rasterio expects (C, H, W)
    img_save = np.transpose(img, (2, 0, 1))
    
    # Update profile
    profile.update({
        'height': img.shape[0],
        'width': img.shape[1],
        'count': img.shape[2],
        'driver': 'GTiff'
    })
    
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(img_save)

def load_labels(path):
    bboxes = []
    class_labels = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                bbox = list(map(float, parts[1:]))
                bbox = sanitize_bbox(bbox)
                if bbox:
                    class_labels.append(cls_id)
                    bboxes.append(bbox)
    return bboxes, class_labels

def save_labels(path, bboxes, class_labels):
    with open(path, 'w') as f:
        for cls, bbox in zip(class_labels, bboxes):
            f.write(f"{int(cls)} {' '.join(map(str, bbox))}\n")

def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    
    # Copy val and test as is
    for split in ['val', 'test']:
        src = INPUT_DIR / split
        dst = OUTPUT_DIR / split
        if src.exists():
            shutil.copytree(src, dst)
            print(f"Copied {split} set.")

    # Process train
    train_img_dir = INPUT_DIR / 'train' / 'images'
    train_lbl_dir = INPUT_DIR / 'train' / 'labels'
    
    out_img_dir = OUTPUT_DIR / 'train' / 'images'
    out_lbl_dir = OUTPUT_DIR / 'train' / 'labels'
    out_img_dir.mkdir(parents=True)
    out_lbl_dir.mkdir(parents=True)
    
    aug = get_augmentation_pipeline()
    
    image_files = list(train_img_dir.glob("*.tif"))
    print(f"Found {len(image_files)} training images.")
    
    for img_path in tqdm(image_files):
        lbl_path = train_lbl_dir / img_path.with_suffix('.txt').name
        
        # Load data
        try:
            image, profile = load_image(img_path)
            bboxes, class_labels = load_labels(lbl_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
            
        # Determine augmentation factor
        factor = DEFAULT_AUGMENTATION
        for cls in class_labels:
            if cls in TARGET_CLASSES:
                factor = max(factor, TARGET_CLASSES[cls])
        
        # Save original
        save_image(out_img_dir / img_path.name, image, profile)
        if lbl_path.exists():
            shutil.copy(lbl_path, out_lbl_dir / lbl_path.name)
            
        # Generate augmentations
        if len(bboxes) > 0: # Only augment if there are objects
            for i in range(factor):
                try:
                    transformed = aug(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_img = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_cls = transformed['class_labels']
                    
                    # Sanitize bounding boxes
                    aug_bboxes = [sanitize_bbox(bbox) for bbox in aug_bboxes]
                    aug_bboxes = [bbox for bbox in aug_bboxes if bbox is not None] # Remove any None bboxes
                    
                    if len(aug_bboxes) > 0:
                        save_name = f"{img_path.stem}_aug_{i}.tif"
                        save_image(out_img_dir / save_name, aug_img, profile)
                        save_labels(out_lbl_dir / f"{img_path.stem}_aug_{i}.txt", aug_bboxes, aug_cls)
                except Exception as e:
                    print(f"Augmentation error on {img_path.name} iter {i}: {e}")
                    pass

    # Copy classes.txt
    if (INPUT_DIR / 'classes.txt').exists():
        shutil.copy(INPUT_DIR / 'classes.txt', OUTPUT_DIR / 'classes.txt')
        
    print("Augmentation complete.")

if __name__ == "__main__":
    main()
