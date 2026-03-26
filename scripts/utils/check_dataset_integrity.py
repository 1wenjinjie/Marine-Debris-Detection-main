import os
import yaml
from pathlib import Path

def check_dataset_integrity(dataset_root):
    root = Path(dataset_root)
    print(f"Checking dataset integrity for: {root}")
    
    if not root.exists():
        print(f"Error: Dataset root {root} does not exist.")
        return

    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"Warning: Split directory {split_dir} does not exist.")
            continue
            
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists():
            print(f"Error: Images directory {images_dir} missing.")
            continue
            
        if not labels_dir.exists():
            print(f"Error: Labels directory {labels_dir} missing.")
            continue
            
        images = list(images_dir.glob('*.npy'))
        labels = list(labels_dir.glob('*.txt'))
        
        print(f"Split '{split}': Found {len(images)} images and {len(labels)} labels.")
        
        # Check for orphan images (images without labels)
        # Note: Some datasets might have background-only images with empty label files, or no label files.
        # But usually YOLO expects a label file even if empty, or implies no objects if missing (depending on config).
        # Let's check for matching stems.
        
        img_stems = set(img.stem for img in images)
        lbl_stems = set(lbl.stem for lbl in labels)
        
        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems
        
        if missing_labels:
            print(f"  Warning: {len(missing_labels)} images have no corresponding label file.")
            # print(f"  Examples: {list(missing_labels)[:5]}")
            
        if missing_images:
            print(f"  Error: {len(missing_images)} label files have no corresponding image file.")
            
    # Check classes.txt
    classes_file = root / 'classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Found classes.txt with {len(classes)} classes: {classes}")
    else:
        print("Warning: classes.txt not found.")

if __name__ == "__main__":
    # Check the dataset used in overfit (mask_bbox_rescue_split)
    check_dataset_integrity("D:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split")
