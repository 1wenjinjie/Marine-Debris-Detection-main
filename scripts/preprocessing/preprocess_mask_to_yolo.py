"""
Convert MARIDA segmentation masks to YOLO bounding box format.
Extracts connected components for each class and generates bbox annotations.
"""

import os
import numpy as np
from pathlib import Path
import rasterio
from scipy import ndimage
from tqdm import tqdm
import json

# Original class mapping from labels_mapping.txt
ORIGINAL_CLASS_NAMES = {
    1: "Marine Debris",
    2: "Dense Sargassum",
    3: "Sparse Sargassum",
    4: "Natural Organic Material",
    5: "Ship",
    6: "Clouds",
    7: "Marine Water",
    8: "Sediment-Laden Water",
    9: "Foam",
    10: "Turbid Water",
    11: "Shallow Water",
    12: "Waves",
    13: "Cloud Shadows",
    14: "Wakes",
    15: "Mixed Water"
}

# ================================================================================
# CLASS STRATEGY: 7-class fine-grained detection (for research/analysis)
# ================================================================================
# Rationale:
# - Separates Wakes (Class 6) from Ships (Class 4) and Debris (Class 0)
# - Allows the model to learn specific features for Wakes
# - Reduces confusion between Wakes and Marine Debris

# Classes of interest for marine debris detection
MARINE_DEBRIS_CLASSES = [1, 2, 3, 4, 5, 9, 14]

# Mapping from original class ID to YOLO class ID (0-indexed, continuous)
ORIGINAL_TO_YOLO_MAP = {
    1: 0,   # Marine Debris
    2: 1,   # Dense Sargassum
    3: 2,   # Sparse Sargassum
    4: 3,   # Natural Organic Material
    5: 4,   # Ship
    9: 5,   # Foam
    14: 6   # Wakes
}

YOLO_CLASS_NAMES = [
    'Marine_Debris',
    'Dense_Sargassum',
    'Sparse_Sargassum',
    'Natural_Organic',
    'Ship',
    'Foam',
    'Wake'
]

# ================================================================================
# PREVIOUS STRATEGY: 3-class detection (Commented out)
# ================================================================================
"""
# Mapping from original class ID to YOLO class ID (0-indexed, continuous)
# CRITICAL: YOLO requires continuous class IDs starting from 0!
ORIGINAL_TO_YOLO_MAP = {
    1: 0,   # Marine Debris -> Marine_Debris
    2: 1,   # Dense Sargassum -> Floating_Object
    3: 1,   # Sparse Sargassum -> Floating_Object
    4: 1,   # Natural Organic Material -> Floating_Object
    5: 2,   # Ship -> Ship_Wake
    9: 1,   # Foam -> Floating_Object
    14: 2   # Wakes -> Ship_Wake
}

# YOLO class names (must be continuous: 0, 1, 2)
YOLO_CLASS_NAMES = [
    'Marine_Debris',      # Class 0: Plastic debris, trash
    'Floating_Object',    # Class 1: Sargassum, organic matter, foam
    'Ship_Wake'          # Class 2: Ships and their wakes
]
"""

def mask_to_bboxes(mask, class_id, min_area=10, bbox_padding=5):
    """
    Convert binary mask to bounding boxes using connected components.
    
    Args:
        mask: Binary mask for a single class
        class_id: YOLO class ID (0-indexed, continuous)
        min_area: Minimum area (pixels) for valid objects
        bbox_padding: Pixels to add around bbox to avoid cutting edges (default: 5)
        
    Returns:
        List of bboxes in YOLO format [class_id, x_center, y_center, width, height]
        
    Rationale for bbox_padding:
        - Connected component extraction gives tight bounding boxes
        - Adding padding ensures we don't cut object edges
        - Important for small objects where every pixel matters
        - Typical value: 5-10 pixels for 512x512 images
    """
    # Label connected components
    labeled_mask, num_features = ndimage.label(mask)
    
    height, width = mask.shape
    bboxes = []
    
    for i in range(1, num_features + 1):
        # Get component
        component = (labeled_mask == i)
        
        # Filter by area (remove noise)
        area = np.sum(component)
        if area < min_area:
            continue
        
        # Find bounding box
        rows, cols = np.where(component)
        if len(rows) == 0:
            continue
            
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        
        # Add padding while staying within image bounds
        # Rationale: Prevents cutting object edges, improves detection
        x_min = max(0, x_min - bbox_padding)
        y_min = max(0, y_min - bbox_padding)
        x_max = min(width - 1, x_max + bbox_padding)
        y_max = min(height - 1, y_max + bbox_padding)
        
        # Calculate bbox dimensions
        bbox_width_px = x_max - x_min + 1
        bbox_height_px = y_max - y_min + 1
        
        # Filter extremely thin bboxes (likely artifacts)
        # Rationale: bbox with width or height < 3 pixels are usually noise
        if bbox_width_px < 3 or bbox_height_px < 3:
            continue
        
        # Convert to YOLO format (normalized coordinates [0, 1])
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        bbox_width = bbox_width_px / width
        bbox_height = bbox_height_px / height
        
        # Sanity check: ensure coordinates are valid
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                0 < bbox_width <= 1 and 0 < bbox_height <= 1):
            print(f"Warning: Invalid bbox for class {class_id}: "
                  f"center=({x_center:.3f}, {y_center:.3f}), "
                  f"size=({bbox_width:.3f}, {bbox_height:.3f})")
            continue
        
        bboxes.append([class_id, x_center, y_center, bbox_width, bbox_height])
    
    return bboxes

def convert_mask_to_yolo(mask_path, output_txt_path, classes_of_interest=None, 
                         min_area=None, bbox_padding=5):
    """
    Convert segmentation mask to YOLO format annotations.
    
    Args:
        mask_path: Path to label mask .tif file
        output_txt_path: Path to output .txt file
        classes_of_interest: List of class IDs to process (None = all)
        min_area: Minimum area for valid objects (None = auto-calculate)
        bbox_padding: Padding pixels around bboxes
        
    Rationale for dynamic min_area:
        - 256x256 images: min_area = 10 pixels (0.015% of image)
        - 512x512 images: min_area = 40 pixels (0.015% of image)
        - Scales with image size to maintain consistent filtering
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.int32)
        height, width = mask.shape
    
    # Auto-calculate minimum area based on image size
    if min_area is None:
        # Set to 0.015% of image area (empirically good for filtering noise)
        min_area = max(10, int(0.00015 * height * width))
    
    all_bboxes = []
    class_bbox_counts = {}  # Track bboxes per class for statistics
    
    # Get unique classes in mask
    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes > 0]  # Exclude background (0)
    
    if classes_of_interest is not None:
        unique_classes = [c for c in unique_classes if c in classes_of_interest]
    
    # Process each class
    for class_id in unique_classes:
        # Map original class ID to YOLO class ID
        yolo_class_id = ORIGINAL_TO_YOLO_MAP.get(class_id)
        
        if yolo_class_id is None:
            # Skip classes not in our mapping (background classes)
            continue
        
        class_mask = (mask == class_id)
        bboxes = mask_to_bboxes(class_mask, yolo_class_id, min_area, bbox_padding)
        all_bboxes.extend(bboxes)
        
        # Track statistics
        if yolo_class_id not in class_bbox_counts:
            class_bbox_counts[yolo_class_id] = 0
        class_bbox_counts[yolo_class_id] += len(bboxes)
    
    # Write to file
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, 'w') as f:
        for bbox in all_bboxes:
            f.write(' '.join(map(str, bbox)) + '\n')
    
    return len(all_bboxes), class_bbox_counts

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert MARIDA masks to YOLO format.')
    parser.add_argument('--input-dir', type=str, help='Input patches directory')
    parser.add_argument('--output-dir', type=str, help='Output base directory')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path('D:/OceanEdge_Group/Marine-Debris-Detection-main')
    
    if args.input_dir:
        input_patches = Path(args.input_dir)
    else:
        # Find latest 6-band processed data
        import glob
        pattern = str(base_dir / 'MARIDA_6band*')
        matching_dirs = glob.glob(pattern)
        if matching_dirs:
            input_base = Path(max(matching_dirs, key=lambda x: os.path.getmtime(x)))
            print(f"Using input: {input_base}")
        else:
            input_base = base_dir / 'MARIDA_6band_256'
            if not input_base.exists():
                input_base = base_dir / 'MARIDA'
        input_patches = input_base / 'patches'
    
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = base_dir / 'MARIDA_YOLO'
    
    # Get all scene directories
    scene_dirs = [d for d in input_patches.iterdir() if d.is_dir()]

    print(f"Found {len(scene_dirs)} scenes to process")
    print(f"Converting masks to YOLO bbox format (per patch)")
    print(f"\nClass Strategy: {len(YOLO_CLASS_NAMES)}-class detection")
    for i, name in enumerate(YOLO_CLASS_NAMES):
        print(f"  {i}: {name}")

    stats = {
        'total_patches': 0,
        'patches_with_objects': 0,
        'total_objects': 0,
        'class_distribution': {i: 0 for i in range(len(YOLO_CLASS_NAMES))}
    }

    # Process each scene per patch label *_cl.tif
    for scene_dir in tqdm(scene_dirs, desc="Converting masks"):
        # 遍历每个patch标签
        label_files = sorted(scene_dir.glob('*_cl.tif'))
        for label_file in label_files:
            patch_base = label_file.stem.replace('_cl', '')  # e.g., S2_..._0
            output_txt = output_base / 'labels' / f"{patch_base}.txt"
            num_objects, class_counts = convert_mask_to_yolo(
                label_file,
                output_txt,
                classes_of_interest=MARINE_DEBRIS_CLASSES,
                min_area=None,
                bbox_padding=5
            )
            stats['total_patches'] += 1
            if num_objects > 0:
                stats['patches_with_objects'] += 1
                stats['total_objects'] += num_objects
                for class_id, count in class_counts.items():
                    stats['class_distribution'][class_id] += count

            # 关联图像，优先 *_6band.tif，其次原始 *.tif
            img_candidates = [
                scene_dir / f"{patch_base}_6band.tif",
                scene_dir / f"{patch_base}.tif",
            ]
            img_file = next((p for p in img_candidates if p.exists()), None)
            if img_file is not None:
                output_img_dir = output_base / 'images'
                output_img_dir.mkdir(parents=True, exist_ok=True)
                output_img = output_img_dir / f"{patch_base}.tif"
                if not output_img.exists():
                    try:
                        os.symlink(img_file.resolve(), output_img)
                    except Exception:
                        import shutil
                        shutil.copy2(img_file, output_img)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"Output saved to: {output_base}")
    print(f"\nStatistics:")
    print(f"  Total patches: {stats['total_patches']}")
    print(f"  Patches with objects: {stats['patches_with_objects']}")
    print(f"  Total objects detected: {stats['total_objects']}")
    avg = stats['total_objects'] / max(stats['total_patches'], 1)
    print(f"  Average objects per patch: {avg:.2f}")

    print(f"\nClass Distribution:")
    for class_id, count in sorted(stats['class_distribution'].items()):
        class_name = YOLO_CLASS_NAMES[class_id]
        percentage = 100 * count / max(stats['total_objects'], 1)
        print(f"  Class {class_id} ({class_name}): {count} objects ({percentage:.2f}%)")

    # Save statistics
    stats_file = output_base / 'conversion_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Create class names file for YOLO (continuous 0, 1, 2, ...)
    class_file = output_base / 'classes.txt'
    with open(class_file, 'w') as f:
        for name in YOLO_CLASS_NAMES:
            f.write(name + '\n')

    # Save class mapping for reference
    mapping_file = output_base / 'class_mapping.json'
    mapping_info = {
        'yolo_classes': YOLO_CLASS_NAMES,
        'original_to_yolo': {str(k): v for k, v in ORIGINAL_TO_YOLO_MAP.items()},
        'original_class_names': {str(k): v for k, v in ORIGINAL_CLASS_NAMES.items()}
    }
    with open(mapping_file, 'w') as f:
        json.dump(mapping_info, f, indent=2)

if __name__ == "__main__":
    main()
