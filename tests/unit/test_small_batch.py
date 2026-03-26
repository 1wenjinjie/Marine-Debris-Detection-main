"""
Small batch test script for MARIDA preprocessing.
Tests the pipeline on 2-3 scenes to verify correctness before full processing.
"""

import os
import sys
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage
from tqdm import tqdm
import json

# Import the updated class mapping
ORIGINAL_CLASS_NAMES = {
    1: "Marine Debris",
    2: "Dense Sargassum",
    3: "Sparse Sargassum",
    4: "Natural Organic Material",
    5: "Ship",
    9: "Foam",
    14: "Wakes"
}

# 3-class strategy (recommended)
ORIGINAL_TO_YOLO_MAP = {
    1: 0,   # Marine Debris -> Marine_Debris
    2: 1,   # Dense Sargassum -> Floating_Object
    3: 1,   # Sparse Sargassum -> Floating_Object
    4: 1,   # Natural Organic Material -> Floating_Object
    5: 2,   # Ship -> Ship_Wake
    9: 1,   # Foam -> Floating_Object
    14: 2   # Wakes -> Ship_Wake
}

YOLO_CLASS_NAMES = [
    'Marine_Debris',
    'Floating_Object',
    'Ship_Wake'
]

MARINE_DEBRIS_CLASSES = [1, 2, 3, 4, 5, 9, 14]

def mask_to_bboxes(mask, class_id, min_area=10, bbox_padding=5):
    """Convert binary mask to bounding boxes."""
    labeled_mask, num_features = ndimage.label(mask)
    
    height, width = mask.shape
    bboxes = []
    
    for i in range(1, num_features + 1):
        component = (labeled_mask == i)
        area = np.sum(component)
        
        if area < min_area:
            continue
        
        rows, cols = np.where(component)
        if len(rows) == 0:
            continue
            
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        
        # Add padding
        x_min = max(0, x_min - bbox_padding)
        y_min = max(0, y_min - bbox_padding)
        x_max = min(width - 1, x_max + bbox_padding)
        y_max = min(height - 1, y_max + bbox_padding)
        
        bbox_width_px = x_max - x_min + 1
        bbox_height_px = y_max - y_min + 1
        
        # Filter thin bboxes
        if bbox_width_px < 3 or bbox_height_px < 3:
            continue
        
        # Convert to YOLO format
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        bbox_width = bbox_width_px / width
        bbox_height = bbox_height_px / height
        
        # Sanity check
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                0 < bbox_width <= 1 and 0 < bbox_height <= 1):
            continue
        
        bboxes.append([class_id, x_center, y_center, bbox_width, bbox_height])
    
    return bboxes

def convert_mask_to_yolo(mask_path, output_txt_path, min_area=None, bbox_padding=5):
    """Convert segmentation mask to YOLO format."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.int32)
        height, width = mask.shape
    
    # Auto-calculate minimum area
    if min_area is None:
        min_area = max(10, int(0.00015 * height * width))
    
    all_bboxes = []
    class_bbox_counts = {}
    
    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes > 0]
    unique_classes = [c for c in unique_classes if c in MARINE_DEBRIS_CLASSES]
    
    for class_id in unique_classes:
        yolo_class_id = ORIGINAL_TO_YOLO_MAP.get(class_id)
        
        if yolo_class_id is None:
            continue
        
        class_mask = (mask == class_id)
        bboxes = mask_to_bboxes(class_mask, yolo_class_id, min_area, bbox_padding)
        all_bboxes.extend(bboxes)
        
        if yolo_class_id not in class_bbox_counts:
            class_bbox_counts[yolo_class_id] = 0
        class_bbox_counts[yolo_class_id] += len(bboxes)
    
    # Write to file
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, 'w') as f:
        for bbox in all_bboxes:
            f.write(' '.join(map(str, bbox)) + '\n')
    
    return len(all_bboxes), class_bbox_counts

def test_small_batch(num_scenes=3):
    """Test preprocessing on a small batch of scenes."""
    base_dir = Path('/Users/wen11/Documents/OceanEdge_Group/Marine_dataset')
    input_patches = base_dir / 'MARIDA' / 'patches'
    output_base = base_dir / 'MARIDA_YOLO_test'
    
    # Get scene directories
    scene_dirs = sorted([d for d in input_patches.iterdir() if d.is_dir()])
    
    # Select scenes with different numbers of files (variety)
    test_scenes = []
    scene_file_counts = [(d, len(list(d.glob('*.tif')))) for d in scene_dirs]
    scene_file_counts.sort(key=lambda x: x[1])
    
    # Pick small, medium, large scenes
    if len(scene_file_counts) >= 3:
        test_scenes = [
            scene_file_counts[5][0],      # Small scene
            scene_file_counts[30][0],     # Medium scene
            scene_file_counts[-5][0]      # Large scene
        ]
    else:
        test_scenes = [s[0] for s in scene_file_counts[:num_scenes]]
    
    print(f"\n{'='*60}")
    print(f"SMALL BATCH TEST: Processing {len(test_scenes)} scenes")
    print(f"{'='*60}")
    
    stats = {
        'total_scenes': 0,
        'total_patches': 0,
        'scenes_with_objects': 0,
        'total_objects': 0,
        'class_distribution': {i: 0 for i in range(len(YOLO_CLASS_NAMES))},
        'scene_details': []
    }
    
    for scene_dir in test_scenes:
        scene_name = scene_dir.name
        print(f"\n📂 Processing: {scene_name}")
        
        # Find all patch files in this scene
        patch_files = sorted([f for f in scene_dir.glob('*.tif') 
                             if not f.name.endswith('_cl.tif') 
                             and not f.name.endswith('_conf.tif')])
        
        scene_stats = {
            'scene_name': scene_name,
            'num_patches': len(patch_files),
            'patches_with_objects': 0,
            'total_objects': 0,
            'class_counts': {i: 0 for i in range(len(YOLO_CLASS_NAMES))}
        }
        
        for patch_file in tqdm(patch_files, desc=f"  Converting {scene_name}"):
            patch_stem = patch_file.stem
            
            # Find corresponding label
            label_file = scene_dir / f"{patch_stem}_cl.tif"
            if not label_file.exists():
                continue
            
            # Convert to YOLO
            output_txt = output_base / 'labels' / f"{patch_stem}.txt"
            try:
                num_objects, class_counts = convert_mask_to_yolo(
                    label_file, output_txt,
                    min_area=None,
                    bbox_padding=5
                )
                
                stats['total_patches'] += 1
                scene_stats['total_objects'] += num_objects
                
                if num_objects > 0:
                    scene_stats['patches_with_objects'] += 1
                    
                    for class_id, count in class_counts.items():
                        stats['class_distribution'][class_id] += count
                        scene_stats['class_counts'][class_id] += count
                
                # Copy image
                output_img = output_base / 'images' / f"{patch_stem}.tif"
                os.makedirs(output_img.parent, exist_ok=True)
                if not output_img.exists():
                    os.symlink(patch_file.resolve(), output_img)
                    
            except Exception as e:
                print(f"    ❌ Error processing {patch_stem}: {e}")
                continue
        
        stats['total_scenes'] += 1
        stats['total_objects'] += scene_stats['total_objects']
        if scene_stats['total_objects'] > 0:
            stats['scenes_with_objects'] += 1
        
        stats['scene_details'].append(scene_stats)
        
        # Print scene summary
        print(f"  ✅ Patches: {scene_stats['num_patches']}, "
              f"With objects: {scene_stats['patches_with_objects']}, "
              f"Total objects: {scene_stats['total_objects']}")
    
    # Print overall results
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"\n📊 Overall Statistics:")
    print(f"  Scenes processed: {stats['total_scenes']}")
    print(f"  Patches processed: {stats['total_patches']}")
    print(f"  Scenes with objects: {stats['scenes_with_objects']}")
    print(f"  Total objects detected: {stats['total_objects']}")
    
    if stats['total_objects'] > 0:
        print(f"\n🎯 Class Distribution:")
        for class_id in range(len(YOLO_CLASS_NAMES)):
            count = stats['class_distribution'][class_id]
            percentage = 100 * count / stats['total_objects']
            print(f"  Class {class_id} ({YOLO_CLASS_NAMES[class_id]}): "
                  f"{count} objects ({percentage:.2f}%)")
    
    # Save results
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save statistics
    stats_file = output_base / 'test_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save class names
    class_file = output_base / 'classes.txt'
    with open(class_file, 'w') as f:
        for name in YOLO_CLASS_NAMES:
            f.write(name + '\n')
    
    # Save class mapping
    mapping_file = output_base / 'class_mapping.json'
    mapping_info = {
        'yolo_classes': YOLO_CLASS_NAMES,
        'original_to_yolo': {str(k): v for k, v in ORIGINAL_TO_YOLO_MAP.items()},
        'original_class_names': {str(k): v for k, v in ORIGINAL_CLASS_NAMES.items()}
    }
    with open(mapping_file, 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    print(f"\n📁 Output saved to: {output_base}")
    print(f"  - Statistics: {stats_file}")
    print(f"  - Classes: {class_file}")
    print(f"  - Mapping: {mapping_file}")
    
    print(f"\n{'='*60}")
    if stats['total_objects'] > 0:
        print("✅ TEST PASSED: Objects detected successfully!")
        print("\nNext steps:")
        print("  1. Run visualization: python visualize_dataset.py --dataset MARIDA_YOLO_test --num_samples 5")
        print("  2. If results look good, run full pipeline: python run_preprocessing_pipeline.py")
    else:
        print("⚠️  WARNING: No objects detected!")
        print("This could mean:")
        print("  - The selected scenes contain only background")
        print("  - min_area threshold is too high")
        print("  - Class mapping needs adjustment")
    print(f"{'='*60}\n")
    
    return stats

if __name__ == "__main__":
    stats = test_small_batch(num_scenes=3)
    
    # Return success if objects were found
    sys.exit(0 if stats['total_objects'] > 0 else 1)
