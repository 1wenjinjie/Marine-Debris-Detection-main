"""
Validation script for preprocessed MARIDA dataset.
Checks data integrity, format correctness, and statistics.
"""

import os
import sys
from pathlib import Path
import numpy as np
import rasterio
import json
from collections import Counter
from tqdm import tqdm

def validate_yolo_format(label_path, num_classes):
    """
    Validate YOLO format label file.
    
    Returns:
        (is_valid, error_messages, bbox_count)
    """
    errors = []
    bbox_count = 0
    
    if not label_path.exists():
        return True, [], 0  # Empty label file is valid
    
    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"Line {line_num}: Expected 5 values, got {len(parts)}")
                continue
            
            try:
                class_id, x, y, w, h = map(float, parts)
                
                # Validate class ID
                if not (0 <= class_id < num_classes):
                    errors.append(f"Line {line_num}: Class ID {class_id} out of range [0, {num_classes})")
                
                # Validate coordinates
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    errors.append(f"Line {line_num}: Center ({x}, {y}) out of range [0, 1]")
                
                if not (0 < w <= 1 and 0 < h <= 1):
                    errors.append(f"Line {line_num}: Size ({w}, {h}) out of range (0, 1]")
                
                bbox_count += 1
                
            except ValueError as e:
                errors.append(f"Line {line_num}: Cannot parse values: {e}")
    
    return len(errors) == 0, errors, bbox_count

def validate_image(image_path, expected_channels=4, expected_size=512):
    """
    Validate image file.
    
    Returns:
        (is_valid, error_messages, image_info)
    """
    errors = []
    info = {}
    
    try:
        with rasterio.open(image_path) as src:
            info['channels'] = src.count
            info['height'] = src.height
            info['width'] = src.width
            info['dtype'] = str(src.dtypes[0])
            
            # Check channels
            if expected_channels and src.count != expected_channels:
                errors.append(f"Expected {expected_channels} channels, got {src.count}")
            
            # Check size
            if expected_size and (src.height != expected_size or src.width != expected_size):
                errors.append(f"Expected {expected_size}x{expected_size}, got {src.height}x{src.width}")
            
            # Check data validity
            data = src.read(1)
            info['min'] = float(np.min(data))
            info['max'] = float(np.max(data))
            info['mean'] = float(np.mean(data))
            
            if np.any(np.isnan(data)):
                errors.append("Contains NaN values")
            if np.any(np.isinf(data)):
                errors.append("Contains Inf values")
                
    except Exception as e:
        errors.append(f"Failed to read image: {e}")
        return False, errors, info
    
    return len(errors) == 0, errors, info

def validate_dataset(dataset_dir, num_classes=3, expected_channels=None, expected_size=None):
    """
    Validate entire dataset directory.
    
    Args:
        dataset_dir: Path to dataset directory containing images/ and labels/
        num_classes: Expected number of classes
        expected_channels: Expected number of image channels
        expected_size: Expected image size
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'
    
    print(f"\n{'='*60}")
    print(f"Validating dataset: {dataset_dir.name}")
    print(f"{'='*60}")
    
    # Check directory structure
    if not images_dir.exists():
        print(f"❌ ERROR: Images directory not found: {images_dir}")
        return False
    
    if not labels_dir.exists():
        print(f"❌ ERROR: Labels directory not found: {labels_dir}")
        return False
    
    # Get all image files
    image_files = list(images_dir.glob('*.tif')) + list(images_dir.glob('*.png'))
    
    if not image_files:
        print(f"❌ ERROR: No image files found in {images_dir}")
        return False
    
    print(f"\n✅ Found {len(image_files)} images")
    
    # Validation statistics
    stats = {
        'total_images': len(image_files),
        'valid_images': 0,
        'valid_labels': 0,
        'images_with_errors': 0,
        'labels_with_errors': 0,
        'total_bboxes': 0,
        'class_distribution': Counter(),
        'empty_labels': 0,
        'missing_labels': 0,
        'image_info': {},
        'errors': []
    }
    
    # Validate each image and its label
    print("\nValidating files...")
    for image_path in tqdm(image_files):
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"
        
        # Validate image
        img_valid, img_errors, img_info = validate_image(
            image_path, expected_channels, expected_size
        )
        
        if img_valid:
            stats['valid_images'] += 1
            if not stats['image_info']:
                stats['image_info'] = img_info
        else:
            stats['images_with_errors'] += 1
            stats['errors'].append(f"Image {stem}: {'; '.join(img_errors)}")
        
        # Validate label
        if not label_path.exists():
            stats['missing_labels'] += 1
            stats['errors'].append(f"Label {stem}.txt: Missing")
            continue
        
        label_valid, label_errors, bbox_count = validate_yolo_format(
            label_path, num_classes
        )
        
        if label_valid:
            stats['valid_labels'] += 1
            stats['total_bboxes'] += bbox_count
            
            if bbox_count == 0:
                stats['empty_labels'] += 1
            else:
                # Count class distribution
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(float(line.split()[0]))
                            stats['class_distribution'][class_id] += 1
        else:
            stats['labels_with_errors'] += 1
            for error in label_errors:
                stats['errors'].append(f"Label {stem}.txt: {error}")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Validation Results")
    print(f"{'='*60}")
    
    print(f"\n📊 Image Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Valid images: {stats['valid_images']} ({100*stats['valid_images']/stats['total_images']:.1f}%)")
    print(f"  Images with errors: {stats['images_with_errors']}")
    
    if stats['image_info']:
        info = stats['image_info']
        print(f"\n  Image properties:")
        print(f"    Size: {info['height']}x{info['width']}")
        print(f"    Channels: {info['channels']}")
        print(f"    Data type: {info['dtype']}")
        print(f"    Value range: [{info['min']:.4f}, {info['max']:.4f}]")
        print(f"    Mean value: {info['mean']:.4f}")
    
    print(f"\n📋 Label Statistics:")
    print(f"  Total labels: {stats['total_images']}")
    print(f"  Valid labels: {stats['valid_labels']} ({100*stats['valid_labels']/stats['total_images']:.1f}%)")
    print(f"  Labels with errors: {stats['labels_with_errors']}")
    print(f"  Missing labels: {stats['missing_labels']}")
    print(f"  Empty labels: {stats['empty_labels']} ({100*stats['empty_labels']/stats['total_images']:.1f}%)")
    
    print(f"\n🎯 Object Statistics:")
    print(f"  Total bounding boxes: {stats['total_bboxes']}")
    print(f"  Average bboxes per image: {stats['total_bboxes']/stats['total_images']:.2f}")
    
    if stats['class_distribution']:
        print(f"\n  Class distribution:")
        for class_id in sorted(stats['class_distribution'].keys()):
            count = stats['class_distribution'][class_id]
            percentage = 100 * count / stats['total_bboxes']
            print(f"    Class {class_id}: {count} objects ({percentage:.2f}%)")
    
    # Print errors
    if stats['errors']:
        print(f"\n⚠️  Errors found ({len(stats['errors'])} total):")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    # Overall validation result
    all_valid = (stats['images_with_errors'] == 0 and 
                 stats['labels_with_errors'] == 0 and 
                 stats['missing_labels'] == 0)
    
    print(f"\n{'='*60}")
    if all_valid:
        print("✅ PASSED: Dataset validation successful!")
    else:
        print("❌ FAILED: Dataset has errors that need to be fixed")
    print(f"{'='*60}\n")
    
    return all_valid, stats

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate MARIDA preprocessing outputs.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory containing MARIDA_YOLO* folders (defaults to script directory)."
    )
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=None,
        help="Expected number of channels in each image (omit to skip the check)."
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=None,
        help="Expected square image size in pixels (omit to skip the check)."
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of YOLO classes to use when validating labels."
    )
    args = parser.parse_args()

    base_dir = args.base_dir

    if not base_dir.exists():
        print(f"❌ Base directory does not exist: {base_dir}")
        return False
    
    # Check which dataset to validate
    yolo_split_dir = base_dir / 'MARIDA_YOLO_split'
    yolo_dir = base_dir / 'MARIDA_YOLO'
    
    results = {}
    
    # Validate split dataset if exists
    if yolo_split_dir.exists():
        print("\n" + "="*60)
        print("VALIDATING SPLIT DATASET")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            split_dir = yolo_split_dir / split
            if split_dir.exists():
                valid, stats = validate_dataset(
                    split_dir,
                    num_classes=args.num_classes,
                    expected_channels=args.expected_channels,
                    expected_size=args.expected_size
                )
                results[split] = {'valid': valid, 'stats': stats}
    
    # Validate unsplit dataset if exists
    elif yolo_dir.exists():
        valid, stats = validate_dataset(
            yolo_dir,
            num_classes=args.num_classes,
            expected_channels=args.expected_channels,
            expected_size=args.expected_size
        )
        results['all'] = {'valid': valid, 'stats': stats}
    
    else:
        print("❌ No processed dataset found!")
        print("Please run preprocessing pipeline first.")
        return
    
    # Save validation results
    output_file = base_dir / 'validation_results.json'
    with open(output_file, 'w') as f:
        # Convert Counter objects to dicts for JSON serialization
        for split_name, result in results.items():
            if 'stats' in result and 'class_distribution' in result['stats']:
                result['stats']['class_distribution'] = dict(result['stats']['class_distribution'])
        json.dump(results, f, indent=2)
    
    print(f"\nValidation results saved to: {output_file}")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    all_passed = all(r['valid'] for r in results.values())
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED!")
        print("\nYou can proceed to training with confidence.")
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("\nPlease review the errors above and fix them before training.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
