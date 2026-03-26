"""
Split MARIDA dataset into train/val/test sets by scene.
Ensures no data leakage by splitting at the scene level.
"""

import os
import shutil
from pathlib import Path
import random
import json
from tqdm import tqdm

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test sets by scene.
    
    Args:
        input_dir: Path to MARIDA_YOLO directory
        output_dir: Path to output directory
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all label files
    labels_dir = input_path / 'labels'
    images_dir = input_path / 'images'
    
    if not labels_dir.exists() or not images_dir.exists():
        raise ValueError(f"Labels or images directory not found in {input_dir}")
    
    # Get all scenes
    label_files = list(labels_dir.glob('*.txt'))
    scene_names = [f.stem for f in label_files]
    
    print(f"Found {len(scene_names)} scenes")
    
    # Shuffle scenes
    random.shuffle(scene_names)
    
    # Calculate split indices
    n_total = len(scene_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split scenes
    train_scenes = scene_names[:n_train]
    val_scenes = scene_names[n_train:n_train + n_val]
    test_scenes = scene_names[n_train + n_val:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_scenes)} scenes ({len(train_scenes)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_scenes)} scenes ({len(val_scenes)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_scenes)} scenes ({len(test_scenes)/n_total*100:.1f}%)")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files to respective splits
    split_mapping = {
        'train': train_scenes,
        'val': val_scenes,
        'test': test_scenes
    }
    
    stats = {'train': 0, 'val': 0, 'test': 0}
    missing_images = []
    
    for split_name, scenes in split_mapping.items():
        print(f"\nCopying {split_name} set...")
        for scene in tqdm(scenes):
            # Copy label
            label_src = labels_dir / f"{scene}.txt"
            label_dst = output_path / split_name / 'labels' / f"{scene}.txt"
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
                
                # Count objects
                with open(label_src, 'r') as f:
                    stats[split_name] += len(f.readlines())
            
            # Copy image
            img_candidates = list(images_dir.glob(f"{scene}*.tif"))
            if not img_candidates:
                img_candidates = list(images_dir.glob(f"{scene}*.png"))
            if not img_candidates:
                img_candidates = list(images_dir.glob(f"{scene}*.jpg"))

            if img_candidates:
                img_src = img_candidates[0]
                img_dst = output_path / split_name / 'images' / img_src.name
                if not img_dst.exists():
                    # 为兼容Windows权限与GDAL读取，直接复制文件而非符号链接
                    shutil.copy2(img_src, img_dst)
            else:
                missing_images.append(scene)
    
    print(f"\nObject distribution:")
    print(f"  Train: {stats['train']} objects")
    print(f"  Val:   {stats['val']} objects")
    print(f"  Test:  {stats['test']} objects")

    if missing_images:
        print(f"\n[Warn] Missing images for {len(missing_images)} scenes (see split_info.json)")
        split_info_extra = {'missing_images': missing_images}
    else:
        split_info_extra = {}
    
    # Save split info
    split_info = {
        'train': train_scenes,
        'val': val_scenes,
        'test': test_scenes,
        'stats': stats,
        'ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'seed': seed,
        **split_info_extra
    }
    
    with open(output_path / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Copy classes file
    classes_file = input_path / 'classes.txt'
    if classes_file.exists():
        shutil.copy2(classes_file, output_path / 'classes.txt')
    
    print(f"\nDataset split complete!")
    print(f"Output saved to: {output_path}")

def create_yolo_yaml(output_dir):
    """
    Create YAML configuration file for YOLOv11 training.
    """
    output_path = Path(output_dir)
    
    # Read class names
    classes_file = output_path / 'classes.txt'
    if not classes_file.exists():
        print("Warning: classes.txt not found")
        return
    
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Create YAML content
    yaml_content = f"""# MARIDA Marine Debris Detection Dataset
# YOLOv11 Configuration

# Paths (relative to this file)
path: {output_path.resolve()}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}

# Additional info
task: detect
mode: train

# Training parameters (can be overridden)
# epochs: 100
# imgsz: 512
# batch: 16
# device: 0
"""
    
    yaml_file = output_path / 'marida.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML config saved to: {yaml_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Split MARIDA dataset into train/val/test.')
    parser.add_argument('--input-dir', type=str, help='Input MARIDA_YOLO directory')
    parser.add_argument('--output-dir', type=str, help='Output split directory')
    args = parser.parse_args()
    
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        # Prefer exact MARIDA_YOLO as input; fallback to latest non-augmented
        import glob
        exact_dir = Path('D:/OceanEdge_Group/Marine-Debris-Detection-main') / 'MARIDA_YOLO'
        if exact_dir.exists():
            input_dir = exact_dir
            print(f"Using input (exact): {input_dir}")
        else:
            pattern = str(Path('D:/OceanEdge_Group/Marine-Debris-Detection-main') / 'MARIDA_YOLO*')
            matching_dirs = [p for p in glob.glob(pattern) if 'augmented' not in os.path.basename(p)]
            if matching_dirs:
                input_dir = Path(max(matching_dirs, key=lambda x: os.path.getmtime(x)))
                print(f"Using input (latest non-augmented): {input_dir}")
            else:
                input_dir = exact_dir
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('D:/OceanEdge_Group/Marine-Debris-Detection-main') / 'MARIDA_YOLO_split'
    
    # Split dataset
    split_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
    
    # Create YAML config
    create_yolo_yaml(output_dir)

if __name__ == "__main__":
    main()
