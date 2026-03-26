"""
Visualization tool for preprocessed MARIDA dataset.
Displays images with bounding boxes for quality checking.
"""

import os
import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import random

# Class colors for visualization
CLASS_COLORS = [
    '#FF0000',  # Class 0: Red
    '#00FF00',  # Class 1: Green
    '#0000FF',  # Class 2: Blue
    '#FFFF00',  # Class 3: Yellow
    '#FF00FF',  # Class 4: Magenta
    '#00FFFF',  # Class 5: Cyan
    '#FFA500',  # Class 6: Orange
]

def load_image(image_path, normalize=True):
    """
    Load multi-band image.
    
    Args:
        image_path: Path to image file
        normalize: Whether to normalize to [0, 1] range
        
    Returns:
        Image array (H, W, C)
    """
    with rasterio.open(image_path) as src:
        image = src.read()  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
    
    if normalize:
        # Normalize to [0, 1] using percentile for better visualization
        p2, p98 = np.percentile(image, (2, 98))
        image = np.clip((image - p2) / (p98 - p2), 0, 1)
    
    return image

def load_labels(label_path):
    """
    Load YOLO format labels.
    
    Returns:
        List of (class_id, x_center, y_center, width, height)
    """
    labels = []
    
    if not Path(label_path).exists():
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = list(map(float, line.split()))
                if len(parts) == 5:
                    labels.append(parts)
    
    return labels

def yolo_to_corners(x_center, y_center, width, height, img_width, img_height):
    """
    Convert YOLO format to corner coordinates.
    
    Returns:
        (x_min, y_min, width_px, height_px) in pixels
    """
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    return x_min, y_min, width_px, height_px

def visualize_sample(image_path, label_path, class_names=None, save_path=None, show=True):
    """
    Visualize a single image with its bounding boxes.
    
    Args:
        image_path: Path to image
        label_path: Path to label file
        class_names: List of class names
        save_path: Path to save visualization (optional)
        show: Whether to display the plot
    """
    # Load image
    image = load_image(image_path)
    
    # Load labels
    labels = load_labels(label_path)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display RGB channels
    if image.shape[2] >= 3:
        rgb = image[:, :, :3]
    else:
        rgb = np.stack([image[:, :, 0]] * 3, axis=-1)
    
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Visualization', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Display with bounding boxes
    axes[1].imshow(rgb)
    axes[1].set_title(f'Detections ({len(labels)} objects)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Draw bounding boxes
    img_height, img_width = image.shape[:2]
    
    for label in labels:
        class_id, x_center, y_center, width, height = label
        class_id = int(class_id)
        
        # Convert to corner format
        x_min, y_min, width_px, height_px = yolo_to_corners(
            x_center, y_center, width, height, img_width, img_height
        )
        
        # Draw rectangle
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        rect = patches.Rectangle(
            (x_min, y_min), width_px, height_px,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # Add label text
        label_text = f"Class {class_id}"
        if class_names and class_id < len(class_names):
            label_text = class_names[class_id]
        
        axes[1].text(
            x_min, y_min - 5,
            label_text,
            color=color,
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor=color)
        )
    
    # Add image info
    info_text = f"Image: {Path(image_path).stem}\n"
    info_text += f"Size: {img_height}x{img_width}\n"
    info_text += f"Channels: {image.shape[2]}\n"
    info_text += f"Objects: {len(labels)}"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_dataset(dataset_dir, num_samples=10, save_dir=None, class_names=None):
    """
    Visualize random samples from dataset.
    
    Args:
        dataset_dir: Path to dataset directory
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
        class_names: List of class names
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'
    
    # Get all image files
    image_files = list(images_dir.glob('*.tif')) + list(images_dir.glob('*.png'))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    # Randomly sample
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"\nVisualizing {len(sample_files)} samples from {dataset_dir.name}")
    
    for i, image_path in enumerate(sample_files, 1):
        print(f"\nSample {i}/{len(sample_files)}: {image_path.stem}")
        
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        save_path = None
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{dataset_dir.name}_{image_path.stem}.png"
        
        try:
            visualize_sample(
                image_path, label_path,
                class_names=class_names,
                save_path=save_path,
                show=(save_dir is None)  # Only show if not saving
            )
        except Exception as e:
            print(f"Error visualizing {image_path.stem}: {e}")

def load_class_names(dataset_dir):
    """Load class names from classes.txt file."""
    class_file = Path(dataset_dir).parent / 'classes.txt'
    
    if not class_file.exists():
        class_file = Path(dataset_dir) / 'classes.txt'
    
    if class_file.exists():
        with open(class_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MARIDA dataset with bounding boxes')
    parser.add_argument('--dataset', type=str, default='MARIDA_YOLO_split/train',
                      help='Dataset directory to visualize')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                      help='Directory to save visualizations (optional)')
    parser.add_argument('--no_save', action='store_true',
                      help='Do not save visualizations, show interactively')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / args.dataset
    
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        print("\nAvailable datasets:")
        
        yolo_dir = base_dir / 'MARIDA_YOLO'
        if yolo_dir.exists():
            print(f"  - MARIDA_YOLO (unsplit)")
        
        split_dir = base_dir / 'MARIDA_YOLO_split'
        if split_dir.exists():
            for split in ['train', 'val', 'test']:
                split_path = split_dir / split
                if split_path.exists():
                    print(f"  - MARIDA_YOLO_split/{split}")
        
        return
    
    # Load class names
    class_names = load_class_names(dataset_dir)
    if class_names:
        print(f"\nClass names loaded: {class_names}")
    
    # Visualize
    save_dir = None if args.no_save else (base_dir / args.save_dir)
    
    visualize_dataset(
        dataset_dir,
        num_samples=args.num_samples,
        save_dir=save_dir,
        class_names=class_names
    )
    
    print("\n✅ Visualization complete!")
    
    if save_dir:
        print(f"Visualizations saved to: {save_dir}")

if __name__ == "__main__":
    main()
