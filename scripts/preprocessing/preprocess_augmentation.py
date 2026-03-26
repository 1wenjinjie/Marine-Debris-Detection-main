"""
Data augmentation pipeline for marine debris detection.
Applies rotation, flipping, color jitter, and other augmentations.
Enhanced for multispectral satellite imagery.
"""

import os
import numpy as np
from pathlib import Path
import rasterio
from rasterio.transform import Affine
import albumentations as A
import cv2
from tqdm import tqdm
import json


def get_augmentation_pipeline(image_size=512, mode='train', num_channels=6):
    """
    Get augmentation pipeline for training or validation.
    Enhanced for multispectral satellite imagery.
    """
    if mode == 'train':
        # 仅使用与多通道兼容且当前版本API稳定的增强
        geometric_transforms = [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.08, 0.08),
                rotate=(-12, 12),
                shear=(-5, 5),
                fit_output=False,
                p=0.35,
            ),
        ]

        pixel_transforms = [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.2),
        ]

        if num_channels == 3:
            pixel_transforms.insert(1, A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
            ))

        return A.Compose(
            geometric_transforms + pixel_transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3,
            )
        )
    else:
        return A.Compose([], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def load_image_and_labels(image_path, label_path):
    """Load image(HWC) and YOLO labels list."""
    with rasterio.open(image_path) as src:
        image = src.read().transpose(1, 2, 0)
    # 按通道线性拉伸到0-255，便于后续uint8增强与保存
    if image.ndim == 3:
        for i in range(image.shape[2]):
            p2, p98 = np.percentile(image[:, :, i], (2, 98))
            if p98 > p2:
                image[:, :, i] = np.clip((image[:, :, i] - p2) / (p98 - p2) * 255, 0, 255)
        image = image.astype(np.uint8)

    bboxes, class_labels = [], []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    class_labels.append(class_id)
                    bboxes.append(bbox)
    return image, bboxes, class_labels


def _ensure_tif_path(path: Path) -> Path:
    # 将输出统一为.tif
    if path.suffix.lower() != '.tif':
        return path.with_suffix('.tif')
    return path


def _save_multiband_tif(image_hwc: np.ndarray, out_path: Path):
    out_path = _ensure_tif_path(out_path)
    H, W = image_hwc.shape[:2]
    C = 1 if image_hwc.ndim == 2 else image_hwc.shape[2]
    data_chw = image_hwc if image_hwc.ndim == 2 else image_hwc.transpose(2, 0, 1)
    profile = {
        'driver': 'GTiff',
        'height': H,
        'width': W,
        'count': C,
        'dtype': data_chw.dtype,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data_chw)
    return out_path


def save_augmented_data(image, bboxes, class_labels, output_img_path, output_label_path):
    """Save image+labels；多通道统一保存为.tif，3/4通道可写PNG。"""
    output_img_path = Path(output_img_path)
    # 判定通道数
    C = 1 if image.ndim == 2 else image.shape[2]
    if C in (3, 4) and output_img_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
        # 3/4通道可直接PNG
        os.makedirs(output_img_path.parent, exist_ok=True)
        if C == 3:
            cv2.imwrite(str(output_img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:  # 4通道
            cv2.imwrite(str(output_img_path), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    else:
        # 其它情况统一写TIF，避免OpenCV通道限制
        output_img_path = _ensure_tif_path(output_img_path)
        _save_multiband_tif(image, output_img_path)

    # 保存标签
    output_label_path = Path(output_label_path)
    os.makedirs(output_label_path.parent, exist_ok=True)
    with open(output_label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            f.write(f"{class_id} {' '.join(map(str, bbox))}\n")


def augment_dataset(input_dir, output_dir, num_augmentations=5):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    train_images = list((input_path / 'train' / 'images').glob('*.tif'))
    print(f"Found {len(train_images)} training images")
    print(f"Generating {num_augmentations} augmented versions per image")

    aug_pipeline = get_augmentation_pipeline(mode='train')

    stats = {'original': 0, 'augmented': 0, 'total_objects_original': 0, 'total_objects_augmented': 0}

    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 复制 val/test（不做增强），保持多通道，输出为.tif
    for split in ['val', 'test']:
        print(f"\nCopying {split} set...")
        for img_file in tqdm(list((input_path / split / 'images').glob('*.tif'))):
            scene_name = img_file.stem
            label_file = input_path / split / 'labels' / f"{scene_name}.txt"
            image, bboxes, class_labels = load_image_and_labels(img_file, label_file)
            output_img = output_path / split / 'images' / f"{scene_name}.tif"
            output_label = output_path / split / 'labels' / f"{scene_name}.txt"
            save_augmented_data(image, bboxes, class_labels, output_img, output_label)

    # 训练集：保存原图及增强，统一为.tif
    print("\nAugmenting training set...")
    for img_file in tqdm(train_images):
        scene_name = img_file.stem
        label_file = input_path / 'train' / 'labels' / f"{scene_name}.txt"
        try:
            image, bboxes, class_labels = load_image_and_labels(img_file, label_file)
            # 原图
            output_img = output_path / 'train' / 'images' / f"{scene_name}_orig.tif"
            output_label = output_path / 'train' / 'labels' / f"{scene_name}_orig.txt"
            save_augmented_data(image, bboxes, class_labels, output_img, output_label)
            stats['original'] += 1
            stats['total_objects_original'] += len(bboxes)
            # 增强版本
            for aug_idx in range(num_augmentations):
                try:
                    augmented = aug_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']
                    output_img = output_path / 'train' / 'images' / f"{scene_name}_aug{aug_idx}.tif"
                    output_label = output_path / 'train' / 'labels' / f"{scene_name}_aug{aug_idx}.txt"
                    save_augmented_data(aug_image, aug_bboxes, aug_labels, output_img, output_label)
                    stats['augmented'] += 1
                    stats['total_objects_augmented'] += len(aug_bboxes)
                except Exception as e:
                    print(f"\nWarning: Augmentation failed for {scene_name}_aug{aug_idx}: {e}")
                    continue
        except Exception as e:
            print(f"\nWarning: Failed to process {scene_name}: {e}")
            continue

    print(f"\nAugmentation complete!")
    print(f"Statistics:")
    print(f"  Original images: {stats['original']}")
    print(f"  Augmented images: {stats['augmented']}")
    print(f"  Total training images: {stats['original'] + stats['augmented']}")
    print(f"  Original objects: {stats['total_objects_original']}")
    print(f"  Augmented objects: {stats['total_objects_augmented']}")

    with open(output_path / 'augmentation_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    classes_file = input_path / 'classes.txt'
    if classes_file.exists():
        import shutil
        shutil.copy2(classes_file, output_path / 'classes.txt')


def main():
    base_dir = Path('D:/OceanEdge_Group/Marine-Debris-Detection-main')
    import glob
    pattern = str(base_dir / 'MARIDA_YOLO_split*')
    matching_dirs = glob.glob(pattern)
    if matching_dirs:
        input_dir = Path(max(matching_dirs, key=lambda x: os.path.getmtime(x)))
        print(f"Using input: {input_dir}")
    else:
        input_dir = base_dir / 'MARIDA_YOLO_split'
    output_dir = base_dir / 'MARIDA_YOLO_augmented'
    if not input_dir.exists():
        print(f"Error: {input_dir} not found\nPlease run preprocess_split_dataset.py first")
        return
    try:
        import albumentations  # noqa
    except ImportError:
        print("Error: albumentations not installed\nInstall with: pip install albumentations")
        return
    augment_dataset(input_dir=input_dir, output_dir=output_dir, num_augmentations=5)
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
