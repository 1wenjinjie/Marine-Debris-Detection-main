#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的数据增强策略 - 针对多光谱输入优化
基于波段测试结果，增强对不同光谱特征的鲁棒性
"""

import os
import numpy as np
from pathlib import Path
import rasterio
from rasterio.transform import Affine
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import json

def get_enhanced_augmentation_pipeline(image_size=512, mode='train', num_channels=6):
    """
    增强的数据增强管道，针对多光谱输入优化

    Args:
        image_size: 目标图像尺寸
        mode: 'train' 或 'val'
        num_channels: 通道数 (6通道: RGB+NIR+SWIR)
    """

    if mode == 'train':
        # 几何变换 (对所有通道有效)
        geometric_transforms = [
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            # 增强的仿射变换
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-10, 10),
                shear=(-5, 5),
                p=0.3
            ),
        ]

        # 像素级增强 (考虑多光谱特性)
        pixel_transforms = [
            # 亮度/对比度调整 (对所有通道)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),

            # 高斯噪声 (模拟传感器噪声)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

            # 高斯模糊 (模拟大气效应)
            A.GaussianBlur(blur_limit=3, p=0.1),

            # 乘性噪声 (模拟光照变化)
            A.MultiplicativeNoise(
                multiplier=(0.9, 1.1),
                per_channel=True,
                p=0.2
            ),

            # 随机阴影 (模拟云影)
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2
            ),

            # 粗糙 dropout (模拟传感器故障或云覆盖)
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=0,
                p=0.2
            ),

            # 弹性变换 (模拟大气扭曲)
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.1
            ),
        ]

        # 通道特定增强
        if num_channels >= 4:
            # NIR通道dropout (模拟NIR传感器故障)
            pixel_transforms.append(
                A.ChannelDropout(
                    channel_drop_range=(1, 1),
                    fill_value=0,
                    p=0.05
                )
            )

        if num_channels >= 6:
            # SWIR通道增强 dropout
            pixel_transforms.append(
                A.ChannelDropout(
                    channel_drop_range=(1, 1),
                    fill_value=0,
                    p=0.03  # SWIR更稳定，dropout概率更低
                )
            )

        # 光谱增强 (针对多光谱数据)
        spectral_transforms = []

        if num_channels >= 6:
            # 光谱混合增强 (模拟不同材料的光谱特性)
            spectral_transforms.append(
                A.Lambda(
                    image=lambda img, **kwargs: spectral_mix_augmentation(img),
                    p=0.1
                )
            )

        # 组合所有变换
        all_transforms = geometric_transforms + pixel_transforms + spectral_transforms

        return A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            )
        )
    else:
        # 验证模式 - 无增强
        return A.Compose([
            # 验证时不进行任何变换
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

def spectral_mix_augmentation(image):
    """
    光谱混合增强 - 模拟不同材料的光谱混合
    针对海洋垃圾检测的特定场景优化
    """
    if image.shape[2] < 6:
        return image  # 需要至少6个通道

    # 创建增强后的图像副本
    augmented = image.copy().astype(np.float32)

    # 水体光谱特性 (低反射)
    water_spectrum = np.array([0.02, 0.05, 0.08, 0.01, 0.15, 0.12])  # B, G, R, NIR, SWIR1, SWIR2

    # 塑料垃圾光谱特性 (高反射)
    plastic_spectrum = np.array([0.15, 0.25, 0.35, 0.65, 0.45, 0.40])

    # 随机选择混合区域
    h, w = image.shape[:2]
    num_patches = np.random.randint(1, 4)

    for _ in range(num_patches):
        # 随机位置和大小
        x = np.random.randint(0, w-32)
        y = np.random.randint(0, h-32)
        size = np.random.randint(16, 48)

        # 混合系数
        mix_ratio = np.random.uniform(0.1, 0.5)

        # 应用光谱混合
        for c in range(6):  # 只对前6个通道应用
            original = augmented[y:y+size, x:x+size, c]
            # 混合水体和塑料光谱
            mixed = (1 - mix_ratio) * original + mix_ratio * (
                water_spectrum[c] * original + plastic_spectrum[c] * (255 - original)
            )
            augmented[y:y+size, x:x+size, c] = np.clip(mixed, 0, 255)

    return augmented.astype(np.uint8)

def load_multispectral_image_and_labels(image_path, label_path, num_channels=6):
    """
    加载多光谱图像和标签 (6通道: RGB+NIR+SWIR1+SWIR2)

    Args:
        image_path: 图像路径
        label_path: 标签路径
        num_channels: 期望的通道数
    """

    # Sentinel-2波段映射
    band_mapping = {
        1: 'B02',  # Blue
        2: 'B03',  # Green
        3: 'B04',  # Red
        4: 'B08',  # NIR
        5: 'B11',  # SWIR1
        6: 'B12',  # SWIR2
    }

    with rasterio.open(image_path) as src:
        # 读取指定波段
        selected_bands = []
        for i in range(1, num_channels + 1):
            if i <= src.count:
                band_data = src.read(i)
                selected_bands.append(band_data)

        # 如果波段不够，用零填充
        while len(selected_bands) < num_channels:
            selected_bands.append(np.zeros_like(selected_bands[0]))

        # 堆叠为多通道图像
        image = np.stack(selected_bands, axis=-1)

        # 标准化到0-255
        for i in range(num_channels):
            p2, p98 = np.percentile(image[:, :, i], (2, 98))
            image[:, :, i] = np.clip((image[:, :, i] - p2) / (p98 - p2) * 255, 0, 255)

    image = image.astype(np.uint8)

    # 加载YOLO标签
    bboxes = []
    class_labels = []

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

def save_enhanced_augmented_data(image, bboxes, class_labels, output_img_path, output_label_path):
    """保存增强的图像和标签"""
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    cv2.imwrite(str(output_img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    with open(output_label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

def enhance_augmentation_pipeline(input_dir, output_dir, num_channels=6, num_augmentations=7):
    """
    增强的数据增强管道

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        num_channels: 通道数
        num_augmentations: 每个图像的增强版本数
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    train_images = list((input_path / 'train' / 'images').glob('*.tif'))

    print(f"🎯 增强数据增强 - {num_channels}通道输入")
    print(f"📊 找到 {len(train_images)} 个训练图像")
    print(f"🔄 每个图像生成 {num_augmentations} 个增强版本")

    # 获取增强管道
    aug_pipeline = get_enhanced_augmentation_pipeline(
        mode='train',
        num_channels=num_channels
    )

    stats = {
        'original': 0,
        'augmented': 0,
        'total_objects_original': 0,
        'total_objects_augmented': 0,
        'channels': num_channels
    }

    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 复制验证和测试集
    for split in ['val', 'test']:
        print(f"\n📋 复制{split}集...")
        for img_file in tqdm(list((input_path / split / 'images').glob('*.tif'))):
            scene_name = img_file.stem
            label_file = input_path / split / 'labels' / f"{scene_name}.txt"

            image, bboxes, class_labels = load_multispectral_image_and_labels(
                img_file, label_file, num_channels
            )

            output_img = output_path / split / 'images' / f"{scene_name}.png"
            output_label = output_path / split / 'labels' / f"{scene_name}.txt"
            save_enhanced_augmented_data(image, bboxes, class_labels, output_img, output_label)

    # 增强训练集
    print("\n🎨 增强训练集...")
    for img_file in tqdm(train_images):
        scene_name = img_file.stem
        label_file = input_path / 'train' / 'labels' / f"{scene_name}.txt"

        try:
            image, bboxes, class_labels = load_multispectral_image_and_labels(
                img_file, label_file, num_channels
            )

            # 保存原始图像
            output_img = output_path / 'train' / 'images' / f"{scene_name}_orig.png"
            output_label = output_path / 'train' / 'labels' / f"{scene_name}_orig.txt"
            save_enhanced_augmented_data(image, bboxes, class_labels, output_img, output_label)

            stats['original'] += 1
            stats['total_objects_original'] += len(bboxes)

            # 生成增强版本
            for aug_idx in range(num_augmentations):
                try:
                    augmented = aug_pipeline(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )

                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']

                    output_img = output_path / 'train' / 'images' / f"{scene_name}_aug{aug_idx}.png"
                    output_label = output_path / 'train' / 'labels' / f"{scene_name}_aug{aug_idx}.txt"
                    save_enhanced_augmented_data(aug_image, aug_bboxes, aug_labels, output_img, output_label)

                    stats['augmented'] += 1
                    stats['total_objects_augmented'] += len(aug_bboxes)

                except Exception as e:
                    print(f"\n⚠️  增强失败 {scene_name}_aug{aug_idx}: {e}")
                    continue

        except Exception as e:
            print(f"\n⚠️  处理失败 {scene_name}: {e}")
            continue

    print("\n✅ 增强完成!")
    print(f"📊 统计信息:")
    print(f"  • 原始图像: {stats['original']}")
    print(f"  • 增强图像: {stats['augmented']}")
    print(f"  • 总训练图像: {stats['original'] + stats['augmented']}")
    print(f"  • 原始目标数: {stats['total_objects_original']}")
    print(f"  • 增强目标数: {stats['total_objects_augmented']}")
    print(f"  • 输入通道数: {stats['channels']}")

    # 保存统计信息
    with open(output_path / 'enhanced_augmentation_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # 复制类别文件
    classes_file = input_path / 'classes.txt'
    if classes_file.exists():
        import shutil
        shutil.copy2(classes_file, output_path / 'classes.txt')

def main():
    """主函数"""
    base_dir = Path('/Users/wen11/Documents/OceanEdge_Group/Marine_dataset')
    input_dir = base_dir / 'MARIDA_YOLO_split'
    output_dir = base_dir / 'MARIDA_YOLO_enhanced'

    if not input_dir.exists():
        print(f"❌ 错误: {input_dir} 不存在")
        print("请先运行 preprocess_split_dataset.py")
        return

    # 检查albumentations
    try:
        import albumentations
    except ImportError:
        print("❌ 错误: 未安装 albumentations")
        print("请运行: pip install albumentations")
        return

    # 运行增强的数据增强 (6通道: RGB+NIR+SWIR1+SWIR2)
    enhance_augmentation_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        num_channels=6,  # 基于测试结果推荐
        num_augmentations=7  # 增加到7个版本以获得更多多样性
    )

    print(f"\n💾 输出保存到: {output_dir}")

if __name__ == '__main__':
    main()
