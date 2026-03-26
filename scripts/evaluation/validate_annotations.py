#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证YOLO标注准确性：比较自动转换的YOLO bbox与原始MARIDA shapefile标注
帮助判断标注质量和定位准确性
"""

import os
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import json
from pathlib import Path

def load_yolo_labels(label_path):
    """加载YOLO格式标注"""
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])
                    bboxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': w,
                        'height': h
                    })
    return bboxes

def load_shapefile_labels(shp_path, image_bounds):
    """加载shapefile标注并转换为像素坐标"""
    if not os.path.exists(shp_path):
        return []

    # 读取shapefile
    gdf = gpd.read_file(shp_path)

    # 假设标注是多边形，转换为bbox
    bboxes = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is not None:
            # 获取边界框
            bounds = geom.bounds  # (minx, miny, maxx, maxy)

            # 转换为像素坐标 (假设图像坐标系)
            # 这里需要根据实际的地理变换来转换
            # 简化版：假设shapefile坐标与图像像素坐标一致
            minx, miny, maxx, maxy = bounds

            # 转换为YOLO格式 (0-1范围)
            img_width = image_bounds[2] - image_bounds[0]
            img_height = image_bounds[3] - image_bounds[1]

            x_center = ((minx + maxx) / 2 - image_bounds[0]) / img_width
            y_center = ((miny + maxy) / 2 - image_bounds[1]) / img_height
            width = (maxx - minx) / img_width
            height = (maxy - miny) / img_height

            # 确保在0-1范围内
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            width = np.clip(width, 0, 1)
            height = np.clip(height, 0, 1)

            bboxes.append({
                'class_id': 0,  # 假设都是海洋垃圾
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'source': 'shapefile'
            })

    return bboxes

def load_expert_labels_from_mask(mask_path, image_width, image_height):
    """从MARIDA分类mask加载专家标注"""
    expert_bboxes = []
    
    if not os.path.exists(mask_path):
        return expert_bboxes
    
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # 读取分类mask
    
    # 海洋垃圾类别 (根据MARIDA数据集)
    marine_debris_classes = [5, 7, 14]  # 不同类型的海洋垃圾
    
    # 为每个类别找到连通区域
    for class_id in marine_debris_classes:
        class_mask = (mask == class_id).astype(np.uint8)
        
        if class_mask.sum() == 0:
            continue
            
        # 找到边界框
        rows, cols = np.where(class_mask > 0)
        if len(rows) > 0 and len(cols) > 0:
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            
            # 转换为YOLO格式 (0-1范围)
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            
            # 确保在0-1范围内
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            width = np.clip(width, 0, 1)
            height = np.clip(height, 0, 1)
            
            # 映射到YOLO类别 (0: Marine_Debris, 1: Floating_Object, 2: Ship_Wake)
            yolo_class = 0  # 都当作海洋垃圾
            
            expert_bboxes.append({
                'class_id': yolo_class,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'source': 'mask',
                'original_class': class_id
            })
    
    return expert_bboxes

def create_comparison_plot(image_path, yolo_labels, shapefile_labels, output_path):
    """创建YOLO vs Shapefile标注对比图"""

    # 读取图像
    with rasterio.open(image_path) as src:
        image = src.read([4, 3, 2])  # NIR-R-G合成
        image = np.transpose(image, (1, 2, 0))
        height, width = image.shape[:2]

        # 标准化显示
        for i in range(3):
            p2, p98 = np.percentile(image[:, :, i], (2, 98))
            image[:, :, i] = np.clip((image[:, :, i] - p2) / (p98 - p2) * 255, 0, 255)

    image = image.astype(np.uint8)

    # 类别颜色
    colors = ['#FF0000', '#00FF00', '#0000FF']  # Marine_Debris, Floating_Object, Ship_Wake
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'标注准确性验证: {os.path.basename(image_path)}', fontsize=14, fontweight='bold')
    
    # 子图1: 原始图像
    axes[0].imshow(image)
    axes[0].set_title('原始图像 (NIR-R-G)', fontsize=12)
    axes[0].axis('off')
    
    # 子图2: YOLO标注
    axes[1].imshow(image)
    axes[1].set_title('YOLO自动标注', fontsize=12)
    axes[1].axis('off')
    
    # 子图3: 专家标注
    axes[2].imshow(image)
    axes[2].set_title('MARIDA专家标注', fontsize=12)
    axes[2].axis('off')

    # 颜色定义
    yolo_color = '#FF4444'  # 红色 - YOLO
    shp_color = '#4444FF'   # 蓝色 - Shapefile

    # 绘制YOLO bbox
    for bbox in yolo_labels:
        x_center = bbox['x_center'] * width
        y_center = bbox['y_center'] * height
        w = bbox['width'] * width
        h = bbox['height'] * height
        x_min = x_center - w / 2
        y_min = y_center - h / 2

        for ax in axes[1:2]:  # 只在YOLO子图
            rect = Rectangle((x_min, y_min), w, h,
                           linewidth=2, edgecolor=yolo_color, facecolor='none',
                           linestyle='-', alpha=0.8)
            ax.add_patch(rect)

    # 绘制Shapefile bbox
    for bbox in shapefile_labels:
        x_center = bbox['x_center'] * width
        y_center = bbox['y_center'] * height
        w = bbox['width'] * width
        h = bbox['height'] * height
        x_min = x_center - w / 2
        y_min = y_center - h / 2

        for ax in axes[2:3]:  # 只在Shapefile子图
            rect = Rectangle((x_min, y_min), w, h,
                           linewidth=2, edgecolor=shp_color, facecolor='none',
                           linestyle='--', alpha=0.8)
            ax.add_patch(rect)

    # 在第一个子图显示两者对比
    for bbox in yolo_labels:
        x_center = bbox['x_center'] * width
        y_center = bbox['y_center'] * height
        w = bbox['width'] * width
        h = bbox['height'] * height
        x_min = x_center - w / 2
        y_min = y_center - h / 2

        rect = Rectangle((x_min, y_min), w, h,
                       linewidth=2, edgecolor=yolo_color, facecolor='none',
                       linestyle='-', alpha=0.8)
        axes[0].add_patch(rect)

    for bbox in shapefile_labels:
        x_center = bbox['x_center'] * width
        y_center = bbox['y_center'] * height
        w = bbox['width'] * width
        h = bbox['height'] * height
        x_min = x_center - w / 2
        y_min = y_center - h / 2

        rect = Rectangle((x_min, y_min), w, h,
                       linewidth=2, edgecolor=shp_color, facecolor='none',
                       linestyle='--', alpha=0.8)
        axes[0].add_patch(rect)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor=yolo_color, linewidth=2, linestyle='-',
              label='YOLO自动标注'),
        Patch(facecolor='none', edgecolor=shp_color, linewidth=2, linestyle='--',
              label='MARIDA专家标注')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 添加统计信息
    yolo_count = len(yolo_labels)
    shp_count = len(shapefile_labels)
    
    stats_text = (
        f"YOLO标注数量: {yolo_count}\n"
        f"专家标注数量: {shp_count}\n"
        f"差异: {abs(yolo_count - shp_count)}"
    )

    axes[0].text(0.02, 0.98, stats_text,
                transform=axes[0].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 已保存标注对比图: {output_path}")
    return yolo_count, shp_count

def validate_annotations():
    """验证标注准确性"""
    
    # 路径设置
    base_dir = Path('/Users/wen11/Documents/OceanEdge_Group/Marine_dataset')
    test_images_dir = base_dir / 'MARIDA_YOLO_test' / 'images'
    test_labels_dir = base_dir / 'MARIDA_YOLO_test' / 'labels'
    marida_patches_dir = base_dir / 'MARIDA' / 'patches'
    output_dir = base_dir / 'annotation_validation'
    output_dir.mkdir(exist_ok=True)
    
    # 选择几个测试样本
    test_samples = [
        'S2_1-12-19_48MYU_0',
        'S2_27-1-19_16QED_1', 
        'S2_4-9-16_16PCC_20'
    ]
    
    print("🔍 开始验证标注准确性...")
    print("=" * 60)
    
    total_yolo = 0
    total_expert = 0
    
    for sample_name in test_samples:
        image_path = test_images_dir / f"{sample_name}.tif"
        label_path = test_labels_dir / f"{sample_name}.txt"
        
        # 从场景名称提取patch路径
        scene_name = '_'.join(sample_name.split('_')[:-1])  # 去掉最后的编号
        patch_idx = sample_name.split('_')[-1]  # 提取编号
        mask_path = marida_patches_dir / scene_name / f"{sample_name}_cl.tif"
        
        if not image_path.exists():
            print(f"⚠️  图像不存在: {image_path}")
            continue
        
        print(f"\n处理样本: {sample_name}")
        
        # 加载YOLO标注
        yolo_labels = load_yolo_labels(label_path)
        print(f"  YOLO标注数量: {len(yolo_labels)}")
        
        # 获取图像尺寸
        with rasterio.open(image_path) as src:
            image_width, image_height = src.width, src.height
        
        # 加载专家标注 (从mask)
        expert_labels = load_expert_labels_from_mask(mask_path, image_width, image_height)
        print(f"  专家标注数量: {len(expert_labels)}")
        
        # 创建对比图
        output_path = output_dir / f"{sample_name}_annotation_comparison.png"
        yolo_count, expert_count = create_comparison_plot(
            image_path, yolo_labels, expert_labels, output_path
        )
        
        total_yolo += yolo_count
        total_expert += expert_count
    
    print("\n" + "=" * 60)
    print("验证完成统计:")
    print(f"  总YOLO标注: {total_yolo}")
    print(f"  总专家标注: {total_expert}")
    print(f"  平均差异: {abs(total_yolo - total_expert) / len(test_samples):.1f}")
    print(f"\n对比图已保存到: {output_dir}")
    print("\n💡 验证要点:")
    print("  1. 红色实线: YOLO自动转换标注")
    print("  2. 蓝色虚线: MARIDA专家mask标注")
    print("  3. 对比位置重合度判断转换质量")
    print("  4. 数量差异反映检测到的目标多少")

if __name__ == '__main__':
    validate_annotations()
