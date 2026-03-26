#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多光谱可视化脚本：展示RGB vs NIR vs 假彩色合成的区别
帮助理解为什么遥感图像需要多通道而不只是RGB
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

def normalize_band(band, percentile=2):
    """标准化波段到0-1，使用百分位拉伸增强对比度"""
    p_low = np.percentile(band, percentile)
    p_high = np.percentile(band, 100 - percentile)
    band_stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
    return band_stretched

def create_multispectral_comparison(img_path, label_path, output_path, class_mapping):
    """创建多光谱对比图：RGB vs NIR vs 假彩色合成"""
    
    # 读取11波段影像
    with rasterio.open(img_path) as src:
        bands = src.read()
        height, width = bands.shape[1], bands.shape[2]
    
    # 提取关键波段（Sentinel-2波段索引）
    # Band 2 (Blue), 3 (Green), 4 (Red), 8 (NIR)
    blue = bands[1]   # Band 2
    green = bands[2]  # Band 3
    red = bands[3]    # Band 4
    nir = bands[7]    # Band 8
    
    # 标准化
    blue_norm = normalize_band(blue)
    green_norm = normalize_band(green)
    red_norm = normalize_band(red)
    nir_norm = normalize_band(nir)
    
    # 创建不同的合成图像
    # 1. 真彩色RGB
    rgb = np.dstack([red_norm, green_norm, blue_norm])
    
    # 2. NIR灰度图
    nir_gray = nir_norm
    
    # 3. 假彩色合成 (NIR-Red-Green)：植被和水体对比强
    false_color = np.dstack([nir_norm, red_norm, green_norm])
    
    # 4. NDVI (归一化植被指数) - 突出浮游物
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi_norm = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-8)
    
    # 读取YOLO标签
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])
                bboxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': w,
                    'height': h
                })
    
    # 类别颜色
    colors = ['#FF0000', '#00FF00', '#0000FF']  # Marine_Debris, Floating_Object, Ship_Wake
    
    # 创建4子图对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f'Multispectral Analysis: {os.path.basename(img_path)}', fontsize=16, fontweight='bold')
    
    # 子图1: RGB真彩色
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('1. RGB True Color (人眼可见光)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 子图2: NIR近红外
    axes[0, 1].imshow(nir_gray, cmap='gray')
    axes[0, 1].set_title('2. NIR (近红外波段 - 机器视角)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 子图3: 假彩色合成
    axes[1, 0].imshow(false_color)
    axes[1, 0].set_title('3. False Color (NIR-R-G合成)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 子图4: NDVI植被指数
    im = axes[1, 1].imshow(ndvi_norm, cmap='RdYlGn')
    axes[1, 1].set_title('4. NDVI (浮游物/植被指数)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 在所有子图上叠加bbox
    for bbox in bboxes:
        class_id = bbox['class_id']
        x_center = bbox['x_center'] * width
        y_center = bbox['y_center'] * height
        w = bbox['width'] * width
        h = bbox['height'] * height
        x_min = x_center - w / 2
        y_min = y_center - h / 2
        
        color = colors[class_id]
        class_name = class_mapping['yolo_classes'][class_id]
        
        for ax in axes.flat:
            rect = Rectangle((x_min, y_min), w, h, 
                           linewidth=2, edgecolor=color, facecolor='none',
                           linestyle='--', alpha=0.8)
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, class_name, 
                   color=color, fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # 添加图例
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='none', ec=colors[i], linewidth=2, linestyle='--',
                 label=class_mapping['yolo_classes'][i])
        for i in range(len(class_mapping['yolo_classes']))
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
              fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.98))
    
    # 添加说明文字
    explanation = (
        "💡 多光谱分析说明：\n"
        "• RGB (左上)：人眼可见范围，海洋垃圾模糊不清\n"
        "• NIR (右上)：近红外波段，水体吸收强，漂浮物反射强\n"
        "• False Color (左下)：假彩色合成，增强水陆对比\n"
        "• NDVI (右下)：植被指数，绿色=浮游生物，红色=纯水体\n"
        "→ 机器学习模型使用RGB+NIR四通道，比人眼看得更清楚！"
    )
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已保存多光谱对比图: {output_path}")

def main():
    # 加载类别映射
    with open('MARIDA_YOLO_test/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    
    # 创建输出目录
    output_dir = 'multispectral_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择有目标的样本进行对比
    test_samples = [
        ('MARIDA_YOLO_test/images/S2_1-12-19_48MYU_0.tif', 
         'MARIDA_YOLO_test/labels/S2_1-12-19_48MYU_0.txt'),
        ('MARIDA_YOLO_test/images/S2_27-1-19_16QED_1.tif',
         'MARIDA_YOLO_test/labels/S2_27-1-19_16QED_1.txt'),
        ('MARIDA_YOLO_test/images/S2_4-9-16_16PCC_20.tif',
         'MARIDA_YOLO_test/labels/S2_4-9-16_16PCC_20.txt'),
    ]
    
    print("🔬 开始生成多光谱对比分析图...")
    print("=" * 60)
    
    for img_path, label_path in test_samples:
        if os.path.exists(img_path):
            output_name = os.path.basename(img_path).replace('.tif', '_multispectral.png')
            output_path = os.path.join(output_dir, output_name)
            
            try:
                create_multispectral_comparison(img_path, label_path, output_path, class_mapping)
            except Exception as e:
                print(f"❌ 处理失败 {img_path}: {e}")
    
    print("=" * 60)
    print(f"✅ 多光谱分析完成！请查看 {output_dir}/ 目录")
    print("\n💡 对比要点：")
    print("  1. RGB图像模糊 → 人眼难以识别")
    print("  2. NIR图像对比强 → 水体vs漂浮物清晰")
    print("  3. NDVI指数 → 突出有机物和植被")
    print("  4. 机器学习模型利用所有通道 → 比人眼更准确！")

if __name__ == '__main__':
    main()
