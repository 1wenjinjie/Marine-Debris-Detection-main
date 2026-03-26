#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多光谱输入通道优化策略
比较不同波段组合对海洋垃圾检测的影响
"""

import os
import numpy as np
import rasterio
from rasterio.transform import Affine
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_multispectral_image(image_path, band_config):
    """
    加载多光谱图像，支持不同波段组合

    Args:
        image_path: 图像路径
        band_config: 波段配置字典

    Returns:
        image: 多通道图像数组
        band_names: 波段名称列表
    """
    with rasterio.open(image_path) as src:
        # Sentinel-2波段映射
        band_mapping = {
            'B02': 1,  # Blue
            'B03': 2,  # Green
            'B04': 3,  # Red
            'B08': 4,  # NIR
            'B11': 5,  # SWIR1
            'B12': 6,  # SWIR2
        }

        selected_bands = []
        band_names = []

        for band_name in band_config['bands']:
            if band_name in band_mapping:
                band_idx = band_mapping[band_name]
                if band_idx <= src.count:
                    band_data = src.read(band_idx)
                    selected_bands.append(band_data)
                    band_names.append(band_name)

        if not selected_bands:
            raise ValueError(f"No valid bands found for config: {band_config}")

        # 堆叠波段
        image = np.stack(selected_bands, axis=-1)

        # 应用指数计算
        if 'indices' in band_config:
            for idx_name, idx_func in band_config['indices'].items():
                index_band = idx_func(image)
                image = np.concatenate([image, index_band[:, :, np.newaxis]], axis=-1)
                band_names.append(idx_name)

        # 标准化
        for i in range(image.shape[2]):
            p2, p98 = np.percentile(image[:, :, i], (2, 98))
            image[:, :, i] = np.clip((image[:, :, i] - p2) / (p98 - p2) * 255, 0, 255)

        return image.astype(np.uint8), band_names

def calculate_ndvi(image):
    """计算NDVI (红外-红)/(红外+红)"""
    nir = image[:, :, 3]  # 假设NIR在第4通道
    red = image[:, :, 2]  # 假设红在第3通道
    return (nir.astype(float) - red.astype(float)) / (nir.astype(float) + red.astype(float) + 1e-8)

def calculate_ndwi(image):
    """计算NDWI (绿-红外)/(绿+红外)"""
    green = image[:, :, 1]  # 假设绿在第2通道
    nir = image[:, :, 3]   # 假设NIR在第4通道
    return (green.astype(float) - nir.astype(float)) / (green.astype(float) + nir.astype(float) + 1e-8)

def calculate_fdi(image):
    """计算浮游藻指数 FDI (红外-红)/(红外+红) - 类似NDVI但用于水体"""
    nir = image[:, :, 3]
    red = image[:, :, 2]
    return (nir.astype(float) - red.astype(float)) / (nir.astype(float) + red.astype(float) + 1e-8)

def create_band_configurations():
    """创建不同的波段配置进行测试"""

    configs = {
        'RGB': {
            'bands': ['B04', 'B03', 'B02'],  # R, G, B
            'name': 'RGB (3通道)',
            'description': '传统RGB输入'
        },

        'RGB_NIR': {
            'bands': ['B04', 'B03', 'B02', 'B08'],  # R, G, B, NIR
            'name': 'RGB+NIR (4通道)',
            'description': '当前使用的配置'
        },

        'RGB_SWIR': {
            'bands': ['B04', 'B03', 'B02', 'B11', 'B12'],  # R, G, B, SWIR1, SWIR2
            'name': 'RGB+SWIR (5通道)',
            'description': '添加SWIR波段区分材料'
        },

        'RGB_NIR_SWIR': {
            'bands': ['B04', 'B03', 'B02', 'B08', 'B11', 'B12'],  # R, G, B, NIR, SWIR1, SWIR2
            'name': 'RGB+NIR+SWIR (6通道)',
            'description': '完整多光谱输入'
        },

        'RGB_NIR_NDVI': {
            'bands': ['B04', 'B03', 'B02', 'B08'],
            'indices': {
                'NDVI': calculate_ndvi
            },
            'name': 'RGB+NIR+NDVI (5通道)',
            'description': '添加植被指数'
        },

        'RGB_NIR_NDWI': {
            'bands': ['B04', 'B03', 'B02', 'B08'],
            'indices': {
                'NDWI': calculate_ndwi
            },
            'name': 'RGB+NIR+NDWI (5通道)',
            'description': '添加水体指数'
        },

        'RGB_NIR_INDICES': {
            'bands': ['B04', 'B03', 'B02', 'B08'],
            'indices': {
                'NDVI': calculate_ndvi,
                'NDWI': calculate_ndwi,
                'FDI': calculate_fdi
            },
            'name': 'RGB+NIR+3指数 (7通道)',
            'description': '添加多个光谱指数'
        },

        'ALL_BANDS': {
            'bands': ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],  # B, G, R, NIR, SWIR1, SWIR2
            'indices': {
                'NDVI': calculate_ndvi,
                'NDWI': calculate_ndwi
            },
            'name': '全波段+指数 (8通道)',
            'description': '最大信息量配置'
        }
    }

    return configs

def create_comparison_visualization(image_path, configs, output_dir):
    """创建不同配置的对比可视化"""

    sample_name = Path(image_path).stem
    output_dir.mkdir(exist_ok=True)

    # 为每个配置创建子图
    n_configs = len(configs)
    n_cols = min(4, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f'Multispectral Input Configuration Comparison: {sample_name}', fontsize=16, fontweight='bold')

    results = {}

    for idx, (config_key, config) in enumerate(configs.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        try:
            # 加载图像
            image, band_names = load_multispectral_image(image_path, config)

            # 创建RGB合成图用于显示
            if image.shape[2] >= 3:
                # NIR-R-G合成 (如果有NIR)
                if 'B08' in band_names:
                    nir_idx = band_names.index('B08')
                    red_idx = band_names.index('B04') if 'B04' in band_names else 0
                    green_idx = band_names.index('B03') if 'B03' in band_names else 1

                    display_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                    display_img[:, :, 0] = image[:, :, nir_idx]    # R: NIR
                    display_img[:, :, 1] = image[:, :, red_idx]    # G: Red
                    display_img[:, :, 2] = image[:, :, green_idx]  # B: Green
                else:
                    # 普通RGB
                    display_img = image[:, :, :3]
            else:
                # 单通道灰度图
                display_img = cv2.cvtColor(image[:, :, :1], cv2.COLOR_GRAY2RGB)

            ax.imshow(display_img)
            ax.set_title(f'{config["name"]}\n({image.shape[2]} channels)', fontsize=10)
            ax.axis('off')

            results[config_key] = {
                'success': True,
                'channels': image.shape[2],
                'band_names': band_names,
                'config': config
            }

        except Exception as e:
            ax.text(0.5, 0.5, f'Loading Failed\n{str(e)[:50]}...',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
            ax.set_title(f'{config["name"]}\n(Failed)', fontsize=10, color='red')
            ax.axis('off')

            results[config_key] = {
                'success': False,
                'error': str(e),
                'config': config
            }

    # 隐藏多余的子图
    for idx in range(n_configs, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    output_path = output_dir / f'{sample_name}_band_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return results

def analyze_band_effectiveness(results, output_dir):
    """分析不同波段配置的有效性"""

    analysis = {
        'total_configs': len(results),
        'successful_configs': sum(1 for r in results.values() if r['success']),
        'channel_counts': [r['channels'] for r in results.values() if r['success']],
        'config_performance': {}
    }

    # 分析每个配置
    for config_key, result in results.items():
        if result['success']:
            analysis['config_performance'][config_key] = {
                'channels': result['channels'],
                'bands': result['band_names'],
                'description': result['config']['description']
            }

    # 保存分析结果
    analysis_path = output_dir / 'band_analysis.json'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return analysis

def run_multispectral_optimization():
    """运行多光谱输入优化测试"""

    print("🚀 开始多光谱输入优化测试...")
    print("=" * 70)

    # 路径设置
    base_dir = Path('/Users/wen11/Documents/OceanEdge_Group/Marine_dataset')
    test_images_dir = base_dir / 'MARIDA_YOLO_test' / 'images'
    output_dir = base_dir / 'multispectral_optimization'
    output_dir.mkdir(exist_ok=True)

    # 测试样本 - 增加更多样本以获得更好的统计
    test_samples = [
        # S2_1-12-19_48MYU 场景
        'S2_1-12-19_48MYU_0.tif',
        'S2_1-12-19_48MYU_1.tif',
        'S2_1-12-19_48MYU_2.tif',
        'S2_1-12-19_48MYU_3.tif',
        
        # S2_27-1-19_16QED 场景
        'S2_27-1-19_16QED_0.tif',
        'S2_27-1-19_16QED_1.tif',
        'S2_27-1-19_16QED_2.tif',
        'S2_27-1-19_16QED_5.tif',
        'S2_27-1-19_16QED_10.tif',
        
        # S2_4-9-16_16PCC 场景
        'S2_4-9-16_16PCC_0.tif',
        'S2_4-9-16_16PCC_10.tif',
        'S2_4-9-16_16PCC_20.tif',
        'S2_4-9-16_16PCC_30.tif',
        'S2_4-9-16_16PCC_40.tif'
    ]

    # 创建波段配置
    configs = create_band_configurations()

    print(f"📊 测试 {len(configs)} 种波段配置:")
    for key, config in configs.items():
        print(f"  • {key}: {config['description']}")

    print(f"\n🖼️  测试 {len(test_samples)} 个样本图像")

    all_results = {}

    for sample_file in test_samples:
        sample_path = test_images_dir / sample_file
        if not sample_path.exists():
            print(f"⚠️  样本不存在: {sample_path}")
            continue

        print(f"\n🔍 处理样本: {sample_file}")
        sample_key = sample_file.replace('.tif', '')

        # 创建对比可视化
        results = create_comparison_visualization(sample_path, configs, output_dir)
        all_results[sample_key] = results

        # 统计成功率
        success_count = sum(1 for r in results.values() if r['success'])
        print(f"  ✅ 成功配置: {success_count}/{len(configs)}")

    # 总体分析
    print(f"\n📈 总体分析...")
    analysis = analyze_band_effectiveness(all_results[list(all_results.keys())[0]], output_dir)

    print("\n📊 分析结果:")
    print(f"  • 总配置数: {analysis['total_configs']}")
    print(f"  • 成功配置数: {analysis['successful_configs']}")
    print(f"  • 通道数范围: {min(analysis['channel_counts'])} - {max(analysis['channel_counts'])}")

    print("\n🎯 推荐配置:")
    recommendations = [
        ("RGB_NIR", "基础多光谱，平衡性能"),
        ("RGB_NIR_SWIR", "最佳区分度，推荐使用"),
        ("RGB_NIR_INDICES", "添加光谱指数，增强特征")
    ]

    for config, reason in recommendations:
        if config in analysis['config_performance']:
            perf = analysis['config_performance'][config]
            print(f"  • {config} ({perf['channels']}通道): {reason}")

    print(f"\n💾 结果保存到: {output_dir}")
    print("  • 对比图: *_band_comparison.png")
    print("  • 分析报告: band_analysis.json")

if __name__ == '__main__':
    run_multispectral_optimization()
