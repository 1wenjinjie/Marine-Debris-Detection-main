#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证多光谱优化输出的正确性
检查每种波段配置是否生成了正确的通道数和内容
"""

import os
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import cv2
import json

def verify_band_loading(image_path, band_config, config_name):
    """
    验证单个波段配置的加载是否正确
    
    Args:
        image_path: 图像路径
        band_config: 波段配置
        config_name: 配置名称
        
    Returns:
        验证结果字典
    """
    result = {
        'config': config_name,
        'expected_bands': band_config['bands'],
        'status': 'unknown',
        'errors': []
    }
    
    try:
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
            
            # 检查图像是否有足够的波段
            total_bands = src.count
            result['total_available_bands'] = total_bands
            
            # 验证每个需要的波段是否存在
            loaded_bands = []
            for band_name in band_config['bands']:
                if band_name in band_mapping:
                    band_idx = band_mapping[band_name]
                    if band_idx <= total_bands:
                        band_data = src.read(band_idx)
                        loaded_bands.append(band_name)
                        
                        # 检查波段数据质量
                        if np.all(band_data == 0):
                            result['errors'].append(f"Band {band_name} is all zeros")
                        elif np.isnan(band_data).any():
                            result['errors'].append(f"Band {band_name} contains NaN values")
                    else:
                        result['errors'].append(f"Band {band_name} (index {band_idx}) exceeds available bands ({total_bands})")
                else:
                    result['errors'].append(f"Band {band_name} not in mapping")
            
            result['loaded_bands'] = loaded_bands
            result['expected_channels'] = len(band_config['bands'])
            result['loaded_channels'] = len(loaded_bands)
            
            # 如果有光谱指数，增加通道数
            if 'indices' in band_config:
                result['expected_channels'] += len(band_config['indices'])
                result['indices'] = list(band_config['indices'].keys())
            
            # 状态判断
            if len(result['errors']) == 0 and result['loaded_channels'] == len(band_config['bands']):
                result['status'] = 'success'
            elif result['loaded_channels'] > 0:
                result['status'] = 'partial'
            else:
                result['status'] = 'failed'
                
    except Exception as e:
        result['status'] = 'error'
        result['errors'].append(f"Exception: {str(e)}")
    
    return result

def create_verification_report(image_path, output_dir):
    """
    创建详细的验证报告，包括可视化
    """
    from test_multispectral_optimization import create_band_configurations, load_multispectral_image
    
    sample_name = Path(image_path).stem
    configs = create_band_configurations()
    
    # 验证结果
    verification_results = {}
    
    print(f"\n{'='*70}")
    print(f"Verifying: {sample_name}")
    print(f"{'='*70}")
    
    # 检查原始图像信息
    with rasterio.open(image_path) as src:
        print(f"\n📊 Original Image Info:")
        print(f"  • Total bands: {src.count}")
        print(f"  • Data type: {src.dtypes[0]}")
        print(f"  • Shape: {src.width} x {src.height}")
        print(f"  • CRS: {src.crs}")
        
        # 显示每个波段的统计信息
        print(f"\n  Band Statistics:")
        for i in range(1, min(src.count + 1, 7)):  # 最多显示6个波段
            band_data = src.read(i)
            print(f"    Band {i}: min={band_data.min():.2f}, max={band_data.max():.2f}, "
                  f"mean={band_data.mean():.2f}, std={band_data.std():.2f}")
    
    print(f"\n🔍 Testing Band Configurations:")
    print(f"{'-'*70}")
    
    # 测试每个配置
    for config_key, config in configs.items():
        # 基本验证
        basic_result = verify_band_loading(image_path, config, config_key)
        
        # 尝试实际加载
        try:
            image, band_names = load_multispectral_image(image_path, config)
            basic_result['actual_shape'] = image.shape
            basic_result['actual_channels'] = image.shape[2]
            basic_result['band_names'] = band_names
            
            # 检查通道数是否匹配
            if basic_result['actual_channels'] == basic_result['expected_channels']:
                basic_result['channel_match'] = True
            else:
                basic_result['channel_match'] = False
                basic_result['errors'].append(
                    f"Channel mismatch: expected {basic_result['expected_channels']}, "
                    f"got {basic_result['actual_channels']}"
                )
                
            # 检查数据质量
            for i, bname in enumerate(band_names):
                band_data = image[:, :, i]
                if np.all(band_data == 0):
                    basic_result['errors'].append(f"Channel {i} ({bname}) is all zeros")
                elif band_data.max() == band_data.min():
                    basic_result['errors'].append(f"Channel {i} ({bname}) has no variation")
                    
        except Exception as e:
            basic_result['errors'].append(f"Loading failed: {str(e)}")
            basic_result['status'] = 'error'
        
        verification_results[config_key] = basic_result
        
        # 打印结果
        status_symbol = '✅' if basic_result['status'] == 'success' else '⚠️' if basic_result['status'] == 'partial' else '❌'
        print(f"\n{status_symbol} {config_key}: {config['name']}")
        print(f"  Expected: {basic_result['expected_channels']} channels")
        if 'actual_channels' in basic_result:
            print(f"  Actual: {basic_result['actual_channels']} channels")
            print(f"  Band names: {', '.join(basic_result['band_names'])}")
            print(f"  Image shape: {basic_result['actual_shape']}")
        if basic_result['errors']:
            print(f"  Errors:")
            for error in basic_result['errors']:
                print(f"    - {error}")
    
    # 创建可视化对比验证
    create_channel_verification_visualization(
        image_path, configs, verification_results, output_dir, sample_name
    )
    
    # 保存验证报告
    report_path = output_dir / f'{sample_name}_verification_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(verification_results, f, indent=2, ensure_ascii=False)
    
    return verification_results

def create_channel_verification_visualization(image_path, configs, results, output_dir, sample_name):
    """
    创建通道验证的可视化
    """
    from test_multispectral_optimization import load_multispectral_image
    
    # 选择几个关键配置进行详细可视化
    key_configs = ['RGB', 'RGB_NIR', 'RGB_NIR_SWIR', 'RGB_NIR_INDICES', 'ALL_BANDS']
    
    n_configs = len(key_configs)
    fig, axes = plt.subplots(n_configs, 4, figsize=(20, 5*n_configs))
    
    fig.suptitle(f'Channel Verification: {sample_name}', fontsize=16, fontweight='bold')
    
    for idx, config_key in enumerate(key_configs):
        if config_key not in configs:
            continue
            
        config = configs[config_key]
        result = results.get(config_key, {})
        
        try:
            image, band_names = load_multispectral_image(image_path, config)
            
            # 显示前3个通道的RGB合成
            ax = axes[idx, 0]
            if image.shape[2] >= 3:
                rgb_composite = image[:, :, :3]
                ax.imshow(rgb_composite)
            else:
                ax.imshow(image[:, :, 0], cmap='gray')
            ax.set_title(f'{config["name"]}\nRGB Composite')
            ax.axis('off')
            
            # 显示第4通道 (通常是NIR)
            ax = axes[idx, 1]
            if image.shape[2] >= 4:
                ax.imshow(image[:, :, 3], cmap='RdYlGn')
                ax.set_title(f'Channel 4: {band_names[3] if len(band_names) > 3 else "N/A"}')
            else:
                ax.text(0.5, 0.5, 'No Channel 4', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Channel 4: N/A')
            ax.axis('off')
            
            # 显示第5通道 (SWIR1 or Index)
            ax = axes[idx, 2]
            if image.shape[2] >= 5:
                ax.imshow(image[:, :, 4], cmap='viridis')
                ax.set_title(f'Channel 5: {band_names[4] if len(band_names) > 4 else "N/A"}')
            else:
                ax.text(0.5, 0.5, 'No Channel 5', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Channel 5: N/A')
            ax.axis('off')
            
            # 显示配置信息
            ax = axes[idx, 3]
            ax.axis('off')
            
            info_text = f"Configuration: {config_key}\n"
            info_text += f"Description: {config['description']}\n\n"
            info_text += f"Expected Channels: {result.get('expected_channels', 'N/A')}\n"
            info_text += f"Actual Channels: {result.get('actual_channels', 'N/A')}\n\n"
            info_text += f"Band Names:\n"
            for i, bname in enumerate(band_names[:8]):  # 最多显示8个
                info_text += f"  {i+1}. {bname}\n"
            
            if result.get('errors'):
                info_text += f"\nErrors:\n"
                for error in result['errors'][:3]:  # 最多显示3个错误
                    info_text += f"  - {error[:40]}...\n"
            
            status = result.get('status', 'unknown')
            color = 'green' if status == 'success' else 'orange' if status == 'partial' else 'red'
            
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
            
        except Exception as e:
            for col in range(4):
                ax = axes[idx, col]
                ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
                ax.axis('off')
    
    plt.tight_layout()
    output_path = output_dir / f'{sample_name}_channel_verification.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Verification visualization saved: {output_path}")

def run_verification():
    """运行完整的验证流程"""
    
    print("🔍 Starting Band Configuration Verification")
    print("="*70)
    
    # 路径设置
    base_dir = Path('/Users/wen11/Documents/OceanEdge_Group/Marine_dataset')
    test_images_dir = base_dir / 'MARIDA_YOLO_test' / 'images'
    output_dir = base_dir / 'multispectral_verification'
    output_dir.mkdir(exist_ok=True)
    
    # 选择几个代表性样本进行详细验证
    test_samples = [
        'S2_1-12-19_48MYU_0.tif',
        'S2_27-1-19_16QED_1.tif',
        'S2_4-9-16_16PCC_20.tif'
    ]
    
    all_results = {}
    
    for sample_file in test_samples:
        sample_path = test_images_dir / sample_file
        if not sample_path.exists():
            print(f"⚠️  Sample not found: {sample_path}")
            continue
        
        results = create_verification_report(sample_path, output_dir)
        all_results[sample_file] = results
    
    # 总结报告
    print(f"\n{'='*70}")
    print("📊 VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    total_configs = 0
    successful_configs = 0
    partial_configs = 0
    failed_configs = 0
    
    for sample, results in all_results.items():
        print(f"\n{sample}:")
        for config_key, result in results.items():
            total_configs += 1
            status = result['status']
            if status == 'success':
                successful_configs += 1
                symbol = '✅'
            elif status == 'partial':
                partial_configs += 1
                symbol = '⚠️'
            else:
                failed_configs += 1
                symbol = '❌'
            
            match = '✓' if result.get('channel_match', False) else '✗'
            print(f"  {symbol} {config_key}: {result.get('actual_channels', '?')}/{result.get('expected_channels', '?')} channels {match}")
    
    print(f"\n{'='*70}")
    print(f"Overall Results:")
    print(f"  Total configurations tested: {total_configs}")
    print(f"  ✅ Successful: {successful_configs}")
    print(f"  ⚠️  Partial: {partial_configs}")
    print(f"  ❌ Failed: {failed_configs}")
    print(f"  Success rate: {successful_configs/total_configs*100:.1f}%")
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("  • Verification reports: *_verification_report.json")
    print("  • Channel visualizations: *_channel_verification.png")

if __name__ == '__main__':
    run_verification()
