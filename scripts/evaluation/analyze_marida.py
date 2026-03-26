import os
import pandas as pd
from collections import Counter

# 数据集根目录
dataset_root = 'MARIDA'

# 路径
patches_dir = os.path.join(dataset_root, 'patches')
shapefiles_dir = os.path.join(dataset_root, 'shapefiles')
splits_dir = os.path.join(dataset_root, 'splits')
labels_mapping_file = os.path.join(dataset_root, 'labels_mapping.txt')

def analyze_dataset():
    print("=== MARIDA 数据集结构分析 ===\n")

    # 1. 标签映射
    print("1. 标签映射 (labels_mapping.txt):")
    if os.path.exists(labels_mapping_file):
        with open(labels_mapping_file, 'r') as f:
            content = f.read().strip()
            if content:
                print(content)
            else:
                print("文件为空。")
    else:
        print("文件不存在。")
    print()

    # 2. Patches 分析
    print("2. Patches 目录分析:")
    if os.path.exists(patches_dir):
        scenes = [d for d in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, d))]
        print(f"场景数量: {len(scenes)}")
        print("示例场景:", scenes[:5])

        total_patches = 0
        patch_types = Counter()
        for scene in scenes:
            scene_path = os.path.join(patches_dir, scene)
            files = os.listdir(scene_path)
            patches_in_scene = len([f for f in files if f.endswith('.tif')]) // 3  # 每个补丁有3个文件
            total_patches += patches_in_scene
            for f in files:
                if '_cl.tif' in f:
                    patch_types['classification'] += 1
                elif '_conf.tif' in f:
                    patch_types['confidence'] += 1
                elif '.tif' in f and not '_cl' in f and not '_conf' in f:
                    patch_types['image'] += 1

        print(f"总补丁数量: {total_patches}")
        print(f"文件类型统计: {dict(patch_types)}")
    else:
        print("patches 目录不存在。")
    print()

    # 3. Shapefiles 分析
    print("3. Shapefiles 目录分析:")
    if os.path.exists(shapefiles_dir):
        shp_files = [f for f in os.listdir(shapefiles_dir) if f.endswith('.shp')]
        print(f"Shapefile 数量: {len(shp_files)}")
        print("示例 Shapefile:", shp_files[:5])

        # 简单统计文件大小
        if shp_files:
            example_shp = os.path.join(shapefiles_dir, shp_files[0])
            size = os.path.getsize(example_shp)
            print(f"示例 Shapefile 大小: {size} bytes")
    else:
        print("shapefiles 目录不存在。")
    print()

    # 4. Splits 分析
    print("4. Splits 目录分析:")
    if os.path.exists(splits_dir):
        split_files = os.listdir(splits_dir)
        print(f"分割文件: {split_files}")

        for split_file in split_files:
            split_path = os.path.join(splits_dir, split_file)
            with open(split_path, 'r') as f:
                lines = f.readlines()
                print(f"{split_file}: {len(lines)} 个补丁")
    else:
        print("splits 目录不存在。")
    print()

    # 5. 总体统计
    print("5. 总体统计:")
    print("- 数据集类型: 弱监督像素级语义分割")
    print("- 卫星: Sentinel-2")
    print("- 目标: 海洋垃圾检测 + 海域特征")
    print("- 图像格式: GeoTIFF (.tif)")
    print("- 标注格式: Shapefile (.shp) + 分类掩码 (_cl.tif)")
    print("- 分割: 训练/验证/测试")

if __name__ == "__main__":
    analyze_dataset()</content>
<parameter name="filePath">d:\OceanEdge_Group\Marine-Debris-Detection-main\analyze_marida.py