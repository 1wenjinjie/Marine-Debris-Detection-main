"""
MARIDA数据集影像显示脚本
读取多波段.tif文件并合成RGB真彩色图像进行显示
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import glob

def read_tif_image(tif_path):
    """
    读取.tif文件的所有波段
    
    参数:
        tif_path: .tif文件路径
    
    返回:
        numpy数组，形状为(波段数, 高度, 宽度)
    """
    dataset = gdal.Open(tif_path)
    if dataset is None:
        raise ValueError(f"无法打开文件: {tif_path}")
    
    # 获取波段数
    band_count = dataset.RasterCount
    
    # 读取所有波段
    bands = []
    for i in range(1, band_count + 1):
        band = dataset.GetRasterBand(i)
        band_array = band.ReadAsArray()
        bands.append(band_array)
    
    # 堆叠成三维数组
    image = np.stack(bands, axis=0)
    
    dataset = None  # 关闭数据集
    
    return image

def create_rgb_image(multi_band_image, rgb_bands=(2, 1, 0)):
    """
    从多波段影像中提取RGB波段并进行归一化
    
    参数:
        multi_band_image: 多波段影像数组，形状为(波段数, 高度, 宽度)
        rgb_bands: RGB对应的波段索引 (R, G, B)，默认为(2, 1, 0)
                   对于Sentinel-2: B4(红)、B3(绿)、B2(蓝)通常是前三个波段
    
    返回:
        RGB图像，形状为(高度, 宽度, 3)，值范围[0, 1]
    """
    # 提取RGB波段
    r = multi_band_image[rgb_bands[0]]
    g = multi_band_image[rgb_bands[1]]
    b = multi_band_image[rgb_bands[2]]
    
    # 堆叠成RGB图像
    rgb = np.stack([r, g, b], axis=-1)
    
    # 归一化到0-1范围
    # 使用2%和98%分位数进行拉伸，增强对比度
    percentile_2 = np.percentile(rgb, 2)
    percentile_98 = np.percentile(rgb, 98)
    
    rgb_normalized = np.clip((rgb - percentile_2) / (percentile_98 - percentile_2), 0, 1)
    
    return rgb_normalized

def display_image(tif_path, rgb_bands=(2, 1, 0), save_path=None):
    """
    显示单张影像的RGB真彩色图像
    
    参数:
        tif_path: .tif文件路径
        rgb_bands: RGB对应的波段索引
        save_path: 保存图像的路径（可选）
    """
    print(f"正在读取: {tif_path}")
    
    # 读取影像
    multi_band_image = read_tif_image(tif_path)
    print(f"影像形状: {multi_band_image.shape} (波段数, 高度, 宽度)")
    
    # 创建RGB图像
    rgb_image = create_rgb_image(multi_band_image, rgb_bands)
    
    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.title(os.path.basename(tif_path))
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"图像已保存至: {save_path}")
    
    plt.show()

def display_multiple_images(tif_paths, rgb_bands=(2, 1, 0), cols=3):
    """
    显示多张影像的RGB真彩色图像
    
    参数:
        tif_paths: .tif文件路径列表
        rgb_bands: RGB对应的波段索引
        cols: 每行显示的图像数量
    """
    num_images = len(tif_paths)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, tif_path in enumerate(tif_paths):
        print(f"正在读取 [{idx+1}/{num_images}]: {tif_path}")
        
        try:
            # 读取影像
            multi_band_image = read_tif_image(tif_path)
            
            # 创建RGB图像
            rgb_image = create_rgb_image(multi_band_image, rgb_bands)
            
            # 显示图像
            axes[idx].imshow(rgb_image)
            axes[idx].set_title(os.path.basename(tif_path), fontsize=8)
            axes[idx].axis('off')
        except Exception as e:
            print(f"处理 {tif_path} 时出错: {e}")
            axes[idx].text(0.5, 0.5, f'Error\n{os.path.basename(tif_path)}', 
                          ha='center', va='center')
            axes[idx].axis('off')
    
    # 隐藏多余的子图
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    # 设置数据集路径
    dataset_path = "/Users/wen11/Documents/OceanEdge_Group/Marine_dataset/MARIDA"
    patches_path = os.path.join(dataset_path, "patches")
    
    # 示例1: 显示单张影像
    print("="*60)
    print("示例1: 显示单张影像")
    print("="*60)
    
    # 获取第一个文件夹中的第一张影像（非_cl和_conf文件）
    first_folder = os.path.join(patches_path, "S2_1-12-19_48MYU")
    tif_files = [f for f in glob.glob(os.path.join(first_folder, "*.tif")) 
                 if not f.endswith('_cl.tif') and not f.endswith('_conf.tif')]
    
    if tif_files:
        display_image(tif_files[0])
    
    # 示例2: 显示同一文件夹中的多张影像
    print("\n" + "="*60)
    print("示例2: 显示同一文件夹中的前6张影像")
    print("="*60)
    
    display_multiple_images(tif_files[:6], cols=3)
    
    # 示例3: 显示不同文件夹中的影像
    print("\n" + "="*60)
    print("示例3: 显示不同场景的影像")
    print("="*60)
    
    # 获取不同文件夹中的影像
    folders = [f for f in os.listdir(patches_path) 
               if os.path.isdir(os.path.join(patches_path, f))][:3]
    
    sample_images = []
    for folder in folders:
        folder_path = os.path.join(patches_path, folder)
        tif_files = [f for f in glob.glob(os.path.join(folder_path, "*.tif")) 
                     if not f.endswith('_cl.tif') and not f.endswith('_conf.tif')]
        if tif_files:
            sample_images.append(tif_files[0])
    
    if sample_images:
        display_multiple_images(sample_images, cols=3)

if __name__ == "__main__":
    print("MARIDA数据集影像显示工具")
    print("=" * 60)
    main()
