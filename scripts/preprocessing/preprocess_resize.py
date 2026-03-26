"""
Resize MARIDA images and masks from 256x256 to 512x512.
This script processes all patches and their corresponding labels.
"""

import os
import numpy as np
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

def resize_image(input_path, output_path, target_size=(512, 512)):
    """
    Resize a multi-band raster image using bilinear interpolation.
    
    Args:
        input_path: Path to input .tif file
        output_path: Path to output .tif file
        target_size: Tuple of (height, width)
    """
    with rasterio.open(input_path) as src:
        # Calculate scale factors
        scale_y = target_size[0] / src.height
        scale_x = target_size[1] / src.width
        
        # Read and resample data
        data = src.read(
            out_shape=(src.count, target_size[0], target_size[1]),
            resampling=Resampling.bilinear
        )
        
        # Update metadata
        transform = src.transform * src.transform.scale(
            (src.width / target_size[1]),
            (src.height / target_size[0])
        )
        
        meta = src.meta.copy()
        meta.update({
            'height': target_size[0],
            'width': target_size[1],
            'transform': transform,
            'compress': 'lzw',  # LZW压缩以节省磁盘空间
            'tiled': True,      # 使用瓦片存储提高I/O性能
            'blockxsize': 256,
            'blockysize': 256
        })
        
        # Write output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data)

def resize_mask(input_path, output_path, target_size=(512, 512)):
    """
    Resize a label mask using nearest neighbor interpolation.
    
    Args:
        input_path: Path to input label .tif file
        output_path: Path to output label .tif file
        target_size: Tuple of (height, width)
    """
    with rasterio.open(input_path) as src:
        # Read and resample using nearest neighbor for labels
        data = src.read(
            out_shape=(src.count, target_size[0], target_size[1]),
            resampling=Resampling.nearest
        )
        
        # Update metadata
        transform = src.transform * src.transform.scale(
            (src.width / target_size[1]),
            (src.height / target_size[0])
        )
        
        meta = src.meta.copy()
        meta.update({
            'height': target_size[0],
            'width': target_size[1],
            'transform': transform,
            'compress': 'lzw',  # LZW压缩以节省磁盘空间
            'tiled': True,      # 使用瓦片存储提高I/O性能
            'blockxsize': 256,
            'blockysize': 256
        })
        
        # Write output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data)

def main():
    # Paths
    base_dir = Path('D:/OceanEdge_Group/Marine-Debris-Detection-main/MARIDA')
    patches_dir = base_dir / 'patches'
    output_base = Path('D:/OceanEdge_Group/Marine-Debris-Detection-main/MARIDA_resized')
    
    target_size = (512, 512)
    
    # Get all scene directories
    scene_dirs = [d for d in patches_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(scene_dirs)} scenes to process")
    print(f"Resizing to {target_size[0]}x{target_size[1]}")
    
    # Process each scene
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_name = scene_dir.name
        
        # Process S2 image
        s2_file = scene_dir / f"{scene_name}_S2.tif"
        if s2_file.exists():
            output_s2 = output_base / 'patches' / scene_name / f"{scene_name}_S2.tif"
            resize_image(s2_file, output_s2, target_size)
        
        # Process label
        label_file = scene_dir / f"{scene_name}_cl.tif"
        if label_file.exists():
            output_label = output_base / 'patches' / scene_name / f"{scene_name}_cl.tif"
            resize_mask(label_file, output_label, target_size)
        
        # Process confidence (if needed)
        conf_file = scene_dir / f"{scene_name}_conf.tif"
        if conf_file.exists():
            output_conf = output_base / 'patches' / scene_name / f"{scene_name}_conf.tif"
            resize_mask(conf_file, output_conf, target_size)
    
    print(f"\nResizing complete! Output saved to: {output_base}")
    
    # Verify a sample
    sample_scene = scene_dirs[0].name
    sample_output = output_base / 'patches' / sample_scene / f"{sample_scene}_S2.tif"
    if sample_output.exists():
        with rasterio.open(sample_output) as src:
            print(f"\nVerification - Sample image shape: {src.shape}")
            print(f"Expected: {target_size}")

if __name__ == "__main__":
    main()
