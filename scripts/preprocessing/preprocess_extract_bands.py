"""
Extract 3/6-channel images (RGB or RGB + NIR + SWIR) from 11-band Sentinel-2 data.
Bands: B4 (Red), B3 (Green), B2 (Blue), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
"""

import os
import numpy as np
from pathlib import Path
import rasterio
from tqdm import tqdm
import argparse
import concurrent.futures
import warnings

# Sentinel-2 band mapping (1-indexed in file)
# Band 1: B1 (Coastal aerosol)
# Band 2: B2 (Blue)
# Band 3: B3 (Green)
# Band 4: B4 (Red)
# Band 5: B5 (Red edge 1)
# Band 6: B6 (Red edge 2)
# Band 7: B7 (Red edge 3)
# Band 8: B8 (NIR)
# Band 9: B8A (Narrow NIR)
# Band 10: B11 (SWIR 1)
# Band 11: B12 (SWIR 2)

# RGB + NIR + SWIR indices (0-indexed in numpy array)
BAND_INDICES = {
    'blue': 1,   # B2
    'green': 2,  # B3
    'red': 3,    # B4
    'nir': 7,    # B8
    'swir1': 9,  # B11
    'swir2': 10  # B12
}

def extract_bands(input_path, output_path, bands=6, normalize=False, nodata=None, normalize_threshold=2.0):
    """
    Extract RGB or RGB+NIR+SWIR bands from 11-band Sentinel-2 image.
    
    Args:
        input_path: Path to input 11-band .tif file
        output_path: Path to output .tif file
        bands: Number of bands to extract (3 for RGB, 6 for RGB+NIR+SWIR)
        normalize: Whether to normalize to 0-1 range
        nodata: NoData value to set
    """
    with rasterio.open(input_path) as src:
        # 基本校验：确保至少有 11 个波段
        if src.count < 11:
            print(f"[Skip] {input_path} has {src.count} bands (<11)")
            return False

        # Read specific bands (1-indexed for rasterio)
        blue = src.read(BAND_INDICES['blue'] + 1)
        green = src.read(BAND_INDICES['green'] + 1)
        red = src.read(BAND_INDICES['red'] + 1)
        
        if bands == 6:
            nir = src.read(BAND_INDICES['nir'] + 1)
            swir1 = src.read(BAND_INDICES['swir1'] + 1)
            swir2 = src.read(BAND_INDICES['swir2'] + 1)
            # Stack in RGB-NIR-SWIR order
            data = np.stack([red, green, blue, nir, swir1, swir2], axis=0)
        elif bands == 3:
            # Stack in RGB order
            data = np.stack([red, green, blue], axis=0)
        else:
            raise ValueError("bands must be 3 or 6")
        
        data = data.astype('float32')

        def needs_normalization(raw_dtype: str, max_value: float) -> bool:
            """Heuristically determine whether normalization should run."""
            is_float = np.issubdtype(np.dtype(raw_dtype), np.floating)
            if is_float and max_value <= normalize_threshold:
                return False
            if max_value <= normalize_threshold:
                return False
            return True

        apply_norm = normalize
        max_value = float(data.max()) if normalize else None
        if normalize and not needs_normalization(src.dtypes[0], max_value):
            apply_norm = False
            print(
                f"[Skip normalize] {input_path.name}: dtype={src.dtypes[0]}, max={max_value:.4f}"
            )

        if apply_norm:
            data /= 10000.0
            data = np.clip(data, 0, 1)
        
        # Update metadata
        meta = src.meta.copy()
        meta.update({
            'count': bands,
            'dtype': 'float32'
        })
        if nodata is not None:
            meta['nodata'] = nodata
        
        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data.astype('float32'))
            if bands == 6:
                dst.set_band_description(1, 'Red (B4)')
                dst.set_band_description(2, 'Green (B3)')
                dst.set_band_description(3, 'Blue (B2)')
                dst.set_band_description(4, 'NIR (B8)')
                dst.set_band_description(5, 'SWIR1 (B11)')
                dst.set_band_description(6, 'SWIR2 (B12)')
            elif bands == 3:
                dst.set_band_description(1, 'Red (B4)')
                dst.set_band_description(2, 'Green (B3)')
                dst.set_band_description(3, 'Blue (B2)')
    return True

def process_scene(scene_dir, input_patches, output_base, bands, normalize, nodata, normalize_threshold):
    """Process a single scene directory."""
    scene_name = scene_dir.name
    processed = 0
    skipped = 0
    copied_labels = 0
    
    # 匹配该场景下的所有影像（排除 _cl 和 _conf）
    image_files = [p for p in scene_dir.glob('*.tif')
                   if not p.name.endswith('_cl.tif') and not p.name.endswith('_conf.tif')]
    if not image_files:
        return 0, 1, 0  # processed, skipped, copied_labels
    
    for img_path in image_files:
        # 输出文件名：原名 + _band.tif
        suffix = f"_{bands}band" if bands == 6 else "_rgb"
        out_name = f"{img_path.stem}{suffix}.tif"
        output_s2 = output_base / 'patches' / scene_name / out_name
        ok = extract_bands(
            img_path,
            output_s2,
            bands=bands,
            normalize=normalize,
            nodata=nodata,
            normalize_threshold=normalize_threshold,
        )
        if ok:
            processed += 1
            
            # 复制对应标签（同名 + _cl.tif）
            label_path = scene_dir / f"{img_path.stem}_cl.tif"
            if label_path.exists():
                with rasterio.open(label_path) as src:
                    meta = src.meta.copy()
                    data = src.read()
                    output_label = output_base / 'patches' / scene_name / label_path.name
                    output_label.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(output_label, 'w', **meta) as dst:
                        dst.write(data)
                copied_labels += 1
        else:
            skipped += 1
    
    return processed, skipped, copied_labels

def main():
    parser = argparse.ArgumentParser(description='Extract RGB or RGB+NIR+SWIR bands from Sentinel-2 images.')
    parser.add_argument('--input-dir', type=str, help='Input patches directory')
    parser.add_argument('--output-dir', type=str, help='Output base directory')
    parser.add_argument('--bands', type=int, choices=[3, 6], default=6, help='Number of bands to extract (3 for RGB, 6 for RGB+NIR+SWIR)')
    parser.add_argument('--normalize', action='store_true', help='Normalize to 0-1 range')
    parser.add_argument('--nodata', type=float, help='NoData value to set')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--normalize-threshold', type=float, default=2.0,
                        help='If max value <= threshold (and dtype float), skip extra normalization')
    args = parser.parse_args()
    
    # Default paths if not provided
    base_dir = Path('D:/OceanEdge_Group/Marine-Debris-Detection-main')
    if not args.input_dir:
        resized_dir = base_dir / 'MARIDA_resized' / 'patches'
        if resized_dir.exists():
            input_patches = resized_dir
            print("Using resized 512x512 images")
        else:
            input_patches = base_dir / 'MARIDA' / 'patches'
            print("Using original 256x256 images")
    else:
        input_patches = Path(args.input_dir)
    
    if not args.output_dir:
        suffix = "_6band" if args.bands == 6 else "_rgb"
        if input_patches.name == 'patches' and input_patches.parent.name == 'MARIDA_resized':
            output_base = base_dir / f'MARIDA{suffix}'
        else:
            output_base = base_dir / f'MARIDA{suffix}_256'
    else:
        output_base = Path(args.output_dir)
    
    if not input_patches.exists():
        print(f"[Error] Input patches not found: {input_patches}")
        return
    
    # Get all scene directories
    scene_dirs = [d for d in input_patches.iterdir() if d.is_dir()]
    
    print(f"Found {len(scene_dirs)} scenes to process")
    band_desc = "RGB+NIR+SWIR" if args.bands == 6 else "RGB"
    print(f"Extracting {args.bands} bands ({band_desc})")
    if args.normalize:
        print("Normalizing to 0-1 range")
    if args.nodata is not None:
        print(f"Setting NoData value to {args.nodata}")
    
    total_processed = 0
    total_skipped = 0
    total_copied_labels = 0
    
    # Process scenes
    if args.parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [
                executor.submit(
                    process_scene,
                    scene_dir,
                    input_patches,
                    output_base,
                    args.bands,
                    args.normalize,
                    args.nodata,
                    args.normalize_threshold,
                )
                for scene_dir in scene_dirs
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(scene_dirs), desc="Extracting bands"):
                processed, skipped, copied_labels = future.result()
                total_processed += processed
                total_skipped += skipped
                total_copied_labels += copied_labels
    else:
        for scene_dir in tqdm(scene_dirs, desc="Extracting bands"):
            processed, skipped, copied_labels = process_scene(
                scene_dir,
                input_patches,
                output_base,
                args.bands,
                args.normalize,
                args.nodata,
                args.normalize_threshold,
            )
            total_processed += processed
            total_skipped += skipped
            total_copied_labels += copied_labels
    
    print(f"\nBand extraction complete!")
    print(f"  Output base: {output_base}")
    print(f"  Images processed: {total_processed}")
    print(f"  Labels copied:   {total_copied_labels}")
    print(f"  Skipped:         {total_skipped}")
    
    # Verify a sample
    try:
        pattern = '*_6band.tif' if args.bands == 6 else '*_rgb.tif'
        sample = next((output_base / 'patches').rglob(pattern))
        with rasterio.open(sample) as src:
            print(f"\nVerification - Sample image: {sample}")
            print(f"  Shape: {src.shape}")
            print(f"  Bands: {src.count}")
            print(f"  Data type: {src.dtypes[0]}")
            print(f"  NoData: {src.nodata}")
            print(f"  Descriptions: {list(src.descriptions)}")
            if args.normalize:
                print(f"  Data range: {src.read().min():.4f} - {src.read().max():.4f}")
    except StopIteration:
        print("[Warn] No output files found for verification.")

if __name__ == "__main__":
    main()
