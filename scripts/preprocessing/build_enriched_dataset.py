import os
import sys
import shutil
import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
from rasterio.features import rasterize
from tqdm import tqdm
import cv2

# Fix OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
MARIDA_ROOT = Path("MARIDA")
OUTPUT_ROOT = Path("MARIDA_YOLO_Enriched_Dilated")
PATCH_SIZE = 256

# Class Mapping (7 Classes)
ORIGINAL_TO_YOLO_MAP = {
    1: 0,   # Marine Debris
    2: 1,   # Dense Sargassum
    3: 2,   # Sparse Sargassum
    4: 3,   # Natural Organic Material
    5: 4,   # Ship
    9: 5,   # Foam
    14: 6   # Wake
}

def build_enriched_dataset():
    print("Building Enriched Dataset (Mining all Debris & Wakes)...")
    
    # Create directories
    for split in ['train', 'val']:
        (OUTPUT_ROOT / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / split / 'labels').mkdir(parents=True, exist_ok=True)
        
    # 1. Identify "Precious" Files (Containing Debris or Wakes)
    precious_files = set()
    shapefile_dir = MARIDA_ROOT / "shapefiles"
    
    print("Scanning shapefiles for precious classes...")
    for shp in shapefile_dir.glob("*.shp"):
        try:
            gdf = gpd.read_file(shp)
            class_col = None
            for col in ['Class', 'class', 'id', 'ID', 'gridcode']:
                if col in gdf.columns:
                    class_col = col
                    break
            
            if class_col:
                # Check if file contains Class 1 (Debris) or Class 14 (Wake)
                has_debris = len(gdf[gdf[class_col] == 1]) > 0
                has_wake = len(gdf[gdf[class_col] == 14]) > 0
                
                if has_debris or has_wake:
                    # Filename format: S2_DATE_TILE.shp -> DATE_TILE
                    # But patches are named DATE_TILE_X.tif
                    # We need to match the base name.
                    base_name = shp.stem
                    if base_name.startswith("S2_"):
                        base_name = base_name[3:]
                    precious_files.add(base_name)
        except:
            pass
            
    print(f"Found {len(precious_files)} source scenes containing Debris or Wakes.")
    
    # 2. Process Patches
    # The patches are in SUBDIRECTORIES of MARIDA/patches
    # e.g. MARIDA/patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0.tif
    
    patch_dir = MARIDA_ROOT / "patches"
    # Use rglob to find all TIFs recursively
    all_tifs = list(patch_dir.rglob("*.tif"))
    
    print(f"Processing {len(all_tifs)} patches...")
    
    count_debris = 0
    count_wakes = 0
    
    for tif_path in tqdm(all_tifs):
        # Check if this patch is from a precious scene
        # Filename: 1-12-19_48MYU_0.tif -> Scene: 1-12-19_48MYU
        scene_name = "_".join(tif_path.stem.split("_")[:-1])
        
        # Load corresponding shapefile to generate mask
        # We need to find the shapefile for this scene.
        # Shapefiles are named S2_SCENE.shp
        shp_path = shapefile_dir / f"S2_{scene_name}.shp"
        if not shp_path.exists():
            # Try without S2_
            shp_path = shapefile_dir / f"{scene_name}.shp"
            
        if not shp_path.exists():
            continue
            
        # We need to check if this specific PATCH contains the object.
        # This requires rasterizing the shapefile onto the patch coordinates.
        # This is slow to do for every patch.
        # Alternative: The patches are crops of the scene.
        # We don't have the georeference of the patch easily unless we read the TIF.
        
        # Let's read the TIF to get transform
        with rasterio.open(tif_path) as src:
            transform = src.transform
            height, width = src.shape[-2:]
            crs = src.crs
            
            # Read Shapefile
            try:
                gdf = gpd.read_file(shp_path)
                if gdf.crs != crs:
                    gdf = gdf.to_crs(crs)
            except:
                continue
                
            # Filter for relevant classes
            class_col = None
            for col in ['Class', 'class', 'id', 'ID', 'gridcode']:
                if col in gdf.columns:
                    class_col = col
                    break
            if not class_col: continue
            
            # Filter GDF to only features that intersect the patch bounds
            patch_bounds = rasterio.transform.array_bounds(height, width, transform)
            # (left, bottom, right, top)
            from shapely.geometry import box
            patch_box = box(*patch_bounds)
            
            # Spatial Index intersection is faster
            try:
                possible_matches_index = list(gdf.sindex.intersection(patch_box.bounds))
                possible_matches = gdf.iloc[possible_matches_index]
                
                # Fix invalid geometries before intersection
                possible_matches = possible_matches.copy()
                possible_matches['geometry'] = possible_matches.geometry.buffer(0)
                
                precise_matches = possible_matches[possible_matches.intersects(patch_box)]
            except Exception as e:
                # print(f"Geometry error in {tif_path.name}: {e}")
                continue
            
            if precise_matches.empty:
                continue
                
            # Now we have features in this patch.
            # Check if any are Debris (1) or Wake (14)
            has_debris = 1 in precise_matches[class_col].values
            has_wake = 14 in precise_matches[class_col].values
            
            target_split = "train" # Default to train for enrichment
            
            # If it has Debris, we definitely want it.
            # If it has Wake, we definitely want it.
            # If it has neither, we skip it (to keep dataset focused) or add it if it was in original split.
            
            if not (has_debris or has_wake):
                continue
                
            # Generate Labels
            labels = []
            
            # Rasterize each feature to get bounding box
            # Iterate over features in this patch
            for idx, row in precise_matches.iterrows():
                cls_id = row[class_col]
                if cls_id not in ORIGINAL_TO_YOLO_MAP:
                    continue
                    
                yolo_cls = ORIGINAL_TO_YOLO_MAP[cls_id]
                geom = row.geometry
                
                # Clip geometry to patch
                clipped_geom = geom.intersection(patch_box)
                if clipped_geom.is_empty: continue
                
                # Get pixel coordinates
                # We can use rasterio.features.rasterize to get a mask, then find bbox
                # Or transform coordinates manually.
                # Rasterize is safer for complex polygons.
                
                mask = rasterize(
                    [(clipped_geom, 1)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )
                
                # DILATION STRATEGY:
                # Artificially enlarge small objects to make them visible to YOLO
                # A 1-pixel object is invisible after 8x downsampling.
                # Dilating by 1 iteration (3x3 kernel) makes it 3x3.
                # Dilating by 2 iterations makes it 5x5.
                if yolo_cls == 0: # Only dilate Debris
                    kernel = np.ones((3,3), np.uint8)
                    # Dilate 1 iteration -> 3x3 min size
                    mask = cv2.dilate(mask, kernel, iterations=1)

                # Find bbox of mask
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) == 0: continue
                
                # Filter by pixel area (after dilation)
                # Since we dilated, even a 1px point is now 9px (if 3x3).
                # So we can keep everything that is now > 0.
                # But let's still filter tiny fragments if any.
                pixel_area = len(y_indices)
                if pixel_area < 4: # Should be at least 9 if it was a point
                    continue

                xmin, xmax = x_indices.min(), x_indices.max()
                ymin, ymax = y_indices.min(), y_indices.max()
                
                # Calculate dimensions in pixels (inclusive of the last pixel)
                bbox_w_px = xmax - xmin + 1
                bbox_h_px = ymax - ymin + 1
                
                # Filter by bbox dimensions (optional, but good for thin noise)
                if bbox_w_px < 2 or bbox_h_px < 2:
                    continue

                # Normalize
                # Pixel coordinates are 0-indexed. Pixel i spans [i, i+1].
                # BBox spans [xmin, xmax+1] and [ymin, ymax+1].
                x_center = ((xmin + xmax + 1) / 2) / width
                y_center = ((ymin + ymax + 1) / 2) / height
                w = bbox_w_px / width
                h = bbox_h_px / height
                
                labels.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            if not labels:
                continue
                
            # Save Image and Label
            # Copy TIF
            out_img_path = OUTPUT_ROOT / target_split / "images" / tif_path.name
            shutil.copy(tif_path, out_img_path)
            
            # Save Label
            out_lbl_path = OUTPUT_ROOT / target_split / "labels" / tif_path.with_suffix(".txt").name
            with open(out_lbl_path, "w") as f:
                f.write("\n".join(labels))
                
            if has_debris: count_debris += 1
            if has_wake: count_wakes += 1
            
    print(f"Enrichment Complete.")
    print(f"Total Debris Patches: {count_debris}")
    print(f"Total Wake Patches: {count_wakes}")
    
    # Create YAML
    yaml_content = f"""
path: {OUTPUT_ROOT.absolute()}
train: train/images
val: train/images  # Use train for val temporarily to maximize training data, or split later
test: train/images
names:
  0: Marine_Debris
  1: Dense_Sargassum
  2: Sparse_Sargassum
  3: Natural_Organic
  4: Ship
  5: Foam
  6: Wake
channels: 6
"""
    with open(OUTPUT_ROOT / "marida_enriched.yaml", "w") as f:
        f.write(yaml_content)

if __name__ == "__main__":
    build_enriched_dataset()
