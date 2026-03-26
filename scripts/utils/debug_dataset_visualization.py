import cv2
import numpy as np
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt

def visualize_sample():
    # Path to a sample file
    sample_path = Path("MARIDA/patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0.tif")
    
    if not sample_path.exists():
        print(f"File not found: {sample_path}")
        # Try to find any tif file
        files = list(Path("MARIDA/patches").rglob("*.tif"))
        if files:
            sample_path = files[0]
            print(f"Using found file: {sample_path}")
        else:
            print("No TIF files found.")
            return

    # Load using the logic from MaskRescueNPYDataset
    with rasterio.open(sample_path) as src:
        arr = src.read() # (C, H, W)
        arr = np.transpose(arr, (1, 2, 0)) # (H, W, C)
    
    print(f"Original shape: {arr.shape}")
    print(f"Original range: {arr.min()} - {arr.max()}")
    
    # Select bands [3, 2, 1, 7, 9, 10]
    if arr.shape[-1] >= 11:
        arr = arr[..., [3, 2, 1, 7, 9, 10]]
    
    print(f"Selected bands shape: {arr.shape}")
    
    # Apply the fix: Scale x 1500
    arr_scaled = arr * 1500.0
    
    print(f"Scaled range (x1500): {arr_scaled.min()} - {arr_scaled.max()}")
    
    # Clip
    arr_clipped = np.clip(arr_scaled, 0, 255)
    
    # Visualize RGB (first 3 channels)
    rgb = arr_clipped[..., :3].astype(np.uint8)
    
    # Save
    cv2.imwrite("debug_sample_visualization_x1500.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("Saved debug_sample_visualization_x1500.png")
    
    # Calculate histogram stats
    print(f"Mean value: {np.mean(arr_clipped)}")
    print(f"Median value: {np.median(arr_clipped)}")
    print(f"95th percentile: {np.percentile(arr_clipped, 95)}")

if __name__ == "__main__":
    visualize_sample()
