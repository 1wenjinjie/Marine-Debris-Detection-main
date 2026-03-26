import os
import numpy as np
import glob
import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
import cv2

# Configuration
DATA_DIR = 'mask_bbox_rescue_split/train'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')
OUTPUT_DIR = 'spectral_analysis'
NUM_SAMPLES = 500  # Limit number of images to process for speed (set to None for all)

# Band Definitions (Sentinel-2)
BANDS = ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2']
# Wavelengths for plotting (approximate center wavelength in nm)
WAVELENGTHS = [665, 560, 490, 842, 1610, 2190] 
# Note: Plotting usually follows wavelength order. 
# Our channel order in .npy is: R, G, B, NIR, SWIR1, SWIR2
# Let's reorder them for plotting to be physically intuitive: B, G, R, NIR, SWIR1, SWIR2
PLOT_ORDER_INDICES = [2, 1, 0, 3, 4, 5] 
PLOT_LABELS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

CLASSES = {
    0: 'Marine Debris',
    1: 'Dense Sargassum',
    2: 'Sparse Sargassum',
    3: 'Natural Organic',
    4: 'Ship',
    5: 'Foam',
    6: 'Wake',
    7: 'Water' # Virtual class for background
}

def load_data_and_extract_pixels():
    pixel_data = {k: [] for k in CLASSES.keys()}
    
    npy_files = glob.glob(os.path.join(IMAGES_DIR, "*.npy"))
    if NUM_SAMPLES:
        npy_files = npy_files[:NUM_SAMPLES]
        
    print(f"Processing {len(npy_files)} images for spectral analysis...")
    
    for fpath in tqdm(npy_files):
        # Load Image
        try:
            arr = np.load(fpath)
        except:
            continue
            
        # Handle dimensions (C, H, W) -> (H, W, C)
        if arr.ndim == 3 and arr.shape[0] in (6, 7, 11, 12) and arr.shape[0] < arr.shape[2]:
            arr = arr.transpose(1, 2, 0)
            
        # Extract 6 bands (Same logic as training)
        if arr.shape[-1] >= 11:
            img = arr[..., [3, 2, 1, 7, 9, 10]]
        elif arr.shape[-1] == 7:
            img = arr[..., :6]
        elif arr.shape[-1] == 6:
            img = arr
        else:
            continue
            
        h, w, _ = img.shape
        
        # Load Labels
        label_name = os.path.basename(fpath).replace('.npy', '.txt')
        label_path = os.path.join(LABELS_DIR, label_name)
        
        mask = np.zeros((h, w), dtype=np.uint8) # 0=Water
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                cls_id = int(parts[0])
                
                # Parse bbox (normalized xywh)
                cx, cy, bw, bh = map(float, parts[1:])
                
                # Convert to pixel coords
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                # Clip to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    # Extract pixels for this object
                    obj_pixels = img[y1:y2, x1:x2, :].reshape(-1, 6)
                    
                    # Randomly sample pixels to avoid memory explosion (e.g., max 100 pixels per object)
                    if len(obj_pixels) > 50:
                        indices = np.random.choice(len(obj_pixels), 50, replace=False)
                        sampled_pixels = obj_pixels[indices]
                    else:
                        sampled_pixels = obj_pixels
                        
                    pixel_data[cls_id].extend(sampled_pixels)
                    
                    # Mark mask as occupied
                    mask[y1:y2, x1:x2] = 1
        
        # Extract Water pixels (Background)
        # Only sample if there is enough water
        water_pixels = img[mask == 0].reshape(-1, 6)
        if len(water_pixels) > 100:
            # Sample fewer water pixels per image to balance
            indices = np.random.choice(len(water_pixels), 50, replace=False)
            pixel_data[7].extend(water_pixels[indices])

    return pixel_data

def plot_spectral_signatures(pixel_data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Prepare data for plotting
    means = {}
    stds = {}
    
    plt.figure(figsize=(12, 8))
    # sns.set_style("whitegrid")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Define colors for classes
    palette = {
        'Marine Debris': 'red',
        'Dense Sargassum': 'green',
        'Sparse Sargassum': 'lightgreen',
        'Natural Organic': 'brown',
        'Ship': 'black',
        'Foam': 'cyan',
        'Wake': 'blue',
        'Water': 'navy'
    }
    
    print("Calculating statistics and plotting...")
    
    for cls_id, pixels in pixel_data.items():
        cls_name = CLASSES[cls_id]
        if not pixels:
            print(f"No samples for {cls_name}")
            continue
            
        data = np.array(pixels)
        # data shape: (N, 6) -> R, G, B, NIR, SWIR1, SWIR2
        
        # Reorder to wavelength order: B, G, R, NIR, SWIR1, SWIR2
        data_ordered = data[:, PLOT_ORDER_INDICES]
        
        # Calculate stats
        mean_vals = np.mean(data_ordered, axis=0)
        std_vals = np.std(data_ordered, axis=0)
        
        means[cls_name] = mean_vals
        stds[cls_name] = std_vals
        
        # Plot
        x = np.arange(len(PLOT_LABELS))
        plt.plot(x, mean_vals, marker='o', linewidth=2, label=cls_name, color=palette.get(cls_name, 'gray'))
        
        # Add error bands (optional, can be messy with many classes)
        # plt.fill_between(x, mean_vals - std_vals*0.2, mean_vals + std_vals*0.2, alpha=0.1, color=palette.get(cls_name, 'gray'))

    plt.xticks(x, PLOT_LABELS, fontsize=12)
    plt.ylabel("Normalized Reflectance (Pixel Value)", fontsize=14)
    plt.xlabel("Spectral Band", fontsize=14)
    plt.title("Spectral Signatures of Marine Objects (MARIDA)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, 'spectral_signatures.png')
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")
    
    # Save separate plot for Debris vs Foam vs Water (The Key Comparison)
    plt.figure(figsize=(10, 6))
    focus_classes = ['Marine Debris', 'Foam', 'Water', 'Wake']
    for cls_name in focus_classes:
        if cls_name in means:
            plt.plot(x, means[cls_name], marker='o', linewidth=2, label=cls_name, color=palette.get(cls_name))
            # plt.fill_between(x, means[cls_name] - stds[cls_name]*0.2, means[cls_name] + stds[cls_name]*0.2, alpha=0.1, color=palette.get(cls_name))
            
    plt.xticks(x, PLOT_LABELS, fontsize=12)
    plt.ylabel("Reflectance", fontsize=14)
    plt.title("Spectral Separation: Debris vs Foam vs Water", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'spectral_comparison_focus.png'), dpi=300)

if __name__ == "__main__":
    data = load_data_and_extract_pixels()
    plot_spectral_signatures(data)
