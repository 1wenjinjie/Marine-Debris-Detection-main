import numpy as np
import glob
import os

files = glob.glob('mask_bbox_rescue_split/train/images/*.npy')
if files:
    arr = np.load(files[0])
    print(f"File: {files[0]}")
    print(f"Shape: {arr.shape}")
else:
    print("No files found")