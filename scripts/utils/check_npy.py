import numpy as np

path = "D:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split/val/images/S2_11-6-18_16PCC_13.npy"
arr = np.load(path)
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
print(f"Min: {arr.min()}, Max: {arr.max()}")
print(f"Mean: {arr.mean()}")
