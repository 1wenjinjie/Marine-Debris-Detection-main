import os
import random
import numpy as np

root = r"D:/OceanEdge_Group/Marine-Debris-Detection-main/mask_bbox_rescue_split/train/images"
files = [f for f in os.listdir(root) if f.endswith('.npy')]
random.seed(0)
print(f"total train npy: {len(files)}")

per_channel_sum = None
per_channel_sq = None
pixel_count = 0

subset = random.sample(files, min(50, len(files)))
for name in subset:
    arr = np.load(os.path.join(root, name)).reshape(-1, 7).astype("float64")
    ch_sum = arr.sum(axis=0)
    ch_sq = (arr ** 2).sum(axis=0)
    if per_channel_sum is None:
        per_channel_sum = ch_sum
        per_channel_sq = ch_sq
    else:
        per_channel_sum += ch_sum
        per_channel_sq += ch_sq
    pixel_count += arr.shape[0]

sample_path = os.path.join(root, random.choice(files))
sample = np.load(sample_path)
print(f"sample={os.path.basename(sample_path)} dtype={sample.dtype}, min={sample.min():.3f}, max={sample.max():.3f}")
for idx in range(sample.shape[-1]):
    print(f"  channel {idx}: min={sample[..., idx].min():.4f}, max={sample[..., idx].max():.4f}, mean={sample[..., idx].mean():.6f}")
per_channel_mean = per_channel_sum / pixel_count
per_channel_std = (per_channel_sq / pixel_count - per_channel_mean ** 2) ** 0.5
for idx, (mean, std) in enumerate(zip(per_channel_mean, per_channel_std)):
    print(f"channel {idx}: mean={mean:.4f}, std={std:.4f}")
