import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

CHANNELS = 6
SAMPLE_PER_FILE = 4096
ROOT = Path('mask_bbox_rescue_split')
OUTPUT = Path('marida_runs/channel_stats_6ch.json')

image_dirs = [ROOT / 'train' / 'images', ROOT / 'val' / 'images']
files = []
for d in image_dirs:
    files.extend(sorted(str(p) for p in d.glob('*.npy')))

rng = np.random.default_rng(42)
sample_accum = []

for path in tqdm(files, desc='Sampling pixels'):
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3:
        continue
    if arr.shape[0] in (6, 7) and arr.shape[0] <= arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)
    if arr.shape[-1] < CHANNELS:
        continue
    data = arr[..., :CHANNELS].reshape(-1, CHANNELS)
    n = data.shape[0]
    k = min(SAMPLE_PER_FILE, n)
    idx = rng.choice(n, size=k, replace=False)
    sample_accum.append(data[idx])

samples = np.concatenate(sample_accum, axis=0)
percentiles = np.percentile(samples, [1, 25, 50, 75, 99], axis=0)
mean = samples.mean(axis=0)
std = samples.std(axis=0)

stats = {
    'channels': CHANNELS,
    'sampled_pixels': int(samples.shape[0]),
    'mean': mean.tolist(),
    'std': std.tolist(),
    'percentiles': {
        'p01': percentiles[0].tolist(),
        'p25': percentiles[1].tolist(),
        'p50': percentiles[2].tolist(),
        'p75': percentiles[3].tolist(),
        'p99': percentiles[4].tolist(),
    }
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(stats, indent=2))
print(f'Saved stats to {OUTPUT} (pixels={stats["sampled_pixels"]})')
