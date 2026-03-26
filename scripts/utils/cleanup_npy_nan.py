import numpy as np
from pathlib import Path

SPLITS = [
    Path("MARIDA_YOLO_6band_split/train/images"),
    Path("MARIDA_YOLO_6band_split/val/images"),
    Path("MARIDA_YOLO_6band_split/test/images"),
]

if __name__ == "__main__":
    total_files = 0
    cleaned_files = 0
    replaced_values = 0

    for split in SPLITS:
        for path in sorted(split.glob("*.npy")):
            data = np.load(path)
            if not (np.isnan(data).any() or np.isinf(data).any()):
                total_files += 1
                continue

            mask = np.isnan(data) | np.isinf(data)
            replaced_values += int(mask.sum())
            data[mask] = 0.0
            np.save(path, data)
            cleaned_files += 1
            total_files += 1

    print(f"processed={total_files}")
    print(f"cleaned={cleaned_files}")
    print(f"values_replaced={replaced_values}")
