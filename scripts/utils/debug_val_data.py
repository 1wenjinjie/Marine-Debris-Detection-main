import sys
import numpy as np
from pathlib import Path
from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset

def inspect_validation_data():
    print("Initializing dataset loader for validation set...")
    try:
        # Initialize dataset pointing to the validation split
        # Note: We use the same class but point to the validation list
        # The yaml defines val: val.txt, which contains paths relative to dataset_root
        dataset_root = Path("mask_bbox_rescue_split")
        val_list = dataset_root / "val.txt"
        
        if not val_list.exists():
            print(f"Error: Validation list not found at {val_list}")
            return

        # Create a minimal dataset instance just to load images
        # We pass the val.txt path as the img_path argument to get_img_files logic
        # Mock data dict to satisfy YOLODataset.__init__
        mock_data = {"channels": 7, "names": ["c0","c1","c2","c3","c4","c5","c6"]}
        
        ds = MaskRescueNPYDataset(
            img_path=str(val_list),
            imgsz=512,
            batch_size=4,
            augment=False,
            hyp=None,
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix="[ValCheck] ",
            data=mock_data
        )
        
        print(f"Found {len(ds.im_files)} validation samples.")
        
        # Check first 10 samples for stats
        stats = {'min': [], 'max': [], 'mean': [], 'nans': 0, 'infs': 0}
        
        for i in range(min(10, len(ds.im_files))):
            im, _, _ = ds.load_image(i)
            # im is (H, W, C)
            
            stats['min'].append(np.min(im))
            stats['max'].append(np.max(im))
            stats['mean'].append(np.mean(im))
            
            if np.isnan(im).any():
                stats['nans'] += 1
                print(f"Sample {i} ({ds.im_files[i]}) contains NaNs!")
            if np.isinf(im).any():
                stats['infs'] += 1
                print(f"Sample {i} ({ds.im_files[i]}) contains Infs!")
                
        print("\n--- Validation Data Statistics (First 10 samples) ---")
        print(f"Global Min: {min(stats['min']):.4f}")
        print(f"Global Max: {max(stats['max']):.4f}")
        print(f"Average Mean: {np.mean(stats['mean']):.4f}")
        print(f"Samples with NaNs: {stats['nans']}")
        print(f"Samples with Infs: {stats['infs']}")
        
        # Check labels for the first sample
        labels = ds.get_labels()
        if len(labels) > 0:
            print(f"\nFirst sample label: {labels[0]}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_validation_data()
