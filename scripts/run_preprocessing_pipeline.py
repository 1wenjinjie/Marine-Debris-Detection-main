"""
Master preprocessing pipeline for MARIDA dataset.
Runs all preprocessing steps in sequence:
1. Extract RGB+NIR+SWIR bands (6 channels)
2. Convert masks to YOLO format
3. Split dataset (train/val/test)
4. Apply data augmentation
"""

import sys
from pathlib import Path
import subprocess
import datetime
import json
import argparse

def run_script(script_path, description):
    """
    Run a preprocessing script and handle errors.
    
    Args:
        script_path: Path to script
        description: Description of the step
    """
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False

def create_metadata(output_dir, step_name, params=None):
    """
    Create metadata.json for the output directory.
    """
    metadata = {
        "step": step_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "parameters": params or {}
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

def main():
    print("Starting MARIDA preprocessing pipeline...")
    parser = argparse.ArgumentParser(description='MARIDA Dataset Preprocessing Pipeline')
    parser.add_argument('--yes', action='store_true', help='Auto yes to prompts')
    args = parser.parse_args()
    
    # Base directory
    base_dir = Path('D:/OceanEdge_Group/Marine-Debris-Detection-main')
    
    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M")
    
    # Check if MARIDA directory exists
    marida_dir = base_dir / 'MARIDA'
    if not marida_dir.exists():
        print(f"Error: MARIDA dataset not found at {marida_dir}")
        return
    
    print("="*60)
    print("MARIDA Dataset Preprocessing Pipeline")
    print("="*60)
    print(f"\nBase directory: {base_dir}")
    print(f"Run timestamp: {timestamp}")
    print("\nThis pipeline will:")
    print("  1. Extract RGB+NIR+SWIR bands (6 channels)")
    print("  2. Convert segmentation masks to YOLO bounding boxes")
    print("  3. Split dataset into train/val/test (70/20/10)")
    print("  4. Apply data augmentation to training set")
    print("\nNote: Each step may take several minutes depending on dataset size.")
    
    if args.yes:
        response = 'y'
    else:
        response = input("\nProceed with preprocessing? (y/n): ")
    if response.lower() != 'y':
        print("Preprocessing cancelled.")
        return
    
    # Define preprocessing steps with timestamped outputs
    steps = [
        # {
        #     'script': 'preprocess_resize.py',
        #     'description': 'Image Resizing (256x256 -> 512x512)',
        #     'output_dir': f'MARIDA_resized_{timestamp}'
        # },
        {
            'script': 'preprocess_extract_bands.py',
            'description': 'Band Extraction (RGB+NIR+SWIR, 6 channels)',
            'output_dir': f'MARIDA_6band_{timestamp}'
        },
        {
            'script': 'preprocess_mask_to_yolo.py',
            'description': 'Mask to YOLO Conversion',
            'output_dir': f'MARIDA_YOLO_{timestamp}'
        },
        {
            'script': 'preprocess_split_dataset.py',
            'description': 'Dataset Splitting',
            'output_dir': f'MARIDA_YOLO_split_{timestamp}'
        },
        {
            'script': 'preprocess_augmentation.py',
            'description': 'Data Augmentation',
            'output_dir': f'MARIDA_YOLO_augmented_{timestamp}'
        }
    ]
    
    # Run each step
    completed_steps = 0
    for step in steps:
        script_path = base_dir / step['script']
        
        # Run the script
        if not script_path.exists():
            print(f"\nWarning: Script not found: {script_path}")
            continue
        
        success = run_script(script_path, step['description'])
        if success:
            completed_steps += 1
            # Create metadata
            output_dir = base_dir / step['output_dir']
            create_metadata(output_dir, step['description'], {"timestamp": timestamp})
        else:
            print(f"\nError in step: {step['description']}")
            response = input("Continue with remaining steps? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Summary
    print(f"\n{'='*60}")
    print("Preprocessing Pipeline Complete")
    print(f"{'='*60}")
    print(f"Completed {completed_steps}/{len(steps)} steps")
    
    # Check final output
    final_output = base_dir / f'MARIDA_YOLO_augmented_{timestamp}'
    if final_output.exists():
        print(f"\n✓ Final dataset ready at: {final_output}")
        print("\nNext steps:")
        print("  1. Check dataset config: MARIDA_YOLO_split_<timestamp>/marida.yaml")
        print("  2. Modify model config for 6 channels if needed")
        print("  3. Start training with: yolo detect train data=marida.yaml model=yolov8n.pt")
    else:
        print("\nWarning: Final output directory not found")
        print("Some preprocessing steps may have failed")

if __name__ == "__main__":
    main()
