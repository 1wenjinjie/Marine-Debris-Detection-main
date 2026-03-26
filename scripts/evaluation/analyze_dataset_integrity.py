import os
import yaml
import glob
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
# import seaborn as sns

# Configuration
DATA_YAML = "data_npy_6ch.yaml"

def load_file_list(txt_path, root_path):
    """Loads file list from a txt file, resolving paths."""
    if not os.path.exists(txt_path):
        # Try relative to root
        txt_path = os.path.join(root_path, txt_path)
    
    if not os.path.exists(txt_path):
        print(f"Warning: Could not find {txt_path}")
        return []

    with open(txt_path, 'r') as f:
        files = [line.strip() for line in f.readlines() if line.strip()]
    
    # Resolve to absolute paths
    resolved_files = []
    for f in files:
        p = Path(f)
        if not p.is_absolute():
            p = Path(root_path) / p
        resolved_files.append(p)
    return resolved_files

def get_scene_id(filename):
    """
    Extracts scene ID from filename.
    Assumes format: S2_DD-MM-YY_CODE_PATCHID.npy
    Example: S2_18-9-20_16PCC_47.npy -> S2_18-9-20_16PCC
    """
    name = filename.stem
    parts = name.split('_')
    if len(parts) >= 4:
        return "_".join(parts[:-1])
    return name # Fallback

def analyze_classes(file_list, class_names):
    """Counts class instances in the given file list."""
    counts = Counter()
    
    for file_path in file_list:
        # Infer label path
        # .../images/name.npy -> .../labels/name.txt
        # or adjacent folder
        label_path = None
        
        # Strategy 1: Standard YOLO
        if file_path.parent.name == 'images':
            candidate = file_path.parent.parent / 'labels' / file_path.with_suffix('.txt').name
            if candidate.exists():
                label_path = candidate
        
        # Strategy 2: Same folder (flat structure)
        if label_path is None:
            candidate = file_path.with_suffix('.txt')
            if candidate.exists():
                label_path = candidate
                
        # Strategy 3: Adjacent 'labels' folder
        if label_path is None:
             candidate = file_path.parent / 'labels' / file_path.with_suffix('.txt').name
             if candidate.exists():
                 label_path = candidate

        if label_path and label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < len(class_names):
                            counts[class_names[cls_id]] += 1
    return counts

def main():
    print("--- Dataset Integrity & Distribution Analysis ---")
    
    # 1. Load Config
    with open(DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_root = config.get('path', '.')
    class_names = config['names']
    
    print(f"Dataset Root: {dataset_root}")
    print(f"Classes: {class_names}")
    
    # 2. Load Splits
    splits = {
        'train': load_file_list(config.get('train'), dataset_root),
        'val': load_file_list(config.get('val'), dataset_root),
        'test': load_file_list(config.get('test'), dataset_root)
    }
    
    # 3. Check for Exact Leakage (File Overlap)
    print("\n[1. Exact File Leakage Check]")
    sets = {k: set(f.name for f in v) for k, v in splits.items()}
    
    leakage_tv = sets['train'].intersection(sets['val'])
    leakage_tt = sets['train'].intersection(sets['test'])
    leakage_vt = sets['val'].intersection(sets['test'])
    
    if leakage_tv:
        print(f"CRITICAL: Found {len(leakage_tv)} files in both Train and Val!")
    else:
        print("PASS: No overlap between Train and Val.")
        
    if leakage_tt:
        print(f"CRITICAL: Found {len(leakage_tt)} files in both Train and Test!")
    else:
        print("PASS: No overlap between Train and Test.")
        
    if leakage_vt:
        print(f"CRITICAL: Found {len(leakage_vt)} files in both Val and Test!")
    else:
        print("PASS: No overlap between Val and Test.")

    # 4. Check for Spatial/Scene Leakage
    print("\n[2. Spatial/Scene Leakage Check]")
    print("Checking if crops from the same satellite scene exist in different splits...")
    
    scene_sets = {k: set(get_scene_id(f) for f in v) for k, v in splits.items()}
    
    scene_leak_tv = scene_sets['train'].intersection(scene_sets['val'])
    scene_leak_tt = scene_sets['train'].intersection(scene_sets['test'])
    
    if scene_leak_tv:
        print(f"WARNING: {len(scene_leak_tv)} Scenes are shared between Train and Val.")
        print(f"  (This means the model is tested on the same water conditions it trained on.)")
        print(f"  Shared Scenes: {list(scene_leak_tv)[:5]}...")
    else:
        print("EXCELLENT: Train and Val are spatially independent (different scenes).")
        
    if scene_leak_tt:
        print(f"WARNING: {len(scene_leak_tt)} Scenes are shared between Train and Test.")
    else:
        print("EXCELLENT: Train and Test are spatially independent.")

    # 5. Class Distribution Analysis
    print("\n[3. Class Distribution Analysis]")
    stats = {}
    for split_name, file_list in splits.items():
        print(f"Analyzing {split_name} ({len(file_list)} files)...")
        counts = analyze_classes(file_list, class_names)
        stats[split_name] = counts
        print(f"  {dict(counts)}")
        
    # 6. Generate Report
    print("\n[4. Summary & Recommendations]")
    
    # Check Balance
    train_counts = stats['train']
    if not train_counts:
        print("Error: No labels found in training set.")
        return

    total_train_objects = sum(train_counts.values())
    print(f"Total Training Objects: {total_train_objects}")
    
    # Calculate ratios
    ratios = {k: v/total_train_objects for k, v in train_counts.items()}
    
    # Identify imbalance
    print("Class Imbalance Report:")
    for cls, ratio in ratios.items():
        print(f"  {cls}: {ratio:.2%}")
        
    if ratios.get('Marine Debris', 0) < 0.1:
        print("  -> ALERT: Marine Debris is a minority class (<10%). Consider oversampling or focal loss.")
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
