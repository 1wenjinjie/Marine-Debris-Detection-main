import argparse
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

random.seed(42)

CLASS_NAMES = [
    'Marine_Debris',
    'Dense_Sargassum',
    'Sparse_Sargassum',
    'Natural_Organic',
    'Ship',
    'Foam',
    'Wake'
]

def derive_scene_id(stem: str) -> str:
    parts = stem.split('_')
    if parts and parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return stem

def load_patches(source_labels: Path):
    scenes = defaultdict(list)
    for label_file in source_labels.glob('*.txt'):
        stem = label_file.stem
        scene = derive_scene_id(stem)
        scenes[scene].append(stem)
    return scenes

def ensure_split_dirs(root: Path):
    for split in ['train', 'val', 'test']:
        (root / split / 'images').mkdir(parents=True, exist_ok=True)
        (root / split / 'labels').mkdir(parents=True, exist_ok=True)

def copy_sample(stem: str, source_images: Path, source_labels: Path, dest_root: Path, split: str):
    img_src = next((p for p in source_images.glob(f'{stem}.*') if p.suffix.lower() in ['.tif', '.png', '.jpg']), None)
    if img_src is None:
        raise FileNotFoundError(f'Image for {stem} not found in {source_images}')
    label_src = source_labels / f'{stem}.txt'
    if not label_src.exists():
        raise FileNotFoundError(f'Label for {stem} not found in {source_labels}')

    img_dst = dest_root / split / 'images' / img_src.name
    lbl_dst = dest_root / split / 'labels' / label_src.name
    shutil.copy2(img_src, img_dst)
    shutil.copy2(label_src, lbl_dst)


def count_objects(label_path: Path, class_hist: dict):
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    for line in lines:
        cls = int(line.split()[0])
        class_hist[cls] += 1
    return len(lines)

def split_dataset(source_root: Path, dest_root: Path, train_ratio=0.7, val_ratio=0.2, seed=42):
    random.seed(seed)
    source_images = source_root / 'images'
    source_labels = source_root / 'labels'
    if not source_images.exists() or not source_labels.exists():
        raise ValueError(f'Source directories missing: {source_images}, {source_labels}')

    scenes = load_patches(source_labels)
    scene_ids = list(scenes.keys())
    random.shuffle(scene_ids)

    total = len(scene_ids)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    train_ids = scene_ids[:n_train]
    val_ids = scene_ids[n_train:n_train + n_val]
    test_ids = scene_ids[n_train + n_val:]

    ensure_split_dirs(dest_root)

    stats = {split: {'patches': 0, 'objects': 0, 'class_hist': {i: 0 for i in range(len(CLASS_NAMES))}} for split in ['train', 'val', 'test']}

    def assign(split_name, ids):
        for scene in ids:
            for stem in scenes[scene]:
                copy_sample(stem, source_images, source_labels, dest_root, split_name)
                stats[split_name]['patches'] += 1
                label_path = dest_root / split_name / 'labels' / f'{stem}.txt'
                stats[split_name]['objects'] += count_objects(label_path, stats[split_name]['class_hist'])

    assign('train', train_ids)
    assign('val', val_ids)
    assign('test', test_ids)

    # Write classes.txt
    with open(dest_root / 'classes.txt', 'w') as f:
        for name in CLASS_NAMES:
            f.write(name + '\n')

    # Write split info
    split_info = {
        'ratios': {'train': train_ratio, 'val': val_ratio, 'test': 1 - train_ratio - val_ratio},
        'counts': stats,
        'total_scenes': total,
        'train_scenes': len(train_ids),
        'val_scenes': len(val_ids),
        'test_scenes': len(test_ids),
        'seed': seed
    }
    with open(dest_root / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Split enriched dataset into train/val/test by scene.')
    parser.add_argument('--source', type=str, default='MARIDA_YOLO_Enriched_Dilated/train', help='Source directory containing images/labels folders')
    parser.add_argument('--output', type=str, default='MARIDA_YOLO_Enriched_Dilated_split', help='Destination root directory')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source)
    dest_root = Path(args.output)
    if dest_root.exists():
        shutil.rmtree(dest_root)
    split_dataset(source_root, dest_root, args.train_ratio, args.val_ratio, args.seed)
    print(f'Split complete. Output at {dest_root.resolve()}')

if __name__ == '__main__':
    main()
