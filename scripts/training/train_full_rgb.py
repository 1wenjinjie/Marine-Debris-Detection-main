import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
import torch

# --- Monkey Patch Start ---
# Force Ultralytics to use our custom Dataset class for .npy loading
from ultralytics.data import build
from ultralytics.models.yolo.detect import train as detect_train
from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset

# Store original function
_original_build_yolo_dataset = build.build_yolo_dataset

def _patched_build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """
    Monkey-patched build_yolo_dataset to force MaskRescueNPYDataset usage.
    """
    print(f"[MonkeyPatch] Intercepted build_yolo_dataset for mode={mode}")
    return MaskRescueNPYDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',  # Augmentation only in train mode
        hyp=cfg,
        rect=rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0,
        prefix=colorstr(f'{mode}: ') if mode in ('train', 'val') else '',
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0
    )

# Apply patch to both locations
build.build_yolo_dataset = _patched_build_yolo_dataset
detect_train.build_yolo_dataset = _patched_build_yolo_dataset
print("DEBUG: Applied monkey patch to ultralytics.data.build and ultralytics.models.yolo.detect.train")

from ultralytics.utils import colorstr
# --- Monkey Patch End ---

def train_full_rgb():
    # Fix for OpenMP error
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Load model
    model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data='marida_rescue_split.yaml',
        epochs=150,
        imgsz=640,
        batch=16,
        project='marida_runs',
        name='yolo11n_rgb_full',
        device='0',
        workers=4,
        patience=20, # Early stopping
        save=True,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        cos_lr=True, # Cosine learning rate scheduler
        close_mosaic=10, # Disable mosaic augmentation for final 10 epochs
    )
    
    print("Training completed.")

if __name__ == '__main__':
    train_full_rgb()
