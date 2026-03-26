import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset
import torch

class CustomTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """Build Custom Dataset."""
        return MaskRescueNPYDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls,
            stride=int(self.stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: ") if mode == "train" else "",
            channels=3, # Force 3 channels for RGB
            data=self.data
        )

def train():
    args = dict(
        model='yolo11n.pt',
        data='overfit_rgb.yaml',
        epochs=100,
        imgsz=256,
        batch=16,
        project='marida_overfit_runs',
        name='yolo11n_rgb_overfit_custom',
        device='0',
        workers=0,
        exist_ok=True,
        amp=True,
        optimizer='AdamW',
        lr0=0.001,
        patience=0, # Disable early stopping
        augment=False, # Disable augmentation for overfitting check
        hsv_h=0.0, # Disable color jitter
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    
    trainer = CustomTrainer(overrides=args)
    trainer.train()

if __name__ == '__main__':
    train()
