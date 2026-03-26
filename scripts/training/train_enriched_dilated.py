import sys
import os
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

# Ensure we can import from local
sys.path.append(os.getcwd())
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train():
    args = dict(
        model="yolo11n_6ch.pt",
    data="MARIDA_YOLO_Enriched_Dilated_split/marida_enriched_split.yaml",
        epochs=100,
        imgsz=256,
        batch=16,
        device=0,
        project="marida_enriched_runs",
        name="yolo11n_6ch_enriched_dilated_fixed_v3",
        exist_ok=True,
        amp=True,
        patience=30,
        save=True,
        workers=4,
        val=True,
        cache=False,
        # Augmentation
        mosaic=1.0,
        mixup=0.5,
        copy_paste=0.5,
        degrees=10.0,
        fliplr=0.5,
        flipud=0.5
    )
    
    print("Starting training with Dilated Enriched Dataset...")
    print(f"Model: {args['model']}")
    print(f"Data: {args['data']}")
    
    trainer = MaskRescueTrainer(overrides=args)
    trainer.train()

if __name__ == "__main__":
    train()
