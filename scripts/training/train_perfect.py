import sys
import os

# Fix OpenMP conflict error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

# Ensure we can import from local
sys.path.append(os.getcwd())

def train():
    args = dict(
        model="yolo11n_6ch.pt",
        data="MARIDA_YOLO_Perfect/marida_perfect.yaml",
        epochs=50,
        imgsz=256,
        batch=16,
        device=0,
        project="marida_perfect_runs",
        name="yolo11n_6ch_perfect",
        exist_ok=True,
        amp=True,
        patience=10,
        save=True,
        workers=4,
        val=True,
        cache=False # Disable cache to ensure we load TIFs correctly
    )
    
    print("Starting training with MaskRescueTrainer...")
    print(f"Model: {args['model']}")
    print(f"Data: {args['data']}")
    
    trainer = MaskRescueTrainer(overrides=args)
    trainer.train()

if __name__ == "__main__":
    train()
