"""
Phase 3 Training: Weighted Fine-tuning.
Starting from: yolov11n_6ch_rebalance_v2_50e (Phase 2 Best)
Goal: Fix Class Confusion (Wakes -> Marine Debris) by increasing Classification Loss weight.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

def main():
    # Load the best model from Phase 2
    # This model already knows how to find objects, but confuses Wakes with Debris
    model_path = "runs_marida/yolov11n_6ch_rebalance_v2_50e/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading Phase 2 model: {model_path}")
    model = YOLO(model_path)

    overrides = {
        "data": "data_npy_6ch_rebalanced.yaml", # Continue using the rebalanced dataset
        "epochs": 30,           # Shorter run, just for fine-tuning
        "batch": 16,            # Smaller batch size might help generalization
        "imgsz": 640,
        "device": "0",
        "project": "runs_marida",
        "name": "yolov11n_6ch_phase3_weighted",
        "task": "detect",
        "workers": 0,
        
        # --- Fine-tuning Strategy ---
        "lr0": 0.001,           # Very low learning rate (Fine-tuning)
        "lrf": 0.01,            # Final LR will be very small
        "warmup_epochs": 1.0,
        "optimizer": "AdamW",
        
        # --- CRITICAL: Loss Weighting ---
        # Default cls gain is usually 0.5. We increase it to force class separation.
        "cls": 4.0,             # 8x default! Penalize wrong class heavily.
        "box": 5.0,             # Keep box gain high too (default 7.5)
        "dfl": 1.5,             # Default 1.5
        
        # --- Augmentation ---
        # Reduce augmentation slightly to let model see "real" shapes more often
        "mosaic": 0.5,          # Reduce mosaic probability (was 1.0 or 0.0)
        "mixup": 0.0,
    }

    print("Starting Phase 3: Weighted Fine-tuning...")
    print(f"Focus: Increasing 'cls' loss gain to {overrides['cls']} to fix Wakes/Debris confusion.")
    
    # Use MaskRescueTrainer to handle 6-channel NPY data
    model.train(trainer=MaskRescueTrainer, **overrides)

if __name__ == "__main__":
    main()
