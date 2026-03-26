import sys
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    sys.path.insert(0, str(Path.cwd()))
    from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer
    
    print("="*80)
    print("Experiment A: 10-Channel Baseline (CIoU)")
    print("="*80)
    
    # Prepare 10ch weights
    model = YOLO("yolo11n.pt")
    state_dict = model.model.state_dict()
    for key in list(state_dict.keys()):
        if "model.0.conv.weight" in key:
            original = state_dict[key]
            expanded = original.repeat(1, 4, 1, 1)[:, :10, :, :]
            state_dict[key] = expanded
            torch.save({"model": model.model, "state_dict": state_dict}, "yolo11n_10ch.pt")
            print(" Created 10-channel weights")
            break
    
    # Load and train
    model = YOLO("yolo11n_10ch.pt")
    print(" Starting training: 50 epochs, batch 8")
    print("="*80)
    
    results = model.train(
        data="configs/datasets/marida_mask_rescue_10ch.yaml",
        epochs=50,
        imgsz=256,
        batch=8,
        device=0,
        project="runs_marida",
        name="exp_a_10ch_ciou_baseline",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        patience=50,
        save_period=10,
        verbose=True,
        workers=0,
        trainer=MaskRescueTrainer
    )
    
    print("\n" + "="*80)
    print(" Experiment A Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
