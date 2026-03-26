"""10-Channel Spectral-Enhanced Training Script for MARIDA
Adds FDI/NDVI/RedEdge spectral indices to the 7-channel baseline.
Based on MARIDA_data.pdf Section 5.1 methodology.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
from prepare_multichannel_yolo_weights import prepare_weights
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default="marida_mask_rescue_10ch.yaml")
    parser.add_argument("--base-weights", type=Path, default="yolo11n.pt")
    parser.add_argument("--weights", type=Path, default="yolo11n_10ch_init.pt")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="yolov11n_10ch_spectral")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Prepare 10-channel weights
    if not args.weights.exists():
        print(f"Creating 10-channel weights from {args.base_weights}")
        prepare_weights(
            base_model_path=str(args.base_weights),
            output_path=str(args.weights),
            target_channels=10
        )
    else:
        print(f"Using existing 10-channel weights: {args.weights}")
    
    # Train with custom trainer
    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data-config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="runs_marida",
        trainer=MaskRescueTrainer,
        amp=True,
        val=True
    )
