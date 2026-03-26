"""Load the trained YOLO model and run inference on the PNG test folder, saving visualizations.

Usage (PowerShell):
    python run_inference_visualize.py --model marida_runs/yolo11n_rgb_full/weights/best.pt --src marida_runs/yolo11n_rgb_full/visualize_test_png --out_dir marida_runs/yolo11n_rgb_full/visualizations
"""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="Path to trained .pt file")
    p.add_argument("--src", type=Path, required=True, help="Folder containing PNGs to run inference on")
    p.add_argument("--out_dir", type=Path, required=True, help="Folder to save visualization results")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))

    # Use Ultralytics predict API to run over a folder of PNGs; save=True stores results under runs/predict by default.
    # We set project and name so outputs go into the requested out_dir (by pointing project to its parent and name to last folder).
    project = str(args.out_dir.parent)
    name = args.out_dir.name

    # Run prediction and save annotated images
    model.predict(source=str(args.src), imgsz=args.imgsz, conf=args.conf, save=True, project=project, name=name, exist_ok=True)
    print(f"Predictions saved to {args.out_dir}")

if __name__ == '__main__':
    main()
