import sys
import os
import torch
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure local imports work
sys.path.append(os.getcwd())

from ultralytics_plugins.mask_rescue_validator import MaskRescueValidator

def validate_enriched():
    # Paths
    model_path = "marida_enriched_runs/yolo11n_6ch_enriched/weights/best.pt"
    data_path = "MARIDA_YOLO_Enriched/marida_enriched.yaml"
    
    print(f"Validating Enriched Model: {model_path}")
    print(f"Data: {data_path}")

    # Load model
    model = YOLO(model_path)
    
    # Get config
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = data_path
    args.model = model_path
    args.batch = 16
    args.imgsz = 256
    args.split = "val" # Use val split of enriched dataset
    args.device = 0
    args.project = "marida_enriched_runs"
    args.name = "val_enriched_set"
    args.exist_ok = True
    
    # Instantiate Validator
    validator = MaskRescueValidator(args=args)
    
    # Run validation
    validator(model=model_path)

if __name__ == "__main__":
    validate_enriched()
