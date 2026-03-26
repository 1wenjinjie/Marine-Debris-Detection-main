import os
import cv2
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.data import build
from ultralytics.models.yolo.detect import val as detect_val
from train_full_6ch import _patched_build_yolo_dataset
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.metrics import box_iou

# 1. Setup Environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 2. Apply Monkey Patch
build.build_yolo_dataset = _patched_build_yolo_dataset
detect_val.build_yolo_dataset = _patched_build_yolo_dataset

# Constants
MODEL_PATH = 'marida_runs/yolo11n_6ch_full/weights/best.pt'
DATA_YAML = 'marida_rescue_split_6ch.yaml'
VAL_IMAGES_DIR = 'mask_bbox_rescue_split/val/images'
VAL_LABELS_DIR = 'mask_bbox_rescue_split/val/labels'
OUTPUT_DIR = 'error_analysis'
CLASSES = {
    0: 'Marine_Debris',
    1: 'Dense_Sargassum',
    2: 'Sparse_Sargassum',
    3: 'Natural_Organic',
    4: 'Ship',
    5: 'Foam',
    6: 'Wake'
}

def load_npy_image(path):
    """Load .npy file and preprocess for model input (6ch) and display (RGB)."""
    try:
        arr = np.load(path)
    except:
        return None, None

    if arr.ndim != 3:
        return None, None
    
    # Transpose if needed (C, H, W) -> (H, W, C)
    if arr.shape[0] in (6, 7, 11, 12) and arr.shape[0] < arr.shape[2]:
        arr = arr.transpose(1, 2, 0)
        
    # Select bands
    if arr.shape[-1] >= 11:
        arr_6ch = arr[..., [3, 2, 1, 7, 9, 10]]
        arr_3ch = arr[..., [3, 2, 1]]
    elif arr.shape[-1] == 7:
        arr_6ch = arr[..., :6]
        arr_3ch = arr[..., :3]
    else:
        arr_6ch = arr[..., :6] if arr.shape[-1] >= 6 else arr
        arr_3ch = arr[..., :3]

    # Scale and Clip (x1500 -> 0-255)
    def preprocess(a):
        a = a * 1500.0
        a = np.clip(a, 0.0, 255.0)
        return a.astype(np.float32)

    proc_6ch = preprocess(arr_6ch)
    proc_3ch = preprocess(arr_3ch)
    
    # Display image (RGB -> BGR for cv2)
    img_display = proc_3ch.astype(np.uint8)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
    img_display = cv2.resize(img_display, (640, 640))
    
    # Model Input
    input_6ch = cv2.resize(proc_6ch, (640, 640))
    input_6ch = input_6ch / 255.0
    input_6ch = input_6ch.transpose(2, 0, 1) # HWC -> CHW
    input_6ch = np.expand_dims(input_6ch, axis=0) # Add batch dim
    
    return img_display, torch.from_numpy(input_6ch).float()

def load_labels(label_path):
    """Load YOLO labels from .txt file."""
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    labels = []
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        # xywh normalized
        bbox = [float(x) for x in parts[1:]]
        labels.append({'cls': cls, 'bbox': bbox})
    return labels

def analyze_errors():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    
    # 1. Run Standard Validation for Confusion Matrix
    print("Running standard validation...")
    model.val(data=DATA_YAML, split='val', project=OUTPUT_DIR, name='val_metrics', plots=True)
    
    # 2. Custom Error Analysis
    print("Running custom error analysis on validation set...")
    os.makedirs(os.path.join(OUTPUT_DIR, 'worst_cases'), exist_ok=True)
    
    npy_files = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.npy"))
    
    # Store errors: list of (filename, error_score, type)
    # error_score could be confidence of FP, or 1.0 for FN
    errors_by_class = {k: [] for k in CLASSES.keys()}
    
    for fpath in npy_files:
        fname = os.path.basename(fpath)
        label_name = fname.replace('.npy', '.txt')
        label_path = os.path.join(VAL_LABELS_DIR, label_name)
        
        img_display, tensor_6ch = load_npy_image(fpath)
        if img_display is None:
            continue
            
        gt_labels = load_labels(label_path)
        
        # Inference
        results = model(tensor_6ch.to(model.device), verbose=False)[0]
        preds = results.boxes
        
        # Simple matching logic
        # For each GT, find best matching Pred (IoU > 0.5)
        # If no match -> False Negative
        # For each Pred, if no match with GT -> False Positive
        
        # Convert GT to xyxy (pixel coords) for IoU
        h, w = 640, 640
        gt_boxes = []
        gt_classes = []
        for l in gt_labels:
            c, (x, y, bw, bh) = l['cls'], l['bbox']
            x1 = (x - bw/2) * w
            y1 = (y - bh/2) * h
            x2 = (x + bw/2) * w
            y2 = (y + bh/2) * h
            gt_boxes.append([x1, y1, x2, y2])
            gt_classes.append(c)
            
        gt_boxes = torch.tensor(gt_boxes) if gt_boxes else torch.empty((0, 4))
        
        # Check False Negatives (Missed GT)
        if len(gt_boxes) > 0:
            if len(preds) == 0:
                # All GT are FN
                for i, cls in enumerate(gt_classes):
                    errors_by_class[cls].append({
                        'file': fname,
                        'type': 'FN',
                        'score': 1.0, # High error score
                        'img': img_display,
                        'gt': gt_boxes[i],
                        'pred': None
                    })
            else:
                iou = box_iou(gt_boxes, preds.xyxy.cpu())
                # iou shape: (num_gt, num_preds)
                max_iou, max_idx = iou.max(1) # Best match for each GT
                
                for i, (score, idx) in enumerate(zip(max_iou, max_idx)):
                    cls = gt_classes[i]
                    if score < 0.5: # Missed
                        errors_by_class[cls].append({
                            'file': fname,
                            'type': 'FN',
                            'score': 1.0 - float(score),
                            'img': img_display,
                            'gt': gt_boxes[i],
                            'pred': None
                        })
                    else:
                        # Check class mismatch
                        pred_cls = int(preds.cls[idx])
                        if pred_cls != cls:
                            errors_by_class[cls].append({
                                'file': fname,
                                'type': 'ClassMismatch',
                                'score': float(preds.conf[idx]),
                                'img': img_display,
                                'gt': gt_boxes[i],
                                'pred': preds.xyxy[idx].cpu()
                            })

        # Check False Positives (Ghost detections)
        if len(preds) > 0:
            if len(gt_boxes) == 0:
                # All preds are FP
                for i, box in enumerate(preds):
                    cls = int(box.cls)
                    conf = float(box.conf)
                    errors_by_class[cls].append({
                        'file': fname,
                        'type': 'FP',
                        'score': conf,
                        'img': img_display,
                        'gt': None,
                        'pred': box.xyxy[0].cpu()
                    })
            else:
                iou = box_iou(gt_boxes, preds.xyxy.cpu())
                # For each pred, did it match any GT?
                max_iou_pred, _ = iou.max(0)
                
                for i, score in enumerate(max_iou_pred):
                    if score < 0.5: # No GT matched this pred
                        cls = int(preds.cls[i])
                        conf = float(preds.conf[i])
                        errors_by_class[cls].append({
                            'file': fname,
                            'type': 'FP',
                            'score': conf,
                            'img': img_display,
                            'gt': None,
                            'pred': preds.xyxy[i].cpu()
                        })

    # Visualize Top 5 Errors for each class
    print("Generating error visualizations...")
    for cls_id, errs in errors_by_class.items():
        cls_name = CLASSES[cls_id]
        # Sort by score descending (High confidence FP or High score FN)
        errs.sort(key=lambda x: x['score'], reverse=True)
        
        top_errs = errs[:5]
        if not top_errs:
            continue
            
        print(f"Saving top errors for {cls_name}...")
        for i, err in enumerate(top_errs):
            img = err['img'].copy()
            annotator = Annotator(img, line_width=2, example=str(CLASSES))
            
            # Draw GT if exists (Green)
            if err['gt'] is not None:
                annotator.box_label(err['gt'], f"GT: {cls_name}", color=(0, 255, 0))
                
            # Draw Pred if exists (Red)
            if err['pred'] is not None:
                label = f"Pred: {CLASSES.get(int(preds.cls[0]), 'Unknown')}" if err['type'] == 'ClassMismatch' else f"FP: {cls_name}"
                annotator.box_label(err['pred'], f"{err['type']} {err['score']:.2f}", color=(0, 0, 255))
                
            res_img = annotator.result()
            
            # Add title
            cv2.putText(res_img, f"{err['type']} - {err['file']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out_path = os.path.join(OUTPUT_DIR, 'worst_cases', f"{cls_name}_{i+1}_{err['type']}.jpg")
            cv2.imwrite(out_path, res_img)

    print(f"Error analysis complete. Check {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_errors()
