import sys
from pathlib import Path
import torch
import math
from ultralytics import YOLO

def bbox_siou(pred_boxes, target_boxes, eps=1e-7):
    """
    Compute Scylla IoU (SIoU) loss.
    SIoU = IoU - (Angle cost + Distance cost + Shape cost)
    
    Paper: https://arxiv.org/abs/2205.12740
    """
    px1, py1, px2, py2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    tx1, ty1, tx2, ty2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
    
    # Intersection
    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # Union
    pred_area = (px2 - px1) * (py2 - py1)
    target_area = (tx2 - tx1) * (ty2 - ty1)
    union_area = pred_area + target_area - inter_area + eps
    iou = inter_area / union_area
    
    # Centers
    pred_cx = (px1 + px2) / 2
    pred_cy = (py1 + py2) / 2
    target_cx = (tx1 + tx2) / 2
    target_cy = (ty1 + ty2) / 2
    
    # Enclosing box
    enclose_x1 = torch.min(px1, tx1)
    enclose_y1 = torch.min(py1, ty1)
    enclose_x2 = torch.max(px2, tx2)
    enclose_y2 = torch.max(py2, ty2)
    enclose_w = (enclose_x2 - enclose_x1).clamp(min=eps)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=eps)
    enclose_c = (enclose_w ** 2 + enclose_h ** 2).sqrt() + eps
    
    # Distance cost
    rho = ((pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2).sqrt()
    distance_cost = (rho / enclose_c) ** 2
    
    # Angle cost
    sigma = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
    sin_alpha = torch.abs(pred_cy - target_cy) / (sigma.sqrt() + eps)
    sin_beta = torch.abs(pred_cx - target_cx) / (sigma.sqrt() + eps)
    threshold = (2 / math.pi) ** 0.5
    sin_alpha = torch.where(sin_alpha > threshold, sin_beta, sin_alpha)
    angle_cost = 2 * sin_alpha ** 2
    
    # Shape cost
    pred_w = (px2 - px1).clamp(min=eps)
    pred_h = (py2 - py1).clamp(min=eps)
    target_w = (tx2 - tx1).clamp(min=eps)
    target_h = (ty2 - ty1).clamp(min=eps)
    omega_w = torch.abs(pred_w - target_w) / torch.max(pred_w, target_w)
    omega_h = torch.abs(pred_h - target_h) / torch.max(pred_h, target_h)
    omega = (1 - torch.exp(-omega_w)) ** 2 + (1 - torch.exp(-omega_h)) ** 2
    shape_cost = omega
    
    # SIoU formula
    siou = iou - (distance_cost + angle_cost + shape_cost) / 2
    return siou

def main():
    sys.path.insert(0, str(Path.cwd()))
    from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer
    
    print("="*80)
    print("Experiment C: 10-Channel + SIoU Loss")
    print("="*80)
    
    # Patch SIoU
    import ultralytics.utils.metrics as metrics_module
    print(" Patching bbox_iou to use SIoU (Scylla IoU)")
    print("  - Angle cost: Penalizes orientation misalignment")
    print("  - Distance cost: Center point distance penalty")
    print("  - Shape cost: Width/height aspect ratio penalty")
    metrics_module.bbox_iou = lambda box1, box2, *args, **kwargs: bbox_siou(box1, box2)
    
    # Prepare 10ch weights (reuse if exists)
    if Path("yolo11n_10ch.pt").exists():
        print(" Using existing 10-channel weights")
        weights = "yolo11n_10ch.pt"
    else:
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
        weights = "yolo11n_10ch.pt"
    
    # Load and train
    model = YOLO(weights)
    print(" Starting training: 50 epochs, batch 8, SIoU loss")
    print("="*80)
    
    results = model.train(
        data="configs/datasets/marida_mask_rescue_10ch.yaml",
        epochs=50,
        imgsz=256,
        batch=8,
        device=0,
        project="runs_marida",
        name="exp_c_10ch_siou",
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
    print(" Experiment C Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
