#!/usr/bin/env python3
"""Exp-F: 7ch + SIoU Loss - Breakthrough to F1>0.70"""
import sys, os, torch, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

def compute_siou(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1
    inter_x1, inter_y1 = torch.max(b1_x1, b2_x1), torch.max(b1_y1, b2_y1)
    inter_x2, inter_y2 = torch.min(b1_x2, b2_x2), torch.min(b1_y2, b2_y2)
    inter_w, inter_h = (inter_x2 - inter_x1).clamp(0), (inter_y2 - inter_y1).clamp(0)
    inter_area = inter_w * inter_h
    union_area = b1_w * b1_h + b2_w * b2_h - inter_area + eps
    iou = inter_area / union_area
    cw, ch = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1), torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    cx_diff, cy_diff = b2_cx - b1_cx, b2_cy - b1_cy
    sigma = torch.pow(cx_diff ** 2 + cy_diff ** 2, 0.5)
    sin_alpha, sin_beta = torch.abs(cy_diff) / (sigma + eps), torch.abs(cx_diff) / (sigma + eps)
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha > threshold, sin_beta, sin_alpha)
    angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - math.pi / 4), 2)
    rho = (cx_diff ** 2 + cy_diff ** 2) / ((cw ** 2 + ch ** 2) + eps)
    gamma = 2 - angle_cost
    distance_cost = 1 - torch.exp(-gamma * rho)
    omega_w, omega_h = torch.abs(b1_w - b2_w) / torch.max(b1_w, b2_w).clamp(eps), torch.abs(b1_h - b2_h) / torch.max(b1_h, b2_h).clamp(eps)
    theta = 4
    shape_cost = torch.pow(1 - torch.exp(-omega_w), theta) + torch.pow(1 - torch.exp(-omega_h), theta)
    return (iou - distance_cost - 0.5 * shape_cost).squeeze(-1)

def monkey_patch_siou():
    from ultralytics.utils import metrics
    original_bbox_iou = metrics.bbox_iou
    def patched_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        if GIoU or DIoU or CIoU:
            return original_bbox_iou(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps)
        if xywh:
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
            box1 = torch.cat([b1_x1, b1_y1, b1_x2, b1_y2], dim=-1)
            box2 = torch.cat([b2_x1, b2_y1, b2_x2, b2_y2], dim=-1)
        return compute_siou(box1, box2, eps=eps)
    metrics.bbox_iou = patched_bbox_iou
    try:
        from ultralytics.utils import loss as loss_module
        loss_module.bbox_iou = patched_bbox_iou
    except ImportError:
        pass
    print(' SIoU loss injected successfully')

if __name__ == '__main__':
    print('='*70)
    print('Experiment F: 7ch + SIoU Loss')
    print('Target: F1 > 0.70 (Current SOTA: F1=0.6774)')
    print('='*70)
    monkey_patch_siou()
    model = YOLO('yolo11n_7ch_init.pt' if os.path.exists('yolo11n_7ch_init.pt') else 'yolo11n.pt')
    model.train(
        trainer=MaskRescueTrainer,
        data='configs/datasets/marida_mask_rescue_7ch.yaml',
        epochs=50, batch=8, imgsz=512, device='0', workers=8,
        optimizer='auto', lr0=0.01, lrf=0.01,
        project='runs_marida', name='exp_f_7ch_siou',
        patience=50, amp=True, verbose=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=180.0, flipud=0.5, fliplr=0.5,
        mosaic=0.0, mixup=0.0, copy_paste=0.0
    )
    print('\n' + '='*70)
    print('Exp-F training complete!')
    print('Results saved in: runs_marida/exp_f_7ch_siou/')
    print('='*70)
