#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

def iou_distance(boxes, centroids):
    boxes_wh = boxes.reshape(-1, 1, 2)
    centroids_wh = centroids.reshape(1, -1, 2)
    
    intersect_wh = np.minimum(boxes_wh, centroids_wh)
    intersect_area = np.prod(intersect_wh, axis=2)
    
    box_area = np.prod(boxes_wh, axis=2)
    centroid_area = np.prod(centroids_wh, axis=2)
    union_area = box_area + centroid_area - intersect_area
    
    return 1 - (intersect_area / (union_area + 1e-10))

def kmeans_anchors(boxes, k=9, max_iter=300):
    np.random.seed(42)
    centroids = boxes[np.random.choice(len(boxes), k, replace=False)]
    
    prev_assignments = None
    for iteration in range(max_iter):
        distances = iou_distance(boxes, centroids)
        assignments = np.argmin(distances, axis=1)
        
        if prev_assignments is not None and np.all(assignments == prev_assignments):
            break
        prev_assignments = assignments
        
        for i in range(k):
            mask = assignments == i
            if np.sum(mask) > 0:
                centroids[i] = np.mean(boxes[mask], axis=0)
    
    avg_iou = 1 - np.mean(np.min(iou_distance(boxes, centroids), axis=1))
    return centroids, avg_iou

if __name__ == "__main__":
    labels_dir = Path("mask_bbox_rescue_split/train/labels")
    all_boxes = []
    
    for label_file in tqdm(list(labels_dir.glob("*.txt"))):
        lines = label_file.read_text().strip().split("\n")
        for line in lines:
            if line.strip():
                parts = line.split()
                w, h = float(parts[3]), float(parts[4])
                all_boxes.append([w, h])
    
    all_boxes = np.array(all_boxes)
    print(f"Total boxes: {len(all_boxes)}")
    print(f"Box size stats - Mean: {all_boxes.mean(axis=0)}, Std: {all_boxes.std(axis=0)}")
    
    anchors_p5, iou_p5 = kmeans_anchors(all_boxes, k=9)
    anchors_p5 = anchors_p5[np.argsort(anchors_p5[:, 0] * anchors_p5[:, 1])]
    
    print(f"\nOptimal P5 Anchors (avg IoU={iou_p5:.4f}):")
    print(anchors_p5)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(all_boxes[:, 0], all_boxes[:, 1], alpha=0.3, s=1, label="Ground Truth Boxes")
    plt.scatter(anchors_p5[:, 0], anchors_p5[:, 1], c="red", s=200, marker="x", label="K-Means Anchors")
    plt.xlabel("Width (normalized)")
    plt.ylabel("Height (normalized)")
    plt.legend()
    plt.title(f"MARIDA Anchor Optimization (avg IoU={iou_p5:.4f})")
    plt.savefig("marida_anchor_analysis.png", dpi=150)
    print("\nSaved: marida_anchor_analysis.png")
    
    config = {
        "anchors": anchors_p5.tolist(),
        "avg_iou": float(iou_p5),
        "num_boxes": len(all_boxes),
        "box_stats": {
            "mean_w": float(all_boxes[:, 0].mean()),
            "mean_h": float(all_boxes[:, 1].mean()),
            "std_w": float(all_boxes[:, 0].std()),
            "std_h": float(all_boxes[:, 1].std())
        }
    }
    Path("marida_optimized_anchors.yaml").write_text(yaml.dump(config, sort_keys=False))
    print("Saved: marida_optimized_anchors.yaml")
