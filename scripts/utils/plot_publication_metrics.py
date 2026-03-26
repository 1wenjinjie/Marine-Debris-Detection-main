"""Generate publication-ready PR curves and confusion matrices from YOLO predictions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


@dataclass
class Prediction:
    image_id: str
    bbox: np.ndarray  # xyxy
    score: float
    class_id: int


@dataclass
class GroundTruth:
    image_id: str
    bbox: np.ndarray  # xyxy
    class_id: int
    gid: int


@dataclass
class MatchRecord:
    score: float
    pred_class: int
    matched: bool
    matched_gid: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="Path to predictions.json produced by run_val")
    parser.add_argument("labels_dir", type=Path, help="Directory that contains ground-truth YOLO txt labels")
    parser.add_argument("images_dir", type=Path, help="Directory that contains .npy validation images")
    parser.add_argument("data_yaml", type=Path, help="Dataset YAML that defines class names")
    parser.add_argument("output_dir", type=Path, help="Where to save plots and summaries")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a true positive")
    parser.add_argument(
        "--threshold-range",
        type=str,
        default="0.05:0.95:0.01",
        help="Confidence sweep specification (start:stop:step or comma list)",
    )
    parser.add_argument(
        "--category-offset",
        type=int,
        default=1,
        help="Value subtracted from predictions' category_id (COCO exports use 1)",
    )
    parser.add_argument("--dpi", type=int, default=320, help="Figure DPI for saved images")
    return parser.parse_args()


def parse_threshold_spec(spec: str) -> List[float]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty threshold specification")
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError("Range spec must be start:stop:step")
        start, stop, step = map(float, parts)
        if step <= 0:
            raise ValueError("Step must be positive")
        values = []
        current = start
        while current <= stop + 1e-9:
            values.append(round(current, 6))
            current += step
        return values
    return [float(token) for token in spec.split(",") if token]


def xywh_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: float, img_h: float) -> np.ndarray:
    bw = w * img_w
    bh = h * img_h
    cx = xc * img_w
    cy = yc * img_h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def xywh_abs_to_xyxy(x: float, y: float, w: float, h: float) -> np.ndarray:
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    denom = area1 + area2 - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def load_class_names(data_yaml: Path) -> List[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names")
    if isinstance(names, dict):
        ordered = [name for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
        return [str(name) for name in ordered]
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return [str(name) for name in names]
    raise ValueError(f"Unable to parse class names from {data_yaml}")


def load_ground_truth(labels_dir: Path, images_dir: Path) -> Tuple[Dict[str, List[GroundTruth]], Dict[int, GroundTruth], Counter]:
    gts_by_image: Dict[str, List[GroundTruth]] = {}
    gt_lookup: Dict[int, GroundTruth] = {}
    counts: Counter = Counter()
    shape_cache: Dict[str, Tuple[int, int]] = {}
    gid = 0
    label_paths = sorted(labels_dir.glob("*.txt"))
    if not label_paths:
        raise FileNotFoundError(f"No label files found in {labels_dir}")
    for label_path in label_paths:
        image_id = label_path.stem
        entries: List[GroundTruth] = []
        content = label_path.read_text(encoding="utf-8").strip().splitlines()
        if not content:
            gts_by_image[image_id] = []
            continue
        for line in content:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            if image_id not in shape_cache:
                img_path = images_dir / f"{image_id}.npy"
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing image file for {image_id}: {img_path}")
                arr = np.load(img_path, mmap_mode="r")
                shape_cache[image_id] = (arr.shape[1], arr.shape[0])  # (width, height)
                del arr
            img_w, img_h = shape_cache[image_id]
            bbox = xywh_to_xyxy(xc, yc, w, h, img_w, img_h)
            gt = GroundTruth(image_id=image_id, bbox=bbox, class_id=cls, gid=gid)
            entries.append(gt)
            gt_lookup[gid] = gt
            counts[cls] += 1
            gid += 1
        gts_by_image[image_id] = entries
    return gts_by_image, gt_lookup, counts


def load_predictions(
    predictions_path: Path, num_classes: int, category_offset: int
) -> List[Prediction]:
    with predictions_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    predictions: List[Prediction] = []
    for item in raw:
        raw_cls = int(item.get("category_id", -1))
        cls = raw_cls - category_offset
        if cls < 0 or cls >= num_classes:
            continue
        bbox = item.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        xyxy = xywh_abs_to_xyxy(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        predictions.append(
            Prediction(
                image_id=str(item.get("image_id") or Path(item.get("file_name", "")).stem),
                bbox=xyxy,
                score=float(item.get("score", 0.0)),
                class_id=cls,
            )
        )
    predictions.sort(key=lambda p: p.score, reverse=True)
    if not predictions:
        raise RuntimeError(f"No predictions loaded from {predictions_path}")
    return predictions


def match_predictions(
    predictions: Sequence[Prediction],
    gts_by_image: Dict[str, List[GroundTruth]],
    iou_th: float,
) -> List[MatchRecord]:
    gt_used: Dict[Tuple[str, int], bool] = {}
    matches: List[MatchRecord] = []
    for pred in predictions:
        candidates = [gt for gt in gts_by_image.get(pred.image_id, []) if gt.class_id == pred.class_id]
        best_iou = 0.0
        best_gt: GroundTruth | None = None
        for gt in candidates:
            if gt_used.get((gt.image_id, gt.gid)):
                continue
            iou = compute_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        if best_gt is not None and best_iou >= iou_th:
            gt_used[(best_gt.image_id, best_gt.gid)] = True
            matches.append(
                MatchRecord(score=pred.score, pred_class=pred.class_id, matched=True, matched_gid=best_gt.gid)
            )
        else:
            matches.append(MatchRecord(score=pred.score, pred_class=pred.class_id, matched=False, matched_gid=None))
    return matches


def compute_pr_curve(
    records: Sequence[MatchRecord],
    thresholds: Sequence[float],
    total_gts: int,
) -> List[Dict[str, float]]:
    scores = np.array([rec.score for rec in records], dtype=np.float32)
    matched = np.array([1.0 if rec.matched else 0.0 for rec in records], dtype=np.float32)
    order = np.argsort(-scores)
    scores = scores[order]
    matched = matched[order]
    results: List[Dict[str, float]] = []
    for thr in thresholds:
        mask = scores >= thr
        if not np.any(mask):
            precision = 0.0
            recall = 0.0
        else:
            tp = float(matched[mask].sum())
            fp = float(mask.sum()) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / total_gts if total_gts > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append({"threshold": float(thr), "precision": precision, "recall": recall, "f1": f1})
    return results


def build_confusion_matrix(
    records: Sequence[MatchRecord],
    gt_lookup: Dict[int, GroundTruth],
    threshold: float,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    detected_gids = set()
    for rec in records:
        if rec.score < threshold:
            continue
        if rec.matched and rec.matched_gid is not None:
            gt = gt_lookup[rec.matched_gid]
            matrix[gt.class_id, rec.pred_class] += 1
            detected_gids.add(rec.matched_gid)
        else:
            matrix[num_classes, rec.pred_class] += 1
    for gid, gt in gt_lookup.items():
        if gid not in detected_gids:
            matrix[gt.class_id, num_classes] += 1
    return matrix


def compute_class_metrics(matrix: np.ndarray, class_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    num_classes = len(class_names)
    for idx, name in enumerate(class_names):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        support = tp + fn
        predictions = tp + fp
        precision = tp / predictions if predictions > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "predictions": predictions,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def plot_pr_curve(points: Sequence[Dict[str, float]], output_path: Path, dpi: int, best_point: Dict[str, float]) -> None:
    recalls = [pt["recall"] for pt in points]
    precisions = [pt["precision"] for pt in points]
    thresholds = [pt["threshold"] for pt in points]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recalls, precisions, color="#1f77b4", linewidth=2.5)
    ax.scatter(best_point["recall"], best_point["precision"], color="#d62728", s=60, zorder=3)
    ax.annotate(
        f"Best F1={best_point['f1']:.3f}\nconf={best_point['threshold']:.2f}",
        (best_point["recall"], best_point["precision"]),
        textcoords="offset points",
        xytext=(20, -30),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", lw=1.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="#d62728"),
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    sm = ax.scatter(recalls, precisions, c=thresholds, cmap="viridis", s=10)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Confidence threshold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_f1_curve(points: Sequence[Dict[str, float]], output_path: Path, dpi: int) -> None:
    thresholds = [pt["threshold"] for pt in points]
    f1_scores = [pt["f1"] for pt in points]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, f1_scores, color="#2ca02c", linewidth=2.0)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 vs Confidence")
    ax.set_xlim(min(thresholds), max(thresholds))
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], output_path: Path, dpi: int) -> None:
    labels = list(class_names) + ["Background"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Confusion Matrix (best threshold)")

    max_val = matrix.max() if matrix.size else 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > max_val * 0.6 else "black"
            ax.text(j, i, int(value), ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.data_yaml)
    gts_by_image, gt_lookup, gt_counts = load_ground_truth(args.labels_dir, args.images_dir)
    predictions = load_predictions(
        args.predictions,
        num_classes=len(class_names),
        category_offset=args.category_offset,
    )
    records = match_predictions(predictions, gts_by_image, args.iou)

    total_gts = sum(gt_counts.values())
    thresholds = parse_threshold_spec(args.threshold_range)
    pr_points = compute_pr_curve(records, thresholds, total_gts)
    best_point = max(pr_points, key=lambda pt: (pt["f1"], pt["recall"]))

    matrix = build_confusion_matrix(records, gt_lookup, best_point["threshold"], len(class_names))
    class_metrics = compute_class_metrics(matrix, class_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_pr_curve(pr_points, args.output_dir / "publication_pr_curve.png", args.dpi, best_point)
    plot_f1_curve(pr_points, args.output_dir / "f1_vs_conf.png", args.dpi)
    plot_confusion_matrix(matrix, class_names, args.output_dir / "publication_confusion_matrix.png", args.dpi)

    class_distribution = {
        class_names[idx]: int(gt_counts.get(idx, 0)) for idx in range(len(class_names))
    }
    summary = {
        "best_threshold": best_point["threshold"],
        "precision": best_point["precision"],
        "recall": best_point["recall"],
        "f1": best_point["f1"],
        "iou_threshold": args.iou,
        "total_ground_truth": total_gts,
        "class_distribution": class_distribution,
        "class_metrics": class_metrics,
    }
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved publication plots to {args.output_dir}")


if __name__ == "__main__":
    main()
