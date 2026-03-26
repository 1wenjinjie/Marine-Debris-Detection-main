"""Batch inference entrypoint for MaskRescue-style NPY/GeoTIFF inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics_plugins.mask_rescue_dataset import DEFAULT_CHANNEL_STATS


ARRAY_EXTENSIONS = {".npy", ".npz", ".tif", ".tiff"}


def _parse_names(raw_names) -> Dict[int, str]:
    if raw_names is None:
        return {}
    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}
    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}
    raise TypeError(f"Unsupported names format: {type(raw_names)}")


def _resolve_paths(source: Path, dataset_root: Path) -> List[Path]:
    source = Path(source)
    if source.suffix.lower() == ".txt" and source.exists():
        paths: List[Path] = []
        for raw in source.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = (source.parent / candidate).resolve()
            if not candidate.exists():
                alt = (dataset_root / raw).resolve()
                if alt.exists():
                    candidate = alt
                else:
                    LOGGER.warning(f"Skipping missing entry {raw} from {source}")
                    continue
            paths.append(candidate)
        if not paths:
            raise FileNotFoundError(f"{source} resolved to 0 tensors")
        return paths

    if source.is_dir():
        paths = [p for p in sorted(source.rglob("*")) if p.is_file() and p.suffix.lower() in ARRAY_EXTENSIONS]
        if not paths:
            raise FileNotFoundError(f"Directory {source} contained no supported tensors {ARRAY_EXTENSIONS}")
        return paths

    if source.is_file() and source.suffix.lower() in ARRAY_EXTENSIONS:
        return [source]

    rel_candidate = (dataset_root / source).resolve()
    if rel_candidate.exists() and rel_candidate.is_file() and rel_candidate.suffix.lower() in ARRAY_EXTENSIONS:
        return [rel_candidate]

    raise FileNotFoundError(f"Unsupported source specification: {source}")


def _load_channel_stats(channels: int, override: Path | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    stats_path = Path(override) if override else DEFAULT_CHANNEL_STATS.get(channels)
    if not stats_path:
        return None
    stats_path = Path(stats_path)
    if not stats_path.exists():
        LOGGER.warning(f"Channel stats file {stats_path} not found; falling back to default scaling")
        return None
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning(f"Failed to parse {stats_path}: {exc}; falling back to default scaling")
        return None
    p01 = payload.get("percentiles", {}).get("p01")
    p99 = payload.get("percentiles", {}).get("p99")
    if not p01 or not p99 or len(p01) != len(p99):
        LOGGER.warning(f"Channel stats malformed in {stats_path}; using fallback scaling")
        return None
    low = np.array(p01, dtype=np.float32).reshape(1, 1, -1)
    high = np.array(p99, dtype=np.float32).reshape(1, 1, -1)
    span = np.maximum(high - low, 1e-6)
    scale = 255.0 / span
    return low, high, scale


def _load_array(path: Path, channels: int, stats: Tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> np.ndarray:
    import rasterio

    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            arr = src.read().transpose(1, 2, 0)
    else:
        arr = np.load(path, allow_pickle=False)

    if arr.ndim != 3:
        raise ValueError(f"Array at {path} must be 3-D, got {arr.shape}")
    if arr.shape[0] in (6, 7, 11, 13) and arr.shape[0] <= arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)

    if arr.shape[-1] > channels:
        if arr.shape[-1] >= 11:
            if channels == 6:
                arr = arr[..., [3, 2, 1, 7, 9, 10]]
            elif channels == 3:
                arr = arr[..., [3, 2, 1]]
            else:  # default first N
                arr = arr[..., :channels]
        else:
            arr = arr[..., :channels]
    elif arr.shape[-1] < channels:
        raise ValueError(f"{path} has {arr.shape[-1]} channels (expected {channels})")

    if stats:
        low, high, scale = stats
        arr = np.clip(arr, low, high)
        arr = (arr - low) * scale
    else:
        arr = np.clip(arr * 1500.0, 0.0, 255.0)

    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        np.clip(arr, 0.0, 255.0, out=arr)
    return arr.astype(np.float32)


def _letterbox(im: np.ndarray, imgsz: int, color: int = 0) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    h, w = im.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = imgsz - new_w
    dh = imgsz - new_h
    dw /= 2
    dh /= 2
    top, bottom = int(np.floor(dh)), int(np.ceil(dh))
    left, right = int(np.floor(dw)), int(np.ceil(dw))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, (r, r), (left, top)


def _prepare_preview(arr: np.ndarray) -> np.ndarray:
    rgb = arr[..., :3].astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _save_txt(label_path: Path, boxes_xyxy: np.ndarray, confs: Sequence[float], classes: Sequence[int],
              orig_shape: Tuple[int, int], save_conf: bool) -> None:
    h, w = orig_shape
    lines: List[str] = []
    for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, confs, classes):
        x_c = ((x1 + x2) / 2) / w
        y_c = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        if save_conf:
            lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f} {conf:.6f}")
        else:
            lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _collect_stats(records: List[Dict[str, object]]) -> Dict[str, object]:
    totals = {"total_images": len(records), "total_detections": 0}
    counts: Dict[str, int] = {}
    confs: Dict[str, List[float]] = {}
    samples: Dict[str, List[str]] = {}
    for rec in records:
        for det in rec.get("detections", []):
            label = det["label"]
            totals["total_detections"] += 1
            counts[label] = counts.get(label, 0) + 1
            confs.setdefault(label, []).append(det["confidence"])
            samples.setdefault(label, []).append(rec["image_name"])
    avg_conf = {k: float(sum(v) / len(v)) for k, v in confs.items() if v}
    return {
        **totals,
        "counts_by_class": counts,
        "mean_confidence_by_class": avg_conf,
        "sample_paths_by_class": {k: v[:10] for k, v in samples.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Path to YOLO weights (.pt)")
    parser.add_argument("source", type=Path, help="Manifest, directory, or file containing NPY/TIF tensors")
    parser.add_argument("--data", type=Path, default=Path("marida_rescue_7ch.yaml"))
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--channel-stats", type=Path, default=None, help="Optional channel stats JSON to control scaling")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--project", type=str, default="runs_marida")
    parser.add_argument("--name", type=str, default="mask_rescue_infer")
    parser.add_argument("--max-images", type=int, default=None, help="Optionally stop after N images for smoke tests")
    parser.add_argument("--save-img", action="store_true", help="Save annotated PNGs")
    parser.add_argument("--save-txt", action="store_true", help="Export YOLO TXT predictions")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = args.weights.resolve()
    data_path = args.data.resolve()
    source_path = args.source

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    declared_channels = data_cfg.get("channels")
    channels = args.channels or declared_channels or 3
    dataset_root = args.dataset_root or data_cfg.get("path") or data_path.parent
    dataset_root = Path(dataset_root).resolve()
    class_names = _parse_names(data_cfg.get("names"))

    tensors = _resolve_paths(source_path, dataset_root)
    if args.max_images:
        tensors = tensors[: args.max_images]

    stats = _load_channel_stats(channels, args.channel_stats)
    LOGGER.info(
        f"Running inference on {len(tensors)} tensors with {channels} channels (dataset_root={dataset_root})"
    )

    model = YOLO(str(weights_path))
    save_dir = Path(args.project) / args.name
    if args.save_img:
        (save_dir / "images").mkdir(parents=True, exist_ok=True)
    if args.save_txt:
        (save_dir / "labels").mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for idx, tensor_path in enumerate(tensors, start=1):
        arr = _load_array(tensor_path, channels, stats)
        orig_h, orig_w = arr.shape[:2]
        padded, ratio, pad = _letterbox(arr, args.imgsz)
        input_tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0) / 255.0
        device_attr = getattr(model, "device", None)
        if device_attr is not None:
            input_tensor = input_tensor.to(device_attr)
        results = model(
            input_tensor,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
        res = results[0]
        det_record = {"image_path": str(tensor_path), "image_name": tensor_path.name, "detections": []}
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy().copy()
            confs = res.boxes.conf.cpu().numpy().tolist()
            clses = res.boxes.cls.cpu().numpy().astype(int).tolist()
            boxes[:, [0, 2]] -= pad[0]
            boxes[:, [1, 3]] -= pad[1]
            boxes[:, [0, 2]] /= ratio[0]
            boxes[:, [1, 3]] /= ratio[1]
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

            if args.save_txt:
                label_path = save_dir / "labels" / f"{tensor_path.stem}.txt"
                _save_txt(label_path, boxes, confs, clses, (orig_h, orig_w), args.save_conf)

            if args.save_img:
                preview = _prepare_preview(arr)
                annotator = Annotator(preview.copy(), line_width=2, example=list(class_names.values()))
                for box, cls, conf in zip(boxes, clses, confs):
                    label = f"{class_names.get(cls, cls)} {conf:.2f}"
                    annotator.box_label(box, label, color=colors(cls, True))
                out_path = save_dir / "images" / f"{tensor_path.stem}_pred.jpg"
                cv2.imwrite(str(out_path), annotator.result())

            for box, cls, conf in zip(boxes, clses, confs):
                det_record["detections"].append(
                    {
                        "label": class_names.get(cls, str(cls)),
                        "cls_id": cls,
                        "confidence": float(conf),
                        "xyxy": [float(x) for x in box.tolist()],
                    }
                )

        records.append(det_record)
        if idx % 10 == 0 or idx == len(tensors):
            LOGGER.info(f"Processed {idx}/{len(tensors)} tensors")

    stats_summary = _collect_stats(records)
    stats_summary.update(
        {
            "weights": str(weights_path),
            "source": str(source_path),
            "data": str(data_path),
            "conf": args.conf,
            "iou": args.iou,
            "output_dir": str(save_dir),
        }
    )

    summary_path = args.summary_json or (save_dir / "inference_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats_summary, indent=2), encoding="utf-8")
    LOGGER.info(f"Inference summary saved to {summary_path}")


if __name__ == "__main__":
    main()
