from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import rasterio
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为多光谱数据生成知识蒸馏热图通道(基于GT mask距离变换) - 重建版"
    )
    parser.add_argument("--source-dataset", type=Path, default=Path("MARIDA_YOLO_6band_split"))
    parser.add_argument("--target-dataset", type=Path, default=Path("mask_bbox_rescue_split"))
    parser.add_argument("--mask-root", type=Path, default=Path("MARIDA/patches"))
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--summary", type=Path, default=Path("mask_bbox_rescue_split/knowledge_generation_summary.json"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--conf-alpha", type=float, default=0.4)
    parser.add_argument("--conf-scale", type=float, default=0.1)
    return parser.parse_args()


def derive_scene_name(patch_name: str) -> str:
    parts = patch_name.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else patch_name


def load_gt_mask(mask_root: Path, patch_stem: str) -> np.ndarray | None:
    scene = derive_scene_name(patch_stem)
    path = mask_root / scene / f"{patch_stem}_cl.tif"
    if not path.exists():
        return None
    try:
        with rasterio.open(path) as src:
            return src.read(1)
    except Exception:
        return None


def mask_to_heatmap(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    m = dist.max()
    if m > 1e-6:
        return (dist / m).astype(np.float32)
    return dist.astype(np.float32)


def compute_heatmap_stats(hm: np.ndarray) -> Dict:
    return {
        "knowledge_mean": float(hm.mean()),
        "knowledge_max": float(hm.max()),
        "knowledge_coverage": float((hm > 0.01).sum() / hm.size)
    }


def compute_confidence_weight(stats: Dict, alpha: float = 0.4, scale: float = 0.1) -> float:
    return float(alpha + scale * (stats["knowledge_mean"] + stats["knowledge_coverage"]) / 2)


def process_sample(src: Path, dst: Path, mask_root: Path, alpha: float, scale: float, skip: bool) -> Dict:
    if skip and dst.exists():
        try:
            a = np.load(dst)
            if a.ndim == 3 and a.shape[2] == 7:
                hm = a[:, :, 6]
                s = compute_heatmap_stats(hm)
                return {"status": "skipped", **s, "confidence_weight": compute_confidence_weight(s, alpha, scale)}
        except Exception:
            pass
    try:
        a6 = np.load(src)
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    h, w, _ = a6.shape
    patch = src.stem
    mask = load_gt_mask(mask_root, patch)
    if mask is not None and mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    hm = mask_to_heatmap(mask) if mask is not None else np.zeros((h, w), dtype=np.float32)
    a7 = np.concatenate([a6, hm[..., None]], axis=2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, a7.astype(np.float32))
    stats = compute_heatmap_stats(hm)
    return {"status": "ok", **stats, "confidence_weight": compute_confidence_weight(stats, alpha, scale)}


def process_split(src_root: Path, dst_root: Path, mask_root: Path, split: str, alpha: float, scale: float, skip: bool) -> List[Dict]:
    src_dir = src_root / split / "images"
    dst_dir = dst_root / split / "images"
    if not src_dir.exists():
        return []
    files = sorted(src_dir.glob("*.npy"))
    out = []
    for f in tqdm(files, desc=f"{split}"):
        rel = f"{split}/images/{f.name}"
        r = process_sample(f, dst_dir / f.name, mask_root, alpha, scale, skip)
        out.append({"relative_path": rel, **r})
    return out


def main():
    args = parse_args()
    sr = args.source_dataset.resolve()
    tr = args.target_dataset.resolve()
    mr = args.mask_root.resolve()
    summary = args.summary if args.summary.is_absolute() else tr / args.summary
    all_details = []
    for s in args.splits:
        all_details.extend(process_split(sr, tr, mr, s, args.conf_alpha, args.conf_scale, args.skip_existing))
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"details": all_details}, indent=2))


if __name__ == '__main__':
    main()
