"""Run U-Net inference and save Marine_Debris probability mask per tile.

For each tile (in input image dir or manifest), load -> p01/p99 normalize ->
forward U-Net -> softmax -> save channel-1 (Marine_Debris) probability
as float32 (H, W) npy.

Used by E pipeline to generate U-Net prior channel for YOLO 7ch training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_semseg_train import UNet, NC

DATA_ROOT = ROOT / "data" / "processed" / "marida_v2_semseg"
RUNS_DIR = ROOT / "runs_v2_semseg"
N_FOLDS = 5


def fold_of(name_or_path: str) -> int:
    stem = Path(name_or_path).stem
    h = hashlib.sha1(stem.encode()).hexdigest()
    return int(h, 16) % N_FOLDS


def load_unet(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = UNet(in_ch=6, out_ch=NC, base=32).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model


def load_channel_stats():
    s = json.loads((DATA_ROOT / "channel_stats_6ch.json").read_text())
    p01 = np.asarray(s["percentiles"]["p01"], dtype=np.float32)[None, None, :]
    p99 = np.asarray(s["percentiles"]["p99"], dtype=np.float32)[None, None, :]
    return p01, p99


def normalize(img_hw6: np.ndarray, p01, p99):
    img = np.nan_to_num(img_hw6, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.maximum(p99 - p01, 1e-6)
    return np.clip((img - p01) / denom, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def predict_debris_prob(model, img_hw6: np.ndarray, p01, p99, device) -> np.ndarray:
    """Return (H, W) float32 Marine_Debris probability in [0, 1]."""
    x = normalize(img_hw6, p01, p99)
    t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(device)
    logits = model(t)
    prob = F.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
    return prob


def main():
    p = argparse.ArgumentParser(description="U-Net inference, write Marine_Debris prob mask npy")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory containing 6ch (or 7ch -- only first 6 used) npy tiles")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Optional manifest .txt of tile paths (relative to input-dir)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to save (H, W) prob npy files (one per tile, same stem)")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Single U-Net checkpoint (used for all tiles)")
    p.add_argument("--use-kfold", action="store_true",
                   help="Use K-fold cross-prediction: each tile predicted by U-Net that did NOT see it")
    p.add_argument("--kfold-dir-template", type=str,
                   default="unet_b32_kfold{fold}_s0",
                   help="Template for K-fold run dirs under runs_v2_semseg/")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    p01, p99 = load_channel_stats()

    if args.manifest:
        rels = [l.strip() for l in args.manifest.read_text().splitlines() if l.strip()]
        tile_paths = [args.input_dir / r for r in rels]
    else:
        tile_paths = sorted(args.input_dir.rglob("*.npy"))

    if args.use_kfold:
        models = {}
        for f in range(N_FOLDS):
            ckpt_path = RUNS_DIR / args.kfold_dir_template.format(fold=f) / "best.pt"
            if not ckpt_path.exists():
                sys.exit(f"missing K-fold ckpt {ckpt_path}")
            models[f] = load_unet(ckpt_path, device)
        print(f"loaded {len(models)} K-fold models")
    else:
        ckpt_path = args.ckpt or (RUNS_DIR / "unet_b32_s0" / "best.pt")
        if not ckpt_path.exists():
            sys.exit(f"ckpt not found: {ckpt_path}")
        models = {"single": load_unet(ckpt_path, device)}
        print(f"loaded single U-Net from {ckpt_path}")

    n_done = 0
    n_skip = 0
    for tp in tqdm(tile_paths, desc="infer"):
        if not tp.exists():
            n_skip += 1
            continue
        arr = np.load(tp)
        if arr.ndim != 3:
            n_skip += 1
            continue
        if arr.shape[-1] >= 6:
            img6 = arr[..., :6]
        else:
            tqdm.write(f"  skip {tp.name}: not enough channels ({arr.shape})")
            n_skip += 1
            continue

        if args.use_kfold:
            f = fold_of(tp.name)
            model = models[f]
        else:
            model = models["single"]
        prob = predict_debris_prob(model, img6, p01, p99, device)
        out_path = args.output_dir / f"{tp.stem}.npy"
        np.save(out_path, prob)
        n_done += 1

    print(f"\ndone: {n_done} tiles -> {args.output_dir}  (skipped {n_skip})")


if __name__ == "__main__":
    main()
