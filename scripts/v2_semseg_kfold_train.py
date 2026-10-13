"""5-fold cross-prediction U-Net training for leak-free prior generation.

To use U-Net Marine_Debris probability as a prior channel for YOLO without
leak, the prior on TRAIN tiles must come from a U-Net that did NOT see those
tiles during training. We achieve this with K-fold:

  - Partition v2 train (973 tiles) into 5 folds by deterministic hash
  - Train 5 separate U-Nets, each on 4/5 of train + full v2 val
  - Each U-Net's "holdout fold" gets that U-Net's predictions
  - Concatenate all holdout predictions → full v2 train coverage, leak-free

For v2 val / v2 test we use the regular (full-train) seed=0 U-Net (from D),
which is the largest-data U-Net and never saw val/test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_semseg_train import UNet, NC, CLASS_NAMES

DATA_ROOT = ROOT / "data" / "processed" / "marida_v2_semseg"
RUNS_DIR = ROOT / "runs_v2_semseg"
N_FOLDS = 5


def fold_of(tile_relpath: str) -> int:
    """Deterministic fold assignment by sha1 hash of tile name."""
    stem = Path(tile_relpath).stem
    h = hashlib.sha1(stem.encode()).hexdigest()
    return int(h, 16) % N_FOLDS


class FoldedSemSegDataset(Dataset):
    def __init__(self, entries: list[tuple[Path, Path]], channel_stats: dict, augment: bool):
        self.entries = entries
        self.augment = augment
        p01 = np.asarray(channel_stats["percentiles"]["p01"], dtype=np.float32)
        p99 = np.asarray(channel_stats["percentiles"]["p99"], dtype=np.float32)
        self.p01 = p01[None, None, :]
        self.p99 = p99[None, None, :]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        img_path, mask_path = self.entries[i]
        img = np.load(img_path)
        mask = np.load(mask_path)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        denom = np.maximum(self.p99 - self.p01, 1e-6)
        img = np.clip((img - self.p01) / denom, 0.0, 1.0).astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[::-1, :, :])
                mask = np.ascontiguousarray(mask[::-1, :])
            k = np.random.randint(0, 4)
            if k:
                img = np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
                mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))
        return (torch.from_numpy(img.transpose(2, 0, 1)),
                torch.from_numpy(mask.astype(np.int64)))


def load_split_entries(manifest: Path, root: Path):
    lines = [l.strip() for l in manifest.read_text().splitlines() if l.strip()]
    out = []
    for l in lines:
        img = root / l
        mask = root / l.replace("/images/", "/masks/")
        if img.exists() and mask.exists():
            out.append((img, mask, l))
    return out


@torch.no_grad()
def quick_eval(model, loader, device, nc):
    model.eval()
    tp = torch.zeros(nc, device=device)
    fp = torch.zeros(nc, device=device)
    fn = torch.zeros(nc, device=device)
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred = model(img).argmax(dim=1)
        for c in range(nc):
            p = pred == c
            g = mask == c
            tp[c] += (p & g).sum()
            fp[c] += (p & ~g).sum()
            fn[c] += (~p & g).sum()
    prec = (tp / (tp + fp + 1e-7)).cpu().numpy()
    rec = (tp / (tp + fn + 1e-7)).cpu().numpy()
    f1 = (2 * prec * rec / (prec + rec + 1e-7))
    return {CLASS_NAMES[c]: float(f1[c]) for c in range(nc)}


def train_fold(fold_id: int, epochs: int, batch: int, lr: float,
               debris_w: float, algae_w: float, ship_w: float, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}\nFold {fold_id}/{N_FOLDS-1} (seed={seed})\n{'='*60}")

    channel_stats = json.loads((DATA_ROOT / "channel_stats_6ch.json").read_text())
    train_all = load_split_entries(DATA_ROOT / "train.txt", DATA_ROOT)
    val_all = load_split_entries(DATA_ROOT / "val.txt", DATA_ROOT)

    train_entries = [(im, mk) for (im, mk, rel) in train_all if fold_of(rel) != fold_id]
    holdout_entries = [(im, mk) for (im, mk, rel) in train_all if fold_of(rel) == fold_id]
    val_entries = [(im, mk) for (im, mk, _) in val_all]
    print(f"  train (4 folds): {len(train_entries)}")
    print(f"  holdout fold {fold_id}: {len(holdout_entries)}")
    print(f"  val (full v2 val): {len(val_entries)}")

    train_loader = DataLoader(FoldedSemSegDataset(train_entries, channel_stats, augment=True),
                              batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(FoldedSemSegDataset(val_entries, channel_stats, augment=False),
                            batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet(in_ch=6, out_ch=NC, base=32).to(device)
    class_weights = torch.tensor([1.0, debris_w, algae_w, ship_w], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    run_dir = RUNS_DIR / f"unet_b32_kfold{fold_id}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_f1 = -1.0

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_b = 0
        for img, mask in train_loader:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optim.zero_grad()
            logits = model(img)
            loss = criterion(logits, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total_loss += loss.item()
            n_b += 1
        scheduler.step()
        val_f1 = quick_eval(model, val_loader, device, NC)
        debris_f1 = val_f1["Marine_Debris"]
        print(f"  ep {ep:3d}/{epochs}  loss={total_loss/max(n_b,1):.4f}  val_debris_F1={debris_f1:.4f}  ({time.time()-t0:.1f}s)")
        if debris_f1 > best_f1:
            best_f1 = debris_f1
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "fold_id": fold_id, "best_val_debris_F1": best_f1},
                       best_path)

    print(f"[FOLD {fold_id} done] best val_debris_F1={best_f1:.4f}")
    (run_dir / "fold_meta.json").write_text(json.dumps({
        "fold_id": fold_id, "n_train": len(train_entries),
        "n_holdout": len(holdout_entries), "n_val": len(val_entries),
        "best_val_debris_F1": best_f1, "epochs": epochs, "seed": seed,
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--debris-weight", type=float, default=150.0)
    p.add_argument("--algae-weight", type=float, default=30.0)
    p.add_argument("--ship-weight", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for f in args.folds:
        train_fold(f, args.epochs, args.batch, args.lr,
                   args.debris_weight, args.algae_weight, args.ship_weight, args.seed)


if __name__ == "__main__":
    main()
