"""Minimal U-Net training for MARIDA v2-semseg.

Goal: validate whether semantic segmentation (using all 1/2/3 px GT
that detection dropped) can detect Marine_Debris where YOLO can't.

Architecture: 4-stage U-Net (~1.5M params, no pretrain)
Input:  6-channel float32 (B4,B3,B2,B8,B11,B12) p01/p99 normalized -> [0,1]
Output: 4 logits per pixel (bg / Marine_Debris / Algae_Organic / Ship)
Loss:   weighted CE (Marine_Debris weight = 150)
Resolution: native 256x256 (no upsampling)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "data" / "processed" / "marida_v2_semseg"
RUNS_DIR = ROOT / "runs_v2_semseg"
CLASS_NAMES = ["background", "Marine_Debris", "Algae_Organic", "Ship"]
NC = len(CLASS_NAMES)


def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch=6, out_ch=4, base=32):
        super().__init__()
        self.enc1 = conv_bn_relu(in_ch, base)
        self.enc2 = conv_bn_relu(base, base * 2)
        self.enc3 = conv_bn_relu(base * 2, base * 4)
        self.enc4 = conv_bn_relu(base * 4, base * 8)
        self.bottleneck = conv_bn_relu(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = conv_bn_relu(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_bn_relu(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_bn_relu(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_bn_relu(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class SemSegDataset(Dataset):
    def __init__(self, manifest: Path, root: Path, channel_stats: dict, augment: bool):
        self.root = root
        self.augment = augment
        lines = [l.strip() for l in manifest.read_text().splitlines() if l.strip()]
        self.entries = []
        for l in lines:
            img = root / l
            mask = root / l.replace("/images/", "/masks/")
            if img.exists() and mask.exists():
                self.entries.append((img, mask))
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
        # augment: random flip / rotate 90 (geometric only -- spectral preserved)
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
        img_t = torch.from_numpy(img.transpose(2, 0, 1))  # (C, H, W)
        mask_t = torch.from_numpy(mask.astype(np.int64))  # (H, W)
        return img_t, mask_t


@torch.no_grad()
def evaluate(model, loader, device, nc):
    model.eval()
    intersection = torch.zeros(nc, device=device)
    union = torch.zeros(nc, device=device)
    tp = torch.zeros(nc, device=device)
    fp = torch.zeros(nc, device=device)
    fn = torch.zeros(nc, device=device)
    n_pixels = torch.zeros(nc, device=device)
    correct = 0
    total = 0
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(img)
        pred = logits.argmax(dim=1)
        correct += (pred == mask).sum().item()
        total += mask.numel()
        for c in range(nc):
            p = pred == c
            g = mask == c
            intersection[c] += (p & g).sum()
            union[c] += (p | g).sum()
            tp[c] += (p & g).sum()
            fp[c] += (p & ~g).sum()
            fn[c] += (~p & g).sum()
            n_pixels[c] += g.sum()
    iou = (intersection / (union + 1e-7)).cpu().numpy()
    prec = (tp / (tp + fp + 1e-7)).cpu().numpy()
    rec = (tp / (tp + fn + 1e-7)).cpu().numpy()
    f1 = (2 * prec * rec / (prec + rec + 1e-7))
    return {
        "pixel_accuracy": correct / max(total, 1),
        "per_class": {
            CLASS_NAMES[c]: {
                "IoU": float(iou[c]),
                "precision": float(prec[c]),
                "recall": float(rec[c]),
                "F1": float(f1[c]),
                "n_gt_pixels": int(n_pixels[c].item()),
            }
            for c in range(nc)
        },
        "mean_IoU_no_bg": float(iou[1:].mean()),
    }


def train(seed: int, epochs: int, batch: int, lr: float, debris_weight: float,
          algae_weight: float, ship_weight: float, num_workers: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] device={device}, seed={seed}")

    channel_stats = json.loads((DATA_ROOT / "channel_stats_6ch.json").read_text())

    train_ds = SemSegDataset(DATA_ROOT / "train.txt", DATA_ROOT, channel_stats, augment=True)
    val_ds = SemSegDataset(DATA_ROOT / "val.txt", DATA_ROOT, channel_stats, augment=False)
    test_ds = SemSegDataset(DATA_ROOT / "test.txt", DATA_ROOT, channel_stats, augment=False)
    print(f"  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = UNet(in_ch=6, out_ch=NC, base=32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  UNet params: {n_params:,}")

    class_weights = torch.tensor([1.0, debris_weight, algae_weight, ship_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_name = f"unet_b32_s{seed}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  run_dir: {run_dir}")
    print(f"  class_weights: {class_weights.tolist()}")

    best_debris_f1 = -1.0
    best_path = run_dir / "best.pt"
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for img, mask in train_loader:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(img)
                loss = criterion(logits, mask)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optim.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        val_metrics = evaluate(model, val_loader, device, NC)
        debris_f1 = val_metrics["per_class"]["Marine_Debris"]["F1"]
        debris_iou = val_metrics["per_class"]["Marine_Debris"]["IoU"]
        elapsed = time.time() - t0
        print(f"  ep {ep:3d}/{epochs}  loss={train_loss:.4f}  val_debris_F1={debris_f1:.4f}  val_debris_IoU={debris_iou:.4f}  mIoU_no_bg={val_metrics['mean_IoU_no_bg']:.4f}  ({elapsed:.1f}s)")

        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val": val_metrics,
        })

        if debris_f1 > best_debris_f1:
            best_debris_f1 = debris_f1
            torch.save({
                "model": model.state_dict(),
                "epoch": ep,
                "val": val_metrics,
                "arch": "UNet(base=32, in=6, out=4)",
            }, best_path)

    print(f"\n[BEST] val_debris_F1={best_debris_f1:.4f} -> {best_path}")
    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, NC)

    print("\n" + "=" * 60)
    print("TEST METRICS (best-val-debris-F1 checkpoint)")
    print("=" * 60)
    for c in CLASS_NAMES:
        m = test_metrics["per_class"][c]
        print(f"  {c:18s}: IoU={m['IoU']:.4f}  F1={m['F1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  GT_px={m['n_gt_pixels']}")
    print(f"  pixel_accuracy   : {test_metrics['pixel_accuracy']:.4f}")
    print(f"  mIoU (no bg)     : {test_metrics['mean_IoU_no_bg']:.4f}")

    out = {
        "seed": seed,
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "class_weights": class_weights.tolist(),
        "best_val_debris_F1": best_debris_f1,
        "best_epoch": ckpt["epoch"],
        "test": test_metrics,
        "history": history,
    }
    (run_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {run_dir / 'results.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--debris-weight", type=float, default=150.0)
    p.add_argument("--algae-weight", type=float, default=30.0)
    p.add_argument("--ship-weight", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    train(args.seed, args.epochs, args.batch, args.lr,
          args.debris_weight, args.algae_weight, args.ship_weight, args.workers)


if __name__ == "__main__":
    main()
