"""Instance-level Recall evaluation for U-Net predictions.

Converts U-Net pixel predictions into instances via connected components, then
matches against ground-truth instances (the same 13 / N filtered MIN=4 Marine_Debris
instances that YOLO sees). This produces a directly comparable instance Recall
metric.

Two cuts:
  (a) MIN_PIXELS=4 GT instances (what YOLO trains/tests on)  -- direct YOLO comparison
  (b) MIN_PIXELS=1 GT instances (all raw GT)                  -- shows what U-Net can grab that YOLO never could
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_semseg_train import UNet, SemSegDataset, NC, CLASS_NAMES

DATA_ROOT = ROOT / "data" / "processed" / "marida_v2_semseg"
RUNS_DIR = ROOT / "runs_v2_semseg"


def match_instances(pred_mask: np.ndarray, gt_mask: np.ndarray, class_id: int,
                    gt_min_pixels: int):
    """For one tile: count GT instances (>=gt_min_pixels for class) and how many
    are 'hit' by at least one predicted pixel of the same class.

    Returns (n_gt_instances, n_hit_instances).
    """
    gt_bin = (gt_mask == class_id)
    if not gt_bin.any():
        return 0, 0
    labeled, n = ndimage.label(gt_bin)
    pred_bin = (pred_mask == class_id)
    n_gt = 0
    n_hit = 0
    for cid in range(1, n + 1):
        comp = (labeled == cid)
        size = int(comp.sum())
        if size < gt_min_pixels:
            continue
        n_gt += 1
        if (pred_bin & comp).any():
            n_hit += 1
    return n_gt, n_hit


@torch.no_grad()
def eval_seed(seed: int, gt_min_pixels_list: list[int] = (1, 4)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = RUNS_DIR / f"unet_b32_s{seed}" / "best.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = UNet(in_ch=6, out_ch=NC, base=32).to(device).eval()
    model.load_state_dict(ckpt["model"])

    channel_stats = json.loads((DATA_ROOT / "channel_stats_6ch.json").read_text())
    ds = SemSegDataset(DATA_ROOT / "test.txt", DATA_ROOT, channel_stats, augment=False)

    totals = {c: {m: {"gt": 0, "hit": 0} for m in gt_min_pixels_list}
              for c in range(1, NC)}

    for i in range(len(ds)):
        img_t, mask_t = ds[i]
        img = img_t.unsqueeze(0).to(device)
        logits = model(img)
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        gt = mask_t.numpy()
        for c in range(1, NC):
            for m in gt_min_pixels_list:
                n_gt, n_hit = match_instances(pred, gt, c, m)
                totals[c][m]["gt"] += n_gt
                totals[c][m]["hit"] += n_hit

    out = {}
    for c in range(1, NC):
        out[CLASS_NAMES[c]] = {}
        for m in gt_min_pixels_list:
            gt = totals[c][m]["gt"]
            hit = totals[c][m]["hit"]
            out[CLASS_NAMES[c]][f"MIN={m}"] = {
                "gt_instances": gt,
                "detected_instances": hit,
                "instance_recall": hit / gt if gt > 0 else 0.0,
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    all_results = {}
    for s in args.seeds:
        r = eval_seed(s)
        if r is None:
            print(f"[SKIP] seed {s}: no best.pt")
            continue
        all_results[f"s{s}"] = r
        print(f"\n=== Seed {s} (test split, instance-level) ===")
        for cls, m_dict in r.items():
            for m_key, v in m_dict.items():
                print(f"  {cls:18s} {m_key:6s}: {v['detected_instances']:3d}/{v['gt_instances']:3d}  Recall={v['instance_recall']:.4f}")

    if len(all_results) > 1:
        print("\n=== 3-seed aggregate ===")
        seeds = list(all_results.keys())
        for cls in CLASS_NAMES[1:]:
            for m_key in all_results[seeds[0]][cls]:
                recalls = [all_results[s][cls][m_key]["instance_recall"] for s in seeds]
                hits = [all_results[s][cls][m_key]["detected_instances"] for s in seeds]
                gts = [all_results[s][cls][m_key]["gt_instances"] for s in seeds]
                mean = np.mean(recalls)
                std = np.std(recalls)
                print(f"  {cls:18s} {m_key:6s}: hits={hits} / gt={gts[0]} -> Recall {mean:.4f} +- {std:.4f}")

    out_path = RUNS_DIR / "instance_recall_summary.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
