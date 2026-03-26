"""Inspect knowledge summary produced by preprocess_generate_knowledge_channel.py
Prints basic statistics about confidence_weight and optionally writes a histogram PNG.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(path: Path):
    if not path.exists():
        print(f"Summary file not found: {path}")
        return 2
    payload = json.loads(path.read_text())
    details = payload.get('details', [])
    total = len(details)
    have_conf = [d for d in details if 'confidence_weight' in d and d['confidence_weight'] is not None]
    missing = total - len(have_conf)
    print(f"Loaded summary: {path} -> total={total}, with_confidence={len(have_conf)}, missing={missing}")

    if not have_conf:
        print("No confidence_weight entries found. Run preprocess_generate_knowledge_channel.py --skip-existing --summary <path> to populate them.")
        return 0

    weights = [float(d['confidence_weight']) for d in have_conf]
    print(f"weights: min={min(weights):.4f}, max={max(weights):.4f}, mean={statistics.mean(weights):.4f}, median={statistics.median(weights):.4f}")

    # Save histogram
    outdir = Path('runs')
    outdir.mkdir(exist_ok=True)
    outpng = outdir / f"confidence_weights_hist.png"
    plt.figure(figsize=(6,4))
    plt.hist(weights, bins=30, color='C0', edgecolor='k')
    plt.title('Confidence weight distribution')
    plt.xlabel('confidence_weight')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(outpng)
    print(f"Saved histogram to {outpng}")
    return 0


if __name__ == '__main__':
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('mask_bbox_rescue_split/knowledge_generation_summary.json')
    raise SystemExit(main(p))
