"""Generate comparative charts for the six rerun experiments.

This script reads `marida_runs/yolo11n_rerun_summary.json` (best-epoch metrics
recorded during training) and `marida_runs/yolo11n_val_metrics.json`
(post-hoc validation metrics), then plots side-by-side bars and their deltas
for precision, recall, mAP50, and mAP50-95. The resulting figure is saved to
`comparison_results/rerun_channel_diff.png` for meeting slides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
TRAIN_SUMMARY = ROOT / "marida_runs" / "yolo11n_rerun_summary.json"
VAL_SUMMARY = ROOT / "marida_runs" / "yolo11n_val_metrics.json"
OUTPUT_DIR = ROOT / "comparison_results"
OUTPUT_PATH = OUTPUT_DIR / "rerun_channel_diff.png"

RUN_ORDER: List[Tuple[str, str]] = [
    ("3ch", "3ch 基线"),
    ("3ch_rerun2", "3ch 复跑2"),
    ("3ch_lowdata", "3ch 低数据"),
    ("4ch", "4ch NIR"),
    ("5ch", "5ch SWIR"),
    ("6ch", "6ch 全波段"),
    ("6ch_tuned", "6ch 调优")
]

METRICS = [
    ("precision", "metrics/precision(B)", "Precision"),
    ("recall", "metrics/recall(B)", "Recall"),
    ("mAP50", "metrics/mAP50(B)", "mAP50"),
    ("mAP50_95", "metrics/mAP50-95(B)", "mAP50-95"),
]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    train_data = load_json(TRAIN_SUMMARY)
    val_data = load_json(VAL_SUMMARY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(RUN_ORDER))
    width = 0.32

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes_flat = axes.flatten()

    for ax, (train_key, val_key, title) in zip(axes_flat, METRICS):
        train_vals = [train_data[run][train_key] for run, _ in RUN_ORDER]
        val_vals = [val_data[run]["metrics"][val_key] for run, _ in RUN_ORDER]
        delta_vals = np.array(val_vals) - np.array(train_vals)

        ax.bar(x - width / 2, train_vals, width=width, label="训练最好", color="#4C72B0")
        ax.bar(x + width / 2, val_vals, width=width, label="验证", color="#55A868")
        for xi, dv, tv, vv in zip(x, delta_vals, train_vals, val_vals):
            ax.text(
                xi,
                max(tv, vv) + 0.01,
                f"Δ {dv:+.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#C44E52"
            )

        ax.set_title(title)
        ax.set_ylabel("值")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes_flat[0].legend(loc="upper right")
    plt.xticks(x, [label for _, label in RUN_ORDER], rotation=15)
    fig.suptitle("YOLO11n 渠道重跑：训练 vs 验证 指标对比", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    fig.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved comparison figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
