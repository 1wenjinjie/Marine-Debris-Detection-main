"""Generate paper figures from the 10 ablation cells.

Produces (per teacher's figure list in the task plan):
- Figure 5 (Ablation bars):     ablation_mAP50_F1_bars.png
- Figure 6 (FPS-mAP tradeoff):  fps_vs_map50.png
- Figure 3 (mAP comparison):    map50_grouped_bars.png  (per-setting grouping)
- Per-class breakdown:          per_class_AP50_grouped.png
- Leak inflation:               leak_inflation.png

All artifacts saved to docs/paper/figures/.
Source data: the eval JSONs + F1 sweep JSONs + fps_bench JSONs we already wrote.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = ROOT / "docs/paper/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load(p: str | Path) -> dict:
    return json.loads(Path(p).read_text())


MODELS = ["Baseline", "+CBAM", "+BiFPN", "+EIoU", "Full"]
HONEST_RUNS = [
    "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_clean_init",
    "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_cbam_clean_init",
    "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_bifpn_clean_init",
    "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_eiou_clean_init",
    "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_full_clean_init",
]
LEAKY_RUNS = [
    "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_baseline_clean_init",
    "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_cbam_clean_init",
    "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_bifpn_clean_init",
    "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_eiou_clean_init",
    "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_full_clean_init",
]


def collect():
    """Build a unified data dict keyed by (setting, model)."""
    data = {}
    for setting, runs, eval_name in [
        ("Honest", HONEST_RUNS, "v2split_test_eval.json"),
        ("Leaky",  LEAKY_RUNS,  "leaky_test_eval.json"),
    ]:
        for name, run in zip(MODELS, runs):
            d = _load(Path(run) / eval_name)
            f1 = _load(Path(run) / "test_f1_sweep.json")
            fps = _load(Path(run) / "fps_bench.json")
            data[(setting, name)] = {
                "mAP50": d["mAP50"],
                "mAP50_95": d["mAP50_95"],
                "precision": d["precision"],
                "recall": d["recall"],
                "best_f1": f1["best"]["f1"],
                "best_f1_conf": f1["best"]["conf"],
                "fps": fps["fps_median"],
                "latency_ms": fps["latency_ms_median"],
                "per_class": d.get("per_class", {}),
                "params": None,
            }
    return data


def _bar_group(ax, data, key, title, ylabel, ylim=None):
    x = np.arange(len(MODELS))
    width = 0.38
    honest = [data[("Honest", m)][key] for m in MODELS]
    leaky = [data[("Leaky", m)][key] for m in MODELS]
    b1 = ax.bar(x - width / 2, leaky, width, label="Leaky (v1 GT+random split)",
                color="#d6604d", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, honest, width, label="Honest (NDWI+scene-clean)",
                color="#4393c3", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="best", frameon=True)
    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.005,
                    f"{rect.get_height():.3f}", ha="center", va="bottom", fontsize=8)


def fig_ablation_bars(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _bar_group(axes[0], data, "mAP50",   "Figure 5a: mAP@0.5 by ablation cell",   "mAP@0.5",   (0, 0.85))
    _bar_group(axes[1], data, "best_f1", "Figure 5b: Best F1 by ablation cell",   "Best F1",   (0, 0.9))
    fig.suptitle("Architecture ablation matrix under two evaluation settings", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "ablation_mAP50_F1_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_fps_tradeoff(data):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {"Baseline": "#1f77b4", "+CBAM": "#ff7f0e", "+BiFPN": "#2ca02c",
              "+EIoU": "#d62728", "Full": "#9467bd"}
    markers = {"Honest": "o", "Leaky": "s"}
    for setting in ("Honest", "Leaky"):
        for m in MODELS:
            d = data[(setting, m)]
            ax.scatter(d["fps"], d["mAP50"], s=160, c=colors[m], marker=markers[setting],
                       edgecolor="black", linewidth=0.8, alpha=0.85,
                       label=f"{m} ({setting})")
            ax.annotate(f"{m}\n{setting}", (d["fps"], d["mAP50"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=7.5)
    ax.set_xlabel("Inference speed (FPS, RTX 3090 Ti, imgsz=512, batch=1, forward only)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Figure 6: Speed-accuracy tradeoff (10 ablation cells)")
    ax.grid(alpha=0.3, linestyle=":")
    # Two-handle legend just for marker meaning
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                          markeredgecolor="black", markersize=10, label="Honest setting"),
               plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                          markeredgecolor="black", markersize=10, label="Leaky setting")]
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "fps_vs_map50.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_per_class_bars(data):
    classes = ["Marine_Debris", "Ship", "Foam"]  # ones with consistent test support
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    x = np.arange(len(MODELS))
    width = 0.25
    for ax, setting in zip(axes, ("Leaky", "Honest")):
        for i, cls in enumerate(classes):
            vals = []
            for m in MODELS:
                pc = data[(setting, m)]["per_class"].get(cls)
                vals.append(pc["AP50"] if pc else 0.0)
            ax.bar(x + (i - 1) * width, vals, width, label=cls,
                   edgecolor="black", linewidth=0.5)
        ax.set_xticks(x, MODELS)
        ax.set_ylabel("AP@0.5")
        ax.set_title(f"{setting} setting")
        ax.legend()
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.set_ylim(0, 1.05)
    fig.suptitle("Per-class AP@0.5 across ablation matrix", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "per_class_AP50_grouped.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_leak_inflation(data):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    metrics = [("mAP50", "mAP@0.5"), ("best_f1", "Best F1"), ("mAP50_95", "mAP@0.5:0.95")]
    x = np.arange(len(MODELS))
    width = 0.27
    for i, (key, ylabel) in enumerate(metrics):
        diff = [data[("Leaky", m)][key] - data[("Honest", m)][key] for m in MODELS]
        ax.bar(x + (i - 1) * width, diff, width, label=ylabel,
               edgecolor="black", linewidth=0.5)
        for j, d in enumerate(diff):
            ax.text(x[j] + (i - 1) * width, d + 0.01, f"+{d:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel("Leaky minus Honest (inflation by leak)")
    ax.set_title("How much each metric is inflated by GT prior + random split leak")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    out = FIG_DIR / "leak_inflation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_map50_grouped(data):
    """Figure 3-style: grouped bars of mAP50 + mAP50:95 per setting per model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, setting in zip(axes, ("Leaky", "Honest")):
        x = np.arange(len(MODELS))
        width = 0.4
        map50 = [data[(setting, m)]["mAP50"] for m in MODELS]
        map95 = [data[(setting, m)]["mAP50_95"] for m in MODELS]
        ax.bar(x - width / 2, map50, width, label="mAP@0.5",
               color="#4393c3", edgecolor="black", linewidth=0.6)
        ax.bar(x + width / 2, map95, width, label="mAP@0.5:0.95",
               color="#92c5de", edgecolor="black", linewidth=0.6)
        ax.set_xticks(x, MODELS)
        ax.set_ylabel("mAP")
        ax.set_title(f"{setting} setting")
        ax.set_ylim(0, 0.85)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.legend(loc="upper right")
        for j, (a, b) in enumerate(zip(map50, map95)):
            ax.text(x[j] - width / 2, a + 0.008, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(x[j] + width / 2, b + 0.008, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Figure 3: mAP@0.5 and mAP@0.5:0.95 per ablation cell", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "map50_grouped_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_training_curves():
    """Per-epoch mAP50 curve across all ablations (figure 3 in teacher's list)."""
    import csv as _csv
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (title, runs) in zip(axes, [
        ("Leaky setting", LEAKY_RUNS),
        ("Honest setting", HONEST_RUNS),
    ]):
        for name, run in zip(MODELS, runs):
            csv_path = Path(run) / "results.csv"
            if not csv_path.exists():
                continue
            with csv_path.open(encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
            epochs = [int(r["epoch"]) for r in rows]
            # Ultralytics writes "metrics/mAP50(B)"
            key = next((k for k in rows[0] if "mAP50(B)" in k and "95" not in k), None)
            if key is None:
                continue
            map50 = [float(r[key]) for r in rows]
            ax.plot(epochs, map50, marker="o", markersize=3, label=name, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation mAP@0.5")
        ax.set_title(title)
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(loc="best")
    fig.suptitle("Figure 3: Per-epoch validation mAP@0.5 across ablations", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "training_curves_map50.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    print(f"writing to {FIG_DIR}")
    data = collect()
    fig_ablation_bars(data)
    fig_map50_grouped(data)
    fig_per_class_bars(data)
    fig_fps_tradeoff(data)
    fig_leak_inflation(data)
    fig_training_curves()


if __name__ == "__main__":
    main()
