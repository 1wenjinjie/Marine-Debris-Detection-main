"""Aggregate 3-seed results across the 10-cell ablation matrix.

Reads:
- runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_{ablation}_clean_init[_s{1,2}]/v2split_test_eval.json
- runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_{ablation}_clean_init[_s{1,2}]/test_f1_sweep.json
- runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_{ablation}_clean_init[_s{1,2}]/leaky_test_eval.json
- runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_{ablation}_clean_init[_s{1,2}]/test_f1_sweep.json

(The original seed=0 runs don't have an _s0 suffix in their dir name -- they
were the first runs and used the default name.)

Writes:
- docs/paper/v1_ablation_aggregate.json (mean+std per cell, machine-readable)
- docs/paper/v1_ablation_aggregate.md (markdown table for paper)
- docs/paper/figures/ablation_mAP50_F1_bars_with_err.png (error-bar bars)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["Baseline", "+CBAM", "+BiFPN", "+EIoU", "Full"]
ABLATION_KEYS = ["baseline", "cbam", "bifpn", "eiou", "full"]
SEEDS = [0, 1, 2]
FIG_DIR = ROOT / "docs/paper/figures"


def run_dir(setting: str, ab: str, seed: int) -> Path:
    if setting == "honest":
        base = ROOT / "runs/detect/runs_v1_v2split"
        # Honest baseline s=0 was the very first run, trained via
        # v1_train_on_v2split.py and named without the "_baseline_" tag.
        if seed == 0 and ab == "baseline":
            return base / "yolov11n_7ch_v2split_physidx_clean_init"
        name = f"yolov11n_7ch_v2split_physidx_{ab}_clean_init"
    else:  # leaky
        base = ROOT / "runs/detect/runs_v1_leaky"
        name = f"yolov11n_7ch_leaky_{ab}_clean_init"
    if seed == 0:
        return base / name
    return base / f"{name}_s{seed}"


def eval_path(setting: str, run: Path) -> Path:
    name = "v2split_test_eval.json" if setting == "honest" else "leaky_test_eval.json"
    return run / name


def collect_one(setting: str, ab: str) -> dict:
    """Collect mAP50/mAP50:95/P/R/F1 across seeds for one cell."""
    map50, map95, precision, recall, f1, debris_ap50, ship_ap50 = [], [], [], [], [], [], []
    seeds_used = []
    for seed in SEEDS:
        rd = run_dir(setting, ab, seed)
        ep = eval_path(setting, rd)
        if not ep.exists():
            continue
        try:
            d = json.loads(ep.read_text())
        except Exception as e:
            print(f"  WARNING: cannot read {ep}: {e}")
            continue
        map50.append(d["mAP50"])
        map95.append(d["mAP50_95"])
        precision.append(d["precision"])
        recall.append(d["recall"])
        pc = d.get("per_class", {})
        d_ap = (pc.get("Marine_Debris") or {}).get("AP50")
        s_ap = (pc.get("Ship") or {}).get("AP50")
        if d_ap is not None:
            debris_ap50.append(d_ap)
        if s_ap is not None:
            ship_ap50.append(s_ap)
        f1_path = rd / "test_f1_sweep.json"
        if f1_path.exists():
            sweep = json.loads(f1_path.read_text())
            f1.append(sweep["best"]["f1"])
        seeds_used.append(seed)

    def msd(xs):
        if not xs:
            return None, None, []
        return (mean(xs), stdev(xs) if len(xs) > 1 else 0.0, xs)

    return {
        "setting": setting,
        "ablation": ab,
        "n_seeds": len(seeds_used),
        "seeds": seeds_used,
        "mAP50": dict(zip(["mean", "std", "all"], msd(map50))),
        "mAP50_95": dict(zip(["mean", "std", "all"], msd(map95))),
        "precision": dict(zip(["mean", "std", "all"], msd(precision))),
        "recall": dict(zip(["mean", "std", "all"], msd(recall))),
        "f1": dict(zip(["mean", "std", "all"], msd(f1))),
        "Marine_Debris_AP50": dict(zip(["mean", "std", "all"], msd(debris_ap50))),
        "Ship_AP50": dict(zip(["mean", "std", "all"], msd(ship_ap50))),
    }


def main():
    aggregate = {}
    for setting in ("honest", "leaky"):
        for ab in ABLATION_KEYS:
            cell = collect_one(setting, ab)
            aggregate[f"{setting}_{ab}"] = cell

    out_json = ROOT / "docs/paper/v1_ablation_aggregate.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(aggregate, indent=2))
    print(f"saved {out_json}")

    # Markdown table
    lines = ["# 3-seed aggregate (mean +- std)", "",
             "| Model | Setting | n_seeds | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for setting in ("leaky", "honest"):
        for name, ab in zip(MODELS, ABLATION_KEYS):
            c = aggregate[f"{setting}_{ab}"]
            def fmt(d):
                if d.get("mean") is None:
                    return "--"
                return f"{d['mean']:.3f} +- {d['std']:.3f}"
            lines.append(
                f"| {name} | {setting.capitalize()} | {c['n_seeds']} | "
                f"{fmt(c['mAP50'])} | {fmt(c['mAP50_95'])} | {fmt(c['f1'])} | "
                f"{fmt(c['Marine_Debris_AP50'])} | {fmt(c['Ship_AP50'])} |"
            )
    out_md = ROOT / "docs/paper/v1_ablation_aggregate.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_md}")

    # Print to console
    print("\n" + "=" * 70)
    for line in lines:
        print(line)

    # Error-bar figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(MODELS))
    width = 0.38
    for ax, key, title, ylabel, ylim in [
        (axes[0], "mAP50", "mAP@0.5 mean +- std (3 seeds)", "mAP@0.5", (0, 0.85)),
        (axes[1], "f1", "Best F1 mean +- std (3 seeds)", "Best F1", (0, 0.9)),
    ]:
        leaky_m = []
        leaky_s = []
        honest_m = []
        honest_s = []
        for ab in ABLATION_KEYS:
            cl = aggregate[f"leaky_{ab}"][key]
            ch = aggregate[f"honest_{ab}"][key]
            leaky_m.append(cl.get("mean") or 0)
            leaky_s.append(cl.get("std") or 0)
            honest_m.append(ch.get("mean") or 0)
            honest_s.append(ch.get("std") or 0)
        ax.bar(x - width / 2, leaky_m, width, yerr=leaky_s, capsize=4,
               label="Leaky", color="#d6604d", edgecolor="black", linewidth=0.6,
               error_kw={"elinewidth": 1.2, "capthick": 1.2})
        ax.bar(x + width / 2, honest_m, width, yerr=honest_s, capsize=4,
               label="Honest", color="#4393c3", edgecolor="black", linewidth=0.6,
               error_kw={"elinewidth": 1.2, "capthick": 1.2})
        ax.set_xticks(x, MODELS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.legend(loc="upper right")
        for j in range(len(MODELS)):
            ax.text(x[j] - width / 2, leaky_m[j] + leaky_s[j] + 0.005,
                    f"{leaky_m[j]:.3f}", ha="center", va="bottom", fontsize=7)
            ax.text(x[j] + width / 2, honest_m[j] + honest_s[j] + 0.005,
                    f"{honest_m[j]:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Architecture ablation: 3-seed mean +- std", fontsize=12)
    fig.tight_layout()
    out_fig = FIG_DIR / "ablation_mAP50_F1_bars_with_err.png"
    fig.savefig(out_fig, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
