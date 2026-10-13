"""Locked-number figures for the 2026-08-21 advisor report. n=3 seeds shown as points."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

OUT = Path(__file__).resolve().parent
OKABE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "black": "#000000",
}


def _setup_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    available = {f.name for f in fm.fontManager.ttflist}
    family = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def forest(ax, labels, means, lows, highs, seeds, xlabel, title) -> None:
    y = np.arange(len(labels))
    ax.axvline(0.0, color="#888888", lw=0.8, ls="--")
    for i, (mean, lo, hi, seed_vals) in enumerate(zip(means, lows, highs, seeds)):
        color = OKABE["vermillion"] if hi < 0 or lo > 0 and mean < 0 else OKABE["blue"]
        if lo <= 0 <= hi:
            color = OKABE["blue"]
        if lo > 0:
            color = OKABE["green"]
        if hi < 0:
            color = OKABE["vermillion"]
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="D", color=color, ms=6, capsize=3, zorder=3)
        jitter = np.linspace(-0.12, 0.12, len(seed_vals))
        ax.scatter(seed_vals, np.full(len(seed_vals), i) + jitter, s=28, color=OKABE["black"], zorder=4, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()


def fig1_channel() -> None:
    labels = ["NDWI7 − 6ch\n(主对比)", "NDWI7 − shuffled7\n(机制对照)", "shuffled7 − 6ch"]
    means = [-0.0712, -0.0342, -0.0369]
    lows = [-0.2016, -0.1715, -0.0611]
    highs = [0.0190, 0.0120, 0.0302]
    seeds = [
        [-0.0444, -0.0961, -0.0729],
        [0.0408, -0.0221, -0.1214],
        [-0.0853, -0.0740, 0.0485],
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    forest(ax, labels, means, lows, highs, seeds, "Marine_Debris AP50 配对差（scene bootstrap 95% CI）", "图1  通道设计：NDWI 先验不支持增益")
    ax.plot([], [], "D", color=OKABE["blue"], label="均值 + 95% CI")
    ax.scatter([], [], s=28, color=OKABE["black"], label="3 个配对 seed")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, "fig1_channel_contrasts")


def fig2_twoby2() -> None:
    labels = ["Clean\nNDWI", "Clean\nOracle", "Shared\nNDWI", "Shared\nOracle"]
    seeds = [
        [0.5124, 0.4861, 0.4678],
        [0.9375, 0.9659, 0.9448],
        [0.5517, 0.5323, 0.5775],
        [0.9166, 0.9414, 0.9313],
    ]
    means = [float(np.mean(s)) for s in seeds]
    colors = [OKABE["sky"], OKABE["vermillion"], OKABE["blue"], OKABE["orange"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(4)
    ax.bar(x, means, color=colors, width=0.55, alpha=0.85, zorder=2)
    for i, seed_vals in enumerate(seeds):
        jitter = np.linspace(-0.12, 0.12, 3)
        ax.scatter(np.full(3, i) + jitter, seed_vals, s=36, color=OKABE["black"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Marine_Debris AP50（test）")
    ax.set_ylim(0, 1.05)
    ax.set_title("图2  2×2 泄漏归因：高分来自 GT 先验，不是场景泄漏")
    ax.text(0.02, 0.95, "黑点 = 3 个 seed；柱高 = 均值\nOracle 主效应 +0.418（CI 高于 0）", transform=ax.transAxes, va="top", fontsize=8)
    _save(fig, "fig2_leakage_2x2")


def fig3_min() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    ax = axes[0]
    xlabels = ["MIN=1\nGT=206", "MIN=2\nGT=117", "MIN=4\nGT=32", "MIN=8\nGT=7"]
    seeds = [
        [0.3776, 0.4432, 0.4088],
        [0.1880, 0.1749, 0.2383],
        [0.2624, 0.1404, 0.1119],
        [0.0743, 0.0638, 0.0311],
    ]
    means = [float(np.mean(s)) for s in seeds]
    x = np.arange(4)
    ax.bar(x, means, color=OKABE["sky"], width=0.55, alpha=0.85, zorder=2)
    for i, seed_vals in enumerate(seeds):
        ax.scatter(np.full(3, i) + np.linspace(-0.12, 0.12, 3), seed_vals, s=28, color=OKABE["black"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Marine_Debris AP50")
    ax.set_ylim(0, 0.55)
    ax.set_title("匹配定义（描述性）")

    ax2 = axes[1]
    labels = ["train MIN=1\n相对 MIN=4", "train MIN=2\n相对 MIN=4", "train MIN=8\n相对 MIN=4"]
    means = [-0.0537, -0.0471, -0.1514]
    lows = [-0.1189, -0.1155, -0.2673]
    highs = [0.0569, 0.0405, 0.0062]
    seed_list = [
        [-0.1169, -0.0443, 0.0001],
        [-0.1448, -0.0343, 0.0379],
        [-0.2510, -0.1127, -0.0905],
    ]
    forest(ax2, labels, means, lows, highs, seed_list, "固定 MIN=4 任务上的 AP50 差", "训练端 MIN 敏感性")
    fig.suptitle("图3  MIN：改小目标训练无助于固定 MIN=4 任务", y=1.02)
    _save(fig, "fig3_min_sensitivity")


def fig4_external() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    ax = axes[0]
    x = np.arange(3)
    ax.axhline(21, color="#888888", ls=":", lw=0.8, label="全部 21 个靶标")
    ax.scatter(x, [0, 0, 0], s=48, color=OKABE["vermillion"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["seed 0\nτ=0.20", "seed 1\nτ=0.02", "seed 2\nτ=0.10"])
    ax.set_ylabel("HDPE 回收（/21）")
    ax.set_ylim(-1, 22)
    ax.set_title("PLP2021 人工靶标")
    ax.text(1.0, 10.5, "三个 seed 均为 0 / 21\n5 m ≈ 1 px，低于 MIN=4", ha="center", va="center", fontsize=9)

    ax2 = axes[1]
    labels = ["6ch", "NDWI7"]
    seeds = [[0.0407, 0.0391, 0.0313], [0.0661, 0.0384, 0.0413]]
    means = [0.0370, 0.0486]
    x = np.arange(2)
    ax2.bar(x, means, color=[OKABE["blue"], OKABE["orange"]], width=0.45, alpha=0.85, zorder=2)
    for i, seed_vals in enumerate(seeds):
        ax2.scatter(np.full(3, i) + np.linspace(-0.1, 0.1, 3), seed_vals, s=36, color=OKABE["black"], zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Marine_Debris AP50")
    ax2.set_ylim(0, 0.12)
    ax2.set_title("MADOS 官方 test 零样本")
    fig.suptitle("图4  外部验证：两条路径均不支持部署主张", y=1.02)
    _save(fig, "fig4_external")


def main() -> None:
    _setup_font()
    fig1_channel()
    fig2_twoby2()
    fig3_min()
    fig4_external()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
