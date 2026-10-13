"""Generate the 5 patent drawings (CNIPA-style B&W line drawings).
Run:  conda run -n marine python docs/patent/figures/make_patent_figures.py
Outputs: docs/patent/figures/fig1.png ... fig5.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("d:/OceanEdge_Group/Marine-Debris-Detection-main")
OUT = ROOT / "docs/patent/figures"
OUT.mkdir(parents=True, exist_ok=True)

BLACK = "black"
LW = 1.4


def box(ax, xy, w, h, text, ref=None, fs=11):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=LW, edgecolor=BLACK, facecolor="white")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=BLACK)
    if ref is not None:
        ax.text(x + w - 0.04, y + h - 0.04, ref, ha="right", va="top", fontsize=9, color=BLACK)
    return (x + w / 2, y, x + w / 2, y + h)  # bottom-center, top-center helpers


def arrow(ax, p0, p1):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                        linewidth=LW, color=BLACK, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


# ---------------------------------------------------------------- 图1 流程图
def fig1():
    fig, ax = plt.subplots(figsize=(6.2, 9.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.axis("off")
    W, H, X = 7.4, 1.5, 1.3
    steps = [
        (13.9, "S1  获取海面多光谱遥感影像\n（含绿光、近红外等多个光谱波段）", "101"),
        (11.7, "S2  由光谱波段确定性计算物理先验通道 P\nP=clip((1−NDWI)/2,0,1)，不依赖人工标注", "102"),
        (9.5, "S3  光谱波段与先验通道沿通道维拼接\n并逐通道归一化（p01/p99）", "103"),
        (7.3, "S3′  按保持长宽比的填充方式上采样\n使目标尺度落入浅层检测头响应区间", "104"),
        (5.1, "S4  多通道 CNN 检测器联合卷积\n（颈部加权双向特征金字塔融合）", "105"),
        (2.9, "输出 海洋漂浮垃圾实例级边界框\n及对应置信度", "106"),
        (0.7, "F  对置信度施加 Platt 单调校准\np=sigmoid(a·logit(s)+b) → 校准概率", "107"),
    ]
    centers = []
    for y, t, r in steps:
        box(ax, (X, y), W, H, t, ref=r, fs=10.5)
        centers.append((X + W / 2, y, X + W / 2, y + H))
    for i in range(len(centers) - 1):
        arrow(ax, (centers[i][0], centers[i][1]), (centers[i + 1][0], centers[i + 1][3]))
    ax.set_title("图 1  海洋漂浮垃圾多光谱检测方法总体流程图", fontsize=12, pad=6)
    fig.savefig(OUT / "fig1.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 图2 先验通道对照
def fig2():
    name = "S2_18-9-20_16PCC_24"
    img = np.load(ROOT / f"data/processed/mask_bbox_rescue_split_physidx/test/images/{name}.npy")
    lbl = (ROOT / f"data/processed/mask_bbox_rescue_split_physidx/test/labels/{name}.txt").read_text().splitlines()
    Hh, Ww = img.shape[0], img.shape[1]

    def stretch(a):
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    vis = stretch(img[..., :3].mean(axis=-1))   # 可见光亮度（灰度）
    prior = np.clip(img[..., 6], 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    for ax, data, title in (
        (axes[0], vis, "(a) 可见光灰度影像"),
        (axes[1], prior, "(b) 物理先验通道 P（由绿光与近红外波段确定性计算）"),
    ):
        ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        for ln in lbl:
            p = ln.split()
            if not p or p[0] != "0":
                continue
            xc, yc, w, h = (float(v) for v in p[1:5])
            x = (xc - w / 2) * Ww; y = (yc - h / 2) * Hh
            ax.add_patch(Rectangle((x, y), w * Ww, h * Hh, fill=False,
                                   edgecolor="black", linewidth=1.3))
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("black")
    fig.suptitle("图 2  物理先验通道与可见光影像对照（黑框为海洋漂浮垃圾标注）", fontsize=12)
    fig.savefig(OUT / "fig2.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 图3 装置框图
def fig3():
    fig, ax = plt.subplots(figsize=(5.6, 8.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis("off")
    W, X = 7.6, 1.2
    box(ax, (X, 11.4), W, 1.4, "多光谱遥感影像输入", None)
    box(ax, (X, 8.8), W, 1.6, "先验通道构造模块\n（由光谱波段确定性计算先验通道）", "201")
    box(ax, (X, 6.2), W, 1.6, "输入构造模块\n（通道拼接 + 逐通道归一化）", "202")
    box(ax, (X, 3.6), W, 1.6, "检测模块\n（CNN 联合卷积，输出实例级框）", "203")
    box(ax, (X, 1.2), W, 1.4, "海洋漂浮垃圾边界框及置信度输出", None)
    ys = [11.4, 8.8, 6.2, 3.6, 1.2]
    hs = [1.4, 1.6, 1.6, 1.6, 1.4]
    for i in range(len(ys) - 1):
        arrow(ax, (X + W / 2, ys[i]), (X + W / 2, ys[i + 1] + hs[i + 1]))
    ax.set_title("图 3  海洋漂浮垃圾检测装置结构框图", fontsize=12, pad=6)
    fig.savefig(OUT / "fig3.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 图4 可靠性曲线
def fig4():
    d = json.loads((ROOT / "docs/paper/calibration/platt_fit.json").read_text())
    t = d["test"]
    raw, cal = t["reliability_raw"], t["reliability_calibrated"]

    def nz(rel):
        bc = np.array(rel["bin_centers"]); ob = np.array(rel["observed"])
        cnt = np.array(rel["counts"])
        m = cnt > 0
        return bc[m], ob[m]

    raw_x, raw_y = nz(raw)
    cal_x, cal_y = nz(cal)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.2, label="理想校准")
    ax.plot(raw_x, raw_y, color="black", linestyle="--",
            marker="s", markerfacecolor="white", linewidth=LW,
            label=f"校准前 (ECE={raw['ECE']:.3f})")
    ax.plot(cal_x, cal_y, color="black", linestyle="-",
            marker="o", markerfacecolor="black", linewidth=LW,
            label=f"校准后 (ECE={cal['ECE']:.3f})")
    ax.set_xlabel("预测置信度", fontsize=11)
    ax.set_ylabel("实际命中频率", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc="upper left", frameon=True, edgecolor="black")
    ax.grid(True, linewidth=0.4, color="0.8")
    ax.set_title("图 4  置信度校准前后可靠性曲线对比（测试集）", fontsize=12, pad=6)
    fig.savefig(OUT / "fig4.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 图5 网络结构
def fig5():
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    ax.set_xlim(0, 18); ax.set_ylim(0, 10); ax.axis("off")

    box(ax, (0.2, 3.9), 2.4, 2.2, "7 通道输入张量\n(6 光谱波段\n+ 先验通道)", "601", fs=10)
    box(ax, (3.0, 4.2), 1.8, 1.6, "首层卷积\nch=7", "602", fs=10)
    box(ax, (5.2, 2.0), 2.6, 6.0, "主干网络\n多级下采样\n特征提取", "603", fs=10.5)
    box(ax, (9.1, 2.0), 2.4, 6.0, "BiFPN\n加权双向\n特征融合", "604", fs=10.5)
    box(ax, (12.2, 6.6), 2.0, 1.2, "检测头 P5/32", "605", fs=9.5)
    box(ax, (12.2, 4.4), 2.0, 1.2, "检测头 P4/16", "606", fs=9.5)
    box(ax, (12.2, 2.2), 2.0, 1.2, "检测头 P3/8", "607", fs=9.5)
    box(ax, (15.0, 3.9), 2.6, 2.2, "海洋漂浮垃圾\n实例级边界框\n+ 置信度", "608", fs=10)

    arrow(ax, (2.6, 5.0), (3.0, 5.0))
    arrow(ax, (4.8, 5.0), (5.2, 5.0))
    # backbone -> BiFPN, three scales
    for yy, tag in ((7.2, "P5/32"), (5.0, "P4/16"), (2.9, "P3/8")):
        arrow(ax, (7.8, yy), (9.1, yy))
        ax.text(8.45, yy + 0.22, tag, ha="center", va="bottom", fontsize=8.5)
    # BiFPN bidirectional indicator (single double-headed arrow, left interior)
    dbl = FancyArrowPatch((9.5, 2.7), (9.5, 7.3), arrowstyle="<|-|>",
                          mutation_scale=13, linewidth=LW, color=BLACK,
                          shrinkA=0, shrinkB=0)
    ax.add_patch(dbl)
    ax.text(10.45, 8.15, "自顶向下 + 自底向上", ha="center", va="bottom", fontsize=8)
    # BiFPN -> heads
    arrow(ax, (11.5, 7.2), (12.2, 7.2))
    arrow(ax, (11.5, 5.0), (12.2, 5.0))
    arrow(ax, (11.5, 2.8), (12.2, 2.8))
    # heads -> output (converge)
    for yy in (7.2, 5.0, 2.8):
        arrow(ax, (14.2, yy), (15.0, 5.0))
    ax.set_title("图 5  海洋漂浮垃圾检测器网络结构示意图", fontsize=12, pad=6)
    fig.savefig(OUT / "fig5.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1 ok")
    fig2(); print("fig2 ok")
    fig3(); print("fig3 ok")
    fig4(); print("fig4 ok")
    fig5(); print("fig5 ok")
    print("ALL DONE ->", OUT)
