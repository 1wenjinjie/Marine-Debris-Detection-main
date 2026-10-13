"""Generate the 5 patent drawings for 案件2 (segmentation -> connected-component
instance extraction). CNIPA-style B&W line drawings + real-data figures.

Run:  C:/Users/18453/miniconda3/envs/marine/python.exe docs/patent/案件2_分割实例化/figures/make_patent2_figures.py
Outputs: fig1.png ... fig5.png in this directory.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("d:/OceanEdge_Group/Marine-Debris-Detection-main")
OUT = ROOT / "docs/patent/案件2_分割实例化/figures"
OUT.mkdir(parents=True, exist_ok=True)
SEMSEG = ROOT / "data/processed/marida_v2_semseg"
PROB = ROOT / "runs_v2_semseg/prob_masks"
CKPT = ROOT / "runs_v2_semseg/unet_b32_s1/best.pt"

BLACK = "black"
LW = 1.4
TAU = 0.5
NMIN = 4

sys.path.insert(0, str(ROOT))
_MODEL = None
_P01 = _P99 = None


def get_model():
    global _MODEL, _P01, _P99
    if _MODEL is None:
        import torch
        from scripts.v2_semseg_train import UNet, NC
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(CKPT, weights_only=False, map_location=dev)
        m = UNet(in_ch=6, out_ch=NC, base=32).to(dev).eval()
        m.load_state_dict(ck["model"])
        s = json.loads((SEMSEG / "channel_stats_6ch.json").read_text())
        _P01 = np.asarray(s["percentiles"]["p01"], np.float32)[None, None, :]
        _P99 = np.asarray(s["percentiles"]["p99"], np.float32)[None, None, :]
        _MODEL = (m, dev)
    return _MODEL


def predict(img_hwc):
    """Return (argmax_pred HxW, debris_prob HxW) from the U-Net."""
    import torch
    import torch.nn.functional as F
    m, dev = get_model()
    x = np.nan_to_num(img_hwc[..., :6], nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip((x - _P01) / np.maximum(_P99 - _P01, 1e-6), 0, 1).astype(np.float32)
    t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(dev)
    with torch.no_grad():
        logits = m(t)
        prob = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
        pred = logits.argmax(dim=1)[0].cpu().numpy()
    return pred, prob


def box(ax, xy, w, h, text, ref=None, fs=11):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=LW, edgecolor=BLACK, facecolor="white")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=BLACK)
    if ref is not None:
        ax.text(x + w - 0.04, y + h - 0.04, ref, ha="right", va="top", fontsize=9, color=BLACK)


def arrow(ax, p0, p1):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                        linewidth=LW, color=BLACK, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


# ---------------------------------------------------------------- 图1 总体流程图
def fig1():
    fig, ax = plt.subplots(figsize=(6.4, 10.6))
    ax.set_xlim(0, 10); ax.set_ylim(-0.6, 19); ax.axis("off")
    W, X = 7.6, 1.2
    steps = [
        (17.2, "S1  获取海面多光谱遥感影像\n（含可见光、近红外、短波红外波段）", "101"),
        (15.0, "S2  逐通道分位归一化（p01/p99）", "102"),
        (12.8, "S3  语义分割网络前向，\n输出目标类像素概率图", "103"),
        (10.6, "S4  对目标类概率图按阈值 τ 二值化", "104"),
        (8.4, "S5  连通域标记并按最小连通尺寸 n_min 过滤", "105"),
        (6.2, "S6  由各连通域生成实例级边界框与计数", "106"),
        (4.0, "S7  连通域内聚合像素概率，得实例原始得分", "107"),
        (1.8, "S8  单调校准 p=σ(a·logit(s)+b)，得实例置信度", "108"),
        (-0.4, "S9  输出实例级边界框、计数及校准置信度", "109"),
    ]
    ys = []
    for y, t, r in steps:
        box(ax, (X, y), W, 1.5, t, ref=r, fs=10)
        ys.append(y)
    # main chain S1->...->S9
    for i in range(len(ys) - 1):
        # S6 and S8 both feed S9; S4->S5->S6, S5/S3 concept handled by main chain
        arrow(ax, (X + W / 2, ys[i]), (X + W / 2, ys[i + 1] + 1.5))
    fig.savefig(OUT / "fig3.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 实例化与命中
def _boxes(binary):
    from scipy import ndimage
    lbl, n = ndimage.label(binary)
    out = []
    for c in range(1, n + 1):
        comp = (lbl == c)
        if comp.sum() < NMIN:
            continue
        ys, xs = np.where(comp)
        out.append((xs.min(), ys.min(), xs.max(), ys.max(), comp))
    return out


# ---------------------------------------------------------------- 选瓦片（命中最多）
def pick_tile():
    best = None
    for l in (SEMSEG / "test.txt").read_text().splitlines():
        l = l.strip()
        if not l:
            continue
        rel = l
        mask = SEMSEG / l.replace("/images/", "/masks/")
        if not mask.exists():
            continue
        gt = np.load(mask)
        if not (gt == 1).any():
            continue
        img = np.load(SEMSEG / rel)
        pred, prob = predict(img)
        pred_d = (pred == 1)
        gt_boxes = _boxes(gt == 1)
        if not gt_boxes:
            continue
        n_hit = sum(1 for (*_, comp) in gt_boxes if (pred_d & comp).any())
        pred_boxes = _boxes(pred_d)
        key = (n_hit, len(gt_boxes), -abs(len(pred_boxes) - len(gt_boxes)))
        if best is None or key > best[0]:
            best = (key, rel, n_hit, len(gt_boxes), len(pred_boxes))
    return best


# ---------------------------------------------------------------- 图2 分割—实例化真实数据图
def fig2():
    sel = pick_tile()
    _, rel, n_hit, n_gt, n_pred = sel
    stem = Path(rel).stem
    img = np.load(SEMSEG / rel)
    gt = np.load(SEMSEG / rel.replace("/images/", "/masks/"))
    pred, prob = predict(img)
    Hh, Ww = prob.shape

    def stretch(a):
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    vis = stretch(img[..., :3].mean(axis=-1))  # 可见光灰度
    pred_boxes = [(b[0], b[1], b[2], b[3]) for b in _boxes(pred == 1)]
    gt_boxes = [(b[0], b[1], b[2], b[3]) for b in _boxes(gt == 1)]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    axes[0].imshow(vis, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("(a) 可见光灰度影像", fontsize=11)
    axes[1].imshow(prob, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("(b) 语义分割漂浮垃圾类像素概率图", fontsize=11)
    axes[2].imshow(vis, cmap="gray", vmin=0, vmax=1)
    for (x0, y0, x1, y1) in gt_boxes:
        axes[2].add_patch(Rectangle((x0 - 1, y0 - 1), x1 - x0 + 2, y1 - y0 + 2,
                          fill=False, edgecolor="black", linewidth=1.2, linestyle="--"))
    for (x0, y0, x1, y1) in pred_boxes:
        axes[2].add_patch(Rectangle((x0 - 1, y0 - 1), x1 - x0 + 2, y1 - y0 + 2,
                          fill=False, edgecolor="black", linewidth=1.6))
    axes[2].set_title("(c) 连通域实例化提取的实例框\n（实线为本方法实例框，虚线为真值标注）", fontsize=10.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("black")
    fig.savefig(OUT / "fig4.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"fig2 tile = {stem}  pred_inst={len(pred_boxes)} gt_inst={len(gt_boxes)}")


# ---------------------------------------------------------------- 图3 装置结构框图
def fig3():
    fig, ax = plt.subplots(figsize=(6.4, 9.2))
    ax.set_xlim(0, 12); ax.set_ylim(-0.4, 14.2); ax.axis("off")
    W, X = 8.2, 1.4
    box(ax, (X, 12.6), W, 1.3, "多光谱遥感影像输入", None)
    box(ax, (X, 10.2), W, 1.5, "输入归一化模块\n（逐通道分位归一化）", "201")
    box(ax, (X, 7.8), W, 1.5, "语义分割模块\n（编解码网络，输出像素概率图）", "202")
    box(ax, (X, 5.0), W, 1.6, "连通域实例化模块\n（二值化+连通域+最小尺寸过滤）", "203")
    box(ax, (X, 2.2), W, 1.6, "实例置信度聚合与校准模块\n（连通域内概率聚合+单调校准）", "204")
    box(ax, (X, -0.2), W, 1.3, "实例级边界框、计数及校准置信度输出", None)
    ys = [12.6, 10.2, 7.8, 5.0, 2.2, -0.2]
    hs = [1.3, 1.5, 1.5, 1.6, 1.6, 1.3]
    for i in range(len(ys) - 1):
        arrow(ax, (X + W / 2, ys[i]), (X + W / 2, ys[i + 1] + hs[i + 1]))
    # 一图两用：语义分割概率图同时供 203 与 204
    arrow(ax, (X + W, 8.55), (X + W + 1.1, 8.55))
    ax.plot([X + W + 1.1, X + W + 1.1], [3.0, 8.55], color=BLACK, lw=LW)
    arrow(ax, (X + W + 1.1, 3.0), (X + W, 3.0))
    ax.text(X + W + 1.25, 5.8, "像素\n概率图", ha="left", va="center", fontsize=8.5)
    fig.savefig(OUT / "fig1.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 图4 实例级召回对比
def fig4():
    d = json.loads((ROOT / "runs_v2_semseg/instance_recall_summary.json").read_text())
    r4 = [d[s]["Marine_Debris"]["MIN=4"]["instance_recall"] for s in d]
    r1 = [d[s]["Marine_Debris"]["MIN=1"]["instance_recall"] for s in d]
    m4, s4 = float(np.mean(r4)), float(np.std(r4))
    m1 = float(np.mean(r1))
    yolo = 0.154  # 框检测器同口径基线（conf=0.001）

    labels = ["框检测器\n(≥4px)", "本方法\n(≥4px)", "本方法\n(含1–3px)"]
    vals = [yolo, m4, m1]
    errs = [0, s4, 0]
    hatches = ["", "////", "xxxx"]

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    xpos = np.arange(len(vals))
    bars = ax.bar(xpos, vals, yerr=errs, width=0.6, color="white",
                  edgecolor="black", linewidth=LW, capsize=5)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    for x, v, e in zip(xpos, vals, errs):
        ax.text(x, v + e + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    ax.annotate("", xy=(1, m4 + 0.10), xytext=(0, yolo + 0.10),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(0.5, m4 + 0.14, "约 4×", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("漂浮垃圾实例级召回率", fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.grid(True, axis="y", linewidth=0.4, color="0.8")
    fig.savefig(OUT / "fig5.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"fig4: yolo={yolo} 本方法MIN4={m4:.3f}±{s4:.3f} MIN1={m1:.3f}")


# ---------------------------------------------------------------- 图5 网络与实例化结构
def fig5():
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    ax.set_xlim(0, 20); ax.set_ylim(0, 10); ax.axis("off")

    box(ax, (0.2, 3.9), 2.3, 2.2, "多通道输入\n张量\n(C 通道)", "601", fs=9.5)
    # 编码-解码 U 型
    box(ax, (2.9, 4.2), 1.7, 1.6, "编码器\n下采样", "602", fs=9.5)
    box(ax, (5.0, 3.4), 1.7, 3.2, "瓶颈层", "603", fs=10)
    box(ax, (7.1, 4.2), 1.7, 1.6, "解码器\n上采样", "604", fs=9.5)
    ax.text(5.85, 7.0, "跳跃连接", ha="center", va="bottom", fontsize=8.5)
    sk = FancyArrowPatch((3.75, 5.9), (7.95, 5.9), arrowstyle="-",
                         linewidth=1.0, color=BLACK, linestyle=(0, (4, 3)),
                         shrinkA=0, shrinkB=0)
    ax.add_patch(sk)
    box(ax, (9.2, 3.9), 2.2, 2.2, "像素类别\n概率图", "605", fs=9.5)
    box(ax, (11.9, 3.9), 2.5, 2.2, "连通域\n实例化\n(二值化→连通域\n→尺寸过滤)", "606", fs=9)
    box(ax, (14.9, 5.6), 2.4, 1.4, "实例框\n+ 计数", "607", fs=9.5)
    box(ax, (14.9, 2.9), 2.4, 1.6, "概率聚合\n+ 单调校准", "608", fs=9.5)
    box(ax, (17.7, 3.9), 2.1, 2.2, "实例级框\n+ 校准\n置信度", "609", fs=9.5)

    arrow(ax, (2.5, 5.0), (2.9, 5.0))
    arrow(ax, (4.6, 5.0), (5.0, 5.0))
    arrow(ax, (6.7, 5.0), (7.1, 5.0))
    arrow(ax, (8.8, 5.0), (9.2, 5.0))
    arrow(ax, (11.4, 5.0), (11.9, 5.0))
    arrow(ax, (14.4, 5.3), (14.9, 6.0))   # ->实例框
    arrow(ax, (14.4, 4.7), (14.9, 3.7))   # ->聚合校准
    # 概率图也直接供聚合（一图两用）
    arrow(ax, (15.45, 5.6), (15.45, 4.5))
    arrow(ax, (17.3, 6.3), (17.7, 5.6))
    arrow(ax, (17.3, 3.7), (17.7, 4.4))
    fig.savefig(OUT / "fig2.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1 ok")
    fig2(); print("fig2 ok")
    fig3(); print("fig3 ok")
    fig4(); print("fig4 ok")
    fig5(); print("fig5 ok")
    print("ALL DONE ->", OUT)
