"""Instance-level confidence calibration for the segmentation->instance pipeline.

For each predicted Marine_Debris instance (argmax==1 connected component, >=4px),
score s = mean debris softmax prob over the component; label TP=1 if the component
overlaps any GT debris pixel. Fit Platt (a,b) on VAL instances, evaluate reliability
+ ECE on TEST. Honest: if calibrated ECE does not beat raw, we report it and skip
adding the figure to the disclosure.

Run: C:/Users/18453/miniconda3/envs/marine/python.exe docs/patent/案件2_分割实例化/figures/make_calibration_fig.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import minimize

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("d:/OceanEdge_Group/Marine-Debris-Detection-main")
OUT = ROOT / "docs/patent/案件2_分割实例化/figures"
SEMSEG = ROOT / "data/processed/marida_v2_semseg"
CKPT = ROOT / "runs_v2_semseg/unet_b32_s1/best.pt"
NMIN = 4
EPS = 1e-4

sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from scripts.v2_semseg_train import UNet, NC

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ck = torch.load(CKPT, weights_only=False, map_location=dev)
MODEL = UNet(in_ch=6, out_ch=NC, base=32).to(dev).eval()
MODEL.load_state_dict(_ck["model"])
_s = json.loads((SEMSEG / "channel_stats_6ch.json").read_text())
P01 = np.asarray(_s["percentiles"]["p01"], np.float32)[None, None, :]
P99 = np.asarray(_s["percentiles"]["p99"], np.float32)[None, None, :]


def predict(rel):
    img = np.load(SEMSEG / rel)
    x = np.nan_to_num(img[..., :6], nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip((x - P01) / np.maximum(P99 - P01, 1e-6), 0, 1).astype(np.float32)
    t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(dev)
    with torch.no_grad():
        logits = MODEL(t)
        prob = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
        pred = logits.argmax(dim=1)[0].cpu().numpy()
    return pred, prob


def collect(split):
    """Return arrays (scores, labels) over predicted debris instances."""
    scores, labels = [], []
    for l in (SEMSEG / f"{split}.txt").read_text().splitlines():
        l = l.strip()
        if not l:
            continue
        gt = np.load(SEMSEG / l.replace("/images/", "/masks/"))
        gt_d = (gt == 1)
        pred, prob = predict(l)
        lbl, n = ndimage.label(pred == 1)
        for c in range(1, n + 1):
            comp = (lbl == c)
            if comp.sum() < NMIN:
                continue
            s = float(prob[comp].mean())
            tp = 1 if (comp & gt_d).any() else 0
            scores.append(s)
            labels.append(tp)
    return np.asarray(scores), np.asarray(labels, dtype=float)


def fit_platt(s, y):
    z = np.log(np.clip(s, EPS, 1 - EPS) / (1 - np.clip(s, EPS, 1 - EPS)))  # logit(s)

    def nll(ab):
        a, b = ab
        p = 1.0 / (1.0 + np.exp(-(a * z + b)))
        p = np.clip(p, EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    r = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return float(r.x[0]), float(r.x[1])


def apply_platt(s, a, b):
    z = np.log(np.clip(s, EPS, 1 - EPS) / (1 - np.clip(s, EPS, 1 - EPS)))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def reliability(conf, y, nbins=10):
    edges = np.linspace(0, 1, nbins + 1)
    centers, observed, counts = [], [], []
    ece = 0.0
    N = len(y)
    for i in range(nbins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < nbins - 1 else conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        acc = float(y[m].mean())
        cf = float(conf[m].mean())
        centers.append(cf); observed.append(acc); counts.append(int(m.sum()))
        ece += (m.sum() / N) * abs(cf - acc)
    return np.array(centers), np.array(observed), np.array(counts), ece


def main():
    vs, vy = collect("val")
    ts, ty = collect("test")
    print(f"val instances={len(vy)} (TP={int(vy.sum())})  test instances={len(ty)} (TP={int(ty.sum())})")
    if len(vy) < 20 or len(ty) < 20 or vy.sum() == 0 or (vy == 1).all():
        print("NOT ENOUGH / degenerate -> skip calibration figure")
        return
    a, b = fit_platt(vs, vy)
    tp_cal = apply_platt(ts, a, b)
    rx, ro, rc, ece_raw = reliability(ts, ty)
    cx, co, cc, ece_cal = reliability(tp_cal, ty)
    print(f"Platt a={a:.3f} b={b:.3f}")
    print(f"TEST  ECE_raw={ece_raw:.3f}  ECE_calibrated={ece_cal:.3f}  (improve={ece_raw-ece_cal:+.3f})")

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.2, label="理想校准")
    ax.plot(rx, ro, color="black", linestyle="--", marker="s", markerfacecolor="white",
            linewidth=1.4, label=f"校准前 (ECE={ece_raw:.3f})")
    ax.plot(cx, co, color="black", linestyle="-", marker="o", markerfacecolor="black",
            linewidth=1.4, label=f"校准后 (ECE={ece_cal:.3f})")
    ax.set_xlabel("预测实例置信度", fontsize=11)
    ax.set_ylabel("实际命中频率", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc="upper left", frameon=True, edgecolor="black")
    ax.grid(True, linewidth=0.4, color="0.8")
    fig.savefig(OUT / "fig6.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig6.png")
    (OUT / "calibration_result.json").write_text(json.dumps({
        "a": a, "b": b, "ece_raw": ece_raw, "ece_calibrated": ece_cal,
        "val_instances": int(len(vy)), "val_tp": int(vy.sum()),
        "test_instances": int(len(ty)), "test_tp": int(ty.sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
