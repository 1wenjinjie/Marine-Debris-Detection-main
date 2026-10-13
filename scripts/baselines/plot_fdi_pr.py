"""Plot FDI / NDWI-inverse baseline vs YOLO comparison for paper figure.

Two-panel layout:
  Left  : best-F1 bar chart (log y-axis) — direct order-of-magnitude comparison
  Right : PR scatter — FDI/NDWI threshold sweep + YOLO BiFPN threshold sweep
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FDI_JSON = PROJECT_ROOT / "docs" / "paper" / "fdi_baseline" / "fdi_ndwi_sweep_test.json"
YOLO_JSON = PROJECT_ROOT / "docs" / "paper" / "calibration" / "threshold_sweep_test.json"
OUT_FIG = PROJECT_ROOT / "docs" / "paper" / "figures" / "fdi_pr_comparison.png"

# baseline_1024 3-seed Best F1: 0.427 ± 0.048
#   source: docs/paper/v1_imgsz1024_ablation_aggregate.json :: baseline_1024 row
BASELINE_1024_BEST_F1_3SEED_MEAN = 0.427


def main():
    with open(FDI_JSON) as f:
        fdi_data = json.load(f)
    with open(YOLO_JSON) as f:
        yolo_data = json.load(f)

    fdi_best = float(fdi_data["FDI"]["best"]["F1"])
    ndwi_best = float(fdi_data["NDWI_inv"]["best"]["F1"])
    bifpn_dep = next(r for r in yolo_data["raw"] if abs(r["tau"] - 0.25) < 1e-6)
    bifpn_dep_f1 = float(bifpn_dep["F1"])

    bar_data = [
        ("FDI threshold\n(single-seed, non-ML)", fdi_best, "tab:blue"),
        ("NDWI-inv threshold\n(single-seed, non-ML)", ndwi_best, "tab:green"),
        ("YOLO baseline @ 1024\n(3-seed Best F1 mean, n=3)",
         BASELINE_1024_BEST_F1_3SEED_MEAN, "tab:orange"),
        (f"YOLO BiFPN @ 1024\n(seed-0 deployment F1, τ=0.25)",
         bifpn_dep_f1, "tab:red"),
    ]

    fig, (ax_bar, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=140,
                                         gridspec_kw={"width_ratios": [1, 1.1]})

    # ---- left: best F1 bar chart (log scale) ----
    names = [b[0] for b in bar_data]
    vals = [b[1] for b in bar_data]
    colors = [b[2] for b in bar_data]
    y = np.arange(len(names))
    bars = ax_bar.barh(y, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=9)
    ax_bar.set_xscale("log")
    ax_bar.set_xlim(0.0008, 1.0)
    ax_bar.set_xlabel("Best F1 (Marine_Debris, instance-level, IoU≥0.5)", fontsize=10)
    ax_bar.set_title("(a) Best-F1 comparison (73 tiles, 108 GT)", fontsize=11)
    ax_bar.grid(True, axis="x", which="both", alpha=0.3)
    ax_bar.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax_bar.text(v * 1.15, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}" if v < 0.05 else f"{v:.3f}",
                    va="center", fontsize=9, fontweight="bold")

    ax_bar.axvspan(0.0008, 0.01, alpha=0.10, color="gray")
    ax_bar.text(0.0009, len(names) - 0.4,
                "<1% F1 region\n(physical-index baselines)",
                fontsize=7.5, color="dimgray", style="italic")

    # ---- right: PR scatter ----
    for idx_name, color, lbl in [("FDI", "tab:blue", "FDI sweep"),
                                  ("NDWI_inv", "tab:green", "NDWI-inv sweep")]:
        rs = [r["R"] for r in fdi_data[idx_name]["results"]]
        ps = [r["P"] for r in fdi_data[idx_name]["results"]]
        order = np.argsort(rs)
        rs = np.array(rs)[order]
        ps = np.array(ps)[order]
        best = fdi_data[idx_name]["best"]
        ax_pr.plot(rs, ps, "-o", ms=3, lw=1.2, color=color, alpha=0.7,
                   label=f"{lbl} (best F1={best['F1']:.4f})")
        ax_pr.scatter([best["R"]], [best["P"]], s=110, color=color,
                      edgecolor="black", linewidth=1.2, zorder=5)

    raw = yolo_data["raw"]
    cal = yolo_data["calibrated"]
    rs_raw = [r["R"] for r in raw]
    ps_raw = [r["P"] for r in raw]
    ax_pr.plot(rs_raw, ps_raw, "-s", ms=6, lw=1.8, color="tab:red",
               label="YOLO BiFPN @ 1024 (raw conf sweep)")
    # mark the deployment operating point (conf=0.25)
    op = next(r for r in raw if abs(r["tau"] - 0.25) < 1e-6)
    ax_pr.scatter([op["R"]], [op["P"]], s=220, marker="*", color="gold",
                  edgecolor="black", linewidth=1.4, zorder=6,
                  label=f"BiFPN deployment point\n(conf=0.25, F1={op['F1']:.3f})")

    rs_cal = [r["R"] for r in cal]
    ps_cal = [r["P"] for r in cal]
    ax_pr.plot(rs_cal, ps_cal, "--^", ms=4, lw=1.0, color="tab:purple", alpha=0.7,
               label="YOLO BiFPN (Platt-calibrated)")

    ax_pr.set_xlabel("Recall (Marine_Debris instance-level)", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.set_title("(b) Precision-Recall sweeps", fontsize=11)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(loc="upper right", fontsize=8, framealpha=0.95)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"[save] {OUT_FIG}")


if __name__ == "__main__":
    main()
