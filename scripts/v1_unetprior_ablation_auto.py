"""Auto-pilot YOLO ablation on v1 unetprior dataset (U-Net Marine_Debris prob as ch7).

Compares against the existing v1 physidx (NDWI-inverse) results:
- baseline + BiFPN x 3 seed @ imgsz=1024, batch=4, workers=0, ep=30
- Same hyperparams as v1_imgsz1024_ablation_auto.py, only --data changes

Output:
  runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_unetprior_<ab>_imgsz1024_clean_init_s<seed>/
  docs/paper/v1_unetprior_ablation_aggregate.{json,md}
  docs/paper/figures/unetprior_vs_physidx_bars.png
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

PYTHON = "D:/OceanEdge_Group/Text/miniconda3/envs/yolo_gpu/python.exe"
DATA_YAML = "configs/datasets/marida_v1_unetprior_on_v2split_7ch.yaml"
PROJECT = "runs_v1_v2split"
LOG_DIR = ROOT / "runs/detect/runs_v1_v2split/unetprior_ablation_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DONE_MARKER = ROOT / "runs/detect/runs_v1_v2split/.unetprior_ablation_done"

IMGSZ = 1024
BATCH = 4
WORKERS = 0
EPOCHS = 30
ABLATIONS = ["baseline", "bifpn"]
SEEDS = [0, 1, 2]


def run_dir_for(ab: str, seed: int) -> Path:
    if ab == "baseline":
        name = f"yolov11n_7ch_v2split_unetprior_imgsz1024_clean_init_s{seed}"
    else:
        name = f"yolov11n_7ch_v2split_unetprior_{ab}_imgsz1024_clean_init_s{seed}"
    return ROOT / f"runs/detect/{PROJECT}/{name}"


def physidx_run_dir(ab: str, seed: int) -> Path:
    """Counterpart run with NDWI-inverse ch7 (existing imgsz=1024 ablation results)."""
    if ab == "baseline":
        name = f"yolov11n_7ch_v2split_physidx_imgsz1024_clean_init_s{seed}"
    else:
        name = f"yolov11n_7ch_v2split_physidx_{ab}_imgsz1024_clean_init_s{seed}"
    return ROOT / f"runs/detect/{PROJECT}/{name}"


def train_one(ab: str, seed: int) -> Path:
    rd = run_dir_for(ab, seed)
    if (rd / "weights" / "best.pt").exists():
        print(f"[SKIP train] {rd.name}")
        return rd
    print(f"[TRAIN] {rd.name}")
    log = LOG_DIR / f"train_{ab}_s{seed}.log"
    with log.open("w", encoding="utf-8") as f:
        subprocess.run([
            PYTHON, "-X", "utf8", "scripts/v1_ablation_train.py",
            "--ablation", ab,
            "--data", DATA_YAML,
            "--seed", str(seed),
            "--name", rd.name,
            "--imgsz", str(IMGSZ),
            "--batch", str(BATCH),
            "--workers", str(WORKERS),
            "--epochs", str(EPOCHS),
        ], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return rd


def eval_one(rd: Path) -> None:
    w = rd / "weights" / "best.pt"
    out = rd / "v2split_test_eval.json"
    if out.exists() or not w.exists():
        return
    log = LOG_DIR / f"eval_{rd.name}.log"
    with log.open("w", encoding="utf-8") as f:
        subprocess.run([
            PYTHON, "-X", "utf8", "scripts/v2_evaluate.py",
            "--weights", str(w),
            "--data", DATA_YAML,
            "--channels", "7",
            "--split", "test",
            "--imgsz", str(IMGSZ),
            "--batch", str(BATCH),
            "--workers", str(WORKERS),
            "--output", str(out),
        ], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))


def f1_one(rd: Path) -> None:
    w = rd / "weights" / "best.pt"
    out = rd / "test_f1_sweep.json"
    if out.exists() or not w.exists():
        return
    log = LOG_DIR / f"f1_{rd.name}.log"
    with log.open("w", encoding="utf-8") as f:
        subprocess.run([
            PYTHON, "-X", "utf8", "scripts/v1_f1_conf_sweep.py",
            "--weights", str(w),
            "--data", DATA_YAML,
            "--split", "test",
            "--imgsz", str(IMGSZ),
            "--batch", str(BATCH),
            "--output", str(out),
        ], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))


def collect_cell(rd: Path) -> dict:
    ep = rd / "v2split_test_eval.json"
    sp = rd / "test_f1_sweep.json"
    if not ep.exists():
        return {}
    d = json.loads(ep.read_text())
    pc = d.get("per_class", {})
    out = {
        "mAP50": d.get("mAP50"),
        "mAP50_95": d.get("mAP50_95"),
        "Marine_Debris_AP50": (pc.get("Marine_Debris") or {}).get("AP50"),
        "Ship_AP50": (pc.get("Ship") or {}).get("AP50"),
    }
    if sp.exists():
        out["f1"] = json.loads(sp.read_text())["best"]["f1"]
    return out


def msd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "all": []}
    return {"mean": mean(xs), "std": stdev(xs) if len(xs) > 1 else 0.0, "all": xs}


def aggregate() -> None:
    out = {}
    for prior in ["physidx", "unetprior"]:
        out[prior] = {}
        dir_fn = physidx_run_dir if prior == "physidx" else run_dir_for
        for ab in ABLATIONS:
            map50, map95, f1, deb, ship = [], [], [], [], []
            used = []
            for s in SEEDS:
                c = collect_cell(dir_fn(ab, s))
                if not c:
                    continue
                map50.append(c.get("mAP50"))
                map95.append(c.get("mAP50_95"))
                deb.append(c.get("Marine_Debris_AP50"))
                ship.append(c.get("Ship_AP50"))
                if c.get("f1") is not None:
                    f1.append(c["f1"])
                used.append(s)
            out[prior][ab] = {
                "n_seeds": len(used), "seeds": used,
                "mAP50": msd(map50), "mAP50_95": msd(map95), "f1": msd(f1),
                "Marine_Debris_AP50": msd(deb), "Ship_AP50": msd(ship),
            }

    out_json = ROOT / "docs/paper/v1_unetprior_ablation_aggregate.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"saved {out_json}")

    lines = [
        "# U-Net prior vs NDWI-inverse prior, YOLO @ imgsz=1024 (3-seed)",
        "",
        "All honest setting (v1 7ch on v2 scene-clean split), clean init, ep=30, batch=4, workers=0.",
        "Only ch7 changes: NDWI-inverse (physidx) vs U-Net Marine_Debris probability (unetprior, K-fold leak-free).",
        "",
    ]
    for ab in ABLATIONS:
        lines.append(f"## {ab}")
        lines.append("")
        lines.append("| ch7 prior | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        def fmt(d):
            return "--" if d.get("mean") is None else f"{d['mean']:.3f} +- {d['std']:.3f}"
        for prior in ["physidx", "unetprior"]:
            c = out[prior][ab]
            lines.append(
                f"| {prior} | {c['n_seeds']} | {fmt(c['mAP50'])} | {fmt(c['mAP50_95'])} | "
                f"{fmt(c['f1'])} | {fmt(c['Marine_Debris_AP50'])} | {fmt(c['Ship_AP50'])} |"
            )
        lines.append("")

    out_md = ROOT / "docs/paper/v1_unetprior_ablation_aggregate.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_md}")
    print("\n" + "\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ABLATIONS
        x = np.arange(len(labels))
        bar_w = 0.35
        metrics = [
            ("mAP50", "mAP@0.5"),
            ("f1", "Best F1"),
            ("Marine_Debris_AP50", "Marine_Debris AP@0.5"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, (key, ylabel) in zip(axes, metrics):
            for i, prior in enumerate(["physidx", "unetprior"]):
                m = [out[prior][ab][key].get("mean") or 0 for ab in ABLATIONS]
                s = [out[prior][ab][key].get("std") or 0 for ab in ABLATIONS]
                ax.bar(x + (i - 0.5) * bar_w, m, bar_w, yerr=s, capsize=4,
                       label=prior, edgecolor="black", linewidth=0.6)
                for j, (mj, sj) in enumerate(zip(m, s)):
                    ax.text(x[j] + (i - 0.5) * bar_w, mj + sj + 0.005, f"{mj:.3f}",
                            ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x, labels)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} (imgsz=1024)")
            ax.grid(axis="y", alpha=0.3, linestyle=":")
            ax.legend(loc="upper left")
        fig.suptitle("U-Net prior vs NDWI-inverse prior (3-seed mean +- std)", fontsize=12)
        fig.tight_layout()
        out_fig = ROOT / "docs/paper/figures/unetprior_vs_physidx_bars.png"
        fig.savefig(out_fig, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_fig}")
    except Exception as e:
        print(f"figure failed: {e}")


def main():
    t0 = time.time()
    print(f"[START] unetprior YOLO ablation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for ab in ABLATIONS:
        for s in SEEDS:
            rd = train_one(ab, s)
            eval_one(rd)
            f1_one(rd)
    print(f"\n[AGGREGATE]")
    aggregate()
    elapsed = (time.time() - t0) / 60
    DONE_MARKER.write_text(f"done at {time.strftime('%Y-%m-%d %H:%M:%S')}, elapsed {elapsed:.1f} min\n")
    print(f"[DONE] elapsed {elapsed:.1f} min")


if __name__ == "__main__":
    main()
