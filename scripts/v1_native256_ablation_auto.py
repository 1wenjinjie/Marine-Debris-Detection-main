"""Auto-pilot native-256 cross-resolution control.

Goal: distinguish whether the imgsz=1024 gain (Finding F/H) comes from
"more model capacity / P3-head landing on debris box size" vs "letterbox
up-sampling specifically". Train baseline + BiFPN at native imgsz=256
(no up-sampling) x 3 seeds and compare against the existing imgsz=1024
results.

All cells: honest setting (v1 physidx on v2 scene-clean split), clean
init, ep=30, batch=16, workers=0. Run names:
yolov11n_7ch_v2split_physidx_<ab>_imgsz256_clean_init_s<seed>.

Final aggregate compares imgsz={256, 512, 1024} for baseline + BiFPN.
Output:
  docs/paper/v1_native256_aggregate.{json,md}
  docs/paper/figures/native256_vs_1024_bars.png

Designed for unattended run via run_in_background.
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
DATA_YAML = "configs/datasets/marida_v1_physidx_on_v2split_7ch.yaml"
PROJECT = "runs_v1_v2split"
LOG_DIR = ROOT / "runs/detect/runs_v1_v2split/native256_ablation_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DONE_MARKER = ROOT / "runs/detect/runs_v1_v2split/.native256_ablation_done"

IMGSZ = 256
BATCH = 16
WORKERS = 0
EPOCHS = 30

ABLATIONS = ["baseline", "bifpn"]
SEEDS = [0, 1, 2]


def run_dir_for(ab: str, seed: int) -> Path:
    if ab == "baseline":
        name = f"yolov11n_7ch_v2split_physidx_imgsz256_clean_init_s{seed}"
    else:
        name = f"yolov11n_7ch_v2split_physidx_{ab}_imgsz256_clean_init_s{seed}"
    return ROOT / f"runs/detect/{PROJECT}/{name}"


def imgsz1024_run_dir(ab: str, seed: int) -> Path:
    if ab == "baseline":
        name = f"yolov11n_7ch_v2split_physidx_imgsz1024_clean_init_s{seed}"
    else:
        name = f"yolov11n_7ch_v2split_physidx_{ab}_imgsz1024_clean_init_s{seed}"
    return ROOT / f"runs/detect/{PROJECT}/{name}"


def imgsz512_run_dir(ab: str, seed: int) -> Path:
    if seed == 0:
        if ab == "baseline":
            name = "yolov11n_7ch_v2split_physidx_clean_init"
        else:
            name = f"yolov11n_7ch_v2split_physidx_{ab}_clean_init"
    else:
        if ab == "baseline":
            name = f"yolov11n_7ch_v2split_physidx_baseline_clean_init_s{seed}"
        else:
            name = f"yolov11n_7ch_v2split_physidx_{ab}_clean_init_s{seed}"
    return ROOT / f"runs/detect/{PROJECT}/{name}"


def log_path(stage: str, ab: str, seed: int) -> Path:
    return LOG_DIR / f"{stage}_{ab}_s{seed}.log"


def train_one(ab: str, seed: int) -> Path:
    run_dir = run_dir_for(ab, seed)
    name = run_dir.name

    if (run_dir / "weights" / "best.pt").exists():
        print(f"[SKIP train] {name} already has best.pt")
        return run_dir

    print(f"[TRAIN] {name}")
    log = log_path("train", ab, seed)
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run([
            PYTHON, "-X", "utf8", "scripts/v1_ablation_train.py",
            "--ablation", ab,
            "--seed", str(seed),
            "--name", name,
            "--imgsz", str(IMGSZ),
            "--batch", str(BATCH),
            "--workers", str(WORKERS),
            "--epochs", str(EPOCHS),
        ], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    if proc.returncode != 0:
        print(f"[FAIL train] {name} (see {log})")
    else:
        print(f"[OK train] {name}")
    return run_dir


def eval_one(run_dir: Path) -> None:
    weights = run_dir / "weights" / "best.pt"
    out_json = run_dir / "v2split_test_eval.json"
    if out_json.exists():
        print(f"[SKIP eval] {run_dir.name}")
        return
    if not weights.exists():
        print(f"[FAIL eval] {run_dir.name} missing best.pt")
        return
    print(f"[EVAL] {run_dir.name}")
    log = LOG_DIR / f"eval_{run_dir.name}.log"
    with log.open("w", encoding="utf-8") as f:
        subprocess.run([
            PYTHON, "-X", "utf8", "scripts/v2_evaluate.py",
            "--weights", str(weights),
            "--data", DATA_YAML,
            "--channels", "7",
            "--split", "test",
            "--imgsz", str(IMGSZ),
            "--batch", str(BATCH),
            "--workers", str(WORKERS),
            "--output", str(out_json),
        ], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))


def f1_one(run_dir: Path) -> None:
    weights = run_dir / "weights" / "best.pt"
    out_json = run_dir / "test_f1_sweep.json"
    if out_json.exists():
        print(f"[SKIP f1] {run_dir.name}")
        return
    if not weights.exists():
        print(f"[FAIL f1] {run_dir.name} missing best.pt")
        return
    print(f"[F1] {run_dir.name}")
    log = LOG_DIR / f"f1_{run_dir.name}.log"
    with log.open("w", encoding="utf-8") as f:
        subprocess.run([
            PYTHON, "-X", "utf8", "scripts/v1_f1_conf_sweep.py",
            "--weights", str(weights),
            "--data", DATA_YAML,
            "--split", "test",
            "--imgsz", str(IMGSZ),
            "--batch", str(BATCH),
            "--output", str(out_json),
        ], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))


def collect_cell(run_dir: Path) -> dict:
    ep = run_dir / "v2split_test_eval.json"
    sp = run_dir / "test_f1_sweep.json"
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


def aggregate_cross_res() -> dict:
    out = {}
    for ab in ABLATIONS:
        out[ab] = {}
        for resolution, dir_fn in [
            (256, run_dir_for),
            (512, imgsz512_run_dir),
            (1024, imgsz1024_run_dir),
        ]:
            map50, map95, f1, debris, ship = [], [], [], [], []
            used = []
            for seed in SEEDS:
                rd = dir_fn(ab, seed)
                c = collect_cell(rd)
                if not c:
                    continue
                map50.append(c.get("mAP50"))
                map95.append(c.get("mAP50_95"))
                debris.append(c.get("Marine_Debris_AP50"))
                ship.append(c.get("Ship_AP50"))
                if c.get("f1") is not None:
                    f1.append(c["f1"])
                used.append(seed)
            out[ab][resolution] = {
                "n_seeds": len(used), "seeds": used,
                "mAP50": msd(map50), "mAP50_95": msd(map95), "f1": msd(f1),
                "Marine_Debris_AP50": msd(debris), "Ship_AP50": msd(ship),
            }
    return out


def write_md(agg: dict) -> None:
    lines = [
        "# Native-256 vs imgsz=1024 cross-resolution control (3-seed)",
        "",
        "All honest setting, v1 physidx on v2 scene-clean split, clean init, ep=30.",
        "imgsz=256: batch=16, workers=0 (native, no letterbox up-sampling).",
        "imgsz=512: batch=8, workers=0 (2x letterbox up-sampling).",
        "imgsz=1024: batch=4, workers=0 (4x letterbox up-sampling).",
        "",
        "Question: does the imgsz=1024 gain come from (a) more model capacity / P3-head",
        "landing on debris box size, or (b) letterbox up-sampling specifically?",
        "If imgsz=256 already beats imgsz=512 -> (a) dominates. If only imgsz=1024 wins -> (b).",
        "",
        "## Per-architecture cross-resolution table",
        "",
    ]
    for ab in ABLATIONS:
        lines.append(f"### {ab}")
        lines.append("")
        lines.append("| imgsz | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in (256, 512, 1024):
            c = agg[ab][r]
            def fmt(d):
                return "--" if d.get("mean") is None else f"{d['mean']:.3f} +- {d['std']:.3f}"
            lines.append(
                f"| {r} | {c['n_seeds']} | {fmt(c['mAP50'])} | {fmt(c['mAP50_95'])} | "
                f"{fmt(c['f1'])} | {fmt(c['Marine_Debris_AP50'])} | {fmt(c['Ship_AP50'])} |"
            )
        lines.append("")
    out_md = ROOT / "docs/paper/v1_native256_aggregate.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_md}")
    print("\n" + "\n".join(lines))


def write_figure(agg: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        resolutions = [256, 512, 1024]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        metrics = [
            ("mAP50", "mAP@0.5"),
            ("f1", "Best F1"),
            ("Marine_Debris_AP50", "Marine_Debris AP@0.5"),
        ]
        bar_w = 0.35
        x = np.arange(len(resolutions))

        for ax, (key, ylabel) in zip(axes, metrics):
            for i, ab in enumerate(ABLATIONS):
                means = [agg[ab][r][key].get("mean") or 0 for r in resolutions]
                stds = [agg[ab][r][key].get("std") or 0 for r in resolutions]
                ax.bar(x + (i - 0.5) * bar_w, means, bar_w, yerr=stds, capsize=4,
                       label=ab, edgecolor="black", linewidth=0.6)
                for j, (m, s) in enumerate(zip(means, stds)):
                    ax.text(x[j] + (i - 0.5) * bar_w, m + s + 0.005, f"{m:.3f}",
                            ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x, [f"imgsz={r}" for r in resolutions])
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel + " across resolutions")
            ax.grid(axis="y", alpha=0.3, linestyle=":")
            ax.legend(loc="upper left", fontsize=9)

        fig.suptitle(
            "Native-256 vs letterbox-512 vs letterbox-1024 (3-seed mean +- std)",
            fontsize=12,
        )
        fig.tight_layout()
        out_fig = ROOT / "docs/paper/figures/native256_vs_1024_bars.png"
        fig.savefig(out_fig, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_fig}")
    except Exception as e:
        print(f"figure failed: {e}")


def aggregate() -> None:
    agg = aggregate_cross_res()
    out_json = ROOT / "docs/paper/v1_native256_aggregate.json"
    out_json.write_text(json.dumps(agg, indent=2))
    print(f"saved {out_json}")
    write_md(agg)
    write_figure(agg)


def main():
    t0 = time.time()
    print(f"[START] native-256 ablation pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for ab in ABLATIONS:
        for seed in SEEDS:
            rd = train_one(ab, seed)
            eval_one(rd)
            f1_one(rd)

    print(f"\n[AGGREGATE]")
    aggregate()

    elapsed = (time.time() - t0) / 60
    DONE_MARKER.write_text(
        f"done at {time.strftime('%Y-%m-%d %H:%M:%S')}, elapsed {elapsed:.1f} min\n"
    )
    print(f"[DONE] elapsed {elapsed:.1f} min, marker: {DONE_MARKER}")


if __name__ == "__main__":
    main()
