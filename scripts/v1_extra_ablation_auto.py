"""Auto-pilot extra ablation pipeline: 6 new trainings + eval + F1 sweep + aggregate.

Cells (all honest setting, baseline architecture, partial-load yolo11n_7ch_init.pt):
- imgsz=1024 batch=4 epochs=30 x seeds {0,1,2}  -> tests resolution effect on small objects
- imgsz=512  batch=8 epochs=60 x seeds {0,1,2}  -> tests convergence vs current 30 ep

Pipeline per cell:
1. train (MaskRescueTrainer, SOTA hyperparams with overrides)
2. eval  (scripts/v2_evaluate.py, writes <run>/v2split_test_eval.json)
3. F1 sweep (scripts/v1_f1_conf_sweep.py, writes <run>/test_f1_sweep.json)

Final step: aggregate to docs/paper/v1_extra_ablation_aggregate.{json,md} +
              docs/paper/figures/extra_ablation_bars.png + .done marker.

Designed to run unattended via run_in_background. Survives Claude quota gaps.
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
INIT_CKPT = "yolo11n_7ch_init.pt"
PROJECT = "runs_v1_v2split"
LOG_DIR = ROOT / "runs/detect/runs_v1_v2split/extra_ablation_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DONE_MARKER = ROOT / "runs/detect/runs_v1_v2split/.extra_ablation_done"

CELLS = [
    # imgsz=1024 needs workers=0 on Windows: torch multiprocessing pin_memory
    # crashes with shared-memory error at this resolution+batch combo.
    {"tag": "imgsz1024", "imgsz": 1024, "batch": 4, "epochs": 30, "workers": 0},
    {"tag": "ep60",      "imgsz":  512, "batch": 8, "epochs": 60, "workers": 8},
]
SEEDS = [0, 1, 2]


def run_dir_for(tag: str, seed: int) -> Path:
    return ROOT / f"runs/detect/{PROJECT}/yolov11n_7ch_v2split_physidx_{tag}_clean_init_s{seed}"


def log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


def train_one(cell: dict, seed: int) -> Path:
    """Train one cell+seed using an inline Python invocation."""
    tag = cell["tag"]
    run_name = f"yolov11n_7ch_v2split_physidx_{tag}_clean_init_s{seed}"
    out_dir = ROOT / f"runs/detect/{PROJECT}/{run_name}"

    if (out_dir / "weights" / "best.pt").exists():
        print(f"[SKIP train] {run_name} already has best.pt")
        return out_dir

    print(f"[TRAIN] {run_name} (imgsz={cell['imgsz']}, batch={cell['batch']}, epochs={cell['epochs']})")
    script = f"""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, r'{ROOT}')
from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer
from scripts.v1_train_on_v2split import SOTA_HYPERPARAMS

over = dict(SOTA_HYPERPARAMS)
over.update(dict(
    data=r'{DATA_YAML}', device='0',
    project=r'{PROJECT}', name=r'{run_name}',
    seed={seed}, pretrained=True, deterministic=True,
    imgsz={cell['imgsz']}, batch={cell['batch']}, epochs={cell['epochs']},
    workers={cell['workers']},
))

model = YOLO('configs/models/yolov11n_7ch.yaml')
try:
    model.load(r'{INIT_CKPT}')
except Exception as e:
    print('partial-load failed:', e)
model.train(trainer=MaskRescueTrainer, **over)
"""
    log = log_path(f"train_{run_name}")
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [PYTHON, "-X", "utf8", "-c", script],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
        )
    if proc.returncode != 0:
        print(f"[FAIL train] {run_name} (see {log})")
        return out_dir
    print(f"[OK train] {run_name}")
    return out_dir


def eval_one(run_dir: Path, cell: dict) -> None:
    weights = run_dir / "weights" / "best.pt"
    out_json = run_dir / "v2split_test_eval.json"
    if out_json.exists():
        print(f"[SKIP eval] {run_dir.name} already has v2split_test_eval.json")
        return
    if not weights.exists():
        print(f"[FAIL eval] {run_dir.name} missing best.pt")
        return
    print(f"[EVAL] {run_dir.name}")
    log = log_path(f"eval_{run_dir.name}")
    with log.open("w", encoding="utf-8") as f:
        subprocess.run([
            PYTHON, "-X", "utf8", "scripts/v2_evaluate.py",
            "--weights", str(weights),
            "--data", DATA_YAML,
            "--channels", "7",
            "--split", "test",
            "--imgsz", str(cell["imgsz"]),
            "--batch", str(cell["batch"]),
            "--output", str(out_json),
        ], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))


def f1_one(run_dir: Path, cell: dict) -> None:
    weights = run_dir / "weights" / "best.pt"
    out_json = run_dir / "test_f1_sweep.json"
    if out_json.exists():
        print(f"[SKIP f1] {run_dir.name} already has test_f1_sweep.json")
        return
    if not weights.exists():
        print(f"[FAIL f1] {run_dir.name} missing best.pt")
        return
    print(f"[F1] {run_dir.name}")
    log = log_path(f"f1_{run_dir.name}")
    with log.open("w", encoding="utf-8") as f:
        subprocess.run([
            PYTHON, "-X", "utf8", "scripts/v1_f1_conf_sweep.py",
            "--weights", str(weights),
            "--data", DATA_YAML,
            "--split", "test",
            "--imgsz", str(cell["imgsz"]),
            "--batch", str(cell["batch"]),
            "--output", str(out_json),
        ], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))


def aggregate() -> None:
    """Aggregate the 6 cells (2 tags x 3 seeds) into JSON + MD + figure."""
    out = {}
    for cell in CELLS:
        tag = cell["tag"]
        map50, map95, f1, debris, ship = [], [], [], [], []
        used = []
        for seed in SEEDS:
            rd = run_dir_for(tag, seed)
            ep = rd / "v2split_test_eval.json"
            sp = rd / "test_f1_sweep.json"
            if not ep.exists():
                continue
            d = json.loads(ep.read_text())
            map50.append(d["mAP50"])
            map95.append(d["mAP50_95"])
            pc = d.get("per_class", {})
            if (pc.get("Marine_Debris") or {}).get("AP50") is not None:
                debris.append(pc["Marine_Debris"]["AP50"])
            if (pc.get("Ship") or {}).get("AP50") is not None:
                ship.append(pc["Ship"]["AP50"])
            if sp.exists():
                f1.append(json.loads(sp.read_text())["best"]["f1"])
            used.append(seed)

        def msd(xs):
            if not xs:
                return {"mean": None, "std": None, "all": []}
            return {"mean": mean(xs), "std": stdev(xs) if len(xs) > 1 else 0.0, "all": xs}

        out[tag] = {
            "tag": tag, "n_seeds": len(used), "seeds": used,
            "mAP50": msd(map50), "mAP50_95": msd(map95),
            "f1": msd(f1), "Marine_Debris_AP50": msd(debris),
            "Ship_AP50": msd(ship),
        }

    out_json = ROOT / "docs/paper/v1_extra_ablation_aggregate.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"saved {out_json}")

    lines = ["# Extra ablation 3-seed aggregate (mean +- std)", "",
             "All honest setting, baseline architecture. Reference = main 3-seed Baseline Honest: mAP50 0.313 +- 0.038, F1 0.409 +- 0.022.",
             "",
             "| Variant | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(d):
        if d.get("mean") is None:
            return "--"
        return f"{d['mean']:.3f} +- {d['std']:.3f}"
    for cell in CELLS:
        c = out[cell["tag"]]
        lines.append(f"| {cell['tag']} | {c['n_seeds']} | {fmt(c['mAP50'])} | {fmt(c['mAP50_95'])} | {fmt(c['f1'])} | {fmt(c['Marine_Debris_AP50'])} | {fmt(c['Ship_AP50'])} |")
    out_md = ROOT / "docs/paper/v1_extra_ablation_aggregate.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_md}")
    print("\n" + "\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["Baseline\n(ref, ep30, 512)"] + [c["tag"] for c in CELLS]
        m50_m = [0.313] + [out[c["tag"]]["mAP50"]["mean"] or 0 for c in CELLS]
        m50_s = [0.038] + [out[c["tag"]]["mAP50"]["std"] or 0 for c in CELLS]
        f1_m  = [0.409] + [out[c["tag"]]["f1"]["mean"] or 0 for c in CELLS]
        f1_s  = [0.022] + [out[c["tag"]]["f1"]["std"] or 0 for c in CELLS]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        x = np.arange(len(labels))
        for ax, m, s, title, ylabel in [
            (axes[0], m50_m, m50_s, "mAP@0.5 (3 seeds, honest)", "mAP@0.5"),
            (axes[1], f1_m,  f1_s,  "Best F1 (3 seeds, honest)", "Best F1"),
        ]:
            colors = ["#888"] + ["#4393c3"] * len(CELLS)
            ax.bar(x, m, 0.6, yerr=s, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
            for j in range(len(labels)):
                ax.text(x[j], m[j] + s[j] + 0.005, f"{m[j]:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x, labels)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3, linestyle=":")
        fig.suptitle("Extra ablation vs reference baseline", fontsize=11)
        fig.tight_layout()
        out_fig = ROOT / "docs/paper/figures/extra_ablation_bars.png"
        fig.savefig(out_fig, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_fig}")
    except Exception as e:
        print(f"figure failed: {e}")


def main():
    t0 = time.time()
    print(f"[START] extra ablation pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for cell in CELLS:
        for seed in SEEDS:
            run_dir = train_one(cell, seed)
            eval_one(run_dir, cell)
            f1_one(run_dir, cell)
    print(f"\n[AGGREGATE]")
    aggregate()
    elapsed = (time.time() - t0) / 60
    DONE_MARKER.write_text(f"done at {time.strftime('%Y-%m-%d %H:%M:%S')}, elapsed {elapsed:.1f} min\n")
    print(f"[DONE] elapsed {elapsed:.1f} min, marker: {DONE_MARKER}")


if __name__ == "__main__":
    main()
