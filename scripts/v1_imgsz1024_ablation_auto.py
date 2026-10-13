"""Auto-pilot imgsz=1024 ablation: 4 architectures x 3 seeds (baseline already done).

Builds on the just-finished extra ablation. Reuses the existing 3-seed
imgsz=1024 baseline cells, adds CBAM / BiFPN / EIoU / Full at imgsz=1024.

All cells: honest setting, physidx, clean init, imgsz=1024 batch=4 workers=0
ep=30. Output run names: yolov11n_7ch_v2split_physidx_<ablation>_imgsz1024_clean_init_s<seed>.

Final aggregate combines baseline (already done) + 4 new cells = 5x3 matrix,
written to docs/paper/v1_imgsz1024_ablation_aggregate.{json,md} +
docs/paper/figures/imgsz1024_ablation_bars.png.

Designed to run unattended via run_in_background.
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
LOG_DIR = ROOT / "runs/detect/runs_v1_v2split/imgsz1024_ablation_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DONE_MARKER = ROOT / "runs/detect/runs_v1_v2split/.imgsz1024_ablation_done"

# imgsz=1024 settings (workers=0 is mandatory on Windows, see prior crash)
IMGSZ = 1024
BATCH = 4
WORKERS = 0
EPOCHS = 30

# All 5 ablations. Baseline cells are reused from extra ablation (different naming).
ABLATIONS = ["baseline", "cbam", "bifpn", "eiou", "full"]
NEW_ABLATIONS = ["cbam", "bifpn", "eiou", "full"]
SEEDS = [0, 1, 2]


def run_dir_for(ab: str, seed: int) -> Path:
    """Path to a cell's run directory.

    Baseline cells: reuse extra-ablation naming (no _baseline_ infix).
    New cells: yolov11n_7ch_v2split_physidx_<ab>_imgsz1024_clean_init_s<seed>.
    """
    if ab == "baseline":
        name = f"yolov11n_7ch_v2split_physidx_imgsz1024_clean_init_s{seed}"
    else:
        name = f"yolov11n_7ch_v2split_physidx_{ab}_imgsz1024_clean_init_s{seed}"
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


def aggregate() -> None:
    out = {}
    for ab in ABLATIONS:
        map50, map95, f1, debris, ship = [], [], [], [], []
        used = []
        for seed in SEEDS:
            rd = run_dir_for(ab, seed)
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

        out[ab] = {
            "ablation": ab, "imgsz": IMGSZ, "n_seeds": len(used), "seeds": used,
            "mAP50": msd(map50), "mAP50_95": msd(map95), "f1": msd(f1),
            "Marine_Debris_AP50": msd(debris), "Ship_AP50": msd(ship),
        }

    out_json = ROOT / "docs/paper/v1_imgsz1024_ablation_aggregate.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"saved {out_json}")

    lines = ["# imgsz=1024 ablation, 3-seed aggregate (mean +- std)", "",
             "All honest setting, physidx, clean init, imgsz=1024 batch=4 workers=0 ep=30.",
             "Reference = main 3-seed Baseline Honest @ imgsz=512: mAP50 0.313 +- 0.038, F1 0.409 +- 0.022, Debris AP50 0.492 +- 0.056.",
             "",
             "| Ablation | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(d):
        return "--" if d.get("mean") is None else f"{d['mean']:.3f} +- {d['std']:.3f}"
    for ab in ABLATIONS:
        c = out[ab]
        lines.append(f"| {ab} | {c['n_seeds']} | {fmt(c['mAP50'])} | {fmt(c['mAP50_95'])} | {fmt(c['f1'])} | {fmt(c['Marine_Debris_AP50'])} | {fmt(c['Ship_AP50'])} |")
    out_md = ROOT / "docs/paper/v1_imgsz1024_ablation_aggregate.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_md}")
    print("\n" + "\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ABLATIONS
        m50_m = [out[a]["mAP50"]["mean"] or 0 for a in ABLATIONS]
        m50_s = [out[a]["mAP50"]["std"] or 0 for a in ABLATIONS]
        f1_m  = [out[a]["f1"]["mean"]    or 0 for a in ABLATIONS]
        f1_s  = [out[a]["f1"]["std"]     or 0 for a in ABLATIONS]
        deb_m = [out[a]["Marine_Debris_AP50"]["mean"] or 0 for a in ABLATIONS]
        deb_s = [out[a]["Marine_Debris_AP50"]["std"]  or 0 for a in ABLATIONS]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        x = np.arange(len(labels))
        for ax, m, s, title, ylabel, ref_y in [
            (axes[0], m50_m, m50_s, "mAP@0.5 (imgsz=1024, 3 seeds)", "mAP@0.5", 0.313),
            (axes[1], f1_m,  f1_s,  "Best F1 (imgsz=1024, 3 seeds)", "Best F1", 0.409),
            (axes[2], deb_m, deb_s, "Marine_Debris AP@0.5 (imgsz=1024, 3 seeds)", "Debris AP@0.5", 0.492),
        ]:
            ax.bar(x, m, 0.6, yerr=s, capsize=4, color="#4393c3",
                   edgecolor="black", linewidth=0.6)
            ax.axhline(ref_y, color="#888", linestyle="--", linewidth=1.2,
                       label=f"imgsz=512 baseline ({ref_y:.3f})")
            for j in range(len(labels)):
                ax.text(x[j], m[j] + s[j] + 0.005, f"{m[j]:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x, labels)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3, linestyle=":")
            ax.legend(loc="upper right", fontsize=8)
        fig.suptitle("imgsz=1024 architecture ablation, 3-seed mean +- std", fontsize=12)
        fig.tight_layout()
        out_fig = ROOT / "docs/paper/figures/imgsz1024_ablation_bars.png"
        fig.savefig(out_fig, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_fig}")
    except Exception as e:
        print(f"figure failed: {e}")


def main():
    t0 = time.time()
    print(f"[START] imgsz=1024 ablation pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure baseline cells (already done) have their eval/f1 outputs in place
    for seed in SEEDS:
        rd = run_dir_for("baseline", seed)
        if (rd / "weights" / "best.pt").exists():
            eval_one(rd)
            f1_one(rd)

    # Train + eval + F1 for the 4 new architectures
    for ab in NEW_ABLATIONS:
        for seed in SEEDS:
            rd = train_one(ab, seed)
            eval_one(rd)
            f1_one(rd)

    print(f"\n[AGGREGATE]")
    aggregate()

    elapsed = (time.time() - t0) / 60
    DONE_MARKER.write_text(f"done at {time.strftime('%Y-%m-%d %H:%M:%S')}, elapsed {elapsed:.1f} min\n")
    print(f"[DONE] elapsed {elapsed:.1f} min, marker: {DONE_MARKER}")


if __name__ == "__main__":
    main()
