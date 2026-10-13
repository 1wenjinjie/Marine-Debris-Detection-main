"""Auto-pilot imgsz=1024 LEAKY ablation: baseline + BiFPN x 3 seeds.

Strategic comparison cell for paper: does BiFPN @ imgsz=1024 (our new honest
SOTA) still beat baseline under the leaky split? Or does the leak mask the
resolution+architecture interaction?

All cells: leaky data (marida_mask_rescue_7ch.yaml), clean init, imgsz=1024
batch=4 workers=0 ep=30. Output run names:
  yolov11n_7ch_leaky_<ablation>_imgsz1024_clean_init_s<seed>

Final aggregate combines new leaky imgsz=1024 cells + prior leaky imgsz=512
cells for cross-resolution comparison. Output:
  docs/paper/v1_imgsz1024_leaky_aggregate.{json,md}
  docs/paper/figures/imgsz1024_leaky_bars.png
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
DATA_YAML = "configs/datasets/marida_mask_rescue_7ch.yaml"
PROJECT = "runs_v1_leaky"
LOG_DIR = ROOT / "runs/detect/runs_v1_leaky/imgsz1024_leaky_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DONE_MARKER = ROOT / "runs/detect/runs_v1_leaky/.imgsz1024_leaky_done"

IMGSZ = 1024
BATCH = 4
WORKERS = 0
EPOCHS = 30
ABLATIONS = ["baseline", "bifpn"]
SEEDS = [0, 1, 2]


def run_dir_for(ab: str, seed: int) -> Path:
    return ROOT / f"runs/detect/{PROJECT}/yolov11n_7ch_leaky_{ab}_imgsz1024_clean_init_s{seed}"


def log_path(stage: str, ab: str, seed: int) -> Path:
    return LOG_DIR / f"{stage}_{ab}_s{seed}.log"


def train_one(ab: str, seed: int) -> Path:
    run_dir = run_dir_for(ab, seed)
    name = run_dir.name
    if (run_dir / "weights" / "best.pt").exists():
        print(f"[SKIP train] {name}")
        return run_dir
    print(f"[TRAIN] {name}")
    log = log_path("train", ab, seed)
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run([
            PYTHON, "-X", "utf8", "scripts/v1_ablation_train.py",
            "--ablation", ab,
            "--seed", str(seed),
            "--name", name,
            "--project", PROJECT,
            "--data", DATA_YAML,
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
    out_json = run_dir / "leaky_test_eval.json"
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


def collect_cell(ab: str, imgsz_tag: str) -> dict:
    """Collect a cell's results across 3 seeds.

    imgsz_tag = '512' (read from prior leaky aggregate) or '1024' (new runs).
    """
    if imgsz_tag == "1024":
        run_dirs = [run_dir_for(ab, s) for s in SEEDS]
        eval_name = "leaky_test_eval.json"
    else:
        run_dirs = []
        for s in SEEDS:
            base = ROOT / f"runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_{ab}_clean_init"
            run_dirs.append(base if s == 0 else Path(f"{base}_s{s}"))
        eval_name = "leaky_test_eval.json"

    map50, map95, f1, debris, ship = [], [], [], [], []
    used = []
    for s, rd in zip(SEEDS, run_dirs):
        ep = rd / eval_name
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
        used.append(s)

    def msd(xs):
        if not xs:
            return {"mean": None, "std": None, "all": []}
        return {"mean": mean(xs), "std": stdev(xs) if len(xs) > 1 else 0.0, "all": xs}

    return {
        "ablation": ab, "imgsz": int(imgsz_tag), "n_seeds": len(used), "seeds": used,
        "mAP50": msd(map50), "mAP50_95": msd(map95), "f1": msd(f1),
        "Marine_Debris_AP50": msd(debris), "Ship_AP50": msd(ship),
    }


def aggregate() -> None:
    out = {}
    for imgsz_tag in ("512", "1024"):
        for ab in ABLATIONS:
            out[f"{ab}_{imgsz_tag}"] = collect_cell(ab, imgsz_tag)

    out_json = ROOT / "docs/paper/v1_imgsz1024_leaky_aggregate.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"saved {out_json}")

    lines = ["# imgsz=1024 leaky ablation, 3-seed aggregate (mean +- std)", "",
             "Leaky data (marida_mask_rescue_7ch.yaml, 1216 train tiles, GT-derived heatmap as ch7).",
             "imgsz=512 cells reused from prior 3-seed leaky aggregate.",
             "imgsz=1024 cells new (batch=4 workers=0 ep=30).",
             "",
             "| Ablation | imgsz | n | mAP50 | mAP50-95 | Best F1 | Debris AP50 | Ship AP50 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(d):
        return "--" if d.get("mean") is None else f"{d['mean']:.3f} +- {d['std']:.3f}"
    for ab in ABLATIONS:
        for imgsz_tag in ("512", "1024"):
            c = out[f"{ab}_{imgsz_tag}"]
            lines.append(f"| {ab} | {imgsz_tag} | {c['n_seeds']} | {fmt(c['mAP50'])} | {fmt(c['mAP50_95'])} | {fmt(c['f1'])} | {fmt(c['Marine_Debris_AP50'])} | {fmt(c['Ship_AP50'])} |")
    out_md = ROOT / "docs/paper/v1_imgsz1024_leaky_aggregate.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_md}")
    print("\n" + "\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # 4 bars per metric: baseline_512, baseline_1024, bifpn_512, bifpn_1024
        x_labels = ["baseline\n512", "baseline\n1024", "bifpn\n512", "bifpn\n1024"]
        colors = ["#888", "#4393c3", "#d6604d", "#1a5490"]
        groups = [("baseline", "512"), ("baseline", "1024"), ("bifpn", "512"), ("bifpn", "1024")]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        x = np.arange(len(x_labels))
        for ax, key, title, ylabel in [
            (axes[0], "mAP50", "mAP@0.5 (leaky, 3 seeds)", "mAP@0.5"),
            (axes[1], "f1", "Best F1 (leaky, 3 seeds)", "Best F1"),
            (axes[2], "Marine_Debris_AP50", "Marine_Debris AP@0.5 (leaky, 3 seeds)", "Debris AP@0.5"),
        ]:
            m = [out[f"{ab}_{r}"][key]["mean"] or 0 for ab, r in groups]
            s = [out[f"{ab}_{r}"][key]["std"]  or 0 for ab, r in groups]
            ax.bar(x, m, 0.6, yerr=s, capsize=4, color=colors,
                   edgecolor="black", linewidth=0.6)
            for j in range(len(x_labels)):
                ax.text(x[j], m[j] + s[j] + 0.005, f"{m[j]:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x, x_labels)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3, linestyle=":")
        fig.suptitle("Leaky setting: imgsz=512 vs imgsz=1024 (baseline + BiFPN, 3 seeds)", fontsize=12)
        fig.tight_layout()
        out_fig = ROOT / "docs/paper/figures/imgsz1024_leaky_bars.png"
        fig.savefig(out_fig, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_fig}")
    except Exception as e:
        print(f"figure failed: {e}")


def main():
    t0 = time.time()
    print(f"[START] leaky imgsz=1024 pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for ab in ABLATIONS:
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
