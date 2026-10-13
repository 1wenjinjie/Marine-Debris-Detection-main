"""
Aggregate structure-ablation results into Table 1.

Pulls eval_test.json + sweep_results.json + fps_bench.json from each run
and emits:
  - runs_v2/structure_ablation_summary.json   (machine-readable)
  - runs_v2/structure_ablation_table1.md      (paper-ready markdown)

Baseline reports 3-seed mean +/- std (from runs_v2/baseline_6ch_summary.json
when available). Other rows report seed=0 single values; gates/std rules
are described in V2_TEACHER_TASKS_MATRIX.md.

Usage:
  python scripts/v2_structure_ablation_summary.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs_v2"

# (display name, run prefix, seeds to read, imgsz)
# Route C decision (2026-04-30): all rows must be 3-seed for honest
# ablation reporting. CBAM 3-seed proved unstable; no row is the
# "final model" -- the paper's actual contribution is in the engineering
# table (focal+1024+Platt). See V2_TEACHER_TASKS_MATRIX.md.
ROWS = [
    ("Baseline (BCE)",          "baseline_6ch", [0, 1, 2], 512),
    ("+CBAM",                   "cbam_v2",      [0, 1, 2], 512),
    ("+BiFPN-lite",             "bifpn_v2",     [0, 1, 2], 512),
    ("+EIoU",                   "eiou_v2",      [0, 1, 2], 512),
    ("Full (CBAM+BiFPN+EIoU)",  "full_v2",      [0, 1, 2], 512),
]

CONF_LEVELS = ["0.001", "0.005", "0.01", "0.025", "0.05", "0.1"]


def f1_score(p: float, r: float) -> float:
    if p + r <= 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect_seed(prefix: str, seed: int) -> dict | None:
    run_dir = RUNS_DIR / f"{prefix}_s{seed}"
    eval_path = run_dir / "eval_test.json"
    fps_path = run_dir / "fps_bench.json"

    eval_d = load_json(eval_path)
    if eval_d is None:
        return None

    # baseline historical naming: runs are baseline_6ch_s* but sweep dirs
    # are sweep_baseline_s*. Try both forms.
    sweep_candidates = [
        RUNS_DIR / f"sweep_{prefix}_s{seed}" / "sweep_results.json",
        RUNS_DIR / f"sweep_{prefix.replace('_6ch', '')}_s{seed}" / "sweep_results.json",
    ]
    sweep_d = next((load_json(p) for p in sweep_candidates if p.exists()), None) or {}
    fps_d = load_json(fps_path) or {}

    p = eval_d.get("precision", 0.0) or 0.0
    r = eval_d.get("recall", 0.0) or 0.0

    out = {
        "seed": seed,
        "mAP50": eval_d.get("mAP50"),
        "mAP50_95": eval_d.get("mAP50_95"),
        "precision": p,
        "recall": r,
        "f1": f1_score(p, r),
        "marine_debris_ap50": (eval_d.get("per_class") or {}).get("Marine_Debris", {}).get("AP50"),
        "marine_debris_recall": (eval_d.get("per_class") or {}).get("Marine_Debris", {}).get("recall"),
        "algae_organic_ap50": (eval_d.get("per_class") or {}).get("Algae_Organic", {}).get("AP50"),
        "ship_ap50": (eval_d.get("per_class") or {}).get("Ship", {}).get("AP50"),
        "debris_tp_by_conf": {
            conf: (sweep_d.get("threshold_sweep", {}).get(conf, {}).get("tp_strict"))
            for conf in CONF_LEVELS
        },
        "fps_median": fps_d.get("fps_median"),
        "latency_ms_median": fps_d.get("latency_ms_median"),
    }
    return out


def aggregate(seeds_data: list[dict]) -> dict:
    """Return mean/std for each scalar field; nested dicts merge per-key."""
    if not seeds_data:
        return {}
    out = {"n_seeds": len(seeds_data)}
    scalar_keys = [
        "mAP50", "mAP50_95", "precision", "recall", "f1",
        "marine_debris_ap50", "marine_debris_recall",
        "algae_organic_ap50", "ship_ap50",
        "fps_median", "latency_ms_median",
    ]
    for key in scalar_keys:
        vals = [d[key] for d in seeds_data if d.get(key) is not None]
        if not vals:
            out[key] = None
            out[f"{key}_std"] = None
            continue
        out[key] = mean(vals)
        out[f"{key}_std"] = stdev(vals) if len(vals) > 1 else 0.0

    # Per-conf debris TP: list of values across seeds
    out["debris_tp_by_conf"] = {}
    for conf in CONF_LEVELS:
        vals = [
            d["debris_tp_by_conf"].get(conf)
            for d in seeds_data
            if d.get("debris_tp_by_conf", {}).get(conf) is not None
        ]
        if vals:
            out["debris_tp_by_conf"][conf] = {
                "mean": mean(vals),
                "std": stdev(vals) if len(vals) > 1 else 0.0,
                "values": vals,
            }
        else:
            out["debris_tp_by_conf"][conf] = None
    return out


def fmt_pm(mean_v, std_v, n_seeds: int, decimals: int = 3) -> str:
    if mean_v is None:
        return "—"
    if n_seeds <= 1:
        return f"{mean_v:.{decimals}f}"
    return f"{mean_v:.{decimals}f} ± {std_v:.{decimals}f}"


def fmt_int_pm(d: dict | None, n_seeds: int) -> str:
    """Format a debris-TP cell. Uses the number of seeds that actually
    contributed a sweep value (not the row's nominal seed count) so a
    multi-seed row with only one sweep prints a single integer rather
    than a misleading 'X ± 0'."""
    if not d or not d.get("values"):
        return "—"
    n_actual = len(d["values"])
    if n_actual <= 1:
        return f"{int(round(d['values'][0]))}"
    return f"{d['mean']:.1f} ± {d['std']:.1f}"


def render_table(table: dict) -> str:
    lines = [
        "# Table 1: Structure Ablation (imgsz=512, BCE, MARIDA v2 test set)",
        "",
        "| Model | mAP50 | mAP50-95 | P | R | F1 | "
        "Marine_Debris AP50 | Algae AP50 | Ship AP50 | "
        "Debris TP@0.025 | Debris TP@0.05 | Debris TP@0.1 | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table["rows"]:
        n = row["n_seeds"]
        cells = [
            row["display_name"],
            fmt_pm(row.get("mAP50"), row.get("mAP50_std"), n),
            fmt_pm(row.get("mAP50_95"), row.get("mAP50_95_std"), n),
            fmt_pm(row.get("precision"), row.get("precision_std"), n),
            fmt_pm(row.get("recall"), row.get("recall_std"), n),
            fmt_pm(row.get("f1"), row.get("f1_std"), n),
            fmt_pm(row.get("marine_debris_ap50"), row.get("marine_debris_ap50_std"), n),
            fmt_pm(row.get("algae_organic_ap50"), row.get("algae_organic_ap50_std"), n),
            fmt_pm(row.get("ship_ap50"), row.get("ship_ap50_std"), n),
            fmt_int_pm(row["debris_tp_by_conf"].get("0.025"), n),
            fmt_int_pm(row["debris_tp_by_conf"].get("0.05"), n),
            fmt_int_pm(row["debris_tp_by_conf"].get("0.1"), n),
            fmt_pm(row.get("fps_median"), row.get("fps_median_std"), n, decimals=1),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "**Notes:** ",
        "Baseline reports 3-seed mean ± std. Single-seed rows omit ± because "
        "matrix policy forbids reporting std from one seed. ",
        "Debris TP is strict (IoU > 0.5) over 13 Marine_Debris test instances. ",
        "FPS measured with batch=1, on a single GPU, after warmup.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build structure ablation summary (Table 1)")
    ap.add_argument("--out-json", default=str(RUNS_DIR / "structure_ablation_summary.json"))
    ap.add_argument("--out-md", default=str(RUNS_DIR / "structure_ablation_table1.md"))
    args = ap.parse_args()

    table = {"rows": [], "missing_runs": []}
    for display, prefix, seeds, imgsz in ROWS:
        per_seed = []
        for s in seeds:
            d = collect_seed(prefix, s)
            if d is None:
                table["missing_runs"].append(f"{prefix}_s{s}")
                continue
            per_seed.append(d)
        agg = aggregate(per_seed)
        agg["display_name"] = display
        agg["run_prefix"] = prefix
        agg["imgsz"] = imgsz
        agg["seeds_collected"] = [d["seed"] for d in per_seed]
        table["rows"].append(agg)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(table, indent=2))
    md = render_table(table)
    Path(args.out_md).write_text(md, encoding="utf-8")

    print(md)
    print(f"\nJSON: {args.out_json}")
    print(f"MD:   {args.out_md}")
    if table["missing_runs"]:
        print(f"\nMISSING (not yet trained or eval failed):")
        for r in table["missing_runs"]:
            print(f"  - {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
