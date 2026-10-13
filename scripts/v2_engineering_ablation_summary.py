"""
Aggregate engineering-optimization results into Table 2.

Table 2 isolates input-size / loss / post-hoc effects, holding the model
structure fixed at the baseline (no CBAM/BiFPN/EIoU). Outputs:

  runs_v2/engineering_ablation_summary.json   (machine-readable)
  runs_v2/engineering_ablation_table2.md      (paper-ready markdown)

Rows:
  1. Baseline (BCE, imgsz=512)
  2. + imgsz=1024
  3. + focal gamma=2.0, imgsz=1024
  4. + Platt calibration (post-hoc, on row 3 weights)

Usage:
  python scripts/v2_engineering_ablation_summary.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs_v2"

CONF_LEVELS = ["0.001", "0.005", "0.01", "0.025", "0.05", "0.1"]

# (display name, run prefix, seeds, imgsz, calibration_summary or None)
# Calibration summary, when present, overrides debris_tp_by_conf with
# cal_tp from calib_<prefix>_summary.json. Detection metrics stay
# identical to the underlying weights.
ROWS = [
    {
        "name": "Baseline (BCE, 512)",
        "prefix": "baseline_6ch",
        "seeds": [0, 1, 2],
        "imgsz": 512,
        "calib_summary": None,
    },
    {
        "name": "+ imgsz=1024",
        "prefix": "imgsz1024",
        "seeds": [0, 1, 2],
        "imgsz": 1024,
        "calib_summary": None,
    },
    {
        "name": "+ focal γ=2.0, i1024",
        "prefix": "focal_g2.0_i1024",
        "seeds": [0, 1, 2],
        "imgsz": 1024,
        "calib_summary": None,
    },
    {
        "name": "+ Platt calibration",
        "prefix": "focal_g2.0_i1024",
        "seeds": [0, 1, 2],
        "imgsz": 1024,
        "calib_summary": "calib_focal_g2.0_i1024_summary.json",
    },
]


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


def find_sweep(prefix: str, seed: int) -> Path | None:
    candidates = [
        RUNS_DIR / f"sweep_{prefix}_s{seed}" / "sweep_results.json",
        RUNS_DIR / f"sweep_{prefix.replace('_6ch', '')}_s{seed}" / "sweep_results.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def collect_seed(prefix: str, seed: int) -> dict | None:
    eval_p = RUNS_DIR / f"{prefix}_s{seed}" / "eval_test.json"
    eval_d = load_json(eval_p)
    if eval_d is None:
        return None

    sweep_d = load_json(find_sweep(prefix, seed)) if find_sweep(prefix, seed) else {}

    p = eval_d.get("precision", 0.0) or 0.0
    r = eval_d.get("recall", 0.0) or 0.0

    return {
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
    }


def collect_calibrated_tp(calib_summary_path: Path) -> dict[str, dict]:
    """Pull post-Platt cal_tp from a calibration summary; fall back gracefully."""
    d = load_json(calib_summary_path)
    if d is None:
        return {}
    out = {conf: {"values": []} for conf in CONF_LEVELS}
    for r in d.get("results", []):
        ts = r.get("threshold_sweep", {})
        for conf in CONF_LEVELS:
            cal = ts.get(conf, {}).get("cal_tp")
            if cal is not None:
                out[conf]["values"].append(int(cal))
    for conf in CONF_LEVELS:
        vals = out[conf]["values"]
        if vals:
            out[conf]["mean"] = mean(vals)
            out[conf]["std"] = stdev(vals) if len(vals) > 1 else 0.0
        else:
            out[conf] = None
    return out


def find_fps_for_imgsz(imgsz: int) -> dict[str, dict]:
    """Pick fps from whichever batch summary contains the right imgsz."""
    for stem in ("structure_ablation", "engineering"):
        p = RUNS_DIR / f"fps_bench_summary_{stem}.json"
        d = load_json(p)
        if not d:
            continue
        for run_name, info in d.items():
            if info.get("imgsz") == imgsz:
                return {"run": run_name, **info}
    return {}


def aggregate_seeds(seeds_data: list[dict]) -> dict:
    if not seeds_data:
        return {}
    out = {"n_seeds": len(seeds_data)}
    scalar_keys = [
        "mAP50", "mAP50_95", "precision", "recall", "f1",
        "marine_debris_ap50", "marine_debris_recall",
        "algae_organic_ap50", "ship_ap50",
    ]
    for key in scalar_keys:
        vals = [d[key] for d in seeds_data if d.get(key) is not None]
        if not vals:
            out[key] = None
            out[f"{key}_std"] = None
            continue
        out[key] = mean(vals)
        out[f"{key}_std"] = stdev(vals) if len(vals) > 1 else 0.0

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


def fmt_int_pm(d: dict | None, n_seeds: int = 0) -> str:
    if not d or not d.get("values"):
        return "—"
    n_actual = len(d["values"])
    if n_actual <= 1:
        return f"{int(round(d['values'][0]))}"
    return f"{d['mean']:.1f} ± {d['std']:.1f}"


def render_table(table: dict) -> str:
    lines = [
        "# Table 2: Engineering Optimization (no structural change, MARIDA v2 test set)",
        "",
        "| Configuration | imgsz | mAP50 | mAP50-95 | "
        "Marine_Debris AP50 | Algae AP50 | Ship AP50 | "
        "Debris TP@0.025 | Debris TP@0.05 | Debris TP@0.1 | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table["rows"]:
        n = row.get("n_seeds", 0)
        cells = [
            row["display_name"],
            str(row["imgsz"]),
            fmt_pm(row.get("mAP50"), row.get("mAP50_std"), n),
            fmt_pm(row.get("mAP50_95"), row.get("mAP50_95_std"), n),
            fmt_pm(row.get("marine_debris_ap50"), row.get("marine_debris_ap50_std"), n),
            fmt_pm(row.get("algae_organic_ap50"), row.get("algae_organic_ap50_std"), n),
            fmt_pm(row.get("ship_ap50"), row.get("ship_ap50_std"), n),
            fmt_int_pm(row["debris_tp_by_conf"].get("0.025")),
            fmt_int_pm(row["debris_tp_by_conf"].get("0.05")),
            fmt_int_pm(row["debris_tp_by_conf"].get("0.1")),
            (f"{row['fps']:.1f}" if row.get("fps") else "—"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "**Notes:** ",
        "Structure is held fixed at the baseline YOLOv11n-6ch backbone+head. ",
        "imgsz=1024 row is single-seed (s0); other rows are 3-seed where data permits. ",
        "Platt calibration row reuses the focal+1024 weights; detection metrics are identical, only TP@conf changes. ",
        "Debris TP is strict (IoU > 0.5) over 13 Marine_Debris test instances. ",
        "FPS measured at the row's imgsz with batch=1 after warmup (RTX 3090 Ti).",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build engineering ablation summary (Table 2)")
    ap.add_argument("--out-json", default=str(RUNS_DIR / "engineering_ablation_summary.json"))
    ap.add_argument("--out-md", default=str(RUNS_DIR / "engineering_ablation_table2.md"))
    args = ap.parse_args()

    table = {"rows": [], "missing_runs": []}

    for spec in ROWS:
        per_seed = []
        for s in spec["seeds"]:
            d = collect_seed(spec["prefix"], s)
            if d is None:
                table["missing_runs"].append(f"{spec['prefix']}_s{s}")
                continue
            per_seed.append(d)

        agg = aggregate_seeds(per_seed)
        agg["display_name"] = spec["name"]
        agg["imgsz"] = spec["imgsz"]
        agg["seeds_collected"] = [d["seed"] for d in per_seed]

        # Override debris TP from calibration summary if specified
        if spec["calib_summary"]:
            calib_path = RUNS_DIR / spec["calib_summary"]
            cal_tp = collect_calibrated_tp(calib_path)
            if cal_tp:
                agg["debris_tp_by_conf"] = cal_tp

        # FPS lookup by imgsz
        fps_info = find_fps_for_imgsz(spec["imgsz"])
        agg["fps"] = fps_info.get("fps_median")
        agg["fps_run"] = fps_info.get("run")

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
