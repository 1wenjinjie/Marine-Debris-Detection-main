"""
Table 3: scene-level Marine_Debris recovery breakdown.

For each test scene (4 unique scenes containing debris) and each model,
counts how many GT debris instances were recovered (IoU > 0.5) at a set
of practical confidence thresholds. Reports 3-seed mean ± std for the
final model so the reader can see scene-level stability.

Reads sweep_*/sweep_results.json instance_rows directly (each row has
best_conf and max_iou for one GT instance).

Outputs:
  runs_v2/scene_breakdown_summary.json
  runs_v2/scene_breakdown_table3.md

Usage:
  python scripts/v2_make_table3_scene_breakdown.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs_v2"

CONF_LEVELS = [0.025, 0.05, 0.1]
IOU_TP = 0.5

# Models to include in the table:
# (display name, list of sweep dir basenames -- one per seed)
MODELS = [
    ("Baseline (BCE, 512)", [
        "sweep_baseline_s0",
    ]),
    ("+ focal γ=2.0 (i1024)", [
        "sweep_focal_g2.0_i1024_s0",
        "sweep_focal_g2.0_i1024_s1",
        "sweep_focal_g2.0_i1024_s2",
    ]),
]

SCENE_RE = re.compile(r"^(S2_\d+-\d+-\d+_[A-Z0-9]+)_\d+$")


def scene_of_tile(tile_name: str) -> str | None:
    m = SCENE_RE.match(tile_name)
    return m.group(1) if m else None


def load_sweep(name: str) -> dict | None:
    p = RUNS_DIR / name / "sweep_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def per_scene_tp(instance_rows, conf_thr: float) -> dict[str, int]:
    """For each scene, count rows where max_iou > IOU_TP and best_conf >= conf_thr."""
    counts: dict[str, int] = defaultdict(int)
    for row in instance_rows:
        scene = scene_of_tile(row["tile"])
        if scene is None:
            continue
        if row.get("max_iou", 0.0) > IOU_TP and row.get("best_conf", 0.0) >= conf_thr:
            counts[scene] += 1
    return dict(counts)


def per_scene_gt_count(instance_rows) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in instance_rows:
        scene = scene_of_tile(row["tile"])
        if scene is None:
            continue
        counts[scene] += 1
    return dict(counts)


def fmt_pm(values: list[int]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]}"
    m = mean(values)
    s = stdev(values)
    return f"{m:.1f} ± {s:.1f}"


def main() -> int:
    # First pass: get GT counts (consistent across all sweeps - test split)
    any_sweep = next((load_sweep(n) for _, ns in MODELS for n in ns
                      if load_sweep(n) is not None), None)
    if any_sweep is None:
        print("No sweep data found; aborting.")
        return 1
    scenes = sorted(per_scene_gt_count(any_sweep["instance_rows"]).keys())
    n_gt = per_scene_gt_count(any_sweep["instance_rows"])

    # Second pass: collect per-conf, per-scene TP across seeds for each model
    table = {"scenes": scenes, "n_gt": n_gt, "models": []}
    for model_name, sweep_names in MODELS:
        per_seed_counts = {scene: {conf: [] for conf in CONF_LEVELS}
                           for scene in scenes}
        valid_seeds = 0
        for name in sweep_names:
            d = load_sweep(name)
            if d is None:
                continue
            valid_seeds += 1
            for conf in CONF_LEVELS:
                tp = per_scene_tp(d["instance_rows"], conf)
                for scene in scenes:
                    per_seed_counts[scene][conf].append(tp.get(scene, 0))

        table["models"].append({
            "name": model_name,
            "seeds": valid_seeds,
            "per_scene": {
                scene: {
                    str(conf): {
                        "mean": (mean(vals) if vals else None),
                        "std":  (stdev(vals) if len(vals) > 1 else 0.0),
                        "values": vals,
                    }
                    for conf, vals in conf_dict.items()
                }
                for scene, conf_dict in per_seed_counts.items()
            },
        })

    # Render Markdown
    lines = [
        "# Table 3: Scene-level Marine_Debris recovery (test split)",
        "",
        "Each cell is the number of debris GT instances recovered "
        "(IoU > 0.5) at a given confidence threshold. Multi-seed cells "
        "are reported as mean ± std.",
        "",
    ]
    for conf in CONF_LEVELS:
        lines += [
            f"## conf = {conf}",
            "",
            "| Scene | GT count | " + " | ".join(m["name"] for m in table["models"]) + " |",
            "|---|---:|" + "|".join(["---:"] * len(table["models"])) + "|",
        ]
        for scene in scenes:
            row = [
                scene,
                str(n_gt[scene]),
            ]
            for m in table["models"]:
                cell = m["per_scene"][scene][str(conf)]
                row.append(fmt_pm(cell["values"]))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Save
    json_out = RUNS_DIR / "scene_breakdown_summary.json"
    md_out = RUNS_DIR / "scene_breakdown_table3.md"
    json_out.write_text(json.dumps(table, indent=2))
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nJSON: {json_out}")
    print(f"MD:   {md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
