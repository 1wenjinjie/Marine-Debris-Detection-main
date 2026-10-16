"""MADOS external test re-evaluated after removing crops that the T6 duplicate map
identifies as the same Sentinel-2 acquisition as MARIDA patches.

Variants (crop lists derived from runs_mpb/mados_overlap/v2/dup_map_full.csv,
rows with mados_split == test and same_acq == True):
  t6_test_vs_marida_train      : copies of MARIDA *train* patches (expected 155 crops)
  t6_test_vs_marida_trainval   : copies of MARIDA train or val patches
  t6_test_vs_marida_any        : copies of any MARIDA patch
Frozen checkpoints, evaluation settings and paired current-baseline logic are those of
scripts/mpb/rerun_mados_without_overlap.py (imported unchanged). Evaluation only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUP = ROOT / "runs_mpb" / "mados_overlap" / "v2" / "dup_map_full.csv"
CROSSTAB = ROOT / "runs_mpb" / "mados_overlap" / "v2" / "dup_crosstab.csv"
T6_SUMMARY = ROOT / "runs_mpb" / "mados_overlap" / "v2" / "t6_summary.json"
RERUN = ROOT / "scripts" / "mpb" / "rerun_mados_without_overlap.py"
OUT = ROOT / "runs_v2" / "_tmp_eval" / "mados_t6_filtered"
VARIANTS = {
    "t6_test_vs_marida_train": {"train"},
    "t6_test_vs_marida_trainval": {"train", "val"},
    "t6_test_vs_marida_any": {"train", "val", "test"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rerun():
    spec = importlib.util.spec_from_file_location("rerun_mados_without_overlap", RERUN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def crop_sets() -> dict[str, list[str]]:
    expected = json.loads(T6_SUMMARY.read_text(encoding="utf-8"))["dup_map_sha256"]
    if sha256(DUP) != expected:
        raise SystemExit("[error] dup_map_full.csv differs from the T6 summary hash")
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    with DUP.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["mados_split"] == "test" and row["same_acq"] == "True":
                by_split[row["marida_split"]].add(row["mados_id"])
    with CROSSTAB.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["mados_split"] == "test":
                n = int(row["same_acq_mados_crops"])
                if len(by_split[row["marida_split"]]) != n:
                    raise SystemExit(f"[error] crosstab mismatch for test/{row['marida_split']}: "
                                     f"{len(by_split[row['marida_split']])} vs {n}")
    out = {}
    for name, splits in VARIANTS.items():
        out[name] = sorted(set().union(*(by_split[s] for s in splits)))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", default="t6_test_vs_marida_train,t6_test_vs_marida_trainval")
    p.add_argument("--device", default="0")
    p.add_argument("--stage", choices=("prepare", "run"), default="run")
    args = p.parse_args()
    rr = load_rerun()
    sets = crop_sets()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"dup_map_sha256": sha256(DUP), "rerun_script_sha256": sha256(RERUN),
              "script_sha256": sha256(Path(__file__)), "variants": {}}
    for name in args.variants.split(","):
        crops = sets[name]
        ids = rr._parse_ids(",".join(c.removeprefix("Scene_") for c in crops),
                            rr.CROP_PATTERN, "crop")
        if len(ids) != len(crops):
            raise SystemExit(f"[error] {name}: crop id parsing changed the count")
        print(f"[{name}] excluding {len(ids)} MADOS test crops", flush=True)
        protocol, out_dir = rr.prepare(name, ids, [], args.device)
        print(json.dumps(protocol["counts"]), flush=True)
        entry = {"n_excluded_crops": len(ids), "excluded_crops": ids,
                 "counts": protocol["counts"], "output_dir": str(out_dir)}
        if args.stage == "run":
            rr.run(protocol, out_dir, args.device)
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            cells = {}
            for cell, v in summary["cells"].items():
                md = lambda d: d["per_class"]["Marine_Debris"]["AP50"]
                cells[cell] = {
                    "mean_full_MD_AP50": md(v["mean_current_baseline_ap"]),
                    "mean_filtered_MD_AP50": md(v["mean_filtered_ap"]),
                    "mean_delta_MD_AP50": md(v["mean_delta_ap_filtered_minus_current_baseline"]),
                    "per_seed": [{"seed": r["seed"], "full": md(r["current_baseline_ap"]),
                                  "filtered": md(r["filtered_ap"])} for r in v["per_seed"]],
                }
                print(f"  {cell}: full {cells[cell]['mean_full_MD_AP50']:.4f} -> filtered "
                      f"{cells[cell]['mean_filtered_MD_AP50']:.4f}", flush=True)
            entry["cells"] = cells
        report["variants"][name] = entry
    (OUT / f"report_{args.stage}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
