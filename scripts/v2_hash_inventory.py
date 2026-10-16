"""SHA-256 inventory of the v2 analysis scripts and their result files.

Read-only; writes only into --out (must not exist or be empty).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("scripts/v2_relaxed_match.py", "scripts/v2_cv_fold_ap.py",
           "scripts/v2_fdi_baseline.py", "scripts/v2_fdi_sensitivity.py",
           "scripts/v2_error_analysis.py", "scripts/mpb/cv_pipeline.py",
           "scripts/mpb/audit_toolkit.py", "scripts/mpb/run_experiments.py",
           "scripts/mpb/prepare_protocol.py")
RESULT_DIRS = ("runs_v2/_tmp_eval/relaxed_match", "runs_v2/_tmp_eval/cv_fold_ap",
               "runs_v2/_tmp_eval/fdi_baseline", "runs_v2/_tmp_eval/fdi_sensitivity",
               "runs_v2/_tmp_eval/error_analysis")
EXTRA_FILES = ("runs_v2/_tmp_eval/plan_cv_relaxed.md", "runs_cv/cv_stat_results.json",
               "runs_mpb/stat_revision/cv9_fold_assignment.csv",
               "data/processed/marida_mpb_v1/audit/patch_inventory.csv",
               "data/processed/marida_mpb_v1/audit/marine_debris_components.csv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs_v2/_tmp_eval/hash_inventory")
    out = ROOT / p.parse_args().out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"[error] output directory exists and is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    paths = [("script", s) for s in SCRIPTS] + [("input", s) for s in EXTRA_FILES]
    for d in RESULT_DIRS:
        for f in sorted((ROOT / d).glob("*")):
            if f.is_file():
                paths.append(("result", f.relative_to(ROOT).as_posix()))
    for kind, rel in paths:
        f = ROOT / rel
        if not f.is_file():
            rows.append({"kind": kind, "path": rel, "bytes": "", "sha256": "MISSING"})
            continue
        rows.append({"kind": kind, "path": rel, "bytes": f.stat().st_size,
                     "sha256": sha256_file(f)})
        print(f"{kind:6s} {rel}  {rows[-1]['sha256'][:12]}", flush=True)
    with (out / "hashes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "generated_utc.txt").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n", encoding="utf-8")
    print(f"[done] {len(rows)} files -> {out / 'hashes.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
