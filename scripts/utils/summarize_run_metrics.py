"""Utility to summarize Ultralytics run metrics from results.csv."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class MetricSummary:
    run: Path
    best_epoch: int
    best_precision: float
    best_recall: float
    best_map50: float
    best_map5095: float
    last_epoch: int
    last_precision: float
    last_recall: float
    last_map50: float
    last_map5095: float

    def as_rows(self) -> Sequence[tuple[str, float]]:
        return (
            ("best_epoch", self.best_epoch),
            ("best_precision", self.best_precision),
            ("best_recall", self.best_recall),
            ("best_mAP50", self.best_map50),
            ("best_mAP50-95", self.best_map5095),
            ("last_epoch", self.last_epoch),
            ("last_precision", self.last_precision),
            ("last_recall", self.last_recall),
            ("last_mAP50", self.last_map50),
            ("last_mAP50-95", self.last_map5095),
        )


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_results(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No results.csv found at {csv_path}")
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize_run(run_dir: Path) -> MetricSummary:
    rows = load_results(run_dir / "results.csv")
    if not rows:
        raise ValueError(f"No rows found in {run_dir}/results.csv")

    def row_key(row: dict[str, str]) -> float:
        return _to_float(row.get("metrics/mAP50(B)", "nan"))

    best_row = max(rows, key=row_key)
    last_row = rows[-1]

    return MetricSummary(
        run=run_dir,
        best_epoch=int(best_row["epoch"]),
        best_precision=_to_float(best_row["metrics/precision(B)"]),
        best_recall=_to_float(best_row["metrics/recall(B)"]),
        best_map50=_to_float(best_row["metrics/mAP50(B)"]),
        best_map5095=_to_float(best_row["metrics/mAP50-95(B)"]),
        last_epoch=int(last_row["epoch"]),
        last_precision=_to_float(last_row["metrics/precision(B)"]),
        last_recall=_to_float(last_row["metrics/recall(B)"]),
        last_map50=_to_float(last_row["metrics/mAP50(B)"]),
        last_map5095=_to_float(last_row["metrics/mAP50-95(B)"]),
    )


def format_summary(summary: MetricSummary) -> str:
    header = f"Run: {summary.run}"
    lines = [header]
    for key, value in summary.as_rows():
        lines.append(f"  {key:>15s}: {value:.4f}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="One or more run directories containing results.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for run in args.runs:
        summary = summarize_run(run)
        print(format_summary(summary))
        print()


if __name__ == "__main__":
    main()
