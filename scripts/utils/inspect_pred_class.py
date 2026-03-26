"""Collect statistics for a target class within Ultralytics prediction TXT outputs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class PredHit:
    name: str
    conf: float
    line: str


def parse_prediction_file(path: Path, class_id: int) -> list[PredHit]:
    hits: list[PredHit] = []
    if not path.exists():
        return hits
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls = int(float(parts[0]))
            except (ValueError, IndexError):
                continue
            if cls != class_id:
                continue
            conf = float(parts[5]) if len(parts) > 5 else float("nan")
            hits.append(PredHit(name=path.stem, conf=conf, line=line))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_dir", type=Path, help="Directory containing prediction TXT files")
    parser.add_argument("--class-id", type=int, required=True, help="Target class id")
    parser.add_argument("--top", type=int, default=20, help="How many entries to print")
    args = parser.parse_args()

    if not args.labels_dir.exists():
        raise FileNotFoundError(args.labels_dir)

    hits: list[PredHit] = []
    for txt in sorted(args.labels_dir.glob("*.txt")):
        hits.extend(parse_prediction_file(txt, args.class_id))

    if not hits:
        print(f"No predictions found for class {args.class_id} in {args.labels_dir}")
        return

    hits.sort(key=lambda h: h.conf, reverse=True)
    print(f"Found {len(hits)} predictions for class {args.class_id}")
    for hit in hits[: args.top]:
        print(f"  - {hit.name}: conf={hit.conf:.3f} | {hit.line}")


if __name__ == "__main__":
    main()
