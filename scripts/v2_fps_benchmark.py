"""
FPS benchmark for trained MARIDA v2 detectors.

Reports median pure-inference FPS over N forward passes after warmup.
Loads weights via the standard ultralytics API but bypasses dataset
loading - we synthesize a 6-channel float32 tensor of the configured
imgsz and time forward only.

Usage:
  # benchmark a single run
  python scripts/v2_fps_benchmark.py --weights runs_v2/cbam_v2_s0/weights/best.pt --imgsz 512

  # benchmark all structure-ablation runs at imgsz=512
  python scripts/v2_fps_benchmark.py --batch-runs structure_ablation

  # benchmark all engineering-comparison runs (imgsz follows each run)
  python scripts/v2_fps_benchmark.py --batch-runs engineering
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import median, stdev

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Triggers ultralytics_plugins.__init__.py side-effects (CBAM/BiFPN registration)
import ultralytics_plugins  # noqa: F401
from ultralytics import YOLO

RUNS_DIR = ROOT / "runs_v2"

STRUCTURE_RUNS = [
    ("baseline_6ch_s0", 512),
    ("cbam_v2_s0", 512),
    ("bifpn_v2_s0", 512),
    ("eiou_v2_s0", 512),
    ("full_v2_s0", 512),
]

ENGINEERING_RUNS = [
    ("baseline_6ch_s0", 512),
    ("imgsz1024_s0", 1024),
    ("focal_g2.0_i1024_s0", 1024),
]


def benchmark_one(
    weights: Path,
    imgsz: int,
    channels: int = 6,
    warmup: int = 20,
    iters: int = 100,
    device: str = "0",
) -> dict:
    """Run pure forward-pass timing on a single weights file."""
    if not weights.exists():
        return {"weights": str(weights), "error": "weights not found"}

    yolo = YOLO(str(weights))
    model = yolo.model
    dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device != "cpu" else "cpu")
    model = model.to(dev).eval()

    # Synthesise a single batch of float32 tensors matching the training input
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((1, channels, imgsz, imgsz)).astype(np.float32)
    x = torch.from_numpy(x_np).to(dev)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()

    # Timed iterations
    per_iter = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            per_iter.append(time.perf_counter() - t0)

    median_s = median(per_iter)
    return {
        "weights": str(weights),
        "imgsz": imgsz,
        "channels": channels,
        "device": str(dev),
        "warmup": warmup,
        "iters": iters,
        "latency_ms_median": float(median_s * 1000),
        "latency_ms_min": float(min(per_iter) * 1000),
        "latency_ms_max": float(max(per_iter) * 1000),
        "latency_ms_std": float(stdev(per_iter) * 1000) if len(per_iter) > 1 else 0.0,
        "fps_median": float(1.0 / median_s),
    }


def benchmark_run_dir(run_name: str, imgsz: int, **kw) -> dict:
    weights = RUNS_DIR / run_name / "weights" / "best.pt"
    out = benchmark_one(weights, imgsz=imgsz, **kw)
    out["run_name"] = run_name
    bench_path = RUNS_DIR / run_name / "fps_bench.json"
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.write_text(json.dumps(out, indent=2))
    return out


def print_row(name: str, info: dict) -> None:
    if "error" in info:
        print(f"  {name:35s}  ERROR: {info['error']}")
        return
    print(
        f"  {name:35s}  imgsz={info['imgsz']:>4}  "
        f"latency={info['latency_ms_median']:6.2f}ms  fps={info['fps_median']:6.1f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="MARIDA v2 FPS benchmark")
    ap.add_argument("--weights", default="", help="Path to a single best.pt")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch-runs", default="", choices=["", "structure_ablation", "engineering"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if args.weights:
        info = benchmark_one(
            Path(args.weights), imgsz=args.imgsz,
            warmup=args.warmup, iters=args.iters, device=args.device,
        )
        print(json.dumps(info, indent=2))
        return 0

    if args.batch_runs == "structure_ablation":
        runs = STRUCTURE_RUNS
    elif args.batch_runs == "engineering":
        runs = ENGINEERING_RUNS
    else:
        ap.error("provide either --weights or --batch-runs")

    print(f"Benchmarking {len(runs)} runs (warmup={args.warmup}, iters={args.iters})...\n")
    results = {}
    for run_name, imgsz in runs:
        info = benchmark_run_dir(
            run_name, imgsz,
            warmup=args.warmup, iters=args.iters, device=args.device,
        )
        results[run_name] = info
        print_row(run_name, info)

    summary_path = RUNS_DIR / f"fps_bench_summary_{args.batch_runs}.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
