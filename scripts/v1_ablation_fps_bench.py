"""FPS benchmark for the v1 7ch ablation matrix (honest + leaky settings).

Forward-pass timing only (no dataset loading, no NMS). Measures median FPS
over 100 iterations after 20 warmup forwards. Uses a synthetic
(1, 7, 512, 512) float32 input.

Output: <run_dir>/fps_bench.json per checkpoint, plus a combined CSV row.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ultralytics_plugins  # noqa: F401  -> registers CBAM
from ultralytics_plugins.bifpn_modules import install_bifpn_patch
install_bifpn_patch()

from scripts.v2_fps_benchmark import benchmark_one

ABLATIONS = [
    ("honest_baseline", "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_clean_init/weights/best.pt"),
    ("honest_cbam",     "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_cbam_clean_init/weights/best.pt"),
    ("honest_bifpn",    "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_bifpn_clean_init/weights/best.pt"),
    ("honest_eiou",     "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_eiou_clean_init/weights/best.pt"),
    ("honest_full",     "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_full_clean_init/weights/best.pt"),
    ("leaky_baseline",  "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_baseline_clean_init/weights/best.pt"),
    ("leaky_cbam",      "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_cbam_clean_init/weights/best.pt"),
    ("leaky_bifpn",     "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_bifpn_clean_init/weights/best.pt"),
    ("leaky_eiou",      "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_eiou_clean_init/weights/best.pt"),
    ("leaky_full",      "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_full_clean_init/weights/best.pt"),
]


def main():
    out_csv = ROOT / "docs/paper/v1_ablation_fps.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'tag':22s}  {'FPS':>8s}  {'lat_ms_med':>11s}  {'lat_ms_std':>11s}")
    print("=" * 60)
    for tag, weights_str in ABLATIONS:
        wp = Path(weights_str)
        if not wp.exists():
            print(f"  {tag:22s}  MISSING: {wp}")
            continue
        info = benchmark_one(wp, imgsz=512, channels=7, warmup=20, iters=100, device="0")
        info["tag"] = tag
        rows.append(info)
        bench_path = wp.parent.parent / "fps_bench.json"
        bench_path.write_text(json.dumps(info, indent=2))
        print(f"  {tag:22s}  {info['fps_median']:>8.1f}  {info['latency_ms_median']:>11.3f}  {info['latency_ms_std']:>11.3f}")

    # Combined CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "fps_median", "latency_ms_median",
                                          "latency_ms_min", "latency_ms_max", "latency_ms_std",
                                          "imgsz", "channels", "device", "warmup", "iters"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"\nsaved CSV: {out_csv}")


if __name__ == "__main__":
    main()
