"""FPS benchmark at imgsz=1024 for the new ablation cells.

Same protocol as scripts/v1_ablation_fps_bench.py: median FPS over 100 iters
after 20 warmup. Synthetic (1, 7, 1024, 1024) float32 input.

Cells (s=0 representative — FPS is architecture-specific, not seed-specific):
- 5 honest cells (baseline/cbam/bifpn/eiou/full @ imgsz=1024)
- 2 leaky cells (baseline/bifpn @ imgsz=1024)

Output: <run_dir>/fps_bench.json per ckpt + combined CSV.
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

CELLS = [
    ("honest_baseline_1024", "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_imgsz1024_clean_init_s0/weights/best.pt"),
    ("honest_cbam_1024",     "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_cbam_imgsz1024_clean_init_s0/weights/best.pt"),
    ("honest_bifpn_1024",    "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_bifpn_imgsz1024_clean_init_s0/weights/best.pt"),
    ("honest_eiou_1024",     "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_eiou_imgsz1024_clean_init_s0/weights/best.pt"),
    ("honest_full_1024",     "runs/detect/runs_v1_v2split/yolov11n_7ch_v2split_physidx_full_imgsz1024_clean_init_s0/weights/best.pt"),
    ("leaky_baseline_1024",  "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_baseline_imgsz1024_clean_init_s0/weights/best.pt"),
    ("leaky_bifpn_1024",     "runs/detect/runs_v1_leaky/yolov11n_7ch_leaky_bifpn_imgsz1024_clean_init_s0/weights/best.pt"),
]


def main():
    out_csv = ROOT / "docs/paper/v1_imgsz1024_fps.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'tag':28s}  {'FPS':>8s}  {'lat_ms_med':>11s}  {'lat_ms_std':>11s}")
    print("=" * 70)
    for tag, weights_str in CELLS:
        wp = Path(weights_str)
        if not wp.exists():
            print(f"  {tag:28s}  MISSING: {wp}")
            continue
        info = benchmark_one(wp, imgsz=1024, channels=7, warmup=20, iters=100, device="0")
        info["tag"] = tag
        rows.append(info)
        bench_path = wp.parent.parent / "fps_bench.json"
        bench_path.write_text(json.dumps(info, indent=2))
        print(f"  {tag:28s}  {info['fps_median']:>8.1f}  {info['latency_ms_median']:>11.3f}  {info['latency_ms_std']:>11.3f}")

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
