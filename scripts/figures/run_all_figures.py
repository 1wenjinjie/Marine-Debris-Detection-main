#!/usr/bin/env python3
"""生成 MPB 正文图、补图和图文摘要的 Arial 版。

用法:
    python run_all_figures.py [--out-dir ../MPB_Arial] [--png-dpi 600]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

# Ensure Windows Anaconda Library\bin is in PATH for MKL DLLs, and provide safe MPLCONFIGDIR
_py_dir = Path(sys.executable).parent
_lib_bin = _py_dir / "Library" / "bin"
if _lib_bin.is_dir() and str(_lib_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_lib_bin) + os.pathsep + os.environ.get("PATH", "")
if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path(__file__).resolve().parent / ".mpl"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="一键运行全部 MPB 论文配图脚本")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="输出目录（默认: 脚本上一层的 MPB_Arial 目录）",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=600,
        help="同步导出高分辨率 PNG 的 DPI（设为 0 则仅输出矢量 PDF 与图文摘要标准 PNG，默认: 600）",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root.parents[1] / "figures_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = root / "assets" / "mpb_real_imagery_bundle.npz"
    if not bundle.is_file() and (root / "extract_real_assets.py").is_file():
        subprocess.run([sys.executable, str(root / "extract_real_assets.py")], env=os.environ, check=True)

    dpi_args = ["--png-dpi", str(args.png_dpi)]

    jobs = [
        [sys.executable, str(root / "plot_mpb_fig1_streamlined.py"), "--out-dir", str(out_dir), *dpi_args],
        [sys.executable, str(root / "plot_mpb_fig2_fig3.py"), "--out-dir", str(out_dir), *dpi_args],
        [sys.executable, str(root / "plot_mpb_fig1_fig4.py"), "--figure", "4", "--out-dir", str(out_dir), *dpi_args],
        [
            sys.executable,
            str(root / "plot_mpb_fig5_and_ga.py"),
            "--out-dir",
            str(out_dir),
            "--export-ga-png",
            *dpi_args,
        ],
        [sys.executable, str(root / "plot_mpb_fig6.py"), "--out-dir", str(out_dir), *dpi_args],
        [
            sys.executable,
            str(root / "plot_mpb_supplementary.py"),
            "--out-dir",
            str(out_dir / "Supplementary_MPB"),
            *dpi_args,
        ],
    ]

    for cmd in jobs:
        print(">>> Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, env=os.environ, check=True)

    print(f"\n全部图件（Fig. 1–6、Fig. S1–S3 和图文摘要）已输出至 {out_dir}")


if __name__ == "__main__":
    main()
