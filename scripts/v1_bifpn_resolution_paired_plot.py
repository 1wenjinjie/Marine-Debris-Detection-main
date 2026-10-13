from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs/paper/v1_bifpn_resolution_seed_metrics.csv"
AGGREGATE_512 = ROOT / "docs/paper/v1_ablation_aggregate.json"
AGGREGATE_1024 = ROOT / "docs/paper/v1_imgsz1024_ablation_aggregate.json"
OUT_BASENAME = ROOT / "docs/paper/figures/bifpn_resolution_paired_seeds"

SKILL_SCRIPTS = Path(
    os.environ.get(
        "SCIPILOT_FIGURE_SKILL_SCRIPTS",
        Path.home() / ".codex/skills/scipilot-figure-skill/scripts",
    )
)
if not SKILL_SCRIPTS.exists():
    raise RuntimeError(
        "Set SCIPILOT_FIGURE_SKILL_SCRIPTS to the scipilot-figure-skill scripts directory."
    )
sys.path.insert(0, str(SKILL_SCRIPTS))

from export_figure import export_figure
from layout_tools import add_panel_labels, finalize_figure
from setup_style import setup_style
from visual_qa import audit_layout, print_report, render_preview

METRICS = [
    ("Marine_Debris_AP50", "Marine debris AP@0.5"),
    ("mAP50", "mAP@0.5 (all classes)"),
    ("f1", "Best F1"),
]
RESOLUTIONS = [512, 1024]
MODELS = ["Baseline", "BiFPN"]
SEEDS = [0, 1, 2]
COLORS = {"Baseline": "#0072B2", "BiFPN": "#E69F00"}
MARKERS = {"Baseline": "o", "BiFPN": "s"}
X_POSITIONS = {
    (512, "Baseline"): 0.0,
    (512, "BiFPN"): 1.0,
    (1024, "Baseline"): 2.55,
    (1024, "BiFPN"): 3.55,
}
SEED_JITTER = {0: -0.055, 1: 0.0, 2: 0.055}


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    expected = pd.MultiIndex.from_product(
        [RESOLUTIONS, MODELS, SEEDS], names=["resolution", "model", "seed"]
    )
    actual = pd.MultiIndex.from_frame(data[["resolution", "model", "seed"]])
    if len(data) != 12 or set(actual) != set(expected):
        raise ValueError("Expected exactly 12 rows: 2 resolutions x 2 models x 3 seeds.")
    if data[list(dict(METRICS))].isna().any().any():
        raise ValueError("Metric table contains missing values.")
    verify_against_aggregates(data)
    return data


def verify_against_aggregates(data: pd.DataFrame) -> None:
    aggregate_512 = json.loads(AGGREGATE_512.read_text(encoding="utf-8"))
    aggregate_1024 = json.loads(AGGREGATE_1024.read_text(encoding="utf-8"))
    sources = {
        (512, "Baseline"): aggregate_512["honest_baseline"],
        (512, "BiFPN"): aggregate_512["honest_bifpn"],
        (1024, "Baseline"): aggregate_1024["baseline"],
        (1024, "BiFPN"): aggregate_1024["bifpn"],
    }
    for resolution in RESOLUTIONS:
        for model in MODELS:
            subset = data[
                (data["resolution"] == resolution) & (data["model"] == model)
            ].sort_values("seed")
            source = sources[(resolution, model)]
            if subset["seed"].tolist() != source["seeds"]:
                raise ValueError(f"Seed mismatch for {resolution}/{model}.")
            for metric, _ in METRICS:
                if not np.allclose(subset[metric].to_numpy(), source[metric]["all"]):
                    raise ValueError(f"Metric mismatch for {resolution}/{model}/{metric}.")


def paired_summary(data: pd.DataFrame, resolution: int, metric: str) -> dict:
    subset = data[data["resolution"] == resolution]
    baseline = (
        subset[subset["model"] == "Baseline"].sort_values("seed")[metric].to_numpy()
    )
    bifpn = subset[subset["model"] == "BiFPN"].sort_values("seed")[metric].to_numpy()
    differences = bifpn - baseline
    mean_difference = float(np.mean(differences))
    sem = float(stats.sem(differences))
    critical = float(stats.t.ppf(0.975, df=len(differences) - 1))
    ci_low = mean_difference - critical * sem
    ci_high = mean_difference + critical * sem
    test = stats.ttest_rel(bifpn, baseline)
    return {
        "delta": mean_difference,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p": float(test.pvalue),
    }


def resolution_summary(data: pd.DataFrame, model: str, metric: str) -> dict:
    subset = data[data["model"] == model]
    values_512 = (
        subset[subset["resolution"] == 512].sort_values("seed")[metric].to_numpy()
    )
    values_1024 = (
        subset[subset["resolution"] == 1024].sort_values("seed")[metric].to_numpy()
    )
    differences = values_1024 - values_512
    mean_difference = float(np.mean(differences))
    sem = float(stats.sem(differences))
    critical = float(stats.t.ppf(0.975, df=len(differences) - 1))
    ci_low = mean_difference - critical * sem
    ci_high = mean_difference + critical * sem
    test = stats.ttest_rel(values_1024, values_512)
    return {
        "model": model,
        "metric": metric,
        "delta": mean_difference,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p": float(test.pvalue),
    }


def format_p(value: float) -> str:
    return f"{value:.3f}".lstrip("0") if value >= 0.001 else "<.001"


def draw_metric(ax, data: pd.DataFrame, metric: str, title: str) -> list[dict]:
    summaries = []
    for resolution in RESOLUTIONS:
        baseline = data[
            (data["resolution"] == resolution) & (data["model"] == "Baseline")
        ].sort_values("seed")
        bifpn = data[
            (data["resolution"] == resolution) & (data["model"] == "BiFPN")
        ].sort_values("seed")
        for seed in SEEDS:
            baseline_value = float(baseline.loc[baseline["seed"] == seed, metric].iloc[0])
            bifpn_value = float(bifpn.loc[bifpn["seed"] == seed, metric].iloc[0])
            x_baseline = X_POSITIONS[(resolution, "Baseline")] + SEED_JITTER[seed]
            x_bifpn = X_POSITIONS[(resolution, "BiFPN")] + SEED_JITTER[seed]
            ax.plot(
                [x_baseline, x_bifpn],
                [baseline_value, bifpn_value],
                color="#7A7A7A",
                linewidth=0.8,
                alpha=0.78,
                zorder=1,
            )
            for model, value, x_value in (
                ("Baseline", baseline_value, x_baseline),
                ("BiFPN", bifpn_value, x_bifpn),
            ):
                ax.scatter(
                    x_value,
                    value,
                    s=29,
                    marker=MARKERS[model],
                    facecolor=COLORS[model],
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=3,
                )
        for model in MODELS:
            values = data[
                (data["resolution"] == resolution) & (data["model"] == model)
            ][metric].to_numpy()
            x_value = X_POSITIONS[(resolution, model)]
            ax.plot(
                [x_value - 0.15, x_value + 0.15],
                [float(np.mean(values)), float(np.mean(values))],
                color="black",
                linewidth=1.35,
                solid_capstyle="round",
                zorder=4,
            )
        summary = paired_summary(data, resolution, metric)
        summaries.append({"resolution": resolution, "metric": metric, **summary})
        x_left = X_POSITIONS[(resolution, "Baseline")]
        x_right = X_POSITIONS[(resolution, "BiFPN")]
        ax.text(
            (x_left + x_right) / 2,
            0.105,
            (
                f"{resolution}: Δ {summary['delta']:+.3f}, p {format_p(summary['p'])}\n"
                f"CI [{summary['ci_low']:+.3f}, {summary['ci_high']:+.3f}]"
            ),
            ha="center",
            va="center",
            fontsize=5.5,
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "#C9C9C9",
                "linewidth": 0.45,
                "alpha": 0.96,
            },
        )
    ax.axvline(1.775, color="#D0D0D0", linewidth=0.7, linestyle=(0, (2, 2)))
    ax.set_xlim(-0.48, 4.03)
    ax.set_ylim(0, 0.72)
    ax.set_xticks(
        [X_POSITIONS[(resolution, model)] for resolution in RESOLUTIONS for model in MODELS],
        ["Baseline\n512", "BiFPN\n512", "Baseline\n1024", "BiFPN\n1024"],
    )
    ax.set_ylabel(title)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, linestyle=(0, (2, 2)))
    ax.set_axisbelow(True)
    return summaries


def make_figure(data: pd.DataFrame):
    setup_style(journal="nature", lang="en", use_sciplots=False)
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    figure = plt.figure(figsize=(7.2, 3.55))
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[15, 2.35],
        left=0.075,
        right=0.985,
        bottom=0.055,
        top=0.975,
        wspace=0.37,
        hspace=0.20,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    all_summaries = []
    for axis, (metric, title) in zip(axes, METRICS):
        all_summaries.extend(draw_metric(axis, data, metric, title))
    resolution_summaries = [
        resolution_summary(data, model, metric)
        for metric, _ in METRICS
        for model in MODELS
    ]
    p_min = min(summary["p"] for summary in resolution_summaries)
    p_max = max(summary["p"] for summary in resolution_summaries)
    note_axis = figure.add_subplot(grid[1, :])
    note_axis.axis("off")
    note_axis.text(
        0.5,
        0.5,
        (
            "Points are individual seeds (n=3); gray lines pair identical seeds; black bars mark means.\n"
            "Annotated: Δ = BiFPN - Baseline. Resolution contrasts (1024 - 512) are also non-significant "
            f"(p {format_p(p_min)} to {format_p(p_max)}); all 95% CIs include 0.\n"
            "Two-sided paired t-tests (df=2), unadjusted. ns does not establish equivalence."
        ),
        ha="center",
        va="center",
        fontsize=5.9,
        linespacing=1.25,
    )
    finalize_figure(figure, prefer="tight")
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.055,
        top=0.975,
        wspace=0.37,
        hspace=0.20,
    )
    add_panel_labels(
        figure,
        axes=axes,
        style="nature",
        fontsize=8,
        x_offset_pt=2,
        y_offset_pt=-2,
        ha="left",
        va="top",
    )
    return figure, all_summaries, resolution_summaries


def make_grayscale_preview(png_path: Path) -> Path:
    output = png_path.with_name(png_path.stem + "_grayscale.png")
    with Image.open(png_path) as image:
        image.convert("L").save(output, dpi=image.info.get("dpi", (600, 600)))
    return output


def main() -> int:
    data = load_data()
    figure, summaries, resolution_summaries = make_figure(data)
    preview_path = render_preview(
        figure, str(OUT_BASENAME.with_name(OUT_BASENAME.name + "_preview.png")), dpi=150
    )
    verdict = print_report(audit_layout(figure))
    if verdict == "FAIL":
        raise RuntimeError("Visual QA found a blocking issue.")
    exported = export_figure(
        figure,
        str(OUT_BASENAME),
        formats=["pdf", "svg", "png"],
        size_inches=(7.2, 3.55),
        dpi=600,
        grayscale_preview=False,
        tight=False,
    )
    grayscale_path = make_grayscale_preview(OUT_BASENAME.with_suffix(".png"))
    plt.close(figure)
    print("\nPaired comparisons (BiFPN - Baseline):")
    for summary in summaries:
        print(
            f"  {summary['metric']} @ {summary['resolution']}: "
            f"delta={summary['delta']:+.6f}, "
            f"95% CI [{summary['ci_low']:+.6f}, {summary['ci_high']:+.6f}], "
            f"p={summary['p']:.6f}"
        )
    print("\nPaired resolution comparisons (1024 - 512):")
    for summary in resolution_summaries:
        print(
            f"  {summary['metric']} / {summary['model']}: "
            f"delta={summary['delta']:+.6f}, "
            f"95% CI [{summary['ci_low']:+.6f}, {summary['ci_high']:+.6f}], "
            f"p={summary['p']:.6f}"
        )
    print(f"preview={preview_path}")
    for path in exported:
        print(path)
    print(grayscale_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
