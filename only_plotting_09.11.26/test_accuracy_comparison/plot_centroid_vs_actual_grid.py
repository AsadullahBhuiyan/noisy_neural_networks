#!/usr/bin/env python3
"""Six-panel centroid-vs-network accuracy grid: datasets x noise types.

Rows are noise types, columns are datasets. Each panel shows the trained
network accuracy against the analytical (centroid) prediction over the full
noise-probability grid, read from the test_accuracy_summary__*.csv files that
eval_noisy_centroid_and_network_accuracy.py wrote into trained_centroids/.

Unlike plot_centroid_vs_mlp_accuracy.py, which writes one figure per case and
per p grid, this plots every p present in the CSV on one set of axes.

Usage: plot_centroid_vs_actual_grid.py [--activation erf] [--depth 3]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
CENTROID_DIR = HERE / "trained_centroids"

DATASETS = [
    ("mnist", "MNIST"),
    ("fashion_mnist", "Fashion-MNIST"),
    ("kmnist", "KMNIST"),
]
NOISE_TYPES = [
    ("replacement", "Replacement"),
    ("additive_gaussian", "Additive Gaussian"),
]

FIT_P = 0.97
N = 4000
D_LABEL = "28x28"
LOSS = "mse"
WIDTH = 2048

COLUMN_WIDTH = 3.375  # Single-column width; one column per dataset panel.
PANEL_HEIGHT = 2.6

NETWORK_COLOR = "tab:red"
ANALYTICAL_COLOR = "tab:blue"


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "CMU Sans Serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "text.usetex": False,
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "CMU Sans Serif",
            "mathtext.it": "CMU Sans Serif:italic",
            "mathtext.bf": "CMU Sans Serif:bold",
        }
    )


def summary_path(dataset: str, noise_type: str, activation: str, depth: int) -> Path:
    return CENTROID_DIR / (
        f"test_accuracy_summary__dataset={dataset}"
        f"__eval={noise_type}"
        f"__fit={noise_type}_p={FIT_P:.2f}"
        f"__N={N}__d={D_LABEL}"
        f"__loss={LOSS}__act={activation}"
        f"__L={depth}__width={WIDTH}.csv"
    )


def read_case(path: Path) -> list[dict[str, str]]:
    """Every p in the CSV, sorted: deciles plus the dense grid near p = 1."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    rows.sort(key=lambda row: float(row["noise_probability"]))
    seen = [float(row["noise_probability"]) for row in rows]
    if len(set(seen)) != len(seen):
        raise ValueError(f"Duplicate noise probabilities in {path}")
    return rows


def draw_panel(ax: plt.Axes, rows: list[dict[str, str]], title: str) -> None:
    p_values = [float(row["noise_probability"]) for row in rows]
    ax.errorbar(
        p_values,
        [100 * float(row["empirical_mean_test_accuracy"]) for row in rows],
        yerr=[100 * float(row["empirical_std_test_accuracy"]) for row in rows],
        fmt="s-",
        color=NETWORK_COLOR,
        markersize=3,
        linewidth=1.4,
        elinewidth=0.8,
        label="Neural network",
        zorder=1,
    )
    ax.errorbar(
        p_values,
        [100 * float(row["mean_test_accuracy"]) for row in rows],
        yerr=[100 * float(row["std_test_accuracy"]) for row in rows],
        fmt="o--",
        color=ANALYTICAL_COLOR,
        markersize=3,
        linewidth=1.2,
        elinewidth=0.8,
        label="Analytical model",
        zorder=2,
    )
    ax.set_title(title)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.margins(x=0.02)
    ax.set_xticks([i / 10 for i in range(11)])
    # %g drops the trailing zero, so the grid reads 0, 0.1, ... 1 rather than 0.0.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.legend(frameon=True, handlelength=1.8, loc="lower left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", default="erf")
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    configure_plot_style()
    fig, axes = plt.subplots(
        len(NOISE_TYPES), len(DATASETS),
        figsize=(len(DATASETS) * COLUMN_WIDTH, len(NOISE_TYPES) * PANEL_HEIGHT),
        dpi=300,
        constrained_layout=True,
    )
    for row_axes, (noise_type, noise_label) in zip(axes, NOISE_TYPES):
        for ax, (dataset, dataset_label) in zip(row_axes, DATASETS):
            rows = read_case(summary_path(dataset, noise_type, args.activation, args.depth))
            draw_panel(ax, rows, f"{dataset_label} | {noise_label}")

    path = HERE / f"centroid_vs_actual__act={args.activation}__L={args.depth}__p0_to_1.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
