#!/usr/bin/env python3
"""Six-panel accuracy-and-logit comparison: noise types x datasets.

Rows are noise types, columns are datasets. Main panels show trained-network
accuracy against the fitted centroid-based prediction of Eq. (16) over the
high-corruption range; each inset shows ensemble-mean output logits against the
fitted predictions at the fit probability, with R^2.

Styling follows plot_logit_and_test_accuracy_comparison.py. Accuracy comes from
test_accuracy_comparison/trained_centroids/, logits from logit_comparison/.

Usage: plot_accuracy_and_logit_grid.py [--activation relu] [--depth 5]
"""
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HERE = Path(__file__).resolve().parent
ACCURACY_ROOT = HERE / "test_accuracy_comparison" / "trained_centroids"
LOGIT_ROOT = HERE / "logit_comparison"

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
P_MIN = 0.90
N = 4000
D_LABEL = "28x28"
LOSS = "mse"
WIDTH = 2048
MAX_INSET_POINTS = 100_000

COLUMN_WIDTH = 3.375  # Single-column width; one column per dataset panel.
PANEL_HEIGHT = 2.7

RED = "#b31b1b"
THEORY_GRAY = "#404040"
INSET_ORANGE = "#E0A01E"

NETWORK_LABEL = "Neural network"
THEORY_LABEL = "Theoretical prediction"


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "CMU Sans Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8,
            "text.usetex": False,
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "CMU Sans Serif",
            "mathtext.it": "CMU Sans Serif:italic",
            "mathtext.bf": "CMU Sans Serif:bold",
        }
    )


def read_accuracy(dataset: str, noise_type: str, activation: str, depth: int) -> list[dict[str, str]]:
    path = ACCURACY_ROOT / (
        f"test_accuracy_summary__dataset={dataset}"
        f"__eval={noise_type}"
        f"__fit={noise_type}_p={FIT_P:.2f}"
        f"__N={N}__d={D_LABEL}"
        f"__loss={LOSS}__act={activation}"
        f"__L={depth}__width={WIDTH}.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["noise_type"] == noise_type
            and row.get("dataset", dataset) == dataset
            and float(row["noise_probability"]) >= P_MIN
        ]
    if not rows:
        raise ValueError(f"No rows at p >= {P_MIN:g} in {path}")
    return sorted(rows, key=lambda row: float(row["noise_probability"]))


def read_logits(dataset: str, noise_type: str, activation: str,
                depth: int) -> tuple[np.ndarray, np.ndarray]:
    folders = sorted(LOGIT_ROOT.glob(
        f"mlp_ensemble__dataset={dataset}__{noise_type}__p={FIT_P:g}"
        f"__N={N}__d={D_LABEL}__loss={LOSS}__act={activation}"
        f"__L={depth}__width={WIDTH}__*"
    ))
    if not folders:
        raise FileNotFoundError(
            f"No logit folder for dataset={dataset}, noise={noise_type}, "
            f"act={activation}, L={depth}"
        )
    predicted, actual = [], []
    with gzip.open(folders[0] / "centroid_comparison_test10000.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            predicted.append(float(row.get("delta_fit_logit", row["centroid_fit_logit"])))
            actual.append(float(row["ensemble_mean_logit"]))
            if len(predicted) >= MAX_INSET_POINTS:
                break
    return np.array(predicted, dtype=float), np.array(actual, dtype=float)


def draw_inset(ax: plt.Axes, predicted: np.ndarray, actual: np.ndarray) -> None:
    # Residuals are taken about the diagonal, not a refitted line: the inset
    # asks whether prediction equals output, not whether they correlate.
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    lo = float(min(predicted.min(), actual.min()))
    hi = float(max(predicted.max(), actual.max()))
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    inset = ax.inset_axes([0.15, 0.15, 0.4, 0.5])
    inset.scatter(
        predicted, actual,
        s=4, color=INSET_ORANGE, alpha=0.18,
        edgecolors="none", rasterized=True,
    )
    inset.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
    inset.set_xlim(lo, hi)
    inset.set_ylim(lo, hi)
    inset.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    inset.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    inset.tick_params(labelsize=8, pad=1)
    inset.set_xlabel("Predicted outputs", fontsize=8, labelpad=1)
    inset.set_ylabel("Actual outputs", fontsize=8, labelpad=1)
    inset.text(0.05, 0.95, rf"$R^2 = {r2:.3f}$", transform=inset.transAxes,
               fontsize=8, va="top")
    inset.text(0.5, 0.10, rf"$(p = {FIT_P:g})$", transform=inset.transAxes,
               fontsize=8, va="bottom")


def draw_panel(ax: plt.Axes, rows: list[dict[str, str]], dataset: str,
               noise_type: str, title: str, show_ylabel: bool) -> None:
    p_values = [float(row["noise_probability"]) for row in rows]
    network_mean = [100.0 * float(row["empirical_mean_test_accuracy"]) for row in rows]
    network_std = [100.0 * float(row["empirical_std_test_accuracy"]) for row in rows]
    theory_mean = [100.0 * float(row["mean_test_accuracy"]) for row in rows]

    ax.errorbar(
        p_values, network_mean, yerr=network_std,
        marker="o", markersize=3.5, linewidth=1.2, capsize=2.5,
        color=RED, label=NETWORK_LABEL, zorder=3,
    )
    ax.plot(
        p_values, theory_mean,
        linestyle="--", linewidth=1.2, color=THEORY_GRAY,
        label=THEORY_LABEL, zorder=2,
    )

    # Leave room above the p = 0.90 point for the legend without clipping data.
    p09_accuracy = network_mean[min(range(len(p_values)),
                                    key=lambda i: abs(p_values[i] - P_MIN))]
    headroom = 8.0 if dataset == "kmnist" else 15.0
    ax.set_ylim(0, min(100.0, p09_accuracy + headroom))
    ax.set_xlim(P_MIN - 0.004, 1.004)
    ax.set_xlabel(
        r"Corruption probability $p$" if noise_type == "replacement"
        else r"Corruption strength $p$"
    )
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.90, 0.925, 0.95, 0.975, 1.0]))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    if show_ylabel:
        ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    pairs = [(h, l) for want in (NETWORK_LABEL, THEORY_LABEL)
             for h, l in zip(handles, labels) if l == want]
    ax.legend(*zip(*pairs), frameon=True, handlelength=1.8, loc="upper right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()

    configure_plot_style()
    fig, axes = plt.subplots(
        len(NOISE_TYPES), len(DATASETS),
        figsize=(len(DATASETS) * COLUMN_WIDTH, len(NOISE_TYPES) * PANEL_HEIGHT),
        dpi=300,
        constrained_layout=True,
    )
    for row_axes, (noise_type, noise_label) in zip(axes, NOISE_TYPES):
        for column, (ax, (dataset, dataset_label)) in enumerate(zip(row_axes, DATASETS)):
            rows = read_accuracy(dataset, noise_type, args.activation, args.depth)
            draw_panel(ax, rows, dataset, noise_type,
                       f"{dataset_label} | {noise_label}", show_ylabel=column == 0)
            draw_inset(ax, *read_logits(dataset, noise_type, args.activation, args.depth))

    path = HERE / f"accuracy_and_logit_grid__act={args.activation}__L={args.depth}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
