from __future__ import annotations

import csv
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "CMU Sans Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 7,
            "text.usetex": False,
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "CMU Sans Serif",
            "mathtext.it": "CMU Sans Serif:italic",
            "mathtext.bf": "CMU Sans Serif:bold",
        }
    )


HERE = Path(__file__).resolve().parent
JUSTIN_ROOT = HERE.parent.parent / "only_plotting_06.26_justin"
LOGIT_ROOT = JUSTIN_ROOT / "logit_comparison"
ACCURACY_ROOT = JUSTIN_ROOT / "test_accuracy_comparison" / "trained_centroids"

DATASETS = [
    ("mnist", "MNIST"),
    ("fashion_mnist", "Fashion-MNIST"),
    ("kmnist", "KMNIST"),
]
NOISE_TYPES = ["replacement", "additive_gaussian"]

FIT_P = 0.97
P_MIN = 0.90
N = 4000
D_LABEL = "28x28"
LOSS = "mse"
ACTIVATION = "erf"
DEPTH = 3
WIDTH = 2048
MAX_INSET_POINTS = 100_000

RED = "#b31b1b"
THEORY_GRAY = "#404040"
INSET_ORANGE = "#E0A01E"


configure_plot_style()

for noise_type in NOISE_TYPES:
    fig, axes = plt.subplots(
        1, 3,
        figsize=(3 * 3.375, 2.4),
        dpi=300,
        constrained_layout=True,
        sharey=False,
    )

    for ax, (dataset, dataset_label), panel_label in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        accuracy_csv = ACCURACY_ROOT / (
            f"test_accuracy_summary__dataset={dataset}"
            f"__eval={noise_type}"
            f"__fit={noise_type}_p={FIT_P:.2f}"
            f"__N={N}__d={D_LABEL}"
            f"__loss={LOSS}__act={ACTIVATION}"
            f"__L={DEPTH}__width={WIDTH}.csv"
        )

        rows = []
        with accuracy_csv.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["noise_type"] != noise_type:
                    continue
                if row.get("dataset", dataset) != dataset:
                    continue
                if float(row["noise_probability"]) < P_MIN:
                    continue
                rows.append(row)
        rows = sorted(rows, key=lambda r: float(r["noise_probability"]))

        p_vals = [float(r["noise_probability"]) for r in rows]
        neural_mean = [100.0 * float(r["empirical_mean_test_accuracy"]) for r in rows]
        neural_std = [100.0 * float(r["empirical_std_test_accuracy"]) for r in rows]
        theory_mean = [100.0 * float(r["mean_test_accuracy"]) for r in rows]

        p09_accuracy = neural_mean[min(range(len(p_vals)), key=lambda i: abs(p_vals[i] - 0.90))]
        # y_headroom = 10.0 if dataset == "mnist" else 5.0
        # y_headroom = 5.0 if dataset == "kmnist" else (10.0 if dataset == "mnist" else 15.0)
        y_headroom = 8.0 if dataset == "kmnist" else 15.0
        y_top = min(100.0, p09_accuracy + y_headroom)

        ax.errorbar(
            p_vals,
            neural_mean,
            yerr=neural_std,
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            capsize=2.5,
            color=RED,
            label="Neural network",
            zorder=3,
        )
        ax.plot(
            p_vals,
            theory_mean,
            linestyle="--",
            linewidth=1.2,
            color=THEORY_GRAY,
            label="Theoretical prediction [Eq. (16)]",
            zorder=2,
        )

        xlabel = (
            r"Corruption probability $p$"
            if noise_type == "replacement"
            else r"Corruption strength $p$"
        )
        ax.set_xlabel(xlabel)
        ax.set_xlim(P_MIN - 0.004, 1.004)
        ax.set_ylim(0, y_top)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.90, 0.925, 0.95, 0.975, 1.0]))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

        if ax is axes[0]:
            ax.set_ylabel("Test accuracy (%)")
        handles, labels = ax.get_legend_handles_labels()
        order = ["Neural network", "Theoretical prediction [Eq. (16)]"]
        pairs = [(h, l) for l in order for h, lbl in zip(handles, labels) if lbl == l]
        ax.legend(*zip(*pairs), frameon=True, handlelength=1.8, loc="upper right")
        ax.text(
            -0.18, 1.02, panel_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9, clip_on=False,
        )

        ax.text(
            0.56, 0.2,
            dataset_label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
        )

        logit_folders = sorted(
            LOGIT_ROOT.glob(
                f"mlp_ensemble__dataset={dataset}__{noise_type}__p={FIT_P:g}"
                f"__N={N}__d={D_LABEL}__loss={LOSS}__act={ACTIVATION}"
                f"__L={DEPTH}__width={WIDTH}__*"
            )
        )
        if not logit_folders:
            raise FileNotFoundError(
                f"No logit folder found for dataset={dataset}, noise_type={noise_type}"
            )
        logit_csv = logit_folders[0] / "centroid_comparison_test10000.csv.gz"

        xs, ys = [], []
        with gzip.open(logit_csv, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                xs.append(float(row.get("delta_fit_logit", row["centroid_fit_logit"])))
                ys.append(float(row["ensemble_mean_logit"]))
                if len(xs) >= MAX_INSET_POINTS:
                    break

        xs_arr = np.array(xs, dtype=float)
        ys_arr = np.array(ys, dtype=float)
        ss_res = float(np.sum((ys_arr - xs_arr) ** 2))
        ss_tot = float(np.sum((ys_arr - ys_arr.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot

        lo = float(min(xs_arr.min(), ys_arr.min()))
        hi = float(max(xs_arr.max(), ys_arr.max()))
        pad = 0.04 * (hi - lo)
        lo -= pad
        hi += pad

        inset = ax.inset_axes([0.15, 0.15, 0.4, 0.5])
        inset.scatter(
            xs_arr, ys_arr,
            s=4, color=INSET_ORANGE, alpha=0.18,
            edgecolors="none", rasterized=True,
        )
        inset.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
        inset.set_xlim(lo, hi)
        inset.set_ylim(lo, hi)
        inset.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        inset.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        inset.tick_params(labelsize=7, pad=1)
        inset.set_xlabel("Predicted outputs", fontsize=8, labelpad=1)
        inset.set_ylabel("Actual outputs", fontsize=8, labelpad=1)
        inset.text(
            0.05, 0.95,
            rf"$R^2 = {r2:.3f}$",
            transform=inset.transAxes,
            fontsize=8,
            va="top",
        )
        inset.text(
            0.5, 0.10,
            rf"$(p = {FIT_P:g})$",
            transform=inset.transAxes,
            fontsize=8,
            va="bottom",
        )

    output_path = HERE / f"logit_and_test_accuracy_comparison__noise={noise_type}.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {output_path}")
