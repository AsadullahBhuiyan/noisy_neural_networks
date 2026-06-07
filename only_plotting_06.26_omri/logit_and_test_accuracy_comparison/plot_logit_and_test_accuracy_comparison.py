from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jt577")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator


# -----------------------------------------------------------------------------
# Easy-to-edit settings
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
JUSTIN_ROOT = HERE.parent.parent / "only_plotting_06.26_justin"
LOGIT_ROOT = JUSTIN_ROOT / "logit_comparison"
ACCURACY_ROOT = JUSTIN_ROOT / "test_accuracy_comparison" / "trained_centroids"

DATASETS = [
    ("mnist", "MNIST"),
    ("fashion_mnist", "FashionMNIST"),
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

RED = "#c61f26"
THEORY_GRAY = "#4a4a4a"
INSET_ORANGE = "#f2a51a"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


for noise_type in NOISE_TYPES:
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.0), sharey=False)

    for ax, (dataset, dataset_label) in zip(axes, DATASETS):
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
        rows = sorted(rows, key=lambda row: float(row["noise_probability"]))

        p_vals = [float(row["noise_probability"]) for row in rows]
        neural_mean = [100.0 * float(row["empirical_mean_test_accuracy"]) for row in rows]
        neural_std = [100.0 * float(row["empirical_std_test_accuracy"]) for row in rows]
        theory_mean = [100.0 * float(row["mean_test_accuracy"]) for row in rows]
        p09_accuracy = neural_mean[min(range(len(p_vals)), key=lambda idx: abs(p_vals[idx] - 0.90))]
        y_headroom = 10.0 if dataset == "mnist" else 5.0
        y_top = min(100.0, p09_accuracy + y_headroom)

        ax.errorbar(
            p_vals,
            neural_mean,
            yerr=neural_std,
            fmt="o-",
            color=RED,
            ecolor=RED,
            elinewidth=2.0,
            capsize=3.5,
            markersize=6.5,
            linewidth=2.6,
            label="Neural network",
            zorder=3,
        )
        ax.plot(
            p_vals,
            theory_mean,
            "--",
            color=THEORY_GRAY,
            linewidth=2.4,
            label="Prediction",
            zorder=2,
        )

        ax.set_xlabel(r"Corruption probability $p$", fontsize=16)
        ax.set_xlim(P_MIN - 0.004, 1.004)
        ax.set_ylim(0, y_top)
        ax.xaxis.set_major_locator(FixedLocator([0.90, 0.925, 0.95, 0.975, 1.0]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.9", "0.925", "0.95", "0.975", "1"]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(labelsize=15)
        ax.grid(False)
        if ax is axes[0]:
            ax.set_ylabel("Test accuracy (%)", fontsize=16)
            ax.legend(fontsize=12.5, loc="upper right", frameon=True, borderpad=0.45, handlelength=2.2)

        logit_folders = sorted(
            LOGIT_ROOT.glob(
                f"mlp_ensemble__dataset={dataset}__{noise_type}__p={FIT_P:g}"
                f"__N={N}__d={D_LABEL}__loss={LOSS}__act={ACTIVATION}"
                f"__L={DEPTH}__width={WIDTH}__*"
            )
        )
        if not logit_folders:
            raise FileNotFoundError(f"No logit folder found for dataset={dataset}, noise_type={noise_type}")
        logit_csv = logit_folders[0] / "centroid_comparison_test10000.csv.gz"

        xs = []
        ys = []
        with gzip.open(logit_csv, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                xs.append(float(row.get("delta_fit_logit", row["centroid_fit_logit"])))
                ys.append(float(row["ensemble_mean_logit"]))
                if len(xs) >= MAX_INSET_POINTS:
                    break

        y_mean = sum(ys) / len(ys)
        ss_res = sum((y - x) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        r2 = 1.0 - ss_res / ss_tot
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        pad = 0.04 * (hi - lo)
        lo -= pad
        hi += pad

        inset = ax.inset_axes([0.15, 0.15, 0.4, 0.5])
        inset.scatter(xs, ys, s=10, color=INSET_ORANGE, alpha=0.18, linewidths=0)
        inset.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.8)
        inset.set_xlim(lo, hi)
        inset.set_ylim(lo, hi)
        inset.xaxis.set_major_locator(MaxNLocator(nbins=3))
        inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
        inset.tick_params(labelsize=12, width=1.2, length=4)
        for spine in inset.spines.values():
            spine.set_linewidth(1.4)
        inset.set_xlabel("Predicted outputs", fontsize=14, labelpad=1)
        inset.set_ylabel("Actual outputs", fontsize=14, labelpad=1)
        inset.text(0.06, 0.95, rf"$R^2 = {r2:.3f}$", transform=inset.transAxes, fontsize=14, va="top")
        inset.text(0.5, 0.10, rf"$(p = {FIT_P:g})$", transform=inset.transAxes, fontsize=14, va="bottom")
        ax.text(
            0.57,
            0.2,
            dataset_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="center",
        )

    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.18, top=0.94, wspace=0.12)
    output_pdf = HERE / f"logit_and_test_accuracy_comparison__noise={noise_type}.pdf"
    output_png = HERE / f"logit_and_test_accuracy_comparison__noise={noise_type}.png"
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_png}")
