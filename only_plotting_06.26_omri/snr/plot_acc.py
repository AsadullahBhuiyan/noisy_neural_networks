from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "CMU Sans Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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
JUSTIN_SNR_ROOT = HERE.parent.parent / "only_plotting_06.26_justin" / "snr"
CSV_PATH = JUSTIN_SNR_ROOT / "empirical_mean_model_accuracy_summary.csv"

NOISE_TYPES = ["replacement", "additive_gaussian"]

COLORS = ["#2b5f8f", "#dd7f29", "#3a945c", "#a33f6f"]
MARKERS = ["o", "s", "^", "D"]


configure_plot_style()

for noise_type in NOISE_TYPES:
    rows = []
    with CSV_PATH.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if noise_type not in row["folder"]:
                continue
            if not row["N"] or not row["d"] or not row["d_label"]:
                continue
            if not row["mean_model_accuracy"] or not row["std_model_accuracy"]:
                continue
            rows.append(
                {
                    "N": float(row["N"]),
                    "d": float(row["d"]),
                    "d_label": row["d_label"],
                    "mean_model_accuracy": float(row["mean_model_accuracy"]),
                    "std_model_accuracy": float(row["std_model_accuracy"]),
                }
            )

    fig, ax = plt.subplots(figsize=(3.375, 2.4), dpi=300, constrained_layout=True)

    for i, d_value in enumerate(sorted({row["d"] for row in rows})):
        subdf = sorted([row for row in rows if row["d"] == d_value], key=lambda r: r["N"])
        d_label = subdf[0]["d_label"]

        ax.errorbar(
            [row["N"] for row in subdf],
            [100.0 * row["mean_model_accuracy"] for row in subdf],
            yerr=[100.0 * row["std_model_accuracy"] for row in subdf],
            marker=MARKERS[i],
            markersize=3.5,
            linewidth=1.2,
            capsize=2.5,
            color=COLORS[i],
            label=fr"$d = {d_label}$",
        )

    ax.set_xlabel(r"$N$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(sorted({row["N"] for row in rows}))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.set_ylim(10, 75)
    ax.legend(loc="lower right", frameon=True, handlelength=1.8)

    output_path = HERE / f"mean_test_accuracy_vs_N_fixed_d__noise={noise_type}.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {output_path}")
