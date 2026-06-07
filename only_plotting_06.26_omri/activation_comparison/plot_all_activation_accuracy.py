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
JUSTIN_ROOT = HERE.parent.parent / "only_plotting_06.26_justin" / "activation_comparison"
SUMMARY_CSV = JUSTIN_ROOT / "test_accuracy_eval" / "summary_test_accuracy_by_p_activation.csv"

P_MIN = 0.90
P_MAX = 1.00

ACTIVATION_ORDER = ["erf", "gelu", "swish", "tanh"]
ACTIVATION_LABELS = {
    "erf": "erf",
    "gelu": "GELU",
    "swish": "Swish",
    "tanh": "Tanh",
}
COLORS = {
    "erf": "#2b5f8f",
    "gelu": "#3a945c",
    "swish": "#dd7f29",
    "tanh": "#a33f6f",
}
MARKERS = {
    "erf": "o",
    "gelu": "s",
    "swish": "^",
    "tanh": "D",
}


configure_plot_style()

rows = []
with SUMMARY_CSV.open("r", newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        p = float(row["p"])
        if p < P_MIN or p > P_MAX:
            continue
        rows.append(
            {
                "dataset": row.get("dataset", "mnist"),
                "noise_type": row["noise_type"],
                "activation": row["activation"],
                "p": p,
                "mean_accuracy_percent": float(row["mean_accuracy_percent"]),
                "std_accuracy_percent": float(row["std_accuracy_percent"]),
            }
        )

for dataset, noise_type in sorted({(row["dataset"], row["noise_type"]) for row in rows}):
    plot_rows = [row for row in rows if row["dataset"] == dataset and row["noise_type"] == noise_type]

    fig, ax = plt.subplots(figsize=(3.375, 2.4), dpi=300, constrained_layout=True)

    for activation in ACTIVATION_ORDER:
        subrows = sorted(
            [row for row in plot_rows if row["activation"] == activation],
            key=lambda r: r["p"],
        )
        if not subrows:
            continue

        ax.errorbar(
            [row["p"] for row in subrows],
            [row["mean_accuracy_percent"] for row in subrows],
            yerr=[row["std_accuracy_percent"] for row in subrows],
            marker=MARKERS[activation],
            markersize=3.5,
            linewidth=1.2,
            capsize=2.5,
            color=COLORS[activation],
            label=ACTIVATION_LABELS[activation],
        )

    xlabel = (
        r"Corruption probability $p$"
        if noise_type == "replacement"
        else r"Corruption strength $p$"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(P_MIN - 0.006, P_MAX + 0.006)
    ax.set_ylim(6, 82)
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.90, 0.92, 0.94, 0.96, 0.98, 1.00]))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.legend(loc="lower left", frameon=True, handlelength=1.8)

    output_path = HERE / f"activation_accuracy_vs_p__dataset={dataset}__noise={noise_type}.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {output_path}")
