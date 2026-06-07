from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jt577")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter


# -----------------------------------------------------------------------------
# Easy-to-edit settings
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
JUSTIN_ROOT = HERE.parent.parent / "only_plotting_06.26_justin" / "activation_comparison"
SUMMARY_CSV = JUSTIN_ROOT / "test_accuracy_eval" / "summary_test_accuracy_by_p_activation.csv"

P_MIN = 0.90
P_MAX = 1.00
SAVE_PDF = True
SAVE_PNG = True

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


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 17,
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.7,
        "ytick.major.width": 1.7,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


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

    fig, ax = plt.subplots(figsize=(5.45, 3.55))

    for activation in ACTIVATION_ORDER:
        subrows = sorted([row for row in plot_rows if row["activation"] == activation], key=lambda row: row["p"])
        if not subrows:
            continue

        ax.errorbar(
            [row["p"] for row in subrows],
            [row["mean_accuracy_percent"] for row in subrows],
            yerr=[row["std_accuracy_percent"] for row in subrows],
            fmt=f"{MARKERS[activation]}-",
            color=COLORS[activation],
            ecolor=COLORS[activation],
            elinewidth=1.9,
            capsize=3.5,
            markersize=6.2,
            linewidth=2.2,
            label=ACTIVATION_LABELS[activation],
        )

    ax.set_xlabel(r"Corruption probability $p$", fontsize=13)
    ax.set_ylabel("Test accuracy (%)", fontsize=13)
    ax.set_xlim(P_MIN - 0.006, P_MAX + 0.006)
    ax.set_ylim(6, 82)
    ax.xaxis.set_major_locator(FixedLocator([0.90, 0.92, 0.94, 0.96, 0.98, 1.00]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.9", "0.92", "0.94", "0.96", "0.98", "1"]))
    ax.tick_params(labelsize=12)
    ax.legend(
        fontsize=12,
        loc="lower left",
        frameon=True,
        borderpad=0.45,
        handlelength=2.0,
    )

    fig.tight_layout(pad=0.65)

    output_stem = HERE / f"activation_accuracy_vs_p__dataset={dataset}__noise={noise_type}"
    if SAVE_PDF:
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Wrote {output_stem.with_suffix('.pdf')}")
    if SAVE_PNG:
        fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"Wrote {output_stem.with_suffix('.png')}")
    plt.close(fig)
