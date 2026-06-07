from __future__ import annotations

import os
import csv
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jt577")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Easy-to-edit settings
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
JUSTIN_SNR_ROOT = HERE.parent.parent / "only_plotting_06.26_justin" / "snr"
CSV_PATH = JUSTIN_SNR_ROOT / "empirical_mean_model_accuracy_summary.csv"

NOISE_TYPES = ["replacement", "additive_gaussian"]

COLORS = ["#2b5f8f", "#dd7f29", "#3a945c", "#a33f6f"]
MARKERS = ["o", "s", "^", "D"]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.linewidth": 1.7,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 5.5,
        "ytick.major.size": 5.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


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

    fig, ax = plt.subplots(figsize=(5.25, 3.65))

    for i, d_value in enumerate(sorted({row["d"] for row in rows})):
        subdf = sorted([row for row in rows if row["d"] == d_value], key=lambda row: row["N"])
        d_label = subdf[0]["d_label"]

        ax.errorbar(
            [row["N"] for row in subdf],
            [100.0 * row["mean_model_accuracy"] for row in subdf],
            yerr=[100.0 * row["std_model_accuracy"] for row in subdf],
            fmt=f"{MARKERS[i]}-",
            color=COLORS[i],
            ecolor=COLORS[i],
            elinewidth=1.8,
            capsize=3.5,
            markersize=6.0,
            linewidth=2.0,
            label=fr"$d = {d_label}$",
        )

    ax.set_xlabel(r"$N$", fontsize=14)
    ax.set_ylabel("Test accuracy (%)", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.set_xticks(sorted({row["N"] for row in rows}))
    ax.set_ylim(10, 75)
    ax.legend(
        fontsize=12,
        loc="lower right",
        frameon=True,
        borderpad=0.45,
        handlelength=2.0,
    )

    output_pdf = HERE / f"mean_test_accuracy_vs_N_fixed_d__noise={noise_type}.pdf"
    output_png = HERE / f"mean_test_accuracy_vs_N_fixed_d__noise={noise_type}.png"

    fig.tight_layout(pad=0.65)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_png}")
