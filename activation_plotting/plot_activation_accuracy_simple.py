from collections import defaultdict
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jt577")

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def configure_plot_style() -> None:
    """Apply project-default figure styling."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (3.375, 2.4),
            "font.family": "CMU Sans Serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.usetex": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "CMU Sans Serif",
            "mathtext.it": "CMU Sans Serif:italic",
            "mathtext.bf": "CMU Sans Serif:bold",
        }
    )


configure_plot_style()


here = Path(__file__).resolve().parent
csv_path = here / "summary_test_accuracy_by_p_activation.csv"
out_path = here / "simple_activation_accuracy_vs_p.png"
out_pdf_path = here / "simple_activation_accuracy_vs_p.pdf"

colors = {
    "erf": "tab:blue",
    "gelu": "tab:orange",
    "swish": "tab:green",
    "tanh": "tab:red",
}
markers = {
    "erf": "o",
    "gelu": "s",
    "swish": "^",
    "tanh": "D",
}
offsets = {
    "erf": -0.0015,
    "gelu": -0.0005,
    "swish": 0.0005,
    "tanh": 0.0015,
}
legend_labels = {
    "erf": "erf",
    "gelu": "GELU",
    "swish": "Swish",
    "tanh": "Tanh",
}

data = defaultdict(list)
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = float(row["p"])
        mean = float(row["mean_accuracy_percent"])
        std = float(row["std_accuracy_percent"])
        data[row["activation"]].append((p, mean, std))

plt.figure()
for activation, values in sorted(data.items()):
    values = sorted(values)
    p = [v[0] for v in values]
    mean = [v[1] for v in values]
    std = [v[2] for v in values]
    color = colors.get(activation)
    marker = markers.get(activation, "o")
    p_plot = [x + offsets.get(activation, 0.0) for x in p]
    plt.errorbar(
        p_plot,
        mean,
        yerr=std,
        marker=marker,
        color=color,
        label=legend_labels.get(activation, activation),
        linewidth=1.2,
        markersize=3.2,
        capsize=2,
        alpha=0.9,
    )

plt.xlabel(r"Corruption probability $p$")
plt.ylabel("Test accuracy (%)")
plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.legend(frameon=True, handlelength=1.8, loc="lower left")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.savefig(out_pdf_path)
