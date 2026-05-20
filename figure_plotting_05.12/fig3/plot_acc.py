import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

###########
# USER INPUT
parser = argparse.ArgumentParser(description="Plot accuracy vs N for a noise type.")
parser.add_argument(
    "--noise",
    choices=["replacement", "additive"],
    default="replacement",
    help="Noise type to plot (default: replacement).",
)
args = parser.parse_args()

noise_type = "replacement" if args.noise == "replacement" else "additive_gaussian"
###########

here = Path(__file__).resolve().parent
csv_path = here / "empirical_mean_model_accuracy_summary.csv"
outdir = (here / ".." / ".." / "figures").resolve()
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

# Keep only rows matching the selected noise type
df = df[df["folder"].str.contains(noise_type, na=False)].copy()

df = df.dropna(subset=["N", "d", "d_label", "mean_model_accuracy", "std_model_accuracy"]).copy()

# Make sure numeric columns are numeric
df["N"] = pd.to_numeric(df["N"])
df["d"] = pd.to_numeric(df["d"])
df["mean_model_accuracy"] = pd.to_numeric(df["mean_model_accuracy"])
df["std_model_accuracy"] = pd.to_numeric(df["std_model_accuracy"])

FIG_WIDTH = 3.375
FIG_HEIGHT = 2.4
BASE_FONT_SIZE = 9

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "CMU Sans Serif",
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,
        "text.usetex": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "CMU Sans Serif",
        "mathtext.it": "CMU Sans Serif:italic",
        "mathtext.bf": "CMU Sans Serif:bold",
    }
)

plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

styles = [
    {"color": "#1f77b4", "marker": "o"},
    {"color": "#d62728", "marker": "s"},
    {"color": "#2ca02c", "marker": "^"},
    {"color": "#9467bd", "marker": "D"},
]

for i, d_value in enumerate(sorted(df["d"].unique())):
    subdf = df[df["d"] == d_value].sort_values("N")
    d_label = subdf["d_label"].iloc[0]
    d_label_tex = str(d_label).replace("x", "\\times")

    plt.errorbar(
        subdf["N"],
        subdf["mean_model_accuracy"] * 100,
        yerr=subdf["std_model_accuracy"] * 100,
        linestyle="-",
        capsize=2,
        markersize=3,
        linewidth=1,
        label=fr"$d={d_label_tex}$",
        color=styles[i]["color"],
        marker=styles[i]["marker"]
    )

plt.xlabel(r"$N$")
plt.ylabel(r"Test accuracy (%)")
plt.legend(fontsize=8)
plt.tight_layout()
out_path = outdir / f"mean_test_accuracy_vs_N_fixed_d_{noise_type}.pdf"
plt.savefig(out_path)
print(f"Saved: {out_path}")
# plt.show()