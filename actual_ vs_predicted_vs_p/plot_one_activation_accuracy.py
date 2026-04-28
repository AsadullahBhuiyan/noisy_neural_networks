from collections import defaultdict
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jt577")

import matplotlib.pyplot as plt


here = Path(__file__).resolve().parent
csv_path = here / "summary_test_accuracy_by_p_activation.csv"
out_path = here / "one_activation_accuracy_vs_p.png"

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

act_choice = "erf"

data = defaultdict(list)
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = float(row["p"])
        mean = float(row["mean_accuracy_percent"])
        std = float(row["std_accuracy_percent"])
        data[row["activation"]].append((p, mean, std))

plt.figure(figsize=(8, 6))
for activation, values in sorted(data.items()):
    if activation == act_choice:
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
            label="Neural network",
            linewidth=4,
            markersize=10,
            capsize=4,
            alpha=1.0,
        )

plt.axhline(76.42, linewidth=3, linestyle="--", color="r", label="Nearest-class-mean-classifier")
plt.xlabel("p", fontsize=24)
plt.ylabel("Test accuracy (%)", fontsize=24)
plt.ylim(0, 100)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(out_path, dpi=300)
