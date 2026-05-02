from collections import defaultdict
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jt577")

import matplotlib.pyplot as plt


here = Path(__file__).resolve().parent
empirical_csv_path = here / "summary_test_accuracy_by_p_activation.csv"
analytical_csv_path = here / "analytical_abc_accuracy_summary_fit_c.csv"
fit_json_path = here / "analytical_abc_fit_c_to_empirical_erf.json"
out_path = here / "one_activation_accuracy_vs_p_with_fitted_c.png"

act_choice = "erf"


def read_empirical_erf() -> list[tuple[float, float, float]]:
    data = defaultdict(list)
    with empirical_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            data[row["activation"]].append(
                (
                    float(row["p"]),
                    float(row["mean_accuracy_percent"]),
                    float(row["std_accuracy_percent"]),
                )
            )
    return sorted(data[act_choice])


def read_analytical() -> list[tuple[float, float, float]]:
    values = []
    with analytical_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(
                (
                    float(row["p"]),
                    float(row["mean_accuracy_percent"]),
                    float(row["std_accuracy_percent"]),
                )
            )
    return sorted(values)


empirical = read_empirical_erf()
analytical = read_analytical()
with fit_json_path.open("r", encoding="utf-8") as handle:
    fit = json.load(handle)

plt.figure(figsize=(8, 6))

p = [v[0] for v in empirical]
mean = [v[1] for v in empirical]
std = [v[2] for v in empirical]
plt.errorbar(
    p,
    mean,
    yerr=std,
    marker="o",
    color="blue",
    label="Neural network",
    linewidth=5,
    markersize=10,
    capsize=4,
    alpha=1.0,
    zorder=1
)

plt.axhline(76.42, linewidth=3, linestyle="--", color="black", label="Nearest-class-mean classifier")

p = [v[0] for v in analytical]
mean = [v[1] for v in analytical]
plt.plot(
    p,
    mean,
    marker="s",
    color="red",
    label=f"Fitted analytical model",
    linewidth=2.5,
    markersize=4,
    alpha=1.0,
)

plt.xlabel("p", fontsize=24)
plt.ylabel("Test accuracy (%)", fontsize=24)
plt.ylim(0, 100)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
# plt.legend(fontsize=17)
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"Saved plot: {out_path}")
