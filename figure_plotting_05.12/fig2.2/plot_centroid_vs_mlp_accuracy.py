import csv
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent

#################################################
# USER DEFINE
noise_type = "additive_gaussian" # replacement or additive_gaussian
#################################################

CSV_PATH = PROJECT_DIR / f"test_accuracy_summary_{noise_type}.csv"
OUTPUT_PNG = PROJECT_DIR / f"{noise_type}_accuracy_comparison.png"

if noise_type == "additive_gaussian":
    pmin = 0.85
elif noise_type == "replacement":
    pmin = 0.90


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


rows = read_csv(CSV_PATH)
rows = sorted(
    [r for r in rows if r["noise_type"] == noise_type and float(r["noise_probability"]) >= pmin],
    key=lambda r: float(r["noise_probability"])
)

p_vals = [float(r["noise_probability"]) for r in rows]

# Centroid model
centroid_mean = [100.0 * float(r["mean_test_accuracy"]) for r in rows]
centroid_std = [100.0 * float(r["std_test_accuracy"]) for r in rows]

# Empirical network model
empirical_mean = [100.0 * float(r["empirical_mean_test_accuracy"]) for r in rows]
empirical_std = [100.0 * float(r["empirical_std_test_accuracy"]) for r in rows]

plt.figure(figsize=(8, 6))

plt.errorbar(
    p_vals,
    empirical_mean,
    yerr=empirical_std,
    fmt="s-",
    label="Neural Network",
    color="red",
    linewidth=4,
    zorder=1,
)

plt.errorbar(
    p_vals,
    centroid_mean,
    yerr=centroid_std,
    fmt="o--",
    linewidth=3,
    label="Analytical model",
    color="blue",
    zorder=2,
)

plt.xlabel(r"$p$", fontsize=20)
plt.ylabel("Test accuracy (%)", fontsize=20)
plt.ylim(0, 100)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16, loc="upper right")
plt.tight_layout()

OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=300)
print(f"Wrote {OUTPUT_PNG}")