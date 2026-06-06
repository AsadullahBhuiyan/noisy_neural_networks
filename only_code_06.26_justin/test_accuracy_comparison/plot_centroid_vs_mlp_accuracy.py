import csv
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent

DATASET = "kmnist"
NOISE_TYPE = "replacement"
FIT_NOISE_TYPE = "replacement"
FIT_P = 0.97
pmin = 0.90
N = 4000
D_LABEL = "28x28"
LOSS = "mse"
ACTIVATION = "erf"
DEPTH = 3
WIDTH = 2048

CSV_PATH = PROJECT_DIR / "trained_centroids" / (
    f"test_accuracy_summary__dataset={DATASET}"
    f"__eval={NOISE_TYPE}"
    f"__fit={FIT_NOISE_TYPE}_p={FIT_P:.2f}"
    f"__N={N}__d={D_LABEL}"
    f"__loss={LOSS}__act={ACTIVATION}"
    f"__L={DEPTH}__width={WIDTH}.csv"
)
OUTPUT_PNG = PROJECT_DIR / "trained_centroids" / f"{DATASET}_{NOISE_TYPE}_accuracy_comparison.png"


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


rows = read_csv(CSV_PATH)
rows = sorted(
    [
        r
        for r in rows
        if r["noise_type"] == NOISE_TYPE
        and r.get("dataset", DATASET) == DATASET
        and float(r["noise_probability"]) >= pmin
    ],
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
