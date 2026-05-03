from collections import defaultdict
import argparse
import csv
import gzip
import os
from pathlib import Path
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jt577")

import matplotlib.pyplot as plt


DEFAULT_FOLDER = Path(
    "/data2/jt577/2026/04.2026/NN_noise/trained_mlp_nested_ensembles_featureDim/"
    "mlp_ensemble__additive_gaussian__p=0.95__N=1000__d=20x20__loss=mse__act=erf__L=3__width=2048__noiseK=100__initM=10__models=1000"
)


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
parser.add_argument("--test-index", type=int, required=True)
parser.add_argument("--label", type=int, default=None)
parser.add_argument("--bins", type=int, default=20)
args = parser.parse_args()

csv_path = args.folder / "test_outputs.csv.gz"
label = args.label
sums = defaultdict(float)
counts = defaultdict(int)

with gzip.open(csv_path, "rt", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row["test_index"]) != args.test_index:
            continue
        if label is None:
            label = int(row["true_label"])
        noise_index = int(row["noise_dataset_index"])
        sums[noise_index] += float(row[f"logit_{label}"])
        counts[noise_index] += 1

means = [sums[k] / counts[k] for k in sorted(sums)]
if not means:
    raise RuntimeError(f"No rows found for test_index={args.test_index}")
mean_of_means = statistics.mean(means)
std_of_means = statistics.stdev(means) if len(means) > 1 else 0.0

out_path = args.folder / f"test_index_{args.test_index}_label_{label}_noise_dataset_mean_hist.png"

plt.figure(figsize=(7, 5))
plt.hist(means, bins=args.bins, color="tab:blue", edgecolor="black", alpha=0.75)
plt.axvline(mean_of_means, color="black", linestyle="--", linewidth=2)
plt.text(
    0.97,
    0.95,
    f"mean = {mean_of_means:.4f}\nstd = {std_of_means:.4f}",
    transform=plt.gca().transAxes,
    ha="right",
    va="top",
)
plt.xlabel(f"Mean model output for label {label}")
plt.ylabel("Number of noisy datasets")
plt.title(f"Test index {args.test_index}: mean over models per noisy dataset")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"Saved {out_path}")
