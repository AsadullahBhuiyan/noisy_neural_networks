from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



DEFAULT_CSV = Path(
    "/data2/jt577/2026/papers/noisy_networks/05.2026/only_figure_plotting/fig2.1/centroid_comparison_replacement.csv.gz"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Delta E fitted logits vs ensemble mean logits.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=100_000)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    input_csv = args.input_csv if args.input_csv.is_absolute() else project_dir / args.input_csv
    output_png = args.output_png or input_csv.with_suffix("").with_suffix(".png")
    xs, ys, classes = [], [], []

    with gzip.open(input_csv, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row.get("delta_fit_logit", row["centroid_fit_logit"])))
            ys.append(float(row["ensemble_mean_logit"]))
            classes.append(int(row["class_id"]))
            if len(xs) >= args.max_points:
                break

    y_mean = sum(ys) / len(ys)
    ss_res = sum((y - x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot

    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, color="blue", s=20, alpha=0.05, linewidths=0)
    plt.plot([lo, hi], [lo, hi], color="red", linewidth=3.0, linestyle="--")
    plt.text(
        0.06,
        0.94,
        rf"$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        fontsize=26,
        va="top",
    )
    plt.text(
        0.7,
        0.14,
        rf"$(p = 0.95)$",
        transform=plt.gca().transAxes,
        fontsize=26,
        va="top",
    )
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.xlabel("Predicted Outputs", fontsize=26)
    plt.ylabel("Actual Outputs", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
