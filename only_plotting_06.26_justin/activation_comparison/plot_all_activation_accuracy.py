from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jt577")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_CSV = PROJECT_DIR / "test_accuracy_eval" / "summary_test_accuracy_by_p_activation.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot activation-comparison accuracy summaries.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter.")
    parser.add_argument("--noise-type", type=str, default=None, help="Optional noise-type filter.")
    parser.add_argument("--p-min", type=float, default=None)
    parser.add_argument("--p-max", type=float, default=None)
    return parser.parse_args()


def plot_one(df: pd.DataFrame, *, dataset: str, noise_type: str, output_dir: Path) -> Path:
    colors = {
        "erf": "tab:blue",
        "gelu": "tab:orange",
        "swish": "tab:green",
        "tanh": "tab:red",
        "relu": "tab:purple",
        "softplus": "tab:brown",
    }
    markers = {
        "erf": "o",
        "gelu": "s",
        "swish": "^",
        "tanh": "D",
        "relu": "P",
        "softplus": "X",
    }

    plt.figure(figsize=(8, 6))
    for activation, subdf in sorted(df.groupby("activation")):
        subdf = subdf.sort_values("p")
        plt.errorbar(
            subdf["p"],
            subdf["mean_accuracy_percent"],
            yerr=subdf["std_accuracy_percent"],
            marker=markers.get(activation, "o"),
            color=colors.get(activation),
            label=activation,
            linewidth=4,
            markersize=10,
            capsize=4,
            alpha=0.8,
        )

    plt.title(f"{dataset}, {noise_type}")
    plt.xlabel("p", fontsize=24)
    plt.ylabel("Test accuracy (%)", fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"activation_accuracy_vs_p__dataset={dataset}__noise={noise_type}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def main() -> None:
    args = parse_args()
    summary_csv = args.summary_csv.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else summary_csv.parent

    df = pd.read_csv(summary_csv)
    if "dataset" not in df.columns:
        df["dataset"] = "mnist"
    if "noise_type" not in df.columns:
        raise ValueError("Summary CSV is missing noise_type. Regenerate it with evaluate_activation_comparison_test_accuracy.py.")

    for col in ["p", "mean_accuracy_percent", "std_accuracy_percent"]:
        df[col] = pd.to_numeric(df[col])

    if args.dataset is not None:
        df = df[df["dataset"].astype(str) == args.dataset].copy()
    if args.noise_type is not None:
        df = df[df["noise_type"].astype(str) == args.noise_type].copy()
    if args.p_min is not None:
        df = df[df["p"] >= float(args.p_min)].copy()
    if args.p_max is not None:
        df = df[df["p"] <= float(args.p_max)].copy()
    if df.empty:
        raise ValueError("No rows left after filtering.")

    written = []
    for (dataset, noise_type), subdf in sorted(df.groupby(["dataset", "noise_type"])):
        written.append(plot_one(subdf, dataset=str(dataset), noise_type=str(noise_type), output_dir=output_dir))

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
