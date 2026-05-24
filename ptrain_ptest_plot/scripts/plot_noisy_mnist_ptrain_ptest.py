#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "p_train": float(r["p_train"]),
                    "p_test": float(r["p_test"]),
                    "test_accuracy": float(r["test_accuracy"]),
                    "test_loss": float(r["test_loss"]),
                    "run_id": int(r.get("run_id", 0)),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def mean_and_stderr(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    return mean, float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def aggregate_accuracy(rows: list[dict]) -> dict[float, list[tuple[float, float, float]]]:
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)
    for r in rows:
        grouped[(r["p_train"], r["p_test"])].append(r["test_accuracy"])

    series: dict[float, list[tuple[float, float, float]]] = defaultdict(list)
    for (p_train, p_test), vals in grouped.items():
        mean, stderr = mean_and_stderr(vals)
        series[p_train].append((p_test, mean, stderr))
    for p_train in series:
        series[p_train].sort(key=lambda x: x[0])
    return dict(sorted(series.items()))


def make_plot(csv_path: Path, output_png: Path, output_pdf: Path | None) -> None:
    rows = load_rows(csv_path)
    series = aggregate_accuracy(rows)
    p_trains = np.array(sorted(series), dtype=float)
    norm = plt.Normalize(vmin=float(p_trains.min()), vmax=float(p_trains.max()))
    cmap = plt.get_cmap()

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for p_train in p_trains:
        points = series[float(p_train)]
        xs = [p for p, _, _ in points]
        ys = np.array([acc for _, acc, _ in points], dtype=float)
        ses = np.array([stderr for _, _, stderr in points], dtype=float)
        color = cmap(norm(float(p_train)))
        ax.plot(xs, ys, color=color, linewidth=1.6)
        ax.fill_between(xs, ys - ses, ys + ses, color=color, alpha=0.08, linewidth=0)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("p_train")

    ax.set_xlabel("p_test")
    ax.set_ylabel("Test accuracy")
    ax.set_title("MNIST test accuracy vs test replacement noise")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot p_train/p_test noisy MNIST accuracy grid.")
    parser.add_argument("--csv", type=str, required=True, help="Merged CSV from train_noisy_mnist_ptrain_ptest.py")
    parser.add_argument("--output-png", type=str, required=True)
    parser.add_argument("--output-pdf", type=str, default="")
    args = parser.parse_args()
    make_plot(Path(args.csv), Path(args.output_png), Path(args.output_pdf) if args.output_pdf else None)
    print(f"Saved plot to {Path(args.output_png).resolve()}")


if __name__ == "__main__":
    main()
