#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_auc_summary(input_csv: Path) -> list[dict]:
    by_run: dict[tuple[float, int], list[tuple[float, float]]] = defaultdict(list)
    with input_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            by_run[(float(r["p_train"]), int(r["run_id"]))].append(
                (float(r["p_test"]), float(r["test_accuracy"]))
            )

    auc_by_p: dict[float, list[float]] = defaultdict(list)
    for (p_train, _), points in by_run.items():
        points.sort(key=lambda x: x[0])
        xs = np.array([p for p, _ in points], dtype=float)
        ys = np.array([acc for _, acc in points], dtype=float)
        auc_by_p[p_train].append(float(np.trapz(ys, xs)))

    rows: list[dict] = []
    for p_train in sorted(auc_by_p):
        vals = np.array(auc_by_p[p_train], dtype=float)
        rows.append(
            {
                "p_train": p_train,
                "num_runs": int(vals.size),
                "mean_auc_accuracy": float(np.mean(vals)),
                "stderr_auc_accuracy": 0.0 if vals.size <= 1 else float(np.std(vals, ddof=1) / np.sqrt(vals.size)),
                "std_auc_accuracy": 0.0 if vals.size <= 1 else float(np.std(vals, ddof=1)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(summary: list[dict], output_png: Path, output_pdf: Path | None) -> None:
    best = max(summary, key=lambda r: r["mean_auc_accuracy"])
    xs = np.array([r["p_train"] for r in summary], dtype=float)
    ys = np.array([r["mean_auc_accuracy"] for r in summary], dtype=float)
    ses = np.array([r["stderr_auc_accuracy"] for r in summary], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.plot(xs, ys, marker="o", linewidth=1.8, color="#1f77b4")
    ax.fill_between(xs, ys - ses, ys + ses, color="#1f77b4", alpha=0.18, linewidth=0)
    ax.axvline(
        best["p_train"],
        linestyle="--",
        color="black",
        linewidth=1.2,
        alpha=0.8,
        label=f"best p_train={best['p_train']:.2f}",
    )
    ax.scatter([best["p_train"]], [best["mean_auc_accuracy"]], color="black", s=36, zorder=4)
    ax.set_xlabel("p_train")
    ax.set_ylabel("AUC of test accuracy over p_test")
    ax.set_title("Robustness AUC vs training replacement noise")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(max(0.0, float(np.min(ys - ses)) - 0.01), min(1.0, float(np.max(ys + ses)) + 0.01))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AUC of test accuracy over p_test as a function of p_train.")
    parser.add_argument("--csv", type=str, required=True, help="Merged raw p_train/p_test CSV.")
    parser.add_argument("--summary-csv", type=str, required=True, help="Output AUC summary CSV.")
    parser.add_argument("--output-png", type=str, required=True, help="Output AUC plot PNG.")
    parser.add_argument("--output-pdf", type=str, default="", help="Optional output AUC plot PDF.")
    args = parser.parse_args()

    summary = compute_auc_summary(Path(args.csv))
    write_csv(Path(args.summary_csv), summary)
    make_plot(summary, Path(args.output_png), Path(args.output_pdf) if args.output_pdf else None)

    best = max(summary, key=lambda r: r["mean_auc_accuracy"])
    print(
        f"Best p_train={best['p_train']:.6g}, "
        f"mean_auc={best['mean_auc_accuracy']:.6f}, "
        f"stderr={best['stderr_auc_accuracy']:.6f}, "
        f"num_runs={best['num_runs']}"
    )


if __name__ == "__main__":
    main()
