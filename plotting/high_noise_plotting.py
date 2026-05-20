from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (3.375, 2.4),
            "font.family": "CMU Sans Serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.usetex": False,
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "CMU Sans Serif",
            "mathtext.it": "CMU Sans Serif:italic",
            "mathtext.bf": "CMU Sans Serif:bold",
        }
    )


def plot_accuracy_vs_p(
    ax: plt.Axes,
    *,
    csv_path: Path,
    noise_type: str,
    panel_label: str | None,
) -> None:
    """Plot test accuracy vs p for a single noise type."""
    df = pd.read_csv(csv_path)
    subset = df[df["noise_type"] == noise_type].sort_values("noise_probability")
    if subset.empty:
        raise ValueError(f"No rows for noise_type '{noise_type}' in {csv_path}.")

    p = subset["noise_probability"]
    nn_mean = subset["empirical_mean_test_accuracy"] * 100
    nn_std = subset["empirical_std_test_accuracy"] * 100
    an_mean = subset["mean_test_accuracy"] * 100

    ax.errorbar(
        p,
        nn_mean,
        yerr=nn_std,
        marker="o",
        markersize=3.5,
        linewidth=1.2,
        capsize=2.5,
        color="#b31b1b",
        label="Neural network",
        zorder=2,
    )
    ax.plot(
        p,
        an_mean,
        linewidth=1.2,
        linestyle="--",
        color="#404040",
        label="Analytical model",
        zorder=3,
    )
    xlabel = r"Corruption probability $p$" if noise_type == "replacement" else r"Corruption strength $p$"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)

    if noise_type == "replacement":
        ax.set_xticks([0.9, 0.925, 0.95, 0.975, 1.0])
    else:
        ax.set_xticks([0.85, 0.9, 0.95, 1.0])

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.legend(frameon=True, handlelength=1.8, loc="upper right")

    if panel_label:
        ax.text(
            -0.18, 1.02, panel_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9, clip_on=False,
        )


def plot_actual_vs_predicted_scatter(
    ax: plt.Axes,
    *,
    csv_path: Path,
    panel_label: str | None,
    scatter_size: float = 12.0,
    r2_fontsize: int = 8,
    linewidth: float = 1,
    max_points: int = 100_000,
    p_label: str = "0.95",
) -> None:
    """Plot actual vs predicted scatter with diagonal and R² annotation."""
    xs, ys = [], []
    open_fn = gzip.open if str(csv_path).endswith(".gz") else open
    with open_fn(csv_path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row.get("delta_fit_logit", row["centroid_fit_logit"])))
            ys.append(float(row["ensemble_mean_logit"]))
            if len(xs) >= max_points:
                break

    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)

    ss_res = float(np.sum((ys_arr - xs_arr) ** 2))
    ss_tot = float(np.sum((ys_arr - ys_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    lo = float(min(xs_arr.min(), ys_arr.min()))
    hi = float(max(xs_arr.max(), ys_arr.max()))
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad

    ax.scatter(xs_arr, ys_arr, s=scatter_size, alpha=0.05, edgecolors="none", color="#E0A01E", rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="k", linestyle="--", linewidth=linewidth)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted outputs")
    ax.set_ylabel("Actual outputs")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    ax.text(
        0.05, 0.95,
        rf"$R^2 = {r2:.3f}$",
        transform=ax.transAxes,
        fontsize=r2_fontsize,
        va="top",
    )
    ax.text(
        0.4, 0.2,
        rf"$(p = {p_label})$",
        transform=ax.transAxes,
        va="top",
        fontsize=r2_fontsize,
    )

    if panel_label:
        ax.text(
            -0.18, 1.02, panel_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9, clip_on=False,
        )


def plot_high_noise_single_column(
    *,
    accuracy_csv: Path,
    scatter_csv: Path,
    output_path: Path,
    noise_type: str = "replacement",
) -> None:
    """Create a single-column figure with an inset scatter panel."""
    df = pd.read_csv(accuracy_csv)
    subset = df[df["noise_type"] == noise_type]
    p_label = str(subset["fit_reference_p"].iloc[0])

    fig, ax = plt.subplots(figsize=(3.375, 2.4), dpi=300, constrained_layout=True)

    plot_accuracy_vs_p(
        ax,
        csv_path=accuracy_csv,
        noise_type=noise_type,
        panel_label=None,
    )

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.tick_params(labelsize=9)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)

    handles, labels = ax.get_legend_handles_labels()
    order = ["Neural network", "Analytical model"]
    ordered = [(h, l) for l in order for h, lbl in zip(handles, labels) if lbl == l]
    h_ord, l_ord = zip(*ordered)
    ax.legend(h_ord, l_ord, frameon=True, handlelength=1.6, loc="upper right",
              bbox_to_anchor=(1.0, 1.0))

    # inset_ax = ax.inset_axes([0.2, 0.18, 0.3, 0.455])
    factor = 1.1
    inset_ax = ax.inset_axes([0.24, 0.18, 0.3*factor, 0.455*factor])
    plot_actual_vs_predicted_scatter(
        inset_ax,
        csv_path=scatter_csv,
        panel_label=None,
        scatter_size=4.0,
        r2_fontsize=8,
        p_label=p_label,
    )
    inset_ax.tick_params(labelsize=7, pad=1)
    inset_ax.xaxis.label.set_size(8)
    inset_ax.yaxis.label.set_size(8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot high-noise accuracy figure.")
    parser.add_argument(
        "--noise",
        choices=["replacement", "additive"],
        default="replacement",
        help="Noise type to plot (default: replacement)",
    )
    args = parser.parse_args()

    configure_plot_style()

    noise_type = "replacement" if args.noise == "replacement" else "additive_gaussian"

    fig_dir = Path(__file__).resolve().parent.parent / "figure_plotting_05.12"
    accuracy_csv = fig_dir / "fig2.2" / f"test_accuracy_summary_{noise_type}.csv"
    scatter_csv = fig_dir / "fig2.1" / f"centroid_comparison_{noise_type}.csv.gz"
    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / f"high_noise_accuracy_with_inset_{args.noise}.pdf"
    )

    plot_high_noise_single_column(
        accuracy_csv=accuracy_csv,
        scatter_csv=scatter_csv,
        output_path=output_path,
        noise_type=noise_type,
    )
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
