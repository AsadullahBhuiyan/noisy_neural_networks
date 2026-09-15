#!/usr/bin/env python3
"""Two-panel figure: (a) log SNR vs log(Nd), (b) test accuracy vs N at fixed d.

Panel (a) reproduces snr_scaling__NOISE__GROUPID.png from the per-folder
summary written by analyze_nested_snr_scaling.py; panel (b) reproduces
mean_test_accuracy_vs_N_fixed_d_NOISE.png from the empirical accuracy summary
used by plot_acc.py. One figure is written per noise type.

Usage: plot_snr_and_accuracy_panels.py [additive_gaussian|replacement]...
       (no argument: both noise types)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SNR_SCALING_ROOT = HERE / "snr_scaling"
ACCURACY_SUMMARY = HERE / "empirical_mean_model_accuracy_summary.csv"

NOISE_TYPES = ("additive_gaussian", "replacement")
COLUMN_WIDTH = 3.375  # Single-column width; two panels give a full-width figure.
PANEL_HEIGHT = 2.6

# Same colour/marker cycle as plot_acc.py, in order of increasing d.
D_STYLES = [
    {"color": "#1f77b4", "marker": "o"},
    {"color": "#d62728", "marker": "s"},
    {"color": "#2ca02c", "marker": "^"},
    {"color": "#9467bd", "marker": "D"},
]


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "CMU Sans Serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "text.usetex": False,
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "CMU Sans Serif",
            "mathtext.it": "CMU Sans Serif:italic",
            "mathtext.bf": "CMU Sans Serif:bold",
        }
    )


def load_snr_cells(noise_type: str) -> pd.DataFrame:
    """One row per (N, d) folder: mean and SD of log SNR across test points."""
    path = SNR_SCALING_ROOT / f"true_class_{noise_type}" / "snr_folder_summary.csv"
    cells = pd.read_csv(path)
    cells = cells.dropna(subset=["log_Nd", "mean_log_snr"]).copy()
    if cells.empty:
        raise ValueError(f"No usable rows in {path}")
    return cells.sort_values("log_Nd")


def load_accuracy(noise_type: str) -> pd.DataFrame:
    df = pd.read_csv(ACCURACY_SUMMARY)
    df = df[df["folder"].str.contains(noise_type, na=False)].copy()
    df = df.dropna(subset=["N", "d", "d_label", "mean_model_accuracy", "std_model_accuracy"])
    for column in ("N", "d", "mean_model_accuracy", "std_model_accuracy"):
        df[column] = pd.to_numeric(df[column])
    if df.empty:
        raise ValueError(f"No {noise_type} rows in {ACCURACY_SUMMARY}")
    return df


def fit_folder_means(cells: pd.DataFrame) -> tuple[float, float]:
    """Equal-weight OLS on folder mean log SNR, matching the original analysis."""
    x = cells["log_Nd"].to_numpy()
    y = cells["mean_log_snr"].to_numpy()
    if len(x) < 2 or len(np.unique(x)) < 2:
        raise ValueError("Need at least two distinct log(Nd) values to fit")
    slope, intercept = np.linalg.lstsq(
        np.column_stack([x, np.ones(len(x))]), y, rcond=None
    )[0]
    return float(slope), float(intercept)


def draw_snr_panel(ax: plt.Axes, cells: pd.DataFrame) -> None:
    slope, intercept = fit_folder_means(cells)
    x = cells["log_Nd"].to_numpy()
    ax.errorbar(
        x,
        cells["mean_log_snr"].to_numpy(),
        yerr=cells["std_log_snr"].to_numpy(),
        fmt="s",
        linestyle="none",
        color="black",
        markersize=3,
        markeredgewidth=0.7,
        ecolor="0.5",
        elinewidth=0.7,
        capsize=2,
        capthick=0.7,
        zorder=5,
    )
    line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        line,
        slope * line + intercept,
        "--",
        color="black",
        linewidth=1.0,
        zorder=4,
    )
    ax.set_xlabel(r"$\log(Nd)$")
    ax.set_ylabel(r"$\log(\mathrm{SNR})$")
    # Equal numerical spans on both axes, so a slope of one reads as a diagonal.
    # Each range is centred separately: log SNR has a nonzero intercept.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    xmid, ymid = np.mean(xlim), np.mean(ylim)
    ax.set_xlim(xmid - span / 2, xmid + span / 2)
    ax.set_ylim(ymid - span / 2, ymid + span / 2)
    ax.text(
        0.96, 0.05,
        fr"Fit slope = ${slope:.2f}$",
        transform=ax.transAxes,
        ha="right", va="bottom",
    )


def draw_accuracy_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    for style, d_value in zip(D_STYLES, sorted(df["d"].unique())):
        sub = df[df["d"] == d_value].sort_values("N")
        ax.errorbar(
            sub["N"],
            sub["mean_model_accuracy"] * 100,
            yerr=sub["std_model_accuracy"] * 100,
            fmt="-",
            capsize=2,
            markersize=3.5,
            linewidth=1.2,
            elinewidth=0.8,
            capthick=0.8,
            label=fr"$d = {sub['d_label'].iloc[0].replace('x', r'\times')}$",
            color=style["color"],
            marker=style["marker"],
        )
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=True, handlelength=1.8, loc="lower right")


def make_figure(noise_type: str) -> None:
    cells = load_snr_cells(noise_type)
    accuracy = load_accuracy(noise_type)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(2 * COLUMN_WIDTH, PANEL_HEIGHT),
        dpi=300,
        constrained_layout=True,
    )
    draw_snr_panel(axes[0], cells)
    draw_accuracy_panel(axes[1], accuracy)

    for ax, panel_label in zip(axes, ["(a)", "(b)"]):
        ax.text(
            -0.16, 1.04, panel_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11, clip_on=False,
        )

    path = HERE / f"snr_and_test_accuracy_{noise_type}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    print(f"Wrote {path}")
    plt.close(fig)


def main() -> None:
    requested = sys.argv[1:] or list(NOISE_TYPES)
    unknown = [name for name in requested if name not in NOISE_TYPES]
    if unknown:
        raise SystemExit(f"Unknown noise type(s): {', '.join(unknown)}; "
                         f"choose from {', '.join(NOISE_TYPES)}")
    configure_plot_style()
    for noise_type in requested:
        make_figure(noise_type)


if __name__ == "__main__":
    main()
