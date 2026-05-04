from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def configure_plot_style() -> None:
    """Apply project-default figure styling."""
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
    activation: str,
    baseline: float,
    panel_label: str | None,
    analytical_csv: Path | None = None,
) -> None:
    """Plot test accuracy vs p for a single activation."""
    df = pd.read_csv(csv_path)
    required_cols = {"p", "activation", "mean_accuracy_percent", "std_accuracy_percent"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing_cols)}")

    act_df = df[df["activation"] == activation].copy()
    if act_df.empty:
        raise ValueError(f"No rows for activation '{activation}' in {csv_path}.")
    act_df = act_df.sort_values("p")

    ax.errorbar(
        act_df["p"],
        act_df["mean_accuracy_percent"],
        yerr=act_df["std_accuracy_percent"],
        marker="o",
        markersize=3.5,
        linewidth=1.2,
        capsize=2.5,
        color="#b31b1b",
        label="Neural network",
        zorder=2,
    )

    if analytical_csv is not None:
        an_df = pd.read_csv(analytical_csv).sort_values("p")
        ax.plot(
            an_df["p"],
            an_df["mean_accuracy_percent"],
            linewidth=1.2,
            color="#404040",
            label="Analytical model",
            zorder=3,
        )

    ax.axhline(
        baseline,
        linewidth=1.2,
        linestyle="--",
        color="tab:gray",
    )
    ax.set_xlabel(r"Corruption probability $p$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks([0.9, 0.925, 0.95, 0.975, 1.0])
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.legend(frameon=True, handlelength=1.8, loc="upper right")

    if panel_label:
        ax.text(
            -0.18,
            1.02,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            clip_on=False,
        )


def plot_actual_vs_predicted_scatter(
    ax: plt.Axes,
    *,
    csv_path: Path,
    json_path: Path,
    panel_label: str | None,
    scatter_size: float = 12.0,
    r2_fontsize: int = 8,
    linewidth: float = 1,
) -> None:
    """Plot predicted vs actual scatter with diagonal and Pearson r annotation."""
    database = pd.read_json(json_path)
    fit = database["summary"]["simple_delta_fit_raw"]
    a = float(fit["a"])
    b = float(fit["b"])

    data = pd.read_csv(csv_path)
    required_cols = {"empirical_mean", "simple_delta_xstar_x_centered"}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing_cols)}")

    target = data["empirical_mean"].astype(float)
    delta = data["simple_delta_xstar_x_centered"].astype(float)
    x = target
    y = a * delta + b

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad

    ax.scatter(x, y, s=scatter_size, alpha=0.05, edgecolors="none", color="#E0A01E", rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="k", linestyle="--", linewidth=linewidth)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual outputs")
    ax.set_ylabel("Predicted outputs")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    r_value = float(np.corrcoef(x, y)[0, 1])
    if np.isfinite(r_value):
        r2_value = r_value ** 2
        ax.text(
            -0.5,
            0.92,
            rf"$R^2 = {r2_value:.3f}$",
            fontsize=r2_fontsize,
        )

    ax.text(
        0.4,
        0.2,
        r"$(p = 0.95)$",
        transform=ax.transAxes,
        va="top",
        fontsize=r2_fontsize,
    )
        

    if panel_label:
        ax.text(
            -0.18,
            1.02,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            clip_on=False,
        )


def plot_high_noise_figure(
    *,
    accuracy_csv: Path,
    comparison_csv: Path,
    comparison_json: Path,
    output_path: Path,
    activation: str = "erf",
    baseline: float = 76.42,
    analytical_csv: Path | None = None,
) -> None:
    """Create the combined high-noise figure with two panels."""
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.4), dpi=300, constrained_layout=True)

    plot_accuracy_vs_p(
        axes[0],
        csv_path=accuracy_csv,
        activation=activation,
        baseline=baseline,
        panel_label="(a)",
        analytical_csv=analytical_csv,
    )
    plot_actual_vs_predicted_scatter(
        axes[1],
        csv_path=comparison_csv,
        json_path=comparison_json,
        panel_label="(b)",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_high_noise_single_column(
    *,
    accuracy_csv: Path,
    comparison_csv: Path,
    comparison_json: Path,
    output_path: Path,
    activation: str = "erf",
    baseline: float = 76.42,
    analytical_csv: Path | None = None,
) -> None:
    """Create a single-column figure with an inset scatter panel."""
    fig, ax = plt.subplots(figsize=(3.375, 2.4), dpi=300, constrained_layout=True)

    plot_accuracy_vs_p(
        ax,
        csv_path=accuracy_csv,
        activation=activation,
        baseline=baseline,
        panel_label=None,
        analytical_csv=analytical_csv,
    )
    line = ax.lines[0] if ax.lines else None
    if line is not None:
        line.set_clip_on(True)

    ax.set_ylim(0, 80)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.legend(frameon=True, handlelength=1.6, loc="upper right", bbox_to_anchor=(1.0, 0.96))

    inset_ax = ax.inset_axes([0.2, 0.18, 0.3, 0.455])
    plot_actual_vs_predicted_scatter(
        inset_ax,
        csv_path=comparison_csv,
        json_path=comparison_json,
        panel_label=None,
        scatter_size=8.0,
        r2_fontsize=8,
    )
    inset_ax.tick_params(labelsize=6, pad=1)
    inset_ax.xaxis.label.set_size(8)
    inset_ax.yaxis.label.set_size(8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    configure_plot_style()

    repo_root = Path(__file__).resolve().parent.parent
    accuracy_csv = repo_root / "actual_ vs_predicted_vs_p" / "summary_test_accuracy_by_p_activation.csv"
    analytical_csv = repo_root / "actual_ vs_predicted_vs_p" / "analytical_abc_accuracy_summary_fit_c.csv"
    comparison_base = repo_root / "actual_vs_predicted_scatter" / "subset__test=1000"
    comparison_csv = comparison_base / "per_test_class_comparison.csv"
    comparison_json = comparison_base / "comparison_database.json"
    output_path = repo_root / "figures" / "high_noise_accuracy_with_inset.pdf"

    plot_high_noise_single_column(
        accuracy_csv=accuracy_csv,
        comparison_csv=comparison_csv,
        comparison_json=comparison_json,
        output_path=output_path,
        analytical_csv=analytical_csv,
    )
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
