#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "CMU Sans Serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.usetex": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "CMU Sans Serif",
            "mathtext.it": "CMU Sans Serif:italic",
            "mathtext.bf": "CMU Sans Serif:bold",
        }
    )


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "p_train": float(r["p_train"]),
                    "p_test": float(r["p_test"]),
                    "test_accuracy": float(r["test_accuracy"]),
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


def aggregate(
    rows: list[dict], x_var: str, group_var: str
) -> dict[float, list[tuple[float, float, float]]]:
    """Aggregate test_accuracy into series keyed by group_var, x_var on the x-axis."""
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)
    for r in rows:
        grouped[(r[group_var], r[x_var])].append(r["test_accuracy"])

    series: dict[float, list[tuple[float, float, float]]] = defaultdict(list)
    for (group_val, x_val), vals in grouped.items():
        mean, stderr = mean_and_stderr(vals)
        series[group_val].append((x_val, mean, stderr))
    for g in series:
        series[g].sort(key=lambda t: t[0])
    return dict(sorted(series.items()))


def compute_auc_summary(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_trains, mean_auc, stderr_auc) where AUC is over p_test per run."""
    by_run: dict[tuple[float, int], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        by_run[(r["p_train"], r["run_id"])].append((r["p_test"], r["test_accuracy"]))

    auc_by_ptrain: dict[float, list[float]] = defaultdict(list)
    for (p_train, _), points in by_run.items():
        points.sort(key=lambda t: t[0])
        xs = np.array([p for p, _ in points], dtype=float)
        ys = np.array([acc for _, acc in points], dtype=float)
        auc_by_ptrain[p_train].append(float(np.trapz(ys, xs)))

    p_trains = sorted(auc_by_ptrain)
    means, stderrs = [], []
    for p in p_trains:
        vals = np.array(auc_by_ptrain[p], dtype=float)
        means.append(float(np.mean(vals)))
        stderrs.append(0.0 if vals.size <= 1 else float(np.std(vals, ddof=1) / np.sqrt(vals.size)))
    return np.array(p_trains), np.array(means), np.array(stderrs)


def draw_curves_panel(
    ax: plt.Axes,
    fig: plt.Figure,
    series: dict,
    group_vals: np.ndarray,
    norm: plt.Normalize,
    cmap,
    x_label: str,
    cbar_label: str,
    show_max: bool = False,
) -> None:
    for g in group_vals:
        points = series[float(g)]
        xs = np.array([x for x, _, _ in points])
        ys = 100.0 * np.array([acc for _, acc, _ in points])
        color = cmap(norm(float(g)))
        ax.plot(xs, ys, color=color, linewidth=0.8)
        if show_max:
            best = int(np.argmax(ys))
            ax.scatter(xs[best], ys[best], color=color, s=12, zorder=5, linewidths=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 105.0)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))


def draw_auc_panel(ax: plt.Axes, rows: list[dict]) -> None:
    xs, ys, ses = compute_auc_summary(rows)
    ys_pct = 100.0 * ys
    ses_pct = 100.0 * ses
    best_idx = int(np.argmax(ys_pct))

    ax.plot(xs, ys_pct, linewidth=1.5, color="0.35")
    ax.fill_between(xs, ys_pct - ses_pct, ys_pct + ses_pct, color="0.3", alpha=0.15, linewidth=0)
    ax.axvline(xs[best_idx], linestyle="--", color="0.35", linewidth=1,
               label=rf"$p_\mathrm{{train}}^\mathrm{{opt}}\approx{xs[best_idx]:g}$")
    ax.legend(frameon=False, handlelength=1.5)

    ax.set_xlabel(r"Train corruption $p_\mathrm{train}$")
    ax.set_ylabel(r"AUC of test accuracy (%)")
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))


def make_plot(
    csv_path: Path,
    output_path: Path,
    mode: str,
    p_filter: list[float] | None,
    n_curves: int | None = None,
    sbs: bool = False,
) -> None:
    rows = load_rows(csv_path)

    if mode == "vs-p-test":
        x_var, group_var = "p_test", "p_train"
        x_label = r"Test corruption $p_\mathrm{test}$"
        cbar_label = r"Train corruption $p_\mathrm{train}$"
    else:
        x_var, group_var = "p_train", "p_test"
        x_label = r"Train corruption $p_\mathrm{train}$"
        cbar_label = r"Test corruption $p_\mathrm{test}$"

    series = aggregate(rows, x_var=x_var, group_var=group_var)

    if p_filter is not None:
        series = {k: v for k, v in series.items() if any(abs(k - pf) < 1e-9 for pf in p_filter)}
        if not series:
            raise ValueError(f"No series matched --p-values {p_filter}. Available: {sorted(aggregate(rows, x_var=x_var, group_var=group_var))}")
    elif n_curves is not None and len(series) > n_curves:
        all_keys = sorted(series)
        indices = np.round(np.linspace(0, len(all_keys) - 1, n_curves)).astype(int)
        series = {all_keys[i]: series[all_keys[i]] for i in indices}

    group_vals = np.array(sorted(series), dtype=float)
    norm = plt.Normalize(vmin=float(group_vals.min()), vmax=float(group_vals.max()))
    cmap = plt.get_cmap("viridis")

    width = 6.75 if sbs else 3.375
    ncols = 2 if sbs else 1
    fig, axes = plt.subplots(1, ncols, figsize=(width, 2.08), constrained_layout=True,
                             gridspec_kw={"wspace": 0.05} if sbs else {})
    ax_curves = axes[0] if sbs else axes

    draw_curves_panel(ax_curves, fig, series, group_vals, norm, cmap, x_label, cbar_label,
                      show_max=(mode == "vs-p-train"))
    if sbs:
        draw_auc_panel(axes[1], rows)
        for ax, label in zip(axes, ("(a)", "(b)")):
            ax.text(-0.19, 0.95, label, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path.resolve()}")


_SCRIPTS_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _SCRIPTS_DIR.parent
_DEFAULT_CSV = _MODULE_DIR / "data" / "ptrain_ptest_full_merged.csv"
_FIGURES_DIR = _MODULE_DIR / "figures"


def default_output(mode: str, sbs: bool = False) -> Path:
    stem = "ptrain_ptest_curves_" + mode.replace("-", "_")
    if sbs:
        stem += "_sbs"
    return _FIGURES_DIR / f"{stem}.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot test accuracy as curves of p_train or p_test."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=f"CSV with p_train, p_test, test_accuracy columns (default: {_DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (.pdf or .png); inferred from --mode if omitted",
    )
    parser.add_argument(
        "--mode",
        choices=["vs-p-test", "vs-p-train"],
        default="vs-p-test",
        help=(
            "vs-p-test (default): x-axis=p_test, one curve per p_train value; "
            "vs-p-train: x-axis=p_train, one curve per p_test value"
        ),
    )
    parser.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=None,
        metavar="P",
        help="Restrict to this subset of the grouped variable (e.g. --p-values 0.0 0.25 0.5 0.75 1.0)",
    )
    parser.add_argument(
        "--sbs",
        action="store_true",
        help="Side-by-side: add a second panel with AUC of test accuracy vs p_train (figure doubled in width)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else _DEFAULT_CSV
    output_path = Path(args.output) if args.output else default_output(args.mode, sbs=args.sbs)

    n_curves = 20 if (args.mode == "vs-p-train" and args.p_values is None) else None

    configure_plot_style()
    make_plot(
        csv_path=csv_path,
        output_path=output_path,
        mode=args.mode,
        p_filter=args.p_values,
        n_curves=n_curves,
        sbs=args.sbs,
    )


if __name__ == "__main__":
    main()
