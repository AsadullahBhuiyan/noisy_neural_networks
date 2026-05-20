from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (3.375, 2.4),
            "font.family": "CMU Sans Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "text.usetex": False,
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "CMU Sans Serif",
            "mathtext.it": "CMU Sans Serif:italic",
            "mathtext.bf": "CMU Sans Serif:bold",
        }
    )


def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mean_acc(p: float, C: int, a: float = 1.0, n_gh: int = 80) -> float:
    xs, ws = np.polynomial.hermite.hermgauss(n_gh)
    shift = a * (1.0 - p)
    s = 0.0
    for x, w in zip(xs, ws):
        u = math.sqrt(2.0) * x
        s += w * Phi(u + shift) ** (C - 1)
    return s / math.sqrt(math.pi)


def std_acc(p: float, C: int, a: float = 1.0) -> float:
    m = mean_acc(p, C, a)
    return math.sqrt(max(0.0, m * (1.0 - m)))


def main() -> None:
    configure_plot_style()

    a = 10.0
    C_list = [10]
    p_vals = np.linspace(0, 1, 300)

    fig, ax = plt.subplots(constrained_layout=True)

    for C in C_list:
        y = [std_acc(p, C, a) for p in p_vals]
        ax.plot(p_vals, y, linewidth=2, color=[0.35]*3)

    ax.set_xlabel(r"Corruption probability $p$")
    ax.set_ylabel(r"Standard deviation of test accuracy")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))

    output_path = Path(__file__).resolve().parent.parent / "figures" / "non_monotonic_test_acc_stdv.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
