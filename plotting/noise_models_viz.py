from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from torchvision import datasets


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


def apply_replacement_noise(image: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Replace each pixel independently with Uniform(0,1) noise with probability p."""
    keep_mask = rng.random(image.shape) >= p
    return np.where(keep_mask, image, rng.random(image.shape))


def apply_additive_gaussian_noise(image: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise: x_tilde = (1-p)*x + p*xi, xi ~ N(0,1), clipped to [0,1]."""
    xi = rng.standard_normal(image.shape).astype(np.float32)
    return np.clip((1.0 - p) * image + p * xi, 0.0, 1.0)


def _arrow_with_label(
    fig: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    label_side: int,  # +1 = above arrow, -1 = below arrow
    fontsize: float = 7.0,
    color: str = "0.25",
    offset_pts: float = 7.0,
) -> None:
    """Draw a FancyArrowPatch in figure-fraction coords and place a rotated label beside it."""
    fig_w_in, fig_h_in = fig.get_size_inches()
    fig_dpi = fig.dpi

    # Arrow
    arrow = mpatches.FancyArrowPatch(
        start, end,
        transform=fig.transFigure,
        arrowstyle=mpatches.ArrowStyle("->", head_length=4, head_width=2.5),
        color=color,
        lw=0.75,
        shrinkA=2,
        shrinkB=2,
    )
    fig.add_artist(arrow)

    # Compute arrow angle in display (pixel) space for correct text rotation
    start_px = fig.transFigure.transform(start)
    end_px = fig.transFigure.transform(end)
    dx_px = end_px[0] - start_px[0]
    dy_px = end_px[1] - start_px[1]
    angle_deg = np.degrees(np.arctan2(dy_px, dx_px))
    arrow_len_px = np.hypot(dx_px, dy_px)

    # Unit perpendicular (counterclockwise rotation of arrow direction)
    uperp_x = -dy_px / arrow_len_px
    uperp_y = dx_px / arrow_len_px

    # Midpoint in figure fraction + perpendicular offset
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    label_x = mid_x + label_side * uperp_x * offset_pts / (fig_w_in * fig_dpi)
    label_y = mid_y + label_side * uperp_y * offset_pts / (fig_h_in * fig_dpi)

    va = "bottom" if label_side > 0 else "top"
    fig.text(
        label_x, label_y,
        label,
        ha="center", va=va,
        fontsize=fontsize,
        rotation=angle_deg,
        rotation_mode="anchor",
        color=color,
    )


def plot_noise_models_figure(
    data_root: Path,
    output_path: Path,
    seed: int = 8,
    data_index: int = 41,
    p_values: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
) -> None:
    """
    Plot an MNIST digit with two branching arrows showing replacement and
    additive Gaussian noise corruptions for several values of p.

    Layout (in figure-fraction space):
        [Original]──Replacement──▶ [p=0.25] [p=0.50] [p=0.75] [p=1.00]
                 ╲──Additive──────▶ [p=0.25] [p=0.50] [p=0.75] [p=1.00]
    """
    dataset = datasets.MNIST(root=data_root, train=True, download=False)
    original_pil, _ = dataset[data_index]
    original = np.asarray(original_pil, dtype=np.float32) / 255.0

    n_p = len(p_values)
    noise_models = [
        ("Replacement", apply_replacement_noise),
        ("Additive", apply_additive_gaussian_noise),
    ]
    n_models = len(noise_models)

    fig = plt.figure(figsize=(3.375, 1.5), dpi=300)

    # ── Original image ──────────────────────────────────────────────────────
    # Positioned at far left, vertically centered between the two noisy rows.
    # Coordinates are in figure fraction [left, bottom, width, height].
    orig_left, orig_width, orig_height = 0.01, 0.17, 0.54
    orig_bottom = 0.50 - orig_height / 2  # vertically centred at y=0.50
    ax_orig = fig.add_axes([orig_left, orig_bottom, orig_width, orig_height])
    ax_orig.imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    ax_orig.axis("off")
    ax_orig.text(
        0.5, -0.10,
        "Original",
        transform=ax_orig.transAxes,
        ha="center", va="top", fontsize=7.5,
    )

    # ── 2 × n_p grid of noisy images ────────────────────────────────────────
    gs = GridSpec(
        n_models, n_p,
        left=0.39, right=0.99,
        top=0.96, bottom=0.12,
        hspace=0.10, wspace=0.05,
        figure=fig,
    )

    noisy_axes: list[list[plt.Axes]] = []
    for row, (_, noise_fn) in enumerate(noise_models):
        row_axes: list[plt.Axes] = []
        for col, p in enumerate(p_values):
            ax = fig.add_subplot(gs[row, col])
            noise_rng = np.random.default_rng(seed + row * 100 + col)
            noisy = noise_fn(original, p=p, rng=noise_rng)
            ax.imshow(noisy, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row == n_models - 1:
                ax.text(
                    0.5, -0.10,
                    rf"$p={p:g}$",
                    transform=ax.transAxes,
                    ha="center", va="top", fontsize=7,
                )
            row_axes.append(ax)
        noisy_axes.append(row_axes)

    # ── Arrows + labels ──────────────────────────────────────────────────────
    # Compute axis positions only after the canvas layout is resolved.
    fig.canvas.draw()

    orig_pos = ax_orig.get_position()
    start = (orig_pos.x1, (orig_pos.y0 + orig_pos.y1) / 2)

    for row, (label_text, _) in enumerate(noise_models):
        first_pos = noisy_axes[row][0].get_position()
        end = (first_pos.x0, (first_pos.y0 + first_pos.y1) / 2)
        label_side = +1 if row == 0 else -1  # above upper arrow, below lower
        _arrow_with_label(fig, start, end, label_text, label_side=label_side, offset_pts=10.0, fontsize=8)

    # ────────────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def main() -> None:
    configure_plot_style()

    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "figures" / "noise_models_viz.pdf"
    data_root = Path("/Users/omrile/Desktop/ML-Datasets")

    plot_noise_models_figure(
        data_root=data_root,
        output_path=output_path,
        seed=8,
        data_index=50,
    )


if __name__ == "__main__":
    main()
