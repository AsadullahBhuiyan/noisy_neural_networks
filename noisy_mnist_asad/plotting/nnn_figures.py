from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import datasets
from matplotlib.ticker import StrMethodFormatter


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
			"mathtext.fontset": "custom",
			"mathtext.rm": "CMU Sans Serif",
			"mathtext.it": "CMU Sans Serif:italic",
			"mathtext.bf": "CMU Sans Serif:bold",
		}
	)

def plot_accuracy_vs_corruption(csv_path: Path, output_path: Path) -> None:
	"""Plot mean train/test accuracy against corruption probability."""
	df = pd.read_csv(csv_path)

	required_cols = {"p", "mean_train_acc", "mean_test_acc"}
	missing_cols = required_cols - set(df.columns)
	if missing_cols:
		raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing_cols)}")

	fig, ax = plt.subplots()
	ax.plot(
		df["p"],
		df["mean_train_acc"],
		marker="o",
		markersize=3.5,
		linewidth=1.2,
		label="Train",
	)
	ax.plot(
		df["p"],
		df["mean_test_acc"],
		marker="o",
		markersize=3.5,
		linewidth=1.2,
		label="Test",
	)

	ax.set_xlabel(r"Corruption probability $p$")
	ax.set_ylabel("Accuracy")
	# ax.set_xlim(0.0, 1.0)
	# ax.set_ylim(0.05, 1.03)
	ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
	ax.legend(frameon=True, handlelength=1.8)

	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path)
	plt.close(fig)


def apply_replacement_noise(image: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
	"""Replace each pixel with Uniform(0,1) noise with probability p."""
	if not (0.0 <= p <= 1.0):
		raise ValueError(f"p must be in [0, 1], got {p}")

	keep_mask = rng.random(image.shape) >= p
	random_pixels = rng.random(image.shape)
	return np.where(keep_mask, image, random_pixels)


def plot_mnist_replacement_noise_example(
	data_root: Path, output_path: Path, p: float = 0.5, seed: int = 0
) -> None:
	"""Load one MNIST digit, apply replacement noise, and visualize the result."""
	mnist = datasets.MNIST(root=data_root, train=True, download=True)
	original_pil, label = mnist[41]
	original = np.asarray(original_pil, dtype=np.float32) / 255.0

	rng = np.random.default_rng(seed)
	noisy = apply_replacement_noise(original, p=p, rng=rng)

	fig, axes = plt.subplots(1, 2, figsize=(3.375, 1.8), dpi=300)
	for ax, image, title in (
		(axes[0], original, f"Original (label={label})"),
		(axes[1], noisy, rf"Replacement noise, $p={p:.2f}$"),
	):
		ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
		ax.set_title(title)
		ax.axis("off")

	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path)
	plt.close(fig)


def plot_accuracy_with_digit_strip(
	csv_path: Path,
	data_root: Path,
	output_path: Path,
	p_examples: tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00),
	seed: int = 8,
	mnist_id: int = 41,
) -> None:
	"""Plot accuracy vs p with noisy mnist[41] examples below the x-axis."""
	df = pd.read_csv(csv_path)
	required_cols = {"p", "mean_train_acc", "mean_test_acc"}
	missing_cols = required_cols - set(df.columns)
	if missing_cols:
		raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing_cols)}")

	mnist = datasets.MNIST(root=data_root, train=True, download=True)
	original_pil, label = mnist[mnist_id]
	original = np.asarray(original_pil, dtype=np.float32) / 255.0

	fig = plt.figure(figsize=(3.375, 2.5), dpi=300, constrained_layout=True)
	fig.set_constrained_layout_pads(hspace=0.01, h_pad=0.01, w_pad=0.01)
	gs = fig.add_gridspec(
		2,
		len(p_examples),
		height_ratios=(3.0, 0.9),
		hspace=0.05,
		wspace=0.05,
	)

	ax = fig.add_subplot(gs[0, :])
	ax.plot(df["p"], df["mean_train_acc"], marker="o", markersize=3.0, linewidth=1.1, label="Train", c="tab:blue")
	ax.plot(df["p"], df["mean_test_acc"], marker="o", markersize=3.0, linewidth=1.1, label="Test", c="tab:red")
	ax.set_xlabel(r"Corruption probability $p$")#, labelpad=0.2)
	ax.set_ylabel("Accuracy")
	ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
	ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
	ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
	ax.legend(frameon=True, handlelength=1.8, loc="lower left")

	y_mark = np.interp(np.asarray(p_examples), df["p"], df["mean_test_acc"])
	ax.scatter(p_examples, y_mark, s=18, marker="v", color="black", zorder=4)

	fig.text(0.023, 0.165, "Noise\nmodel:", ha="left", va="center")

	for i, p in enumerate(p_examples):
		img_ax = fig.add_subplot(gs[1, i])
		rng = np.random.default_rng(seed + i)
		noisy = apply_replacement_noise(original, p=p, rng=rng)
		img_ax.imshow(noisy, cmap="gray", vmin=0.0, vmax=1.0)
		img_ax.text(0.5, -0.09, rf"$p={p:g}$", transform=img_ax.transAxes, ha="center", va="top")
		img_ax.axis("off")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
	plt.close(fig)


def main() -> None:
	configure_plot_style()

	repo_root = Path(__file__).resolve().parent.parent
	csv_path = repo_root / "mock_data" / "mock_accuracy_vs_corruption_dense.csv"
	output_path = repo_root / "figures" / "accuracy_vs_corruption_with_digit_strip.pdf"
	mnist_data_root = repo_root / "data"

	plot_accuracy_with_digit_strip(
		csv_path=csv_path,
		data_root=mnist_data_root,
		output_path=output_path,
		seed=8,
	)
	print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
	main()
