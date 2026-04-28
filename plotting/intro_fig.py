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
		100.0 * df["mean_train_acc"],
		marker="o",
		markersize=3.5,
		linewidth=1.2,
		label="Train",
	)
	ax.plot(
		df["p"],
		100.0 * df["mean_test_acc"],
		marker="o",
		markersize=3.5,
		linewidth=1.2,
		label="Test",
	)
	ax.axhline(10.0, color="0.35", linestyle=":", linewidth=0.9)
	ax.text(0.48, 10.8, "Random guess", ha="left", va="bottom", fontsize=8)

	ax.set_xlabel(r"Corruption probability $p$")
	ax.set_ylabel("Accuracy (%)")
	ax.set_xlim(left=0.0)
	ax.set_ylim(bottom=0.0)
	# ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
	ax.legend(frameon=True, handlelength=1.8, loc="center left", bbox_to_anchor=(0.02, 0.3))

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


def plot_accuracy_with_digit_strip_panel(
	fig: plt.Figure,
	panel_spec,
	csv_path: Path,
	data_root: Path,
	dataset_cls,
	panel_label: str,
	panel_name: str,
	show_noise_label: bool,
	seed: int,
	data_index: int = 41,
	p_examples: tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00),
) -> None:
	"""Plot one accuracy-vs-p panel with noisy digits below the x-axis."""
	df = pd.read_csv(csv_path)
	required_cols = {"p", "mean_train_acc", "mean_test_acc"}
	missing_cols = required_cols - set(df.columns)
	if missing_cols:
		raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing_cols)}")

	dataset = dataset_cls(root=data_root, train=True, download=True)
	rng = np.random.default_rng(seed)
	original_pil, _ = dataset[data_index]
	original = np.asarray(original_pil, dtype=np.float32) / 255.0

	inner = panel_spec.subgridspec(
		2,
		len(p_examples),
		height_ratios=(3.0, 0.9),
		hspace=0.05,
		wspace=0.05,
	)

	ax = fig.add_subplot(inner[0, :])
	ax.plot(df["p"], 100.0 * df["mean_train_acc"], marker="o", markersize=3.0, linewidth=1.1, label="Train", c="#1402a0")
	ax.plot(df["p"], 100.0 * df["mean_test_acc"], marker="o", markersize=3.0, linewidth=1.1, label="Test", c="#b31b1b")
	ax.axhline(10.0, color="0.35", linestyle=":", linewidth=0.9)
	ax.text(0.48, 13, "Random guess", ha="left", va="bottom", fontsize=8)
	ax.set_xlabel(r"Corruption probability $p$")#, labelpad=0.2)
	ax.set_ylabel("Accuracy (%)")
	# ax.set_xlim(left=0.0)
	ax.set_ylim(bottom=0.0)
	ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
	ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
	# ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
	ax.legend(frameon=True, handlelength=1.8, loc="center left", bbox_to_anchor=(0.02, 0.3))
	ax.text(
		0.05,
		0.65,
		panel_name,
		transform=ax.transAxes,
		ha="left",
		va="center",
		fontsize=9,
		fontweight="bold",
		bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.15),
	)

	y_mark = 100.0 * np.interp(np.asarray(p_examples), df["p"], df["mean_test_acc"])
	ax.scatter(p_examples, y_mark, s=18, marker="v", color="black", zorder=4)

	ax.text(
		-0.17,
		1.02,
		panel_label,
		transform=ax.transAxes,
		ha="left",
		va="top",
		fontsize=9,
		clip_on=False,
	)

	panel_bbox = panel_spec.get_position(fig)

	if show_noise_label:
		fig.text(panel_bbox.x0 - 0.115, panel_bbox.y0 + 0.05, "Noise\nmodel:", ha="left", va="center")

	for i, p in enumerate(p_examples):
		img_ax = fig.add_subplot(inner[1, i])
		noise_rng = np.random.default_rng(seed + i)
		noisy = apply_replacement_noise(original, p=p, rng=noise_rng)
		img_ax.imshow(noisy, cmap="gray", vmin=0.0, vmax=1.0)
		img_ax.text(0.5, -0.09, rf"$p={p:g}$", transform=img_ax.transAxes, ha="center", va="top")
		img_ax.axis("off")


def plot_intro_figure(
	mnist_csv_path: Path,
	kmnist_csv_path: Path,
	data_root: Path,
	output_path: Path,
	seed: int = 8,
	p_examples: tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00),
) -> None:
	"""Plot MNIST and KMNIST corruption figures side by side."""
	fig = plt.figure(figsize=(6.75, 2.5), dpi=300, constrained_layout=True)
	fig.set_constrained_layout_pads(hspace=0.01, h_pad=0.01, w_pad=0.01)
	outer = fig.add_gridspec(1, 3, width_ratios=(1.0, 0.05, 1.0), wspace=0.0)

	plot_accuracy_with_digit_strip_panel(
		fig=fig,
		panel_spec=outer[0],
		csv_path=mnist_csv_path,
		data_root=data_root,
		dataset_cls=datasets.MNIST,
		panel_label="(a)",
		panel_name="MNIST",
		show_noise_label=True,
		seed=seed,
		p_examples=p_examples,
	)
	plot_accuracy_with_digit_strip_panel(
		fig=fig,
		panel_spec=outer[2],
		csv_path=kmnist_csv_path,
		data_root=data_root,
		dataset_cls=datasets.KMNIST,
		panel_label="(b)",
		panel_name="KMNIST",
		show_noise_label=False,
		seed=seed + 1,
		p_examples=p_examples,
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
	plt.close(fig)


def main() -> None:
	configure_plot_style()

	repo_root = Path(__file__).resolve().parent.parent
	mnist_csv_path = repo_root / "mock_data" / "mock_accuracy_vs_corruption_mnist.csv"
	kmnist_csv_path = repo_root / "mock_data" / "mock_accuracy_vs_corruption_kmnist.csv"
	output_path = repo_root / "figures" / "accuracy_vs_corruption_mnist_kmnist.pdf"
	mnist_data_root = repo_root / "noisy_mnist_asad" / "data"

	plot_intro_figure(
		mnist_csv_path=mnist_csv_path,
		kmnist_csv_path=kmnist_csv_path,
		data_root=mnist_data_root,
		output_path=output_path,
		seed=8,
	)
	print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
	main()
