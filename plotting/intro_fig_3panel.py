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
	"""Replace each pixel with Uniform(0,1) noise with probability p."""
	if not (0.0 <= p <= 1.0):
		raise ValueError(f"p must be in [0, 1], got {p}")

	keep_mask = rng.random(image.shape) >= p
	random_pixels = rng.random(image.shape)
	return np.where(keep_mask, image, random_pixels)


def plot_accuracy_with_digit_strip_panel(
	fig: plt.Figure,
	panel_spec,
	df: pd.DataFrame,
	data_root: Path,
	dataset_cls,
	panel_label: str,
	panel_name: str,
	show_noise_label: bool,
	show_y_label: bool,
	seed: int,
	data_index: int = 41,
	p_examples: tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00),
	show_digit_markers: bool = False,
	label_x: float = -0.25,
) -> None:
	"""Plot one accuracy-vs-p panel with noisy digits below the x-axis."""
	required_cols = {"p", "mean_train_accuracy", "mean_test_accuracy"}
	missing_cols = required_cols - set(df.columns)
	if missing_cols:
		raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

	try:
		dataset = dataset_cls(root=data_root, train=True, download=(dataset_cls != datasets.KMNIST))
		original_pil, _ = dataset[data_index]
		original = np.asarray(original_pil, dtype=np.float32) / 255.0
	except Exception:
		if dataset_cls == datasets.KMNIST:
			npz_path = Path(data_root) / "KMNIST" / "kmnist-train-imgs.npz"
			imgs = np.load(npz_path)["arr_0"]
			original = imgs[data_index].astype(np.float32) / 255.0
		else:
			raise

	rng = np.random.default_rng(seed)

	inner = panel_spec.subgridspec(
		2,
		len(p_examples),
		height_ratios=(3.0, 0.9),
		hspace=0.45,
		wspace=0.05,
	)

	ax = fig.add_subplot(inner[0, :])
	ax.plot(df["p"], 100.0 * df["mean_train_accuracy"], marker="o", markersize=3.0, linewidth=1.1, label="Train", c="#1402a0")
	ax.plot(df["p"], 100.0 * df["mean_test_accuracy"], marker="o", markersize=3.0, linewidth=1.1, label="Test", c="#b31b1b")
	ax.axhline(10.0, color="0.35", linestyle=":", linewidth=0.9)
	ax.text(0.48, 13, "Random guess", ha="left", va="bottom", fontsize=8)
	ax.set_xlabel(r"Corruption probability $p$")
	
	if show_y_label:
		ax.set_ylabel("Accuracy (%)")
		
	ax.set_xticks(np.asarray(p_examples, dtype=float))
	ax.set_ylim(0.0, 105.0)
	ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
	ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
	


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

	if show_digit_markers:
		y_mark = 100.0 * np.interp(np.asarray(p_examples), df["p"], df["mean_test_accuracy"])
		ax.scatter(p_examples, y_mark, s=18, marker="v", color="black", zorder=4)

	ax.text(
		label_x,
		1.05,
		panel_label,
		transform=ax.transAxes,
		ha="left",
		va="top",
		fontsize=9,
		clip_on=False,
	)

	for i, p in enumerate(p_examples):
		img_ax = fig.add_subplot(inner[1, i])
		noise_rng = np.random.default_rng(seed + i)
		noisy = apply_replacement_noise(original, p=p, rng=noise_rng)
		img_ax.imshow(noisy, cmap="gray", vmin=0.0, vmax=1.0)
		img_ax.text(0.5, -0.12, rf"$p={p:g}$", transform=img_ax.transAxes, ha="center", va="top", fontsize=6.5)
		img_ax.axis("off")
		
		if show_noise_label and i == 0:
			img_ax.text(-1.1, 0.5, "Noise\nmodel:", transform=img_ax.transAxes, ha="left", va="center")


def plot_intro_figure(
	csv_path: Path,
	data_root: Path,
	output_path: Path,
	seed: int = 8,
	p_examples: tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00),
	show_digit_markers: bool = False,
) -> None:
	"""Plot MNIST, FMNIST, and KMNIST corruption figures side by side."""
	df_all = pd.read_csv(csv_path)

	fig = plt.figure(figsize=(6.75, 2.8), dpi=300)
	
	outer = fig.add_gridspec(1, 3, wspace=0.25)

	# MNIST
	plot_accuracy_with_digit_strip_panel(
		fig=fig,
		panel_spec=outer[0],
		df=df_all[df_all["dataset_name"] == "mnist"],
		data_root=data_root,
		dataset_cls=datasets.MNIST,
		panel_label="(a)",
		panel_name="MNIST",
		show_noise_label=True,
		show_y_label=True,
		seed=seed,
		p_examples=p_examples,
		show_digit_markers=show_digit_markers,
		label_x=-0.26,
	)
	
	# FMNIST
	plot_accuracy_with_digit_strip_panel(
		fig=fig,
		panel_spec=outer[1],
		df=df_all[df_all["dataset_name"] == "fmnist"],
		data_root=data_root,
		dataset_cls=datasets.FashionMNIST,
		panel_label="(b)",
		panel_name="Fashion-MNIST",
		show_noise_label=False,
		show_y_label=False,
		seed=seed + 1,
		p_examples=p_examples,
		show_digit_markers=show_digit_markers,
		label_x=-0.24,
	)

	# KMNIST
	plot_accuracy_with_digit_strip_panel(
		fig=fig,
		panel_spec=outer[2],
		df=df_all[df_all["dataset_name"] == "kmnist"],
		data_root=data_root,
		dataset_cls=datasets.KMNIST,
		panel_label="(c)",
		panel_name="KMNIST",
		show_noise_label=False,
		show_y_label=False,
		seed=seed + 2,
		p_examples=p_examples,
		show_digit_markers=show_digit_markers,
		label_x=-0.24,
	)

	fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.9)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
	plt.close(fig)


def main() -> None:
	configure_plot_style()

	repo_root = Path(__file__).resolve().parent.parent
	csv_path = repo_root / "mnist_data" / "results_summary_mnist-kmnist-fmnist_balanced_noise_curve_r100_e20_p0.00-1.00_w128_d3.csv"
	output_path = repo_root / "figures" / "intro_fig_3panel.pdf"
	data_root = Path("/Users/omrile/Desktop/ML-Datasets")

	plot_intro_figure(
		csv_path=csv_path,
		data_root=data_root,
		output_path=output_path,
		seed=8,
	)
	print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
	main()
