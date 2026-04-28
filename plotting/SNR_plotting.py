from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def load_normalized_snr_curves(
	csv_path: Path,
	*,
	label: int,
	x_col: str,
	std_col: str,
) -> tuple[dict[int, list[tuple[int, float]]], list[int]]:
	"""Load per-test curves and return normalized SNR points and sorted x ticks."""
	df = pd.read_csv(csv_path)
	if "true_label" not in df.columns or "test_index" not in df.columns:
		raise ValueError("Expected columns 'true_label' and 'test_index' in the input CSV.")
	if x_col not in df.columns or std_col not in df.columns:
		raise ValueError(f"Missing columns in {csv_path}: {x_col} or {std_col}.")
	mean_col = f"mean_label_{label}"
	if mean_col not in df.columns:
		raise ValueError(f"Missing column in {csv_path}: {mean_col}.")

	curves: dict[int, list[tuple[int, float]]] = defaultdict(list)
	x_values = set()
	for row in df.itertuples(index=False):
		if int(getattr(row, "true_label")) != label:
			continue
		x_val = int(getattr(row, x_col))
		mean = float(getattr(row, mean_col))
		std = float(getattr(row, std_col))
		snr = mean * mean / (std * std)
		curves[int(getattr(row, "test_index"))].append((x_val, snr))
		x_values.add(x_val)

	return curves, sorted(x_values)


def plot_normalized_snr_panel(
	ax: plt.Axes,
	curves: dict[int, list[tuple[int, float]]],
	*,
	xticks: list[int],
	xlabel: str,
	panel_label: str,
	panel_label_x: float = -0.2,
	panel_label_y: float = 1.02
) -> None:
	"""Plot normalized SNR curves with a diagonal reference line."""
	for pts in curves.values():
		pts = sorted(pts)
		x = np.asarray([p[0] for p in pts], dtype=float)
		snr = np.asarray([p[1] for p in pts], dtype=float)
		a, b = np.polyfit(x, snr, 1)
		ax.plot(
			x,
			(snr - b) / a,
			linewidth=1.1,
			alpha=0.3,
			color="tab:blue",
			marker="o",
			markersize=3.0,
		)

	ax.plot(xticks, xticks, color="black", linestyle="--", linewidth=1.4)
	ax.set_xticks(xticks)
	ax.set_xticklabels([str(v) for v in xticks])
	ax.locator_params(axis="y", nbins=4)
	ax.set_xlabel(xlabel)
	ax.set_ylabel("Normalized SNR")
	ax.text(
		panel_label_x,
		panel_label_y,
		panel_label,
		transform=ax.transAxes,
		ha="left",
		va="top",
		fontsize=9,
		clip_on=False,
	)


def plot_snr_scaling_figure(
	*,
	d_csv_path: Path,
	n_csv_path: Path,
	output_path: Path,
	label_d: int = 9,
	label_n: int = 6,
) -> None:
	"""Plot collapsed normalized SNR vs d and N in a single figure."""
	fig, axes = plt.subplots(1, 2, figsize=(6.75, 1.6), dpi=300, constrained_layout=False)

	# Shared placement and styling for the overlay texts (easy to tweak)
	text_x = 0.65
	text_y = 0.18
	text_bbox = dict(boxstyle="round,pad=0.3", facecolor="0.95", edgecolor="0.8", linewidth=0.6)
	text_kwargs = dict(ha="center", va="bottom", fontsize=8, bbox=text_bbox)
	# Add a bit more horizontal whitespace between the two panels
	fig.subplots_adjust(wspace=0.35)

	curves_d, xticks_d = load_normalized_snr_curves(
		d_csv_path,
		label=label_d,
		x_col="feature_dim",
		std_col="avg_std_all_testpoints_labels_for_d",
	)
	curves_n, xticks_n = load_normalized_snr_curves(
		n_csv_path,
		label=label_n,
		x_col="train_size",
		std_col="avg_std_all_testpoints_labels_for_N",
	)

	plot_normalized_snr_panel(
		axes[0],
		curves_n,
		xticks=xticks_n,
		xlabel="Training set size $N$",
		panel_label="(a)",
		panel_label_x=-0.22,
	)
	# Overlay fixed parameter text for panel (a)
	axes[0].text(
		text_x,
		text_y,
		"Feature dimension\n$d=784$",
		transform=axes[0].transAxes,
		**text_kwargs,
	)
	plot_normalized_snr_panel(
		axes[1],
		curves_d,
		xticks=xticks_d,
		xlabel="Feature dimension $d$",
		panel_label="(b)",
		panel_label_x=-0.25,
	)
	# Overlay fixed parameter text for panel (b)
	axes[1].text(
		text_x,
		text_y,
		"Dataset size\n$N=10{,}000$",
		transform=axes[1].transAxes,
		**text_kwargs,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
	plt.close(fig)


def main() -> None:
	configure_plot_style()

	repo_root = Path(__file__).resolve().parent.parent
	d_csv_path = repo_root / "SNR_d_scaling_plotting" / "nested_by_d_testpoint_means.csv"
	n_csv_path = repo_root / "SNR_N_scaling_plotting" / "nested_by_N_testpoint_means.csv"
	output_path = repo_root / "figures" / "snr_scaling_collapsed.pdf"

	plot_snr_scaling_figure(
		d_csv_path=d_csv_path,
		n_csv_path=n_csv_path,
		output_path=output_path,
	)
	print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
	main()
