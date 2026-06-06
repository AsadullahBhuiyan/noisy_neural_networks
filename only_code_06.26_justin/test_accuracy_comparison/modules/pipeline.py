from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import BasePreparedDataset, add_noise_to_training_data, prepare_clean_train_and_clean_test
from .metrics import acc_from_logits, mse_onehot_from_logits
from .ntk import run_ntk_inference


@dataclass
class HistogramSelectionResult:
    selection_index: int
    test_index: int
    label: int
    true_label: int
    histogram_values: np.ndarray
    histogram_mean: float
    histogram_std: float
    selected_test_logits_runs: np.ndarray
    mean_logits_across_runs: np.ndarray


@dataclass
class MultiRunExperimentResult:
    per_run_metrics: list[dict[str, object]]
    aggregate_metrics: dict[str, dict[str, float]]
    combined_metrics: dict[str, float]
    train_logits_runs: np.ndarray
    test_logits_runs: np.ndarray
    mean_train_logits: np.ndarray
    mean_test_logits: np.ndarray
    histogram_selections: list[HistogramSelectionResult]
    first_run_noisy_train: np.ndarray


def _noise_seed_for_run(cfg: ExperimentConfig, run_index: int) -> int:
    return int(cfg.seed + 111 + run_index)


def _metric_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _normalize_int_list(value: int | list[int] | tuple[int, ...] | np.ndarray, field_name: str) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [int(item) for item in value]
    else:
        values = [int(value)]

    if not values:
        raise ValueError(f"{field_name} must contain at least one integer.")

    return values


def _resolve_histogram_requests(
    cfg: ExperimentConfig,
    base_data: BasePreparedDataset,
) -> list[tuple[int, int]]:
    test_indices = _normalize_int_list(cfg.histogram_test_index, "histogram_test_index")
    labels = _normalize_int_list(cfg.histogram_label, "histogram_label")

    requests: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for test_index in test_indices:
        if not (0 <= test_index < int(base_data.y_test.shape[0])):
            raise ValueError(
                f"histogram_test_index={test_index} must be in [0, {base_data.y_test.shape[0] - 1}]."
            )

        for label in labels:
            if not (0 <= label < int(cfg.num_classes)):
                raise ValueError(f"histogram_label={label} must be in [0, {cfg.num_classes - 1}].")

            pair = (int(test_index), int(label))
            if pair not in seen:
                requests.append(pair)
                seen.add(pair)

    return requests


def _build_histogram_selections(
    base_data: BasePreparedDataset,
    test_logits_runs: np.ndarray,
    mean_test_logits: np.ndarray,
    histogram_requests: list[tuple[int, int]],
) -> list[HistogramSelectionResult]:
    selections: list[HistogramSelectionResult] = []

    for selection_index, (test_index, label) in enumerate(histogram_requests, start=1):
        histogram_values = np.asarray(test_logits_runs[:, test_index, label], dtype=np.float32)
        selections.append(
            HistogramSelectionResult(
                selection_index=int(selection_index),
                test_index=int(test_index),
                label=int(label),
                true_label=int(base_data.y_test[test_index]),
                histogram_values=histogram_values,
                histogram_mean=float(np.mean(histogram_values)),
                histogram_std=float(np.std(histogram_values, ddof=0)),
                selected_test_logits_runs=np.asarray(test_logits_runs[:, test_index, :], dtype=np.float32),
                mean_logits_across_runs=np.asarray(mean_test_logits[test_index], dtype=np.float32),
            )
        )

    return selections


def run_multi_experiment(
    cfg: ExperimentConfig,
    project_dir: Path,
) -> tuple[BasePreparedDataset, MultiRunExperimentResult]:
    num_runs = max(1, int(cfg.num_independent_runs))
    base_data = prepare_clean_train_and_clean_test(cfg, project_dir=project_dir)
    histogram_requests = _resolve_histogram_requests(cfg, base_data)

    per_run_metrics: list[dict[str, object]] = []
    train_logits_runs = []
    test_logits_runs = []
    first_run_noisy_train = None
    overall_start = time.perf_counter()

    for run_index in range(num_runs):
        run_number = int(run_index + 1)
        noise_seed = _noise_seed_for_run(cfg, run_index)
        print(
            f"Starting independent run {run_number}/{num_runs} with noise_seed={noise_seed}...",
            flush=True,
        )
        run_start = time.perf_counter()
        data = add_noise_to_training_data(base_data, cfg, noise_seed=noise_seed)
        result = run_ntk_inference(cfg, data)
        run_duration = time.perf_counter() - run_start
        elapsed = time.perf_counter() - overall_start
        avg_duration = elapsed / run_number
        remaining_runs = num_runs - run_number
        eta_seconds = avg_duration * remaining_runs

        if first_run_noisy_train is None:
            first_run_noisy_train = data.x_train_noisy.copy()

        per_run_metrics.append(
            {
                "run_index": run_number,
                "noise_seed": int(noise_seed),
                "run_duration_seconds": float(run_duration),
                "train_acc_ntk": float(result.train_acc),
                "train_mse_ntk": float(result.train_mse),
                "test_acc_ntk": float(result.test_acc),
                "test_mse_ntk": float(result.test_mse),
            }
        )
        train_logits_runs.append(result.train_logits)
        test_logits_runs.append(result.test_logits)
        print(
            "Completed independent run "
            f"{run_number}/{num_runs} in {run_duration:.2f}s "
            f"(elapsed {elapsed:.2f}s, estimated remaining {eta_seconds:.2f}s).",
            flush=True,
        )

    train_logits_runs_arr = np.stack(train_logits_runs, axis=0)
    test_logits_runs_arr = np.stack(test_logits_runs, axis=0)
    mean_train_logits = np.mean(train_logits_runs_arr, axis=0)
    mean_test_logits = np.mean(test_logits_runs_arr, axis=0)
    histogram_selections = _build_histogram_selections(
        base_data=base_data,
        test_logits_runs=test_logits_runs_arr,
        mean_test_logits=mean_test_logits,
        histogram_requests=histogram_requests,
    )

    aggregate_metrics = {
        "run_duration_seconds": _metric_summary(
            [float(item["run_duration_seconds"]) for item in per_run_metrics]
        ),
        "train_acc_ntk": _metric_summary([float(item["train_acc_ntk"]) for item in per_run_metrics]),
        "train_mse_ntk": _metric_summary([float(item["train_mse_ntk"]) for item in per_run_metrics]),
        "test_acc_ntk": _metric_summary([float(item["test_acc_ntk"]) for item in per_run_metrics]),
        "test_mse_ntk": _metric_summary([float(item["test_mse_ntk"]) for item in per_run_metrics]),
    }
    combined_metrics = {
        "train_acc_ntk_mean_logits": float(acc_from_logits(mean_train_logits, base_data.y_train)),
        "train_mse_ntk_mean_logits": float(
            mse_onehot_from_logits(mean_train_logits, base_data.y_train, cfg.num_classes)
        ),
        "test_acc_ntk_mean_logits": float(acc_from_logits(mean_test_logits, base_data.y_test)),
        "test_mse_ntk_mean_logits": float(
            mse_onehot_from_logits(mean_test_logits, base_data.y_test, cfg.num_classes)
        ),
    }

    return base_data, MultiRunExperimentResult(
        per_run_metrics=per_run_metrics,
        aggregate_metrics=aggregate_metrics,
        combined_metrics=combined_metrics,
        train_logits_runs=train_logits_runs_arr,
        test_logits_runs=test_logits_runs_arr,
        mean_train_logits=mean_train_logits,
        mean_test_logits=mean_test_logits,
        histogram_selections=histogram_selections,
        first_run_noisy_train=np.asarray(first_run_noisy_train, dtype=np.float32),
    )


def _save_histogram_plot(
    cfg: ExperimentConfig,
    result: MultiRunExperimentResult,
    output_dir: Path,
) -> str | None:
    if int(cfg.num_independent_runs) <= 1 or not result.histogram_selections:
        return None

    matplotlib_dir = output_dir / ".matplotlib"
    cache_dir = output_dir / ".cache"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = output_dir / "selected_test_label_histograms.png"
    num_plots = len(result.histogram_selections)
    num_cols = min(2, num_plots)
    num_rows = int(math.ceil(num_plots / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 4.8 * num_rows))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for axis, selection in zip(axes_arr, result.histogram_selections):
        histogram_values = np.asarray(selection.histogram_values, dtype=np.float64)
        axis.hist(histogram_values, bins="auto", edgecolor="black", alpha=0.8)
        axis.axvline(selection.histogram_mean, color="crimson", linewidth=2.0, label="mean")
        if selection.histogram_std > 0.0:
            axis.axvline(
                selection.histogram_mean - selection.histogram_std,
                color="darkorange",
                linestyle="--",
                linewidth=1.5,
                label="mean +/- std",
            )
            axis.axvline(
                selection.histogram_mean + selection.histogram_std,
                color="darkorange",
                linestyle="--",
                linewidth=1.5,
            )

        axis.set_title(
            f"test_index={selection.test_index}, label={selection.label}, "
            f"true_label={selection.true_label}"
        )
        axis.set_xlabel("Output value")
        axis.set_ylabel("Count")
        axis.grid(alpha=0.25)
        axis.text(
            0.98,
            0.97,
            f"mean={selection.histogram_mean:.6f}\nstd={selection.histogram_std:.6f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        axis.legend(loc="upper left")

    for axis in axes_arr[num_plots:]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return plot_path.name


def save_experiment_outputs(
    cfg: ExperimentConfig,
    base_data: BasePreparedDataset,
    result: MultiRunExperimentResult,
    project_dir: Path,
) -> Path:
    output_dir = cfg.resolve_output_dir(project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    histogram_plot_file = _save_histogram_plot(cfg, result, output_dir)

    summary = {
        "config": asdict(cfg),
        "train_info": base_data.train_info,
        "test_info": base_data.test_info,
        "per_run_metrics": result.per_run_metrics,
        "aggregate_metrics": result.aggregate_metrics,
        "combined_metrics": result.combined_metrics,
        "histogram_plot_file": histogram_plot_file,
        "selected_test_examples": [
            {
                "selection_index": selection.selection_index,
                "test_index": selection.test_index,
                "true_label": selection.true_label,
                "selected_label_for_histogram": selection.label,
                "mean_logits_across_runs": selection.mean_logits_across_runs.tolist(),
                "logits_for_each_run": selection.selected_test_logits_runs.tolist(),
                "histogram_values": selection.histogram_values.tolist(),
                "histogram_mean": selection.histogram_mean,
                "histogram_std": selection.histogram_std,
            }
            for selection in result.histogram_selections
        ],
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    histogram_test_indices = np.asarray(
        [selection.test_index for selection in result.histogram_selections],
        dtype=np.int64,
    )
    histogram_labels = np.asarray(
        [selection.label for selection in result.histogram_selections],
        dtype=np.int64,
    )
    histogram_true_labels = np.asarray(
        [selection.true_label for selection in result.histogram_selections],
        dtype=np.int64,
    )
    histogram_values_selections = np.stack(
        [selection.histogram_values for selection in result.histogram_selections],
        axis=0,
    )
    selected_test_logits_runs_selections = np.stack(
        [selection.selected_test_logits_runs for selection in result.histogram_selections],
        axis=0,
    )
    histogram_means = np.asarray(
        [selection.histogram_mean for selection in result.histogram_selections],
        dtype=np.float32,
    )
    histogram_stds = np.asarray(
        [selection.histogram_std for selection in result.histogram_selections],
        dtype=np.float32,
    )

    np.savez_compressed(
        output_dir / "predictions.npz",
        x_train_clean=base_data.x_train_clean,
        x_train_noisy=result.first_run_noisy_train,
        y_train=base_data.y_train,
        y_train_onehot=base_data.y_train_onehot,
        x_test_clean=base_data.x_test_clean,
        y_test=base_data.y_test,
        y_test_onehot=base_data.y_test_onehot,
        train_logits_ntk=result.mean_train_logits,
        test_logits_ntk=result.mean_test_logits,
        train_logits_ntk_runs=result.train_logits_runs,
        test_logits_ntk_runs=result.test_logits_runs,
        histogram_test_indices=histogram_test_indices,
        histogram_labels=histogram_labels,
        histogram_true_labels=histogram_true_labels,
        histogram_values_selections=histogram_values_selections,
        histogram_means=histogram_means,
        histogram_stds=histogram_stds,
        selected_test_logits_runs_selections=selected_test_logits_runs_selections,
    )

    return output_dir


def run_experiment(cfg: ExperimentConfig) -> tuple[MultiRunExperimentResult, Path]:
    project_dir = Path(__file__).resolve().parents[1]
    base_data, result = run_multi_experiment(cfg, project_dir=project_dir)
    output_dir = save_experiment_outputs(cfg, base_data, result, project_dir)
    return result, output_dir
