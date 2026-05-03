from __future__ import annotations

import csv
import gzip
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from compare_noisy import (
    _activation_bundle,
    _build_pure_noise_pipeline,
    _comparison_device_for_analytics,
    _comparison_summary,
    _resolve_root,
    _save_scatter_plot,
)
from modules.data import DatasetSubset, dataset_to_numpy_matrix, load_mnist, normalize_feature_matrix_unit_rms
from modules.metrics import acc_from_logits
from noise_models import (
    EpsilonCorrectionConfig,
    PredictionCorrection,
    compute_noise_combination_coeffs,
    noise_variance,
    precompute_first_layer_train_stats,
)


################################################################################
# USER INPUTS
#
# Edit only this block for normal use. This script reads saved logits from the
# test_outputs.csv.gz emitted by train_noisy_mlp_ensemble.py, computes empirical
# ensemble mean/variance, and compares them to the first-order perturbative
# analytical prediction from noise_models.py.
################################################################################

TEST_OUTPUTS_PATH = (
    "/data2/jt577/2026/04.2026/NN_noise/trained_mlp_nested_ensembles_trainingSize/mlp_ensemble__additive_gaussian__p=0.95__N=16000__d=28x28__loss=mse__act=erf__L=3__width=2048__noiseK=100__initM=10__models=1000/test_outputs.csv.gz"
)

DATA_ROOT = "data"
OUTPUT_ROOT = "outputs/test_outputs_vs_analytical"

# Use None for every saved model and every saved test point. These are useful
# for a quick smoke test before running the full comparison.
MAX_MODELS = None
TEST_LIMIT = 1000
TEST_INDICES = None  # e.g. [0, 1, 2, 17]; interpreted as saved test_index values.

ALLOW_CPU = False
ANALYTICAL_PROGRESS_EVERY = 250
CSV_PROGRESS_EVERY_ROWS = 1_000_000
CENTER_CLASSWISE_LOGITS = True

SAVE_PER_POINT_CSV = True
SAVE_NPZ = True

################################################################################
# END USER INPUTS
################################################################################


@dataclass
class CSVAnalyticalCompareConfig:
    test_outputs_path: str
    data_root: str = "data"
    output_root: str = "outputs/test_outputs_vs_analytical"
    max_models: int | None = None
    test_limit: int | None = None
    test_indices: list[int] | None = None
    allow_cpu: bool = False
    analytical_progress_every: int = 250
    csv_progress_every_rows: int = 1_000_000
    center_classwise_logits: bool = True
    save_per_point_csv: bool = True
    save_npz: bool = True


@dataclass
class PreparedData:
    x_train_clean: np.ndarray
    y_train_onehot: np.ndarray
    y_train: np.ndarray
    x_test_clean: np.ndarray
    y_test: np.ndarray
    selected_test_indices: np.ndarray
    train_indices: np.ndarray
    num_classes: int


def build_config_from_user_inputs() -> CSVAnalyticalCompareConfig:
    if MAX_MODELS is not None and int(MAX_MODELS) < 1:
        raise ValueError("MAX_MODELS must be None or at least 1.")
    if TEST_LIMIT is not None and int(TEST_LIMIT) < 1:
        raise ValueError("TEST_LIMIT must be None or at least 1.")
    if TEST_INDICES is not None and len(TEST_INDICES) < 1:
        raise ValueError("TEST_INDICES must be None or a non-empty list of saved test_index values.")

    return CSVAnalyticalCompareConfig(
        test_outputs_path=str(TEST_OUTPUTS_PATH),
        data_root=str(DATA_ROOT),
        output_root=str(OUTPUT_ROOT),
        max_models=None if MAX_MODELS is None else int(MAX_MODELS),
        test_limit=None if TEST_LIMIT is None else int(TEST_LIMIT),
        test_indices=None if TEST_INDICES is None else [int(v) for v in TEST_INDICES],
        allow_cpu=bool(ALLOW_CPU),
        analytical_progress_every=int(ANALYTICAL_PROGRESS_EVERY),
        csv_progress_every_rows=int(CSV_PROGRESS_EVERY_ROWS),
        center_classwise_logits=bool(CENTER_CLASSWISE_LOGITS),
        save_per_point_csv=bool(SAVE_PER_POINT_CSV),
        save_npz=bool(SAVE_NPZ),
    )


def _choose_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        "No GPU backend was found. Set ALLOW_CPU = True in the USER INPUTS block "
        "if you intentionally want to run the analytical comparison on CPU."
    )


def _open_text_maybe_gzip(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_metadata(test_outputs_path: Path) -> tuple[Path, dict[str, object]]:
    ensemble_dir = test_outputs_path.parent
    metadata_path = ensemble_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata.json next to {test_outputs_path}")
    return ensemble_dir, _load_json(metadata_path)


def _select_test_indices(
    *,
    num_available: int,
    test_limit: int | None,
    test_indices: list[int] | None,
) -> np.ndarray:
    if test_indices is not None:
        selected = np.asarray(test_indices, dtype=np.int64)
    elif test_limit is not None:
        selected = np.arange(min(int(test_limit), int(num_available)), dtype=np.int64)
    else:
        selected = np.arange(int(num_available), dtype=np.int64)

    if selected.ndim != 1:
        raise ValueError("Selected test indices must be one-dimensional.")
    if selected.size == 0:
        raise ValueError("No test indices were selected.")
    if np.any(selected < 0) or np.any(selected >= int(num_available)):
        raise IndexError(f"Selected test indices must lie in [0, {num_available}).")
    if np.unique(selected).shape[0] != selected.shape[0]:
        raise ValueError("TEST_INDICES contains duplicates.")
    return selected.astype(np.int64)


def _load_exact_data(
    *,
    metadata: dict[str, object],
    data_root: Path,
    selected_test_indices: np.ndarray,
) -> PreparedData:
    saved_cfg = metadata["config"]
    img_hw = tuple(int(v) for v in saved_cfg["img_hw"])
    num_classes = int(saved_cfg["num_classes"])

    train_raw, test_raw = load_mnist(root=data_root, img_hw=img_hw)
    train_indices = np.asarray(metadata["train_indices"], dtype=np.int64)
    train_dataset = DatasetSubset(train_raw, train_indices)
    test_dataset = DatasetSubset(test_raw, selected_test_indices)

    x_train_clean, y_train_onehot, y_train = dataset_to_numpy_matrix(train_dataset, num_classes)
    x_test_clean, _y_test_onehot, y_test = dataset_to_numpy_matrix(test_dataset, num_classes)

    x_train_clean = normalize_feature_matrix_unit_rms(x_train_clean)
    x_test_clean = normalize_feature_matrix_unit_rms(x_test_clean)

    return PreparedData(
        x_train_clean=x_train_clean,
        y_train_onehot=y_train_onehot,
        y_train=y_train,
        x_test_clean=x_test_clean,
        y_test=y_test,
        selected_test_indices=np.asarray(selected_test_indices, dtype=np.int64),
        train_indices=train_indices,
        num_classes=num_classes,
    )


def _logit_columns(fieldnames: list[str]) -> list[str]:
    columns = [name for name in fieldnames if name.startswith("logit_")]
    if not columns:
        raise ValueError("The CSV does not contain logit_* columns.")
    return sorted(columns, key=lambda name: int(name.split("_", 1)[1]))


def _center_classwise_logits(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr - np.mean(arr, axis=-1, keepdims=True)


def _read_csv_ensemble_statistics(
    *,
    test_outputs_path: Path,
    selected_test_indices: np.ndarray,
    num_classes: int,
    max_models: int | None,
    progress_every_rows: int,
    center_classwise_logits: bool,
) -> dict[str, object]:
    selected_test_indices = np.asarray(selected_test_indices, dtype=np.int64)
    test_index_to_position = {int(test_index): pos for pos, test_index in enumerate(selected_test_indices)}
    num_tests = int(selected_test_indices.shape[0])

    model_logits_sum = np.zeros((num_tests, int(num_classes)), dtype=np.float64)
    model_logits_sq_sum = np.zeros_like(model_logits_sum)
    model_counts = np.zeros((num_tests,), dtype=np.int64)
    labels = np.full((num_tests,), -1, dtype=np.int64)

    accepted_model_numbers: set[int] = set()
    seen_model_order: list[int] = []
    per_noise_logits_sum: dict[int, np.ndarray] = {}
    per_noise_centered_logits_sum: dict[int, np.ndarray] = {}
    per_noise_counts: dict[int, np.ndarray] = {}
    per_noise_model_numbers: dict[int, set[int]] = {}
    row_count = 0
    used_row_count = 0
    start = time.perf_counter()

    with _open_text_maybe_gzip(test_outputs_path, "rt") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{test_outputs_path} has no CSV header.")
        logit_cols = _logit_columns(reader.fieldnames)
        if len(logit_cols) != int(num_classes):
            raise ValueError(
                f"Expected {num_classes} logit columns from metadata, found {len(logit_cols)}."
            )

        for row in reader:
            row_count += 1

            if progress_every_rows > 0 and row_count % int(progress_every_rows) == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"Read {row_count:,} CSV rows, used {used_row_count:,} rows "
                    f"from {len(accepted_model_numbers)} models in {elapsed:.1f}s.",
                    flush=True,
                )

            test_index = int(row["test_index"])
            position = test_index_to_position.get(test_index)
            if position is None:
                continue

            model_number = int(row["model_number"])
            if model_number not in accepted_model_numbers:
                if max_models is not None and len(accepted_model_numbers) >= int(max_models):
                    continue
                accepted_model_numbers.add(model_number)
                seen_model_order.append(model_number)

            logits = np.asarray([float(row[col]) for col in logit_cols], dtype=np.float64)
            centered_logits = logits - np.mean(logits)
            noise_dataset_index = int(row.get("noise_dataset_index", int(row["noise_dataset_number"]) - 1))
            if noise_dataset_index not in per_noise_logits_sum:
                per_noise_logits_sum[noise_dataset_index] = np.zeros((num_tests, int(num_classes)), dtype=np.float64)
                per_noise_centered_logits_sum[noise_dataset_index] = np.zeros_like(
                    per_noise_logits_sum[noise_dataset_index]
                )
                per_noise_counts[noise_dataset_index] = np.zeros((num_tests,), dtype=np.int64)
                per_noise_model_numbers[noise_dataset_index] = set()

            model_logits_sum[position] += logits
            model_logits_sq_sum[position] += logits * logits
            model_counts[position] += 1

            per_noise_logits_sum[noise_dataset_index][position] += logits
            per_noise_centered_logits_sum[noise_dataset_index][position] += centered_logits
            per_noise_counts[noise_dataset_index][position] += 1
            per_noise_model_numbers[noise_dataset_index].add(model_number)
            labels[position] = int(row["true_label"])
            used_row_count += 1

    if not accepted_model_numbers:
        raise RuntimeError("No model rows were selected from the CSV.")
    if np.any(model_counts == 0):
        missing = selected_test_indices[model_counts == 0]
        raise RuntimeError(f"The CSV did not contain rows for selected test indices: {missing[:20].tolist()}")
    if np.unique(model_counts).shape[0] != 1:
        raise RuntimeError(
            "Selected test points have unequal model counts. "
            f"Count range: {int(np.min(model_counts))} to {int(np.max(model_counts))}."
        )

    noise_dataset_indices = np.asarray(sorted(per_noise_logits_sum), dtype=np.int64)
    per_noise_means = []
    per_noise_centered_means = []
    init_counts_by_noise = {}
    for noise_dataset_index in noise_dataset_indices:
        counts = per_noise_counts[int(noise_dataset_index)]
        if np.any(counts == 0):
            missing = selected_test_indices[counts == 0]
            raise RuntimeError(
                f"Noise dataset {int(noise_dataset_index)} is missing selected test indices: "
                f"{missing[:20].tolist()}"
            )
        if np.unique(counts).shape[0] != 1:
            raise RuntimeError(
                f"Noise dataset {int(noise_dataset_index)} has unequal init/model counts across "
                f"selected test points. Count range: {int(np.min(counts))} to {int(np.max(counts))}."
            )
        init_count = int(counts[0])
        init_counts_by_noise[str(int(noise_dataset_index))] = init_count
        per_noise_mean = per_noise_logits_sum[int(noise_dataset_index)] / float(init_count)
        per_noise_means.append(per_noise_mean)
        if center_classwise_logits:
            per_noise_centered_means.append(
                per_noise_centered_logits_sum[int(noise_dataset_index)] / float(init_count)
            )
        else:
            per_noise_centered_means.append(per_noise_mean.copy())

    per_noise_means_arr = np.stack(per_noise_means, axis=0)
    per_noise_centered_means_arr = np.stack(per_noise_centered_means, axis=0)

    # Nested estimator: average inits within each noisy dataset, then average
    # those per-noise means with equal weight across noisy datasets.
    mean_logits = np.mean(per_noise_means_arr, axis=0)
    centered_mean_logits = np.mean(per_noise_centered_means_arr, axis=0)
    variance_logits = np.mean((per_noise_means_arr - mean_logits[None, :, :]) ** 2, axis=0)

    num_models = int(model_counts[0])
    flat_model_mean = model_logits_sum / float(num_models)
    flat_model_variance = model_logits_sq_sum / float(num_models) - flat_model_mean * flat_model_mean
    flat_model_variance = np.maximum(flat_model_variance, 0.0)

    return {
        "mean_mlp_ensemble": mean_logits,
        "mean_mlp_ensemble_centered": centered_mean_logits,
        "variance_mlp_ensemble": variance_logits,
        "flat_model_mean_mlp_ensemble": flat_model_mean,
        "flat_model_variance_mlp_ensemble": flat_model_variance,
        "per_noise_mean_mlp_ensemble": per_noise_means_arr,
        "labels_from_csv": labels,
        "model_numbers": np.asarray(seen_model_order, dtype=np.int64),
        "noise_dataset_indices": noise_dataset_indices,
        "init_counts_by_noise_dataset_index": init_counts_by_noise,
        "evaluated_num_noise_datasets": int(noise_dataset_indices.shape[0]),
        "evaluated_num_models": num_models,
        "rows_read": int(row_count),
        "rows_used": int(used_row_count),
    }


def _compute_first_order_analytical_statistics(
    *,
    metadata: dict[str, object],
    data: PreparedData,
    device: torch.device,
    progress_every: int,
    center_classwise_logits: bool,
) -> dict[str, object]:
    saved_cfg = metadata["config"]
    noise_type = str(saved_cfg["noise_type"]).strip().lower()
    if noise_type not in {"additive_gaussian", "replacement"}:
        raise ValueError(
            "Analytical statistics are currently supported only for "
            "noise_type='additive_gaussian' or 'replacement'."
        )

    act, act_p, act_pp, act_ppp, act_pppp = _activation_bundle(str(saved_cfg["activation"]))
    epsilon = 1.0 - float(saved_cfg["noise_probability"])
    num_tests = int(data.x_test_clean.shape[0])
    num_classes = int(saved_cfg["num_classes"])

    mean_first_order = np.empty((num_tests, num_classes), dtype=np.float64)
    variance_first_order = np.empty((num_tests, num_classes), dtype=np.float64)

    y_train_tensor = torch.as_tensor(data.y_train_onehot, dtype=torch.float64, device=device)
    x_train_clean_tensor = torch.as_tensor(data.x_train_clean, dtype=torch.float64, device=device)
    train_stats = precompute_first_layer_train_stats(
        X_train=x_train_clean_tensor,
        y_train=y_train_tensor,
        num_classes=num_classes,
    )

    start = time.perf_counter()
    for test_position in range(num_tests):
        x_test_tensor = torch.as_tensor(data.x_test_clean[test_position], dtype=torch.float64, device=device)
        pure_cfg, pipe = _build_pure_noise_pipeline(
            saved_cfg=saved_cfg,
            x_train_clean_tensor=x_train_clean_tensor,
            x_test_tensor=x_test_tensor,
            act=act,
            act_p=act_p,
            act_pp=act_pp,
            act_ppp=act_ppp,
            act_pppp=act_pppp,
            device=device,
        )
        eps_cfg = EpsilonCorrectionConfig(
            eps=epsilon,
            y_train=y_train_tensor,
            num_classes=num_classes,
            device=device,
            dtype=torch.float64,
        )
        correction = PredictionCorrection(
            pipe,
            pure_cfg,
            eps_cfg,
            train_stats=train_stats,
        )

        pure_mean_scalar = float(pipe.prediction.output_mean.detach().cpu().item())
        mean_first_order[test_position] = (
            pure_mean_scalar
            + correction.output_mean_first_order_correction.detach().cpu().numpy()
        )

        coeffs = compute_noise_combination_coeffs(pipe)
        variance_scalar = float(
            noise_variance(coeffs, pipe, pure_cfg, num_classes=num_classes).detach().cpu().item()
        )
        variance_first_order[test_position] = variance_scalar

        run_number = test_position + 1
        if run_number == 1 or run_number == num_tests or (
            int(progress_every) > 0 and run_number % int(progress_every) == 0
        ):
            elapsed = time.perf_counter() - start
            avg = elapsed / run_number
            eta = avg * (num_tests - run_number)
            print(
                f"Computed analytical statistics for test point {run_number}/{num_tests} "
                f"(elapsed {elapsed:.1f}s, estimated remaining {eta:.1f}s).",
                flush=True,
            )

    if center_classwise_logits:
        centered_mean_first_order = _center_classwise_logits(mean_first_order)
    else:
        centered_mean_first_order = mean_first_order.copy()

    return {
        "epsilon": epsilon,
        "mean_first_order_analytical": mean_first_order,
        "mean_first_order_analytical_centered": centered_mean_first_order,
        "variance_first_order_analytical": variance_first_order,
        "first_order_accuracy": float(acc_from_logits(mean_first_order, data.y_test)),
        "first_order_error_percent": 100.0 * (1.0 - float(acc_from_logits(mean_first_order, data.y_test))),
    }


def _compute_delta_xstar_x_feature(data: PreparedData, *, batch_size: int = 256) -> np.ndarray:
    """
    Compute Delta E^i_alpha[<x_* x_alpha>] for each selected test point and class.

    For each test point x_* and class i:
        C / N sum_alpha y_alpha,i <x_* x_alpha> - 1 / N sum_alpha <x_* x_alpha>

    The dot product is dimension-normalized, matching the kernel convention.
    """
    x_train = np.asarray(data.x_train_clean, dtype=np.float64)
    x_test = np.asarray(data.x_test_clean, dtype=np.float64)
    y_train = np.asarray(data.y_train_onehot, dtype=np.float64)

    num_train, feature_dim = x_train.shape
    num_classes = int(y_train.shape[1])
    out = np.empty((x_test.shape[0], num_classes), dtype=np.float64)
    class_scale = float(num_classes) / float(num_train)

    for start in range(0, int(x_test.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(x_test.shape[0]))
        normalized_dots = np.matmul(x_test[start:end], x_train.T) / float(feature_dim)
        class_means = class_scale * np.matmul(normalized_dots, y_train)
        overall = np.mean(normalized_dots, axis=1, keepdims=True)
        out[start:end] = class_means - overall

    return out


def _fit_simple_delta_predictor(
    *,
    data: PreparedData,
    empirical_mean_raw: np.ndarray,
) -> dict[str, object]:
    feature_raw = _compute_delta_xstar_x_feature(data)
    feature_centered = _center_classwise_logits(feature_raw)
    target_raw = np.asarray(empirical_mean_raw, dtype=np.float64)
    target_centered = _center_classwise_logits(empirical_mean_raw)

    x_raw = feature_raw.reshape(-1)
    y_raw = target_raw.reshape(-1)
    raw_design = np.column_stack([x_raw, np.ones_like(x_raw)])
    (raw_a, raw_b), *_ = np.linalg.lstsq(raw_design, y_raw, rcond=None)
    raw_prediction = float(raw_a) * feature_raw + float(raw_b)
    raw_residual = target_raw - raw_prediction
    raw_ss_res = float(np.sum(raw_residual * raw_residual))
    raw_ss_tot = float(np.sum((y_raw - float(np.mean(y_raw))) ** 2))
    raw_r2 = None if raw_ss_tot == 0.0 else 1.0 - raw_ss_res / raw_ss_tot

    x_centered = feature_centered.reshape(-1)
    y_centered = target_centered.reshape(-1)
    centered_design = np.column_stack([x_centered, np.ones_like(x_centered)])
    (centered_a, centered_b), *_ = np.linalg.lstsq(centered_design, y_centered, rcond=None)

    prediction_centered = float(centered_a) * feature_centered + float(centered_b)
    residual = target_centered - prediction_centered
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y_centered - float(np.mean(y_centered))) ** 2))
    r2 = None if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    return {
        "delta_xstar_x": feature_raw,
        "delta_xstar_x_centered": feature_centered,
        "simple_fit_target_raw": target_raw,
        "simple_fit_prediction_raw": raw_prediction,
        "simple_fit_raw_a": float(raw_a),
        "simple_fit_raw_b": float(raw_b),
        "simple_fit_raw_r2": raw_r2,
        "simple_fit_target_centered": target_centered,
        "simple_fit_prediction_centered": prediction_centered,
        "simple_fit_centered_a": float(centered_a),
        "simple_fit_centered_b": float(centered_b),
        "simple_fit_centered_r2": r2,
    }


def _prepare_matplotlib_cache(output_dir: Path) -> None:
    matplotlib_dir = output_dir / ".matplotlib"
    cache_dir = output_dir / ".cache"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)


def _write_per_point_csv(
    path: Path,
    *,
    selected_test_indices: np.ndarray,
    y_test: np.ndarray,
    empirical_mean: np.ndarray,
    empirical_mean_centered: np.ndarray,
    empirical_variance: np.ndarray,
    analytical_mean: np.ndarray,
    analytical_mean_centered: np.ndarray,
    analytical_variance: np.ndarray,
    simple_delta_xstar_x_centered: np.ndarray,
    simple_fit_target_centered: np.ndarray,
    simple_fit_prediction_centered: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "test_index",
        "true_label",
        "class_id",
        "empirical_mean",
        "analytical_mean",
        "mean_diff",
        "empirical_mean_centered",
        "analytical_mean_centered",
        "centered_mean_diff",
        "simple_delta_xstar_x_centered",
        "simple_fit_target_centered",
        "simple_fit_prediction_centered",
        "simple_fit_centered_diff",
        "empirical_variance",
        "analytical_variance",
        "variance_diff",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for test_position, test_index in enumerate(selected_test_indices):
            for class_id in range(int(empirical_mean.shape[1])):
                row = {
                    "test_index": int(test_index),
                    "true_label": int(y_test[test_position]),
                    "class_id": int(class_id),
                    "empirical_mean": float(empirical_mean[test_position, class_id]),
                    "analytical_mean": float(analytical_mean[test_position, class_id]),
                    "mean_diff": float(empirical_mean[test_position, class_id] - analytical_mean[test_position, class_id]),
                    "empirical_mean_centered": float(empirical_mean_centered[test_position, class_id]),
                    "analytical_mean_centered": float(analytical_mean_centered[test_position, class_id]),
                    "centered_mean_diff": float(
                        empirical_mean_centered[test_position, class_id]
                        - analytical_mean_centered[test_position, class_id]
                    ),
                    "simple_delta_xstar_x_centered": float(
                        simple_delta_xstar_x_centered[test_position, class_id]
                    ),
                    "simple_fit_target_centered": float(
                        simple_fit_target_centered[test_position, class_id]
                    ),
                    "simple_fit_prediction_centered": float(
                        simple_fit_prediction_centered[test_position, class_id]
                    ),
                    "simple_fit_centered_diff": float(
                        simple_fit_target_centered[test_position, class_id]
                        - simple_fit_prediction_centered[test_position, class_id]
                    ),
                    "empirical_variance": float(empirical_variance[test_position, class_id]),
                    "analytical_variance": float(analytical_variance[test_position, class_id]),
                    "variance_diff": float(
                        empirical_variance[test_position, class_id]
                        - analytical_variance[test_position, class_id]
                    ),
                }
                writer.writerow(row)


def _safe_name(path: Path) -> str:
    return path.parent.name if path.name == "test_outputs.csv.gz" else path.stem


def _save_outputs(
    *,
    project_dir: Path,
    cfg: CSVAnalyticalCompareConfig,
    test_outputs_path: Path,
    ensemble_dir: Path,
    metadata: dict[str, object],
    data: PreparedData,
    csv_stats: dict[str, object],
    analytical: dict[str, object],
) -> Path:
    output_root = _resolve_root(cfg.output_root, project_dir)
    output_dir = output_root / _safe_name(test_outputs_path)
    if cfg.max_models is not None:
        output_dir = output_dir / f"subset__models={int(cfg.max_models)}"
    if cfg.test_indices is not None:
        output_dir = output_dir / f"subset__custom_tests={len(cfg.test_indices)}"
    elif cfg.test_limit is not None:
        output_dir = output_dir / f"subset__test={int(cfg.test_limit)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_matplotlib_cache(output_dir)

    empirical_mean_raw = np.asarray(csv_stats["mean_mlp_ensemble"], dtype=np.float64)
    empirical_mean_centered = np.asarray(csv_stats["mean_mlp_ensemble_centered"], dtype=np.float64)
    empirical_variance = np.asarray(csv_stats["variance_mlp_ensemble"], dtype=np.float64)
    analytical_mean_raw = np.asarray(analytical["mean_first_order_analytical"], dtype=np.float64)
    analytical_mean_centered = np.asarray(analytical["mean_first_order_analytical_centered"], dtype=np.float64)
    analytical_variance = np.asarray(analytical["variance_first_order_analytical"], dtype=np.float64)
    simple_fit = _fit_simple_delta_predictor(data=data, empirical_mean_raw=empirical_mean_raw)
    simple_delta_raw = np.asarray(simple_fit["delta_xstar_x"], dtype=np.float64)
    simple_target_raw = np.asarray(simple_fit["simple_fit_target_raw"], dtype=np.float64)
    simple_prediction_raw = np.asarray(simple_fit["simple_fit_prediction_raw"], dtype=np.float64)
    simple_delta_centered = np.asarray(simple_fit["delta_xstar_x_centered"], dtype=np.float64)
    simple_target_centered = np.asarray(simple_fit["simple_fit_target_centered"], dtype=np.float64)
    simple_prediction_centered = np.asarray(simple_fit["simple_fit_prediction_centered"], dtype=np.float64)

    mean_summary = _comparison_summary(empirical_mean_centered, analytical_mean_centered)
    raw_mean_summary = _comparison_summary(empirical_mean_raw, analytical_mean_raw)
    simple_fit_raw_summary = _comparison_summary(simple_target_raw, simple_prediction_raw)
    simple_fit_summary = _comparison_summary(simple_target_centered, simple_prediction_centered)
    variance_summary = _comparison_summary(empirical_variance, analytical_variance)

    mean_plot = _save_scatter_plot(
        output_dir=output_dir,
        file_name="mean_centered_scatter.png",
        x_values=empirical_mean_centered,
        y_values=analytical_mean_centered,
        title="Saved test_outputs Mean vs First-Order Mean",
        x_label="CSV ensemble mean logits (class-centered)",
        y_label="First-order analytical mean logits (class-centered)",
        summary=mean_summary,
    )
    raw_mean_plot = _save_scatter_plot(
        output_dir=output_dir,
        file_name="mean_raw_scatter.png",
        x_values=empirical_mean_raw,
        y_values=analytical_mean_raw,
        title="Saved test_outputs Raw Mean vs First-Order Raw Mean",
        x_label="CSV ensemble mean logits",
        y_label="First-order analytical mean logits",
        summary=raw_mean_summary,
    )
    simple_fit_plot = _save_scatter_plot(
        output_dir=output_dir,
        file_name="simple_delta_fit_centered_scatter.png",
        x_values=simple_target_centered,
        y_values=simple_prediction_centered,
        title="Saved test_outputs Mean vs Fitted Delta Dot-Product Predictor",
        x_label="CSV ensemble mean logits (class-centered)",
        y_label="a * Delta E[<x_* x_alpha>] + b (fitted, class-centered)",
        summary=simple_fit_summary,
    )
    simple_fit_raw_plot = _save_scatter_plot(
        output_dir=output_dir,
        file_name="simple_delta_fit_raw_scatter.png",
        x_values=simple_target_raw,
        y_values=simple_prediction_raw,
        title="Saved test_outputs Raw Mean vs Fitted Delta Dot-Product Predictor",
        x_label="CSV ensemble mean logits",
        y_label="a * Delta E[<x_* x_alpha>] + b (fitted globally)",
        summary=simple_fit_raw_summary,
    )
    variance_plot = _save_scatter_plot(
        output_dir=output_dir,
        file_name="variance_scatter.png",
        x_values=empirical_variance,
        y_values=analytical_variance,
        title="Saved test_outputs Variance vs First-Order Variance",
        x_label="CSV ensemble variance",
        y_label="First-order analytical variance",
        summary=variance_summary,
    )

    if cfg.save_per_point_csv:
        _write_per_point_csv(
            output_dir / "per_test_class_comparison.csv",
            selected_test_indices=data.selected_test_indices,
            y_test=data.y_test,
            empirical_mean=empirical_mean_raw,
            empirical_mean_centered=empirical_mean_centered,
            empirical_variance=empirical_variance,
            analytical_mean=analytical_mean_raw,
            analytical_mean_centered=analytical_mean_centered,
            analytical_variance=analytical_variance,
            simple_delta_xstar_x_centered=simple_delta_centered,
            simple_fit_target_centered=simple_target_centered,
            simple_fit_prediction_centered=simple_prediction_centered,
        )

    if cfg.save_npz:
        np.savez_compressed(
            output_dir / "comparison_arrays.npz",
            selected_test_indices=data.selected_test_indices,
            y_test=data.y_test,
            empirical_mean=empirical_mean_raw,
            empirical_mean_centered=empirical_mean_centered,
            empirical_variance=empirical_variance,
            empirical_flat_model_mean=np.asarray(csv_stats["flat_model_mean_mlp_ensemble"], dtype=np.float64),
            empirical_flat_model_variance=np.asarray(csv_stats["flat_model_variance_mlp_ensemble"], dtype=np.float64),
            empirical_per_noise_mean=np.asarray(csv_stats["per_noise_mean_mlp_ensemble"], dtype=np.float64),
            analytical_mean=analytical_mean_raw,
            analytical_mean_centered=analytical_mean_centered,
            analytical_variance=analytical_variance,
            simple_delta_xstar_x=simple_delta_raw,
            simple_fit_target_raw=simple_target_raw,
            simple_fit_prediction_raw=simple_prediction_raw,
            simple_delta_xstar_x_centered=simple_delta_centered,
            simple_fit_target_centered=simple_target_centered,
            simple_fit_prediction_centered=simple_prediction_centered,
            model_numbers=np.asarray(csv_stats["model_numbers"], dtype=np.int64),
            noise_dataset_indices=np.asarray(csv_stats["noise_dataset_indices"], dtype=np.int64),
        )

    empirical_mean_accuracy = float(acc_from_logits(empirical_mean_raw, data.y_test))
    analytical_accuracy = float(analytical["first_order_accuracy"])
    database = {
        "schema_version": 1,
        "description": (
            "Comparison of saved train_noisy_mlp_ensemble.py test_outputs.csv.gz logits "
            "against the first-order perturbative analytical model in noise_models.py."
        ),
        "requested_config": asdict(cfg),
        "test_outputs_path": str(test_outputs_path),
        "ensemble_dir": str(ensemble_dir),
        "saved_ensemble_config": metadata["config"],
        "dataset": {
            "num_train": int(data.x_train_clean.shape[0]),
            "num_selected_test": int(data.x_test_clean.shape[0]),
            "feature_dim": int(data.x_train_clean.shape[1]),
            "selected_test_indices": data.selected_test_indices.tolist(),
            "train_indices": data.train_indices.tolist(),
        },
        "csv_read": {
            "rows_read": int(csv_stats["rows_read"]),
            "rows_used": int(csv_stats["rows_used"]),
            "evaluated_num_models": int(csv_stats["evaluated_num_models"]),
            "evaluated_num_noise_datasets": int(csv_stats["evaluated_num_noise_datasets"]),
            "model_numbers": np.asarray(csv_stats["model_numbers"], dtype=np.int64).tolist(),
            "noise_dataset_indices": np.asarray(csv_stats["noise_dataset_indices"], dtype=np.int64).tolist(),
            "init_counts_by_noise_dataset_index": dict(csv_stats["init_counts_by_noise_dataset_index"]),
        },
        "empirical_estimator": {
            "mean": (
                "For each noise_dataset_index, average logits over all selected init/model rows; "
                "then average those per-noise means with equal weight across noise datasets."
            ),
            "variance": (
                "variance_mlp_ensemble is the population variance across the per-noise mean logits. "
                "The flat variance over individual model rows is also saved in comparison_arrays.npz "
                "as empirical_flat_model_variance for diagnostics."
            ),
        },
        "centering": {
            "enabled_for_mean_comparison": bool(cfg.center_classwise_logits),
            "description": (
                "For the centered mean comparison, subtract the mean across classes "
                "from each test point's logit vector."
            ),
        },
        "summary": {
            "empirical_ensemble_mean_accuracy": empirical_mean_accuracy,
            "first_order_accuracy": analytical_accuracy,
            "first_order_error_percent": float(analytical["first_order_error_percent"]),
            "epsilon": float(analytical["epsilon"]),
            "mean_centered_comparison": mean_summary,
            "mean_raw_comparison": raw_mean_summary,
            "simple_delta_fit_centered_comparison": simple_fit_summary,
            "simple_delta_fit_raw_comparison": simple_fit_raw_summary,
            "simple_delta_fit_centered": {
                "formula": "output = a * Delta E^i_alpha[<x_* x_alpha>] + b",
                "target": "nested empirical ensemble mean logits after class centering",
                "feature": (
                    "Delta E^i_alpha[<x_* x_alpha>] = C/N sum_alpha y_alpha,i <x_* x_alpha> "
                    "- 1/N sum_alpha <x_* x_alpha>, then class-centered before fitting"
                ),
                "a": float(simple_fit["simple_fit_centered_a"]),
                "b": float(simple_fit["simple_fit_centered_b"]),
                "r2": simple_fit["simple_fit_centered_r2"],
            },
            "simple_delta_fit_raw": {
                "formula": "output = a * Delta E^i_alpha[<x_* x_alpha>] + b",
                "target": "raw nested empirical ensemble mean logits",
                "feature": (
                    "Delta E^i_alpha[<x_* x_alpha>] = C/N sum_alpha y_alpha,i <x_* x_alpha> "
                    "- 1/N sum_alpha <x_* x_alpha>, without class centering"
                ),
                "a": float(simple_fit["simple_fit_raw_a"]),
                "b": float(simple_fit["simple_fit_raw_b"]),
                "r2": simple_fit["simple_fit_raw_r2"],
            },
            "variance_comparison": variance_summary,
        },
        "artifacts": {
            "mean_centered_scatter_plot": mean_plot,
            "mean_raw_scatter_plot": raw_mean_plot,
            "simple_delta_fit_centered_scatter_plot": simple_fit_plot,
            "simple_delta_fit_raw_scatter_plot": simple_fit_raw_plot,
            "variance_scatter_plot": variance_plot,
            "per_test_class_csv": "per_test_class_comparison.csv" if cfg.save_per_point_csv else None,
            "arrays_npz": "comparison_arrays.npz" if cfg.save_npz else None,
        },
    }

    database_path = output_dir / "comparison_database.json"
    database_path.write_text(json.dumps(database, indent=2), encoding="utf-8")
    return database_path


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    cfg = build_config_from_user_inputs()
    test_outputs_path = Path(cfg.test_outputs_path)
    if not test_outputs_path.is_absolute():
        test_outputs_path = project_dir / test_outputs_path
    if not test_outputs_path.exists():
        raise FileNotFoundError(f"TEST_OUTPUTS_PATH does not exist: {test_outputs_path}")

    ensemble_dir, metadata = _load_metadata(test_outputs_path)
    saved_cfg = metadata["config"]
    data_root = _resolve_root(cfg.data_root, project_dir)
    num_full_test = int(metadata.get("num_test", 10_000))
    selected_test_indices = _select_test_indices(
        num_available=num_full_test,
        test_limit=cfg.test_limit,
        test_indices=cfg.test_indices,
    )

    print(f"Reading saved test outputs: {test_outputs_path}", flush=True)
    print(
        f"Matched metadata: p={saved_cfg['noise_probability']:g}, N={metadata['num_train']}, "
        f"d={saved_cfg['img_hw'][0]}x{saved_cfg['img_hw'][1]}, act={saved_cfg['activation']}, "
        f"depth={saved_cfg['depth']}, width={saved_cfg['width']}.",
        flush=True,
    )

    print("Loading clean MNIST train subset and selected clean test points...", flush=True)
    data = _load_exact_data(
        metadata=metadata,
        data_root=data_root,
        selected_test_indices=selected_test_indices,
    )

    runtime_device = _choose_device(cfg.allow_cpu)
    analytical_device = _comparison_device_for_analytics(runtime_device)
    print(f"Using analytical device: {analytical_device}", flush=True)

    print("Streaming CSV to compute empirical MLP ensemble mean and variance...", flush=True)
    csv_stats = _read_csv_ensemble_statistics(
        test_outputs_path=test_outputs_path,
        selected_test_indices=selected_test_indices,
        num_classes=data.num_classes,
        max_models=cfg.max_models,
        progress_every_rows=cfg.csv_progress_every_rows,
        center_classwise_logits=cfg.center_classwise_logits,
    )
    labels_from_csv = np.asarray(csv_stats["labels_from_csv"], dtype=np.int64)
    if not np.array_equal(labels_from_csv, data.y_test):
        raise RuntimeError(
            "Labels in test_outputs.csv.gz do not match labels reconstructed from MNIST. "
            "This usually means test_index does not refer to the original MNIST test index."
        )

    print("Computing first-order analytical mean and variance...", flush=True)
    analytical = _compute_first_order_analytical_statistics(
        metadata=metadata,
        data=data,
        device=analytical_device,
        progress_every=cfg.analytical_progress_every,
        center_classwise_logits=cfg.center_classwise_logits,
    )

    database_path = _save_outputs(
        project_dir=project_dir,
        cfg=cfg,
        test_outputs_path=test_outputs_path,
        ensemble_dir=ensemble_dir,
        metadata=metadata,
        data=data,
        csv_stats=csv_stats,
        analytical=analytical,
    )

    mean_summary = _comparison_summary(
        np.asarray(csv_stats["mean_mlp_ensemble_centered"], dtype=np.float64),
        np.asarray(analytical["mean_first_order_analytical_centered"], dtype=np.float64),
    )
    variance_summary = _comparison_summary(
        np.asarray(csv_stats["variance_mlp_ensemble"], dtype=np.float64),
        np.asarray(analytical["variance_first_order_analytical"], dtype=np.float64),
    )
    output_database = _load_json(database_path)
    simple_summary = output_database["summary"]["simple_delta_fit_centered_comparison"]
    simple_fit_centered = output_database["summary"]["simple_delta_fit_centered"]
    simple_fit_raw = output_database["summary"]["simple_delta_fit_raw"]

    print(f"Saved comparison database: {database_path}", flush=True)
    print(
        "Centered mean comparison: "
        f"MAE={mean_summary['mae']:.6e}, RMSE={mean_summary['rmse']:.6e}, "
        f"corr={mean_summary['correlation']}",
        flush=True,
    )
    print(
        "Variance comparison: "
        f"MAE={variance_summary['mae']:.6e}, RMSE={variance_summary['rmse']:.6e}, "
        f"corr={variance_summary['correlation']}",
        flush=True,
    )
    print(
        "Simple fitted Delta predictor (centered): "
        f"a={simple_fit_centered['a']:.6e}, b={simple_fit_centered['b']:.6e}, "
        f"R2={simple_fit_centered['r2']}, RMSE={simple_summary['rmse']:.6e}",
        flush=True,
    )
    raw_simple_summary = output_database["summary"]["simple_delta_fit_raw_comparison"]
    print(
        "Simple fitted Delta predictor (raw): "
        f"a={simple_fit_raw['a']:.6e}, b={simple_fit_raw['b']:.6e}, "
        f"R2={simple_fit_raw['r2']}, RMSE={raw_simple_summary['rmse']:.6e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
