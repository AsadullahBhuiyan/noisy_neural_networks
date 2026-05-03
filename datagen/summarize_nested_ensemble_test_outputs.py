from __future__ import annotations

import csv
import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


################################################################################
# USER INPUTS
#
# Edit only this block for normal use. For each ensemble folder, this script
# reads test_outputs.csv.gz, averages over initM models within each noisy
# training dataset, then averages those per-noise means. It writes one CSV row
# per ensemble folder, test point, and label.
################################################################################

ENSEMBLE_DIRS = [
    "/data2/jt577/2026/04.2026/NN_noise/trained_mlp_nested_ensembles_featureDim/mlp_ensemble__additive_gaussian__p=0.98__N=10000__d=28x28__loss=mse__act=erf__L=3__width=2048__noiseK=100__initM=10__models=1000",
    "/data2/jt577/2026/04.2026/NN_noise/trained_mlp_nested_ensembles_featureDim/mlp_ensemble__additive_gaussian__p=0.98__N=10000__d=43x43__loss=mse__act=erf__L=3__width=2048__noiseK=100__initM=10__models=1000",
    "/data2/jt577/2026/04.2026/NN_noise/trained_mlp_nested_ensembles_featureDim/mlp_ensemble__additive_gaussian__p=0.98__N=10000__d=55x55__loss=mse__act=erf__L=3__width=2048__noiseK=100__initM=10__models=1000",
    "/data2/jt577/2026/04.2026/NN_noise/trained_mlp_nested_ensembles_featureDim/mlp_ensemble__additive_gaussian__p=0.98__N=10000__d=64x64__loss=mse__act=erf__L=3__width=2048__noiseK=100__initM=10__models=1000"
]

TEST_OUTPUTS_FILENAME = "test_outputs.csv.gz"
OUTPUT_CSV = "outputs/ensemble_test_featureDim_sweep/nested_means_stdvs.csv"

# Use None for all saved test points, or an integer for the first K test_index
# values. TEST_INDICES overrides TEST_LIMIT when provided.
TEST_LIMIT = 100
TEST_INDICES = None  # e.g. [0, 1, 7, 42]

# Use None for all labels, or a list such as [0, 3, 8].
CLASS_IDS = None

# Optional quick-test controls. Leave as None for the full nested ensemble.
MAX_NOISE_DATASETS = None
MAX_INITS_PER_NOISE_DATASET = None

CSV_PROGRESS_EVERY_ROWS = 1_000_000

################################################################################
# END USER INPUTS
################################################################################


@dataclass
class SummaryConfig:
    ensemble_dirs: list[str]
    test_outputs_filename: str
    output_csv: str
    test_limit: int | None
    test_indices: list[int] | None
    class_ids: list[int] | None
    max_noise_datasets: int | None
    max_inits_per_noise_dataset: int | None
    csv_progress_every_rows: int


def build_config_from_user_inputs() -> SummaryConfig:
    if not ENSEMBLE_DIRS:
        raise ValueError("ENSEMBLE_DIRS must contain at least one folder.")
    if TEST_LIMIT is not None and int(TEST_LIMIT) < 1:
        raise ValueError("TEST_LIMIT must be None or at least 1.")
    if TEST_INDICES is not None and len(TEST_INDICES) < 1:
        raise ValueError("TEST_INDICES must be None or a non-empty list.")
    if CLASS_IDS is not None and len(CLASS_IDS) < 1:
        raise ValueError("CLASS_IDS must be None or a non-empty list.")
    if MAX_NOISE_DATASETS is not None and int(MAX_NOISE_DATASETS) < 1:
        raise ValueError("MAX_NOISE_DATASETS must be None or at least 1.")
    if MAX_INITS_PER_NOISE_DATASET is not None and int(MAX_INITS_PER_NOISE_DATASET) < 1:
        raise ValueError("MAX_INITS_PER_NOISE_DATASET must be None or at least 1.")

    return SummaryConfig(
        ensemble_dirs=[str(path) for path in ENSEMBLE_DIRS],
        test_outputs_filename=str(TEST_OUTPUTS_FILENAME),
        output_csv=str(OUTPUT_CSV),
        test_limit=None if TEST_LIMIT is None else int(TEST_LIMIT),
        test_indices=None if TEST_INDICES is None else [int(v) for v in TEST_INDICES],
        class_ids=None if CLASS_IDS is None else [int(v) for v in CLASS_IDS],
        max_noise_datasets=None if MAX_NOISE_DATASETS is None else int(MAX_NOISE_DATASETS),
        max_inits_per_noise_dataset=(
            None if MAX_INITS_PER_NOISE_DATASET is None else int(MAX_INITS_PER_NOISE_DATASET)
        ),
        csv_progress_every_rows=int(CSV_PROGRESS_EVERY_ROWS),
    )


def _project_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else _project_dir() / path


def _open_text_maybe_gzip(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_test_indices(metadata: dict[str, object], cfg: SummaryConfig) -> np.ndarray:
    num_test = int(metadata.get("num_test", 10_000))
    if cfg.test_indices is not None:
        selected = np.asarray(cfg.test_indices, dtype=np.int64)
    elif cfg.test_limit is not None:
        selected = np.arange(min(int(cfg.test_limit), num_test), dtype=np.int64)
    else:
        selected = np.arange(num_test, dtype=np.int64)

    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("Selected test indices must be a non-empty one-dimensional list.")
    if np.any(selected < 0) or np.any(selected >= num_test):
        raise IndexError(f"Selected test indices must lie in [0, {num_test}).")
    if np.unique(selected).shape[0] != selected.shape[0]:
        raise ValueError("Selected test indices contain duplicates.")
    return selected.astype(np.int64)


def _select_class_ids(metadata: dict[str, object], cfg: SummaryConfig) -> np.ndarray:
    num_classes = int(metadata["config"]["num_classes"])
    if cfg.class_ids is None:
        selected = np.arange(num_classes, dtype=np.int64)
    else:
        selected = np.asarray(cfg.class_ids, dtype=np.int64)
    if np.any(selected < 0) or np.any(selected >= num_classes):
        raise IndexError(f"CLASS_IDS must lie in [0, {num_classes}).")
    if np.unique(selected).shape[0] != selected.shape[0]:
        raise ValueError("CLASS_IDS contains duplicates.")
    return selected.astype(np.int64)


def _logit_columns(fieldnames: list[str]) -> list[str]:
    columns = [name for name in fieldnames if name.startswith("logit_")]
    if not columns:
        raise ValueError("The CSV does not contain logit_* columns.")
    return sorted(columns, key=lambda name: int(name.split("_", 1)[1]))


def _row_noise_index(row: dict[str, str]) -> int:
    if row.get("noise_dataset_index") not in {None, ""}:
        return int(row["noise_dataset_index"])
    return int(row["noise_dataset_number"]) - 1


def _row_init_index(row: dict[str, str], model_number: int) -> int:
    if row.get("init_index") not in {None, ""}:
        return int(row["init_index"])
    if row.get("init_number") not in {None, ""}:
        return int(row["init_number"]) - 1
    return int(model_number)


def _accepted_noise_index(
    noise_index: int,
    *,
    accepted_noise_order: list[int],
    accepted_noise_set: set[int],
    max_noise_datasets: int | None,
) -> bool:
    if noise_index in accepted_noise_set:
        return True
    if max_noise_datasets is not None and len(accepted_noise_set) >= int(max_noise_datasets):
        return False
    accepted_noise_set.add(noise_index)
    accepted_noise_order.append(noise_index)
    return True


def _accepted_init_index(
    *,
    noise_index: int,
    init_index: int,
    accepted_inits_by_noise: dict[int, set[int]],
    init_order_by_noise: dict[int, list[int]],
    max_inits_per_noise_dataset: int | None,
) -> bool:
    accepted = accepted_inits_by_noise.setdefault(noise_index, set())
    order = init_order_by_noise.setdefault(noise_index, [])
    if init_index in accepted:
        return True
    if max_inits_per_noise_dataset is not None and len(accepted) >= int(max_inits_per_noise_dataset):
        return False
    accepted.add(init_index)
    order.append(init_index)
    return True


def _summarize_one_folder(
    ensemble_dir: Path,
    cfg: SummaryConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    metadata_path = ensemble_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {ensemble_dir}")
    metadata = _load_json(metadata_path)
    saved_cfg = metadata["config"]
    num_classes = int(saved_cfg["num_classes"])

    test_outputs_path = ensemble_dir / cfg.test_outputs_filename
    if not test_outputs_path.exists():
        raise FileNotFoundError(f"Missing {cfg.test_outputs_filename} in {ensemble_dir}")

    selected_test_indices = _select_test_indices(metadata, cfg)
    selected_class_ids = _select_class_ids(metadata, cfg)
    test_index_to_position = {int(test_index): pos for pos, test_index in enumerate(selected_test_indices)}
    num_selected_tests = int(selected_test_indices.shape[0])

    per_noise_sum: dict[int, np.ndarray] = {}
    per_noise_counts: dict[int, np.ndarray] = {}
    flat_sum = np.zeros((num_selected_tests, num_classes), dtype=np.float64)
    flat_sq_sum = np.zeros_like(flat_sum)
    flat_counts = np.zeros((num_selected_tests,), dtype=np.int64)
    true_labels = np.full((num_selected_tests,), -1, dtype=np.int64)

    accepted_noise_order: list[int] = []
    accepted_noise_set: set[int] = set()
    accepted_inits_by_noise: dict[int, set[int]] = {}
    init_order_by_noise: dict[int, list[int]] = {}
    accepted_model_numbers: set[int] = set()

    row_count = 0
    used_row_count = 0
    start = time.perf_counter()

    with _open_text_maybe_gzip(test_outputs_path, "rt") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{test_outputs_path} has no header.")
        logit_cols = _logit_columns(reader.fieldnames)
        if len(logit_cols) != num_classes:
            raise ValueError(f"Expected {num_classes} logit columns, found {len(logit_cols)}.")

        for row in reader:
            row_count += 1
            if cfg.csv_progress_every_rows > 0 and row_count % int(cfg.csv_progress_every_rows) == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"{ensemble_dir.name}: read {row_count:,} rows, used {used_row_count:,} rows "
                    f"from {len(accepted_noise_set)} noise datasets in {elapsed:.1f}s.",
                    flush=True,
                )

            test_index = int(row["test_index"])
            test_position = test_index_to_position.get(test_index)
            if test_position is None:
                continue

            noise_index = _row_noise_index(row)
            if not _accepted_noise_index(
                noise_index,
                accepted_noise_order=accepted_noise_order,
                accepted_noise_set=accepted_noise_set,
                max_noise_datasets=cfg.max_noise_datasets,
            ):
                continue

            model_number = int(row["model_number"])
            init_index = _row_init_index(row, model_number)
            if not _accepted_init_index(
                noise_index=noise_index,
                init_index=init_index,
                accepted_inits_by_noise=accepted_inits_by_noise,
                init_order_by_noise=init_order_by_noise,
                max_inits_per_noise_dataset=cfg.max_inits_per_noise_dataset,
            ):
                continue

            logits = np.asarray([float(row[col]) for col in logit_cols], dtype=np.float64)
            if noise_index not in per_noise_sum:
                per_noise_sum[noise_index] = np.zeros((num_selected_tests, num_classes), dtype=np.float64)
                per_noise_counts[noise_index] = np.zeros((num_selected_tests,), dtype=np.int64)

            per_noise_sum[noise_index][test_position] += logits
            per_noise_counts[noise_index][test_position] += 1
            flat_sum[test_position] += logits
            flat_sq_sum[test_position] += logits * logits
            flat_counts[test_position] += 1
            true_labels[test_position] = int(row["true_label"])
            accepted_model_numbers.add(model_number)
            used_row_count += 1

    if not per_noise_sum:
        raise RuntimeError(f"No rows were selected from {test_outputs_path}")
    if np.any(flat_counts == 0):
        missing = selected_test_indices[flat_counts == 0]
        raise RuntimeError(f"Missing selected test indices in CSV: {missing[:20].tolist()}")

    noise_indices = np.asarray(sorted(per_noise_sum), dtype=np.int64)
    per_noise_means = []
    init_counts_by_noise: dict[str, int] = {}
    for noise_index in noise_indices:
        counts = per_noise_counts[int(noise_index)]
        if np.any(counts == 0):
            missing = selected_test_indices[counts == 0]
            raise RuntimeError(
                f"Noise dataset {int(noise_index)} is missing selected tests: {missing[:20].tolist()}"
            )
        if np.unique(counts).shape[0] != 1:
            raise RuntimeError(
                f"Noise dataset {int(noise_index)} has unequal init counts across selected tests: "
                f"{int(np.min(counts))} to {int(np.max(counts))}"
            )
        init_count = int(counts[0])
        init_counts_by_noise[str(int(noise_index))] = init_count
        per_noise_means.append(per_noise_sum[int(noise_index)] / float(init_count))

    per_noise_means_arr = np.stack(per_noise_means, axis=0)
    nested_mean = np.mean(per_noise_means_arr, axis=0)
    nested_std_population = np.std(per_noise_means_arr, axis=0, ddof=0)
    nested_std_sample = (
        np.std(per_noise_means_arr, axis=0, ddof=1)
        if per_noise_means_arr.shape[0] > 1
        else np.zeros_like(nested_std_population)
    )
    nested_sem = nested_std_sample / np.sqrt(float(per_noise_means_arr.shape[0]))

    flat_model_count = int(flat_counts[0])
    if np.unique(flat_counts).shape[0] != 1:
        raise RuntimeError(
            f"Flat model counts differ across selected tests: {int(np.min(flat_counts))} to "
            f"{int(np.max(flat_counts))}"
        )
    flat_mean = flat_sum / float(flat_model_count)
    flat_var = flat_sq_sum / float(flat_model_count) - flat_mean * flat_mean
    flat_var = np.maximum(flat_var, 0.0)
    flat_std_population = np.sqrt(flat_var)

    init_counts = np.asarray(list(init_counts_by_noise.values()), dtype=np.int64)
    common_payload = {
        "ensemble_dir": str(ensemble_dir),
        "ensemble_name": ensemble_dir.name,
        "test_outputs_file": str(test_outputs_path),
        "noise_type": str(saved_cfg.get("noise_type", "")),
        "noise_probability": float(saved_cfg.get("noise_probability", np.nan)),
        "train_size": int(metadata.get("num_train", saved_cfg.get("train_size", -1))),
        "feature_dim": int(metadata.get("feature_dim", -1)),
        "img_h": int(saved_cfg["img_hw"][0]),
        "img_w": int(saved_cfg["img_hw"][1]),
        "loss_name": str(saved_cfg.get("loss_name", "")),
        "activation": str(saved_cfg.get("activation", "")),
        "depth": int(saved_cfg.get("depth", -1)),
        "width": int(saved_cfg.get("width", -1)),
        "configured_num_models": int(saved_cfg.get("num_models", -1)),
        "configured_num_noise_datasets": int(saved_cfg.get("num_noise_datasets", -1)),
        "configured_num_inits_per_noise_dataset": int(saved_cfg.get("num_models_per_noise_dataset", -1)),
        "used_num_models": int(len(accepted_model_numbers)),
        "used_num_noise_datasets": int(noise_indices.shape[0]),
        "used_init_count_min": int(np.min(init_counts)),
        "used_init_count_max": int(np.max(init_counts)),
        "used_flat_model_rows_per_test": int(flat_model_count),
        "rows_read": int(row_count),
        "rows_used": int(used_row_count),
    }

    rows: list[dict[str, object]] = []
    for test_position, test_index in enumerate(selected_test_indices):
        for class_id in selected_class_ids:
            rows.append(
                {
                    **common_payload,
                    "test_index": int(test_index),
                    "true_label": int(true_labels[test_position]),
                    "class_id": int(class_id),
                    "nested_mean": float(nested_mean[test_position, class_id]),
                    "nested_std": float(nested_std_sample[test_position, class_id]),
                    "nested_std_sample": float(nested_std_sample[test_position, class_id]),
                    "nested_std_population": float(nested_std_population[test_position, class_id]),
                    "nested_sem": float(nested_sem[test_position, class_id]),
                    "flat_model_mean": float(flat_mean[test_position, class_id]),
                    "flat_model_std_population": float(flat_std_population[test_position, class_id]),
                }
            )

    summary = {
        **common_payload,
        "selected_test_indices": selected_test_indices.tolist(),
        "selected_class_ids": selected_class_ids.tolist(),
        "used_noise_dataset_indices": noise_indices.tolist(),
        "init_counts_by_noise_dataset_index": init_counts_by_noise,
    }
    return rows, summary


def _output_fieldnames() -> list[str]:
    return [
        "ensemble_dir",
        "ensemble_name",
        "test_outputs_file",
        "noise_type",
        "noise_probability",
        "train_size",
        "feature_dim",
        "img_h",
        "img_w",
        "loss_name",
        "activation",
        "depth",
        "width",
        "configured_num_models",
        "configured_num_noise_datasets",
        "configured_num_inits_per_noise_dataset",
        "used_num_models",
        "used_num_noise_datasets",
        "used_init_count_min",
        "used_init_count_max",
        "used_flat_model_rows_per_test",
        "test_index",
        "true_label",
        "class_id",
        "nested_mean",
        "nested_std",
        "nested_std_sample",
        "nested_std_population",
        "nested_sem",
        "flat_model_mean",
        "flat_model_std_population",
        "rows_read",
        "rows_used",
    ]


def main() -> None:
    cfg = build_config_from_user_inputs()
    output_csv = _resolve_path(cfg.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    summaries = []
    for ensemble_dir_text in cfg.ensemble_dirs:
        ensemble_dir = _resolve_path(ensemble_dir_text)
        print(f"Summarizing {ensemble_dir}", flush=True)
        rows, summary = _summarize_one_folder(ensemble_dir, cfg)
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"Done {ensemble_dir.name}: {summary['used_num_noise_datasets']} noise datasets, "
            f"init count range {summary['used_init_count_min']}..{summary['used_init_count_max']}, "
            f"{len(rows)} output rows.",
            flush=True,
        )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_output_fieldnames())
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = output_csv.with_suffix(".summary.json")
    summary_payload = {
        "output_csv": str(output_csv),
        "num_output_rows": int(len(all_rows)),
        "config": {
            "ensemble_dirs": cfg.ensemble_dirs,
            "test_outputs_filename": cfg.test_outputs_filename,
            "test_limit": cfg.test_limit,
            "test_indices": cfg.test_indices,
            "class_ids": cfg.class_ids,
            "max_noise_datasets": cfg.max_noise_datasets,
            "max_inits_per_noise_dataset": cfg.max_inits_per_noise_dataset,
        },
        "ensembles": summaries,
        "nested_estimator": (
            "For each test_index and class_id, the script first averages logits over init/model rows "
            "within each noise_dataset_index. nested_mean is the average of those per-noise means; "
            "nested_std is the sample standard deviation across those per-noise means."
        ),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Saved nested summary CSV to {output_csv}", flush=True)
    print(f"Saved run summary JSON to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
