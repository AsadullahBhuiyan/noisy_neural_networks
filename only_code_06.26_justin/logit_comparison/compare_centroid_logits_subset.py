from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from modules.data import (
    DatasetSubset,
    dataset_to_numpy_matrix,
    load_image_classification_dataset,
    normalize_feature_matrix_zero_mean_unit_rms,
)


DEFAULT_ENSEMBLE_DIR = Path(
    "/data2/jt577/2026/papers/noisy_networks/06.2026/fig2.1/trained_mlp_nested_ensembles/mlp_ensemble__dataset=mnist__replacement__p=0.97__N=4000__d=28x28__loss=mse__act=erf__L=3__width=2048__noiseK=100__initM=50__models=5000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare init-averaged ensemble logits to fitted class centroids.")
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--max-test-examples", type=int, default=10000)
    parser.add_argument("--max-noise-datasets", type=int, default=None)
    return parser.parse_args()


def load_processed_arrays_from_metadata(
    metadata: dict,
    project_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = metadata["config"]
    num_classes = int(cfg["num_classes"])
    dataset = cfg.get("dataset", "mnist")

    data_root = Path(cfg["data_root"])
    data_root = data_root if data_root.is_absolute() else project_dir / data_root

    img_hw = tuple(cfg["img_hw"])
    train_raw, test_raw = load_image_classification_dataset(
        root=data_root,
        img_hw=img_hw,
        dataset_name=dataset,
    )

    train_indices = np.asarray(metadata["train_indices"], dtype=np.int64)
    train_dataset = DatasetSubset(train_raw, train_indices)

    x_train, _, y_train = dataset_to_numpy_matrix(train_dataset, num_classes)
    x_test, _, y_test = dataset_to_numpy_matrix(test_raw, num_classes)

    # Match train_noisy_mlp_ensemble.py exactly. This second normalization is
    # redundant but idempotent for already centered/unit-RMS examples.
    x_train = normalize_feature_matrix_zero_mean_unit_rms(x_train)
    x_test = normalize_feature_matrix_zero_mean_unit_rms(x_test)

    return (
        np.asarray(x_train, dtype=np.float64),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(x_test, dtype=np.float64),
        np.asarray(y_test, dtype=np.int64),
    )


def train_delta_sums(x_train: np.ndarray, y_train: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    dim = int(x_train.shape[1])
    class_sums = np.zeros((num_classes, dim), dtype=np.float64)
    total_sum = np.sum(x_train, axis=0, dtype=np.float64)
    for class_id in range(num_classes):
        class_sums[class_id] = np.sum(x_train[y_train == class_id], axis=0, dtype=np.float64)
    return class_sums, total_sum


def delta_scores(
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_sums: np.ndarray,
    total_sum: np.ndarray,
    num_train: int,
    max_test: int,
) -> tuple[list[int], list[list[float]]]:
    num_test = min(int(max_test), int(x_test.shape[0]))
    x_subset = np.asarray(x_test[:num_test], dtype=np.float64)
    dim = int(x_subset.shape[1])
    num_classes = int(class_sums.shape[0])
    class_scale = float(num_classes) / float(num_train)
    overall_scale = 1.0 / float(num_train)

    class_dots = x_subset @ class_sums.T / float(dim)
    total_dots = x_subset @ total_sum / float(dim)
    scores = class_scale * class_dots - overall_scale * total_dots[:, None]
    return [int(label) for label in y_test[:num_test]], scores.tolist()


def init_averaged_logits(csv_path: Path, num_noise: int, num_test: int, num_classes: int) -> tuple[list, list[int]]:
    sums = [[[0.0] * num_classes for _ in range(num_test)] for _ in range(num_noise)]
    seen_inits = [set() for _ in range(num_noise)]
    with gzip.open(csv_path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        noise_col = header.index("noise_dataset_index")
        init_col = header.index("init_index")
        test_col = header.index("test_index")
        logit_cols = [header.index(f"logit_{class_id}") for class_id in range(num_classes)]
        for line_number, row in enumerate(reader, start=1):
            noise_index = int(row[noise_col])
            if noise_index >= num_noise:
                break
            test_index = int(row[test_col])
            if test_index >= num_test:
                continue
            seen_inits[noise_index].add(int(row[init_col]))
            for class_id, col in enumerate(logit_cols):
                sums[noise_index][test_index][class_id] += float(row[col])
            if line_number % 1_000_000 == 0:
                print(f"  scanned {line_number:,} rows...", flush=True)

    counts = [len(items) for items in seen_inits]
    for noise_index, count in enumerate(counts):
        if count == 0:
            raise ValueError(f"No logits found for noise_dataset_index={noise_index}.")
        for test_index in range(num_test):
            for class_id in range(num_classes):
                sums[noise_index][test_index][class_id] /= count
    return sums, counts


def average_over_noise(logits: list) -> list[list[float]]:
    num_noise = len(logits)
    num_test = len(logits[0])
    num_classes = len(logits[0][0])
    mean_logits = [[0.0] * num_classes for _ in range(num_test)]
    for noise_rows in logits:
        for test_index, row in enumerate(noise_rows):
            for class_id, value in enumerate(row):
                mean_logits[test_index][class_id] += value
    for test_index in range(num_test):
        for class_id in range(num_classes):
            mean_logits[test_index][class_id] /= float(num_noise)
    return mean_logits


def fit_affine(scores: list[list[float]], logits: list) -> tuple[float, float]:
    flat_scores = [value for row in scores for value in row]
    x_sum = sum(flat_scores)
    x2_sum = sum(value * value for value in flat_scores)
    y_sum = sum(value for row in logits for value in row)
    xy_sum = sum(logits[t][c] * scores[t][c] for t in range(len(scores)) for c in range(len(scores[0])))
    n_total = len(flat_scores)
    denom = n_total * x2_sum - x_sum * x_sum
    if denom == 0.0:
        raise ValueError("Cannot fit a,b because centroid scores have zero variance.")
    a = (n_total * xy_sum - x_sum * y_sum) / denom
    b = (y_sum - a * x_sum) / n_total
    return a, b


def write_outputs(
    output_csv: Path,
    summary_json: Path,
    metadata: dict,
    y_test: list[int],
    scores: list,
    logits: list,
    counts: list[int],
    num_noise_used: int,
    a: float,
    b: float,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "test_index",
        "true_label",
        "class_id",
        "delta_e_score",
        "delta_fit_logit",
        "centroid_score",
        "centroid_fit_logit",
        "ensemble_mean_logit",
        "residual",
    ]
    ss_res = ss_tot = y_sum = y2_sum = fit_sum = fit2_sum = fit_y_sum = 0.0
    n_total = len(scores) * len(scores[0])

    with gzip.open(output_csv, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for test_index, row in enumerate(logits):
            for class_id, ensemble_value in enumerate(row):
                fitted = a * scores[test_index][class_id] + b
                residual = ensemble_value - fitted
                y_sum += ensemble_value
                y2_sum += ensemble_value * ensemble_value
                fit_sum += fitted
                fit2_sum += fitted * fitted
                fit_y_sum += fitted * ensemble_value
                ss_res += residual * residual
                row = {
                    "test_index": test_index,
                    "true_label": y_test[test_index],
                    "class_id": class_id,
                    "delta_e_score": scores[test_index][class_id],
                    "delta_fit_logit": fitted,
                    "centroid_score": scores[test_index][class_id],
                    "centroid_fit_logit": fitted,
                    "ensemble_mean_logit": ensemble_value,
                    "residual": residual,
                }
                writer.writerow(row)

    y_mean = y_sum / n_total
    ss_tot = y2_sum - n_total * y_mean * y_mean
    corr_denom = math.sqrt((n_total * fit2_sum - fit_sum * fit_sum) * (n_total * y2_sum - y_sum * y_sum))
    summary = {
        "ensemble_dir": str(output_csv.parent),
        "comparison_csv": str(output_csv),
        "fit": {"a": a, "b": b},
        "metrics": {"mse": ss_res / n_total, "rmse": math.sqrt(ss_res / n_total), "r2": 1.0 - ss_res / ss_tot, "corr": (n_total * fit_y_sum - fit_sum * y_sum) / corr_denom},
        "num_noise_datasets_used": int(num_noise_used),
        "init_counts_per_noise_dataset": counts,
        "num_test": len(scores),
        "num_classes": len(scores[0]),
        "num_csv_rows": n_total,
        "centroid_formula": "f_i(x_*) = a * Delta E^i[<x_* x_alpha>] + b",
        "delta_e_formula": "Delta E^i = C/N sum_alpha y_alpha,i <x_* x_alpha> - 1/N sum_alpha <x_* x_alpha>, with <.,.> = dot/d",
        "source_config": metadata["config"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["fit"] | summary["metrics"], indent=2))
    print(f"Wrote {output_csv}")
    print(f"Wrote {summary_json}")


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    ensemble_dir = args.ensemble_dir if args.ensemble_dir.is_absolute() else project_dir / args.ensemble_dir
    metadata = json.loads((ensemble_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = metadata["config"]
    dim = int(metadata["feature_dim"])
    num_classes = int(cfg["num_classes"])
    num_noise = int(cfg["num_noise_datasets"])
    if args.max_noise_datasets is not None:
        num_noise = min(num_noise, int(args.max_noise_datasets))

    stem = f"centroid_comparison_test{int(args.max_test_examples)}"
    output_csv = args.output_csv or ensemble_dir / f"{stem}.csv.gz"
    summary_json = args.summary_json or ensemble_dir / f"{stem}_summary.json"
    output_csv = output_csv if output_csv.is_absolute() else project_dir / output_csv
    summary_json = summary_json if summary_json.is_absolute() else project_dir / summary_json

    print("Reconstructing clean normalized train/test arrays from metadata...", flush=True)
    x_train, y_train, x_test, y_test_all = load_processed_arrays_from_metadata(metadata, project_dir)
    if int(x_train.shape[1]) != dim:
        raise ValueError(f"Feature-dimension mismatch: metadata says {dim}, reconstructed data has {x_train.shape[1]}.")
    print("Computing Delta E class sums from clean normalized training data...", flush=True)
    class_sums, total_sum = train_delta_sums(x_train, y_train, num_classes)
    y_test, scores = delta_scores(x_test, y_test_all, class_sums, total_sum, len(x_train), int(args.max_test_examples))
    print(f"Averaging logits for {num_noise} noise datasets and {len(scores)} test examples...", flush=True)
    per_noise_logits, counts = init_averaged_logits(
        ensemble_dir / cfg.get("test_outputs_filename", "test_outputs.csv.gz"),
        num_noise,
        len(scores),
        num_classes,
    )
    logits = average_over_noise(per_noise_logits)
    a, b = fit_affine(scores, logits)
    write_outputs(output_csv, summary_json, metadata, y_test, scores, logits, counts, num_noise, a, b)


if __name__ == "__main__":
    main()
