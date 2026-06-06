#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from modules.data import (
    DatasetSubset,
    dataset_to_numpy_matrix,
    load_image_classification_dataset,
    normalize_feature_matrix_zero_mean_unit_rms,
)

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

ROOT_DIR = Path(
    "/data2/jt577/2026/papers/noisy_networks/06.2026/fig2.1/trained_mlp_nested_ensembles"
)

OUTPUT_CSV = ROOT_DIR / "centroid_fit_summary.csv"

# Set to an integer like 1000 if you want to limit how many test examples
# are used per folder. Leave as None to use all test indices found in the CSV.
MAX_TEST_EXAMPLES = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_folder_name(folder_name: str) -> dict[str, object]:
    out: dict[str, object] = {"folder": folder_name}

    match_n = re.search(r"__N=(\d+)", folder_name)
    out["N"] = int(match_n.group(1)) if match_n else None

    match_d = re.search(r"__d=(\d+)x(\d+)", folder_name)
    if match_d:
        h = int(match_d.group(1))
        w = int(match_d.group(2))
        out["d_rows"] = h
        out["d_cols"] = w
        out["d"] = h * w
        out["d_label"] = f"{h}x{w}"
    else:
        out["d_rows"] = None
        out["d_cols"] = None
        out["d"] = None
        out["d_label"] = None

    match_p = re.search(r"__p=([0-9.eE+-]+)", folder_name)
    out["p"] = float(match_p.group(1)) if match_p else None

    match_dataset = re.search(r"__dataset=(.+?)__", folder_name)
    out["dataset"] = match_dataset.group(1) if match_dataset else "mnist"

    match_loss = re.search(r"__loss=([^_]+)", folder_name)
    out["loss"] = match_loss.group(1) if match_loss else None

    match_act = re.search(r"__act=([^_]+)", folder_name)
    out["activation"] = match_act.group(1) if match_act else None

    match_L = re.search(r"__L=(\d+)", folder_name)
    out["depth"] = int(match_L.group(1)) if match_L else None

    match_width = re.search(r"__width=(\d+)", folder_name)
    out["width"] = int(match_width.group(1)) if match_width else None

    return out


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    var_x = float(np.mean((x - x_mean) ** 2))
    if var_x <= 0.0:
        raise ValueError("Centroid values have zero variance; cannot fit affine model.")

    cov_xy = float(np.mean((x - x_mean) * (y - y_mean)))
    a = cov_xy / var_x
    b = y_mean - a * x_mean
    return a, b


def load_processed_arrays_from_metadata(
    metadata: dict,
    project_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = metadata["config"]
    num_classes = int(cfg["num_classes"])

    data_root = Path(cfg["data_root"])
    data_root = data_root if data_root.is_absolute() else project_dir / data_root

    img_hw = tuple(cfg["img_hw"])
    train_raw, test_raw = load_image_classification_dataset(
        root=data_root,
        img_hw=img_hw,
        dataset_name=cfg.get("dataset", "mnist"),
    )

    train_indices = np.asarray(metadata["train_indices"], dtype=np.int64)
    train_dataset = DatasetSubset(train_raw, train_indices)

    x_train, _, y_train = dataset_to_numpy_matrix(train_dataset, num_classes)
    x_test, _, y_test = dataset_to_numpy_matrix(test_raw, num_classes)

    # Match the training script exactly
    x_train = normalize_feature_matrix_zero_mean_unit_rms(x_train)
    x_test = normalize_feature_matrix_zero_mean_unit_rms(x_test)

    return (
        np.asarray(x_train, dtype=np.float64),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(x_test, dtype=np.float64),
        np.asarray(y_test, dtype=np.int64),
    )


def build_centroid_sums_from_matrix(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    dim = int(x_train.shape[1])
    class_sums = np.zeros((num_classes, dim), dtype=np.float64)
    total_sum = np.sum(x_train, axis=0, dtype=np.float64)

    for class_id in range(num_classes):
        class_mask = (y_train == class_id)
        class_sums[class_id] = np.sum(x_train[class_mask], axis=0, dtype=np.float64)

    return class_sums, total_sum


def centroid_scores_for_test_indices(
    x_test: np.ndarray,
    test_indices: list[int],
    class_sums: np.ndarray,
    total_sum: np.ndarray,
    num_train: int,
    num_classes: int,
) -> np.ndarray:
    dim = int(x_test.shape[1])
    class_scale = float(num_classes) / float(num_train)
    overall_scale = 1.0 / float(num_train)

    out = []
    for test_index in test_indices:
        x_star = x_test[int(test_index)]
        total_dot = float(np.dot(x_star, total_sum)) / float(dim)

        scores = []
        for class_id in range(num_classes):
            class_dot = float(np.dot(x_star, class_sums[class_id])) / float(dim)
            score_i = class_scale * class_dot - overall_scale * total_dot
            scores.append(score_i)

        out.append(scores)

    return np.asarray(out, dtype=np.float64)

def compute_mean_and_variance_logits(
    csv_path: Path,
    num_classes: int,
    max_test_examples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    df = pd.read_csv(csv_path)

    if max_test_examples is not None:
        keep_test_indices = sorted(df["test_index"].unique())[: int(max_test_examples)]
        df = df[df["test_index"].isin(keep_test_indices)]

    realization_col = "noise_dataset_index"
    if realization_col not in df.columns:
        raise ValueError(
            f"Missing required column '{realization_col}'. Available columns: {list(df.columns)}"
        )

    logit_cols = [f"logit_{class_id}" for class_id in range(num_classes)]

    # First average over the 10 networks for each noisy dataset realization
    df_means = (
        df.groupby(["test_index", realization_col], sort=True)[logit_cols]
        .mean()
        .reset_index()
    )

    mean_logits_alltest = []
    var_logits_alltest = []
    test_indices_seen = []

    # Then compute mean/variance across noisy dataset realizations
    grouped = df_means.groupby("test_index", sort=True)
    for test_index, group in grouped:
        logits = group[logit_cols].to_numpy(dtype=np.float64)  # (n_realizations, num_classes)

        mean_logits = np.mean(logits, axis=0)
        var_logits = np.mean(logits**2, axis=0) - mean_logits**2

        mean_logits_alltest.append(mean_logits)
        var_logits_alltest.append(var_logits)
        test_indices_seen.append(int(test_index))

    mean_logits_alltest = np.asarray(mean_logits_alltest, dtype=np.float64)
    var_logits_alltest = np.asarray(var_logits_alltest, dtype=np.float64)

    return mean_logits_alltest, var_logits_alltest, test_indices_seen


def process_ensemble_dir(
    ensemble_dir: Path,
    project_dir: Path,
    max_test_examples: int | None = None,
) -> dict[str, object]:
    metadata_path = ensemble_dir / "metadata.json"
    csv_path = ensemble_dir / "test_outputs.csv.gz"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {ensemble_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing test_outputs.csv.gz in {ensemble_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cfg = metadata["config"]

    num_classes = int(cfg["num_classes"])
    feature_dim = int(metadata["feature_dim"])

    # Reconstruct the exact processed train/test features used by training
    x_train, y_train, x_test, y_test = load_processed_arrays_from_metadata(
        metadata=metadata,
        project_dir=project_dir,
    )

    if int(x_train.shape[1]) != feature_dim:
        raise ValueError(
            f"Feature-dimension mismatch for {ensemble_dir.name}: "
            f"metadata says {feature_dim}, reconstructed x_train has {x_train.shape[1]}"
        )

    class_sums, total_sum = build_centroid_sums_from_matrix(
        x_train=x_train,
        y_train=y_train,
        num_classes=num_classes,
    )

    mean_logits_alltest, var_logits_alltest, test_indices_seen = compute_mean_and_variance_logits(
        csv_path=csv_path,
        num_classes=num_classes,
        max_test_examples=max_test_examples,
    )

    centroids = centroid_scores_for_test_indices(
        x_test=x_test,
        test_indices=test_indices_seen,
        class_sums=class_sums,
        total_sum=total_sum,
        num_train=len(x_train),
        num_classes=num_classes,
    )

    a, b = fit_affine(centroids, mean_logits_alltest)

    # c = average std dev across test index and class, matching your earlier script
    var_logits_alltest = np.maximum(var_logits_alltest, 0.0)
    c = float(np.mean(np.sqrt(var_logits_alltest)))

    if c > 0.0:
        snr = float((a / c) ** 2)
    else:
        snr = float("inf")

    # optional fit diagnostics
    x_flat = centroids.reshape(-1)
    y_flat = mean_logits_alltest.reshape(-1)
    y_fit = a * x_flat + b
    residual = y_flat - y_fit
    mse = float(np.mean(residual**2))
    rmse = float(math.sqrt(mse))

    parsed = parse_folder_name(ensemble_dir.name)
    return {
        "folder": ensemble_dir.name,
        "path": str(ensemble_dir),
        "dataset": cfg.get("dataset", parsed["dataset"]),
        "N": parsed["N"],
        "d_label": parsed["d_label"],
        "d_rows": parsed["d_rows"],
        "d_cols": parsed["d_cols"],
        "d": parsed["d"],
        "p": parsed["p"],
        "loss": parsed["loss"],
        "activation": parsed["activation"],
        "depth": parsed["depth"],
        "width": parsed["width"],
        "feature_dim": feature_dim,
        "num_train": int(x_train.shape[0]),
        "num_test_used": len(test_indices_seen),
        "num_classes": num_classes,
        "a": a,
        "b": b,
        "c": c,
        "snr": snr,
        "mse_fit": mse,
        "rmse_fit": rmse,
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parent

    ensemble_dirs = sorted(
        [
            path
            for path in ROOT_DIR.iterdir()
            if path.is_dir() and path.name.startswith("mlp_ensemble__")
        ]
    )

    if not ensemble_dirs:
        raise FileNotFoundError(f"No ensemble folders found in {ROOT_DIR}")

    rows = []
    for ensemble_dir in ensemble_dirs:
        print(f"\nProcessing {ensemble_dir}")
        try:
            row = process_ensemble_dir(
                ensemble_dir=ensemble_dir,
                project_dir=project_dir,
                max_test_examples=MAX_TEST_EXAMPLES,
            )
            rows.append(row)
            print(
                f"Done: N={row['N']}, d={row['d_label']}, "
                f"a={row['a']:.6g}, b={row['b']:.6g}, c={row['c']:.6g}, snr={row['snr']:.6g}"
            )
        except Exception as exc:
            print(f"Skipping {ensemble_dir.name}: {exc}")

    if not rows:
        raise RuntimeError("No folders were processed successfully.")

    summary_df = pd.DataFrame(rows)
    sort_cols = [col for col in ["N", "d", "p", "activation", "depth", "width"] if col in summary_df.columns]
    if sort_cols:
        summary_df = summary_df.sort_values(sort_cols, na_position="last")

    summary_df.to_csv(OUTPUT_CSV, index=False)

    print("\nWrote summary CSV:")
    print(OUTPUT_CSV)
    print(summary_df)


if __name__ == "__main__":
    main()
