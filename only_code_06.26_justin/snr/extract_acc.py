#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

ROOT_DIR = Path(
    "/data2/jt577/2026/papers/noisy_networks/06.2026/fig3/trained_mlp_nested_ensembles"
)

OUTPUT_CSV = ROOT_DIR / "empirical_mean_model_accuracy_summary.csv"

# Use None for all test examples, or an integer like 1000 while debugging.
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

    match_p = re.search(r"__p=([0-9.]+)", folder_name)
    out["p"] = float(match_p.group(1)) if match_p else None

    match_loss = re.search(r"__loss=([^_]+)", folder_name)
    out["loss"] = match_loss.group(1) if match_loss else None

    match_act = re.search(r"__act=([^_]+)", folder_name)
    out["activation"] = match_act.group(1) if match_act else None

    match_L = re.search(r"__L=(\d+)", folder_name)
    out["depth"] = int(match_L.group(1)) if match_L else None

    match_width = re.search(r"__width=(\d+)", folder_name)
    out["width"] = int(match_width.group(1)) if match_width else None

    return out


def compute_mean_model_accuracy_by_noise_dataset(
    csv_path: Path,
    max_test_examples: int | None = None,
) -> dict[str, object]:
    """
    For each noisy dataset:
      1. average logits over the networks trained on that noisy dataset
      2. compute predictions from those mean logits
      3. compute test accuracy of that mean-logit classifier

    Returns summary stats across noisy datasets.
    """
    logit_cols = [f"logit_{i}" for i in range(10)]
    usecols = ["test_index", "noise_dataset_index", "true_label"] + logit_cols
    df = pd.read_csv(csv_path, usecols=usecols)

    if max_test_examples is not None:
        keep_test_indices = np.sort(df["test_index"].unique())[: int(max_test_examples)]
        df = df[df["test_index"].isin(keep_test_indices)]

    # Mean logits within each (noise dataset, test example)
    per_noise = (
        df.groupby(["noise_dataset_index", "test_index"], sort=True, observed=True)
        .agg(
            true_label=("true_label", "first"),
            **{col: (col, "mean") for col in logit_cols},
        )
        .reset_index()
    )

    # Predictions from mean logits
    logits_matrix = per_noise[logit_cols].to_numpy(dtype=np.float64)
    pred = np.argmax(logits_matrix, axis=1)
    true = per_noise["true_label"].to_numpy(dtype=np.int64)
    per_noise["correct"] = (pred == true).astype(np.float64)

    # Accuracy for each noisy dataset
    acc_by_noise = (
        per_noise.groupby("noise_dataset_index", sort=True, observed=True)["correct"]
        .mean()
        .to_numpy(dtype=np.float64)
    )

    # How many models were averaged per noisy dataset?
    model_count_df = (
        df.groupby(["noise_dataset_index", "test_index"], sort=True, observed=True)
        .size()
        .groupby("noise_dataset_index", sort=True, observed=True)
        .mean()
    )
    mean_models_per_noise_dataset = float(model_count_df.mean())

    return {
        "mean_model_accuracy": float(np.mean(acc_by_noise)),
        "median_model_accuracy": float(np.median(acc_by_noise)),
        "std_model_accuracy": float(np.std(acc_by_noise, ddof=1)) if acc_by_noise.size > 1 else 0.0,
        "q25_model_accuracy": float(np.quantile(acc_by_noise, 0.25)),
        "q75_model_accuracy": float(np.quantile(acc_by_noise, 0.75)),
        "num_noise_datasets_used": int(acc_by_noise.size),
        "num_test_used": int(per_noise["test_index"].nunique()),
        "mean_models_per_noise_dataset": mean_models_per_noise_dataset,
    }


def process_ensemble_dir(
    ensemble_dir: Path,
    max_test_examples: int | None = None,
) -> dict[str, object]:
    metadata_path = ensemble_dir / "metadata.json"
    csv_path = ensemble_dir / "test_outputs.csv.gz"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {ensemble_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing test_outputs.csv.gz in {ensemble_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_dim = int(metadata["feature_dim"])

    summary = compute_mean_model_accuracy_by_noise_dataset(
        csv_path=csv_path,
        max_test_examples=max_test_examples,
    )

    parsed = parse_folder_name(ensemble_dir.name)
    return {
        "folder": ensemble_dir.name,
        "path": str(ensemble_dir),
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
        **summary,
    }


def main() -> None:
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
                max_test_examples=MAX_TEST_EXAMPLES,
            )
            rows.append(row)
            print(
                f"Done: N={row['N']}, d={row['d_label']}, "
                f"mean_acc={row['mean_model_accuracy']:.6f}, "
                f"std_acc={row['std_model_accuracy']:.6f}, "
                f"noiseK={row['num_noise_datasets_used']}, "
                f"models/noise≈{row['mean_models_per_noise_dataset']:.2f}"
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