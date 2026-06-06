from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jt577")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.data import normalize_dataset_name


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ENSEMBLE_ROOT = PROJECT_DIR / "trained_mlp_nested_ensembles"
DEFAULT_OUTPUT_DIR = DEFAULT_ENSEMBLE_ROOT / "test_accuracy_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute activation-comparison test accuracies from already-saved "
            "test_outputs.csv.gz files. No checkpoints are loaded."
        )
    )
    parser.add_argument("--ensemble-root", type=Path, default=DEFAULT_ENSEMBLE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to include, e.g. mnist fashion_mnist kmnist. Default: all found.",
    )
    parser.add_argument(
        "--noise-types",
        nargs="+",
        default=None,
        help="Noise types to include, e.g. additive_gaussian replacement. Default: all found.",
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        default=None,
        help="Activation names to include. Default: all found.",
    )
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--reuse-results", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_folder_name(folder_name: str) -> dict[str, object]:
    tokens = folder_name.removeprefix("mlp_ensemble__").split("__")
    out: dict[str, object] = {"folder": folder_name, "dataset": "mnist", "noise_type": None}
    for token in tokens:
        if token.startswith("dataset="):
            out["dataset"] = normalize_dataset_name(token.split("=", 1)[1])
        elif "=" not in token and out["noise_type"] is None:
            out["noise_type"] = token

    patterns = {
        "N": r"__N=(\d+)",
        "d_pair": r"__d=(\d+)x(\d+)",
        "p": r"__p=([0-9.eE+-]+)",
        "loss": r"__loss=([^_]+)",
        "activation": r"__act=([^_]+)",
        "depth": r"__L=(\d+)",
        "width": r"__width=(\d+)",
    }
    match = re.search(patterns["N"], folder_name)
    out["N"] = int(match.group(1)) if match else None
    match = re.search(patterns["d_pair"], folder_name)
    if match:
        rows, cols = int(match.group(1)), int(match.group(2))
        out["d_rows"] = rows
        out["d_cols"] = cols
        out["d"] = rows * cols
        out["d_label"] = f"{rows}x{cols}"
    else:
        out["d_rows"] = out["d_cols"] = out["d"] = out["d_label"] = None
    match = re.search(patterns["p"], folder_name)
    out["p"] = float(match.group(1)) if match else None
    for key in ["loss", "activation"]:
        match = re.search(patterns[key], folder_name)
        out[key] = match.group(1) if match else None
    for key in ["depth", "width"]:
        match = re.search(patterns[key], folder_name)
        out[key] = int(match.group(1)) if match else None
    return out


def _allowed(value: str, allowed: set[str] | None) -> bool:
    return allowed is None or value in allowed


def discover_ensemble_dirs(args: argparse.Namespace) -> list[Path]:
    datasets = {normalize_dataset_name(value) for value in args.datasets} if args.datasets else None
    noise_types = {value.strip().lower() for value in args.noise_types} if args.noise_types else None
    activations = {value.strip().lower() for value in args.activations} if args.activations else None

    dirs = []
    for folder in sorted(path for path in args.ensemble_root.iterdir() if path.is_dir()):
        metadata_path = folder / "metadata.json"
        outputs_path = folder / "test_outputs.csv.gz"
        if not metadata_path.exists() or not outputs_path.exists():
            continue
        metadata = load_json(metadata_path)
        cfg = metadata.get("config", {})
        parsed = parse_folder_name(folder.name)
        dataset = normalize_dataset_name(str(cfg.get("dataset", parsed["dataset"])))
        noise_type = str(cfg.get("noise_type", parsed["noise_type"])).strip().lower()
        activation = str(cfg.get("activation", parsed["activation"])).strip().lower()
        if _allowed(dataset, datasets) and _allowed(noise_type, noise_types) and _allowed(activation, activations):
            dirs.append(folder)
    return dirs


def per_model_accuracy_from_test_outputs(
    csv_path: Path,
    *,
    max_test_examples: int | None,
) -> tuple[list[dict[str, object]], int]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    id_cols = [
        col
        for col in ["model_number", "model_index", "noise_dataset_index", "init_index", "test_index", "true_label"]
        if col in header
    ]
    usecols = list(id_cols)
    if "predicted_label" in header:
        usecols.append("predicted_label")
    else:
        usecols.extend([col for col in header if col.startswith("logit_")])

    df = pd.read_csv(csv_path, usecols=usecols)
    if max_test_examples is not None:
        keep = np.sort(df["test_index"].unique())[: int(max_test_examples)]
        df = df[df["test_index"].isin(keep)].copy()

    if "predicted_label" not in df.columns:
        logit_cols = [col for col in df.columns if col.startswith("logit_")]
        if not logit_cols:
            raise ValueError(f"{csv_path} has neither predicted_label nor logit columns.")
        df["predicted_label"] = df[logit_cols].to_numpy(dtype=np.float64).argmax(axis=1)

    df["correct"] = (df["predicted_label"].astype(int) == df["true_label"].astype(int)).astype(float)
    group_cols = [col for col in ["model_number", "model_index", "noise_dataset_index", "init_index"] if col in df.columns]
    if not group_cols:
        raise ValueError(f"{csv_path} has no model identifier columns.")

    grouped = df.groupby(group_cols, sort=True, observed=True)["correct"].agg(["mean", "count"]).reset_index()
    records = []
    for row in grouped.to_dict("records"):
        records.append(
            {
                **{col: int(row[col]) for col in group_cols},
                "accuracy": float(row["mean"]),
                "accuracy_percent": 100.0 * float(row["mean"]),
                "num_test": int(row["count"]),
            }
        )
    return records, int(df["test_index"].nunique())


def process_ensemble_dir(folder: Path, max_test_examples: int | None) -> list[dict[str, object]]:
    metadata = load_json(folder / "metadata.json")
    cfg = metadata.get("config", {})
    parsed = parse_folder_name(folder.name)
    dataset = normalize_dataset_name(str(cfg.get("dataset", parsed["dataset"])))
    noise_type = str(cfg.get("noise_type", parsed["noise_type"])).strip().lower()
    activation = str(cfg.get("activation", parsed["activation"])).strip().lower()
    p = float(cfg.get("noise_probability", parsed["p"]))

    model_records, num_test = per_model_accuracy_from_test_outputs(
        folder / "test_outputs.csv.gz",
        max_test_examples=max_test_examples,
    )

    common = {
        "folder": folder.name,
        "path": str(folder),
        "dataset": dataset,
        "noise_type": noise_type,
        "p": p,
        "N": int(metadata.get("num_train", parsed["N"])),
        "d_label": parsed["d_label"],
        "d_rows": parsed["d_rows"],
        "d_cols": parsed["d_cols"],
        "d": parsed["d"],
        "loss": cfg.get("loss_name", parsed["loss"]),
        "activation": activation,
        "depth": int(cfg.get("depth", parsed["depth"])),
        "width": int(cfg.get("width", parsed["width"])),
        "num_classes": int(cfg.get("num_classes", 10)),
        "feature_dim": int(metadata["feature_dim"]),
        "num_test_available": num_test,
    }
    return [{**common, **record} for record in model_records]


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def read_per_model_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ["p", "accuracy", "accuracy_percent"]:
                row[key] = float(row[key])
            for key in ["N", "depth", "width", "num_classes", "feature_dim", "num_test", "num_test_available"]:
                if row.get(key) not in {"", None}:
                    row[key] = int(row[key])
            rows.append(row)
    return rows


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    metadata_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    key_fields = ["dataset", "noise_type", "activation", "p", "N", "d_label", "loss", "depth", "width"]
    for record in records:
        key = tuple(record[field] for field in key_fields)
        grouped[key].append(float(record["accuracy"]))
        metadata_by_key[key] = {field: record[field] for field in key_fields}

    summaries = []
    for key, values in sorted(grouped.items(), key=lambda item: item[0]):
        array = np.asarray(values, dtype=np.float64)
        std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        mean = float(np.mean(array))
        summaries.append(
            {
                **metadata_by_key[key],
                "num_models": int(array.size),
                "mean_accuracy": mean,
                "std_accuracy": std,
                "mean_accuracy_percent": 100.0 * mean,
                "std_accuracy_percent": 100.0 * std,
            }
        )
    return summaries


def plot_summary(summaries: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    paths = []
    colors = {"erf": "#2b6cb0", "gelu": "#c05621", "swish": "#2f855a", "tanh": "#b83280"}

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(str(row["dataset"]), str(row["noise_type"]))].append(row)

    for (dataset, noise_type), rows in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
        activations = sorted({str(row["activation"]) for row in rows})
        for activation in activations:
            act_rows = sorted([row for row in rows if row["activation"] == activation], key=lambda row: float(row["p"]))
            x = np.asarray([float(row["p"]) for row in act_rows], dtype=np.float64)
            y = np.asarray([float(row["mean_accuracy_percent"]) for row in act_rows], dtype=np.float64)
            yerr = np.asarray([float(row["std_accuracy_percent"]) for row in act_rows], dtype=np.float64)
            color = colors.get(activation)
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2.0, capsize=4.0, label=activation, color=color)

        ax.set_xlabel("noise p")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(f"{dataset}, {noise_type}: per-model test accuracy")
        ax.grid(True, alpha=0.25)
        ax.legend(title="activation")
        path = output_dir / f"activation_comparison_test_accuracy_vs_p__dataset={dataset}__noise={noise_type}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_model_csv = args.output_dir / "per_model_test_accuracy.csv"
    summary_csv = args.output_dir / "summary_test_accuracy_by_p_activation.csv"
    summary_json = args.output_dir / "summary_test_accuracy_by_p_activation.json"

    if args.reuse_results:
        if not per_model_csv.exists():
            raise FileNotFoundError(f"--reuse-results requested, but {per_model_csv} does not exist.")
        records = read_per_model_csv(per_model_csv)
    else:
        folders = discover_ensemble_dirs(args)
        if not folders:
            raise RuntimeError(f"No matching ensemble folders found under {args.ensemble_root}.")
        print(f"Found {len(folders)} ensemble folders with test_outputs.csv.gz.", flush=True)
        records: list[dict[str, Any]] = []
        for folder in folders:
            folder_records = process_ensemble_dir(folder, args.max_test_examples)
            records.extend(folder_records)
            print(f"  {folder.name}: {len(folder_records)} model accuracies", flush=True)

        fieldnames = [
            "dataset",
            "noise_type",
            "activation",
            "p",
            "N",
            "d_label",
            "d_rows",
            "d_cols",
            "d",
            "loss",
            "depth",
            "width",
            "num_classes",
            "feature_dim",
            "model_number",
            "model_index",
            "noise_dataset_index",
            "init_index",
            "accuracy",
            "accuracy_percent",
            "num_test",
            "num_test_available",
            "folder",
            "path",
        ]
        write_csv(per_model_csv, records, fieldnames)

    summaries = summarize(records)
    write_csv(
        summary_csv,
        summaries,
        [
            "dataset",
            "noise_type",
            "activation",
            "p",
            "N",
            "d_label",
            "loss",
            "depth",
            "width",
            "num_models",
            "mean_accuracy",
            "std_accuracy",
            "mean_accuracy_percent",
            "std_accuracy_percent",
        ],
    )
    summary_json.write_text(
        json.dumps(
            {
                "ensemble_root": str(args.ensemble_root),
                "output_dir": str(args.output_dir),
                "num_model_records": len(records),
                "summary": summaries,
                "config": vars(args),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    plot_paths = plot_summary(summaries, args.output_dir)

    print(f"Wrote per-model results: {per_model_csv}", flush=True)
    print(f"Wrote summary: {summary_csv}", flush=True)
    for path in plot_paths:
        print(f"Wrote plot: {path}", flush=True)


if __name__ == "__main__":
    main()
