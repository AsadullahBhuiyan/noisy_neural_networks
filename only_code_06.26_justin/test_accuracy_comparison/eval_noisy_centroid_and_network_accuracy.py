from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

import pandas as pd

from modules.data import (
    DatasetSubset,
    dataset_to_numpy_matrix,
    load_image_classification_dataset,
    normalize_dataset_name,
    normalize_feature_matrix_zero_mean_unit_rms,
)


PROJECT_DIR = Path(__file__).resolve().parent
FIG22_ENSEMBLE_ROOT = PROJECT_DIR.parent / "fig2.2" / "trained_mlp_nested_ensembles"
FIT_CSV = FIG22_ENSEMBLE_ROOT / "centroid_fit_summary.csv"
OUTPUT_DIR = PROJECT_DIR / "trained_centroids"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute centroid-model and empirical test accuracies across all p values "
            "present in matching ensemble folders."
        )
    )
    parser.add_argument("--ensemble-root", type=Path, default=FIG22_ENSEMBLE_ROOT)
    parser.add_argument("--fit-csv", type=Path, default=FIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    # Which fitted row to use for a,b,c
    parser.add_argument("--fit-noise-type", type=str, default="additive_gaussian")
    parser.add_argument("--fit-p", type=float, default=0.97)

    # Which family of folders to evaluate across all p values
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--eval-noise-type", type=str, default="additive_gaussian")
    parser.add_argument("--N", type=int, default=4000)
    parser.add_argument("--d-label", type=str, default="28x28")
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--activation", type=str, default="erf")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=2048)

    parser.add_argument("--eval-num-samples", type=int, default=20)
    parser.add_argument("--eval-test-limit", type=int, default=0, help="0 means full test set.")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_folder_tokens(folder: str) -> dict[str, str]:
    prefix = "mlp_ensemble__"
    if not folder.startswith(prefix):
        raise ValueError(f"Unrecognized folder format: {folder}")
    tokens = folder[len(prefix):].split("__")
    out: dict[str, str] = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    if "dataset" not in out:
        out["dataset"] = "mnist"
    for token in tokens:
        if "=" not in token:
            out["noise_type"] = token
            break
    return out


def infer_noise_type_from_folder(folder: str) -> str:
    tokens = parse_folder_tokens(folder)
    if "noise_type" not in tokens:
        raise ValueError(f"Could not infer noise type from folder: {folder}")
    return tokens["noise_type"]


def infer_dataset_from_folder(folder: str) -> str:
    return normalize_dataset_name(parse_folder_tokens(folder).get("dataset", "mnist"))


def load_fit_row(
    fit_csv: Path,
    *,
    dataset: str,
    noise_type: str,
    fit_p: float,
    N: int,
    d_label: str,
    loss: str,
    activation: str,
    depth: int,
    width: int,
) -> dict[str, object]:
    df = pd.read_csv(fit_csv)

    if "dataset" not in df.columns:
        df["dataset"] = df["folder"].astype(str).map(infer_dataset_from_folder)
    if "noise_type" not in df.columns:
        df["noise_type"] = df["folder"].astype(str).map(infer_noise_type_from_folder)

    rows = df[
        (df["dataset"].astype(str).map(normalize_dataset_name) == normalize_dataset_name(dataset))
        & (df["noise_type"] == noise_type)
        & (df["p"].astype(float) == float(fit_p))
        & (df["N"].astype(int) == int(N))
        & (df["d_label"].astype(str) == str(d_label))
        & (df["loss"].astype(str) == str(loss))
        & (df["activation"].astype(str) == str(activation))
        & (df["depth"].astype(int) == int(depth))
        & (df["width"].astype(int) == int(width))
    ].copy()

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one fit row for noise_type={noise_type}, p={fit_p}, "
            f"dataset={dataset}, N={N}, d_label={d_label}, loss={loss}, activation={activation}, "
            f"depth={depth}, width={width}. Found {len(rows)} rows."
        )

    return rows.iloc[0].to_dict()


def ensemble_dirs_matching(
    root: Path,
    *,
    dataset: str,
    noise_type: str,
    N: int,
    d_label: str,
    loss: str,
    activation: str,
    depth: int,
    width: int,
) -> list[Path]:
    dataset = normalize_dataset_name(dataset)
    pattern = (
        f"mlp_ensemble__dataset={dataset}__{noise_type}__*__N={N}__d={d_label}"
        f"__loss={loss}__act={activation}__L={depth}__width={width}__*"
    )
    return sorted(path for path in root.glob(pattern) if path.is_dir())


def extract_p_from_folder_name(name: str) -> float:
    marker = "__p="
    start = name.index(marker) + len(marker)
    end = name.index("__", start)
    return float(name[start:end])


def load_metadata_for_reference(
    root: Path,
    *,
    dataset: str,
    noise_type: str,
    N: int,
    d_label: str,
    loss: str,
    activation: str,
    depth: int,
    width: int,
) -> tuple[Path, dict]:
    matches = ensemble_dirs_matching(
        root,
        dataset=dataset,
        noise_type=noise_type,
        N=N,
        d_label=d_label,
        loss=loss,
        activation=activation,
        depth=depth,
        width=width,
    )
    if not matches:
        raise FileNotFoundError("Could not find any matching ensemble directories.")
    metadata = load_json(matches[0] / "metadata.json")
    return matches[0], metadata


def load_processed_arrays_from_metadata(
    metadata: dict,
    ensemble_root: Path,
) -> tuple[list[int], list[list[float]], list[int], list[list[float]]]:
    cfg = metadata["config"]
    dataset = cfg.get("dataset", "mnist")
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        project_dir = ensemble_root.parent  # .../fig2.2
        data_root = project_dir / data_root

    train_raw, test_raw = load_image_classification_dataset(
        root=data_root,
        img_hw=tuple(cfg["img_hw"]),
        dataset_name=dataset,
    )
    train_indices = [int(index) for index in metadata["train_indices"]]
    train_dataset = DatasetSubset(train_raw, train_indices)
    x_train, _, y_train = dataset_to_numpy_matrix(train_dataset, int(cfg["num_classes"]))
    x_test, _, y_test = dataset_to_numpy_matrix(test_raw, int(cfg["num_classes"]))

    # Match train_noisy_mlp_ensemble.py exactly. This pass is redundant but
    # idempotent for already centered/unit-RMS examples.
    x_train = normalize_feature_matrix_zero_mean_unit_rms(x_train)
    x_test = normalize_feature_matrix_zero_mean_unit_rms(x_test)

    return (
        [int(label) for label in y_test],
        x_train.astype(float).tolist(),
        [int(label) for label in y_train],
        x_test.astype(float).tolist(),
    )


def delta_scores(metadata: dict, ensemble_root: Path) -> tuple[list[int], list[list[float]]]:
    cfg = metadata["config"]
    dim = int(metadata["feature_dim"])
    num_classes = int(cfg["num_classes"])
    labels, x_train, train_labels, x_test = load_processed_arrays_from_metadata(metadata, ensemble_root)
    if len(x_train[0]) != dim or len(x_test[0]) != dim:
        raise ValueError("Reconstructed image dimension does not match metadata feature_dim.")

    class_sums = [[0.0] * dim for _ in range(num_classes)]
    total_sum = [0.0] * dim

    for image, label in zip(x_train, train_labels):
        for k, value in enumerate(image):
            class_sums[label][k] += value
            total_sum[k] += value

    scores = []

    class_scale = float(num_classes) / float(len(x_train))
    overall_scale = 1.0 / float(len(x_train))

    for image in x_test:
        total_dot = sum(x * s for x, s in zip(image, total_sum)) / float(dim)
        scores.append(
            [
                class_scale * (sum(x * s for x, s in zip(image, class_sum)) / float(dim))
                - overall_scale * total_dot
                for class_sum in class_sums
            ]
        )

    return labels, scores


def centroid_accuracy_samples(
    *,
    labels: list[int],
    scores: list[list[float]],
    p: float,
    A: float,
    b: float,
    c: float,
    num_samples: int,
    test_limit: int,
    seed: int,
) -> list[float]:
    n_test = min(len(labels), test_limit) if test_limit > 0 else len(labels)
    signal_scale = float(A) * (1.0 - float(p))
    accuracies = []

    for sample_index in range(int(num_samples)):
        rng = random.Random(int(seed) + 1_000_003 * sample_index + int(round(float(p) * 100_000)))
        correct = 0

        for test_index in range(n_test):
            row = scores[test_index]
            true_label = labels[test_index]

            best_class = 0
            best_value = signal_scale * row[0] + float(b) + float(c) * rng.gauss(0.0, 1.0)

            for class_id in range(1, len(row)):
                value = signal_scale * row[class_id] + float(b) + float(c) * rng.gauss(0.0, 1.0)
                if value > best_value:
                    best_value = value
                    best_class = class_id

            if best_class == true_label:
                correct += 1

        accuracies.append(correct / float(n_test))

    return accuracies


def summarize(values: list[float]) -> tuple[float, float, float, float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, std, sem, min(values), max(values)


def empirical_accuracy_from_logits(
    ensemble_dir: Path,
    *,
    test_limit: int,
) -> dict[str, float]:
    csv_path = ensemble_dir / "test_outputs.csv.gz"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)

    if test_limit > 0:
        keep_test_indices = sorted(df["test_index"].unique())[: int(test_limit)]
        df = df[df["test_index"].isin(keep_test_indices)]

    logit_cols = [col for col in df.columns if col.startswith("logit_")]
    if "noise_dataset_index" not in df.columns:
        raise ValueError(f"{ensemble_dir.name}: missing noise_dataset_index")
    if "test_index" not in df.columns or "true_label" not in df.columns:
        raise ValueError(f"{ensemble_dir.name}: missing test_index or true_label")
    if not logit_cols:
        raise ValueError(f"{ensemble_dir.name}: no logit columns found")

    # Average over the 10 trained networks for each fixed noisy realization
    grouped_means = (
        df.groupby(["noise_dataset_index", "test_index", "true_label"], sort=True)[logit_cols]
        .mean()
        .reset_index()
    )

    # One test accuracy per noisy realization
    realization_accuracies = []
    for noise_dataset_index, group in grouped_means.groupby("noise_dataset_index", sort=True):
        logits = group[logit_cols].to_numpy(dtype=float)
        true_labels = group["true_label"].to_numpy(dtype=int)
        pred_labels = logits.argmax(axis=1)
        acc = float((pred_labels == true_labels).mean())
        realization_accuracies.append(acc)

    mean, std, sem, min_acc, max_acc = summarize(realization_accuracies)

    return {
        "num_realizations": len(realization_accuracies),
        "mean_test_accuracy": mean,
        "std_test_accuracy": std,
        "pop_std_test_accuracy": statistics.pstdev(realization_accuracies) if len(realization_accuracies) > 1 else 0.0,
        "sem_test_accuracy": sem,
        "min_test_accuracy": min_acc,
        "max_test_accuracy": max_acc,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_row = load_fit_row(
        args.fit_csv,
        dataset=args.dataset,
        noise_type=args.fit_noise_type,
        fit_p=args.fit_p,
        N=args.N,
        d_label=args.d_label,
        loss=args.loss,
        activation=args.activation,
        depth=args.depth,
        width=args.width,
    )

    a = float(fit_row["a"])
    b = float(fit_row["b"])
    c = float(fit_row["c"])
    A = a / (1.0 - float(args.fit_p))

    print("Using fixed centroid coefficients from fit CSV:")
    print(f"  fit_noise_type = {args.fit_noise_type}")
    print(f"  fit_p          = {args.fit_p}")
    print(f"  a              = {a}")
    print(f"  b              = {b}")
    print(f"  c              = {c}")
    print(f"  A = a/(1-p)    = {A}")

    reference_dir, metadata = load_metadata_for_reference(
        args.ensemble_root,
        dataset=args.dataset,
        noise_type=args.eval_noise_type,
        N=args.N,
        d_label=args.d_label,
        loss=args.loss,
        activation=args.activation,
        depth=args.depth,
        width=args.width,
    )

    print("Computing centroid scores...", flush=True)
    labels, scores = delta_scores(metadata, args.ensemble_root)

    matching_dirs = ensemble_dirs_matching(
        args.ensemble_root,
        dataset=args.dataset,
        noise_type=args.eval_noise_type,
        N=args.N,
        d_label=args.d_label,
        loss=args.loss,
        activation=args.activation,
        depth=args.depth,
        width=args.width,
    )

    if not matching_dirs:
        raise FileNotFoundError("No matching ensemble folders found for evaluation.")

    rows = []
    for ensemble_dir in matching_dirs:
        p = extract_p_from_folder_name(ensemble_dir.name)

        centroid_samples = centroid_accuracy_samples(
            labels=labels,
            scores=scores,
            p=p,
            A=A,
            b=b,
            c=c,
            num_samples=int(args.eval_num_samples),
            test_limit=int(args.eval_test_limit),
            seed=int(args.seed),
        )
        centroid_mean, centroid_std, centroid_sem, centroid_min, centroid_max = summarize(centroid_samples)

        empirical = empirical_accuracy_from_logits(
            ensemble_dir,
            test_limit=int(args.eval_test_limit),
        )

        rows.append(
            {
                "noise_type": args.eval_noise_type,
                "dataset": normalize_dataset_name(args.dataset),
                "noise_probability": p,
                "num_models": int(args.eval_num_samples),
                "mean_test_accuracy": centroid_mean,
                "std_test_accuracy": centroid_std,
                "pop_std_test_accuracy": statistics.pstdev(centroid_samples) if len(centroid_samples) > 1 else 0.0,
                "sem_test_accuracy": centroid_sem,
                "min_test_accuracy": centroid_min,
                "max_test_accuracy": centroid_max,
                "empirical_mean_test_accuracy": empirical["mean_test_accuracy"],
                "empirical_std_test_accuracy": empirical["std_test_accuracy"],
                "empirical_pop_std_test_accuracy": empirical["pop_std_test_accuracy"],
                "empirical_sem_test_accuracy": empirical["sem_test_accuracy"],
                "empirical_min_test_accuracy": empirical["min_test_accuracy"],
                "empirical_max_test_accuracy": empirical["max_test_accuracy"],
                "empirical_num_realizations": empirical["num_realizations"],
                "a_from_fit_row": a,
                "A": A,
                "b": b,
                "c": c,
                "fit_reference_noise_type": args.fit_noise_type,
                "fit_reference_p": float(args.fit_p),
                "eval_test_limit": int(args.eval_test_limit),
                "reference_dir": reference_dir.name,
                "ensemble_dir": ensemble_dir.name,
            }
        )

        print(
            f"p={p:.4f} | centroid mean={centroid_mean:.6f}, "
            f"empirical mean={empirical['mean_test_accuracy']:.6f}, "
            f"empirical std={empirical['std_test_accuracy']:.6f}, "
            f"realizations={empirical['num_realizations']}"
        )

    rows = sorted(rows, key=lambda row: row["noise_probability"])

    output_csv = output_dir / (
        f"test_accuracy_summary__dataset={normalize_dataset_name(args.dataset)}"
        f"__eval={args.eval_noise_type}"
        f"__fit={args.fit_noise_type}_p={args.fit_p:.2f}"
        f"__N={args.N}__d={args.d_label}"
        f"__loss={args.loss}__act={args.activation}"
        f"__L={args.depth}__width={args.width}.csv"
    )

    fieldnames = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_json = output_dir / (
        f"fit_summary__dataset={normalize_dataset_name(args.dataset)}"
        f"__eval={args.eval_noise_type}"
        f"__fit={args.fit_noise_type}_p={args.fit_p:.2f}"
        f"__N={args.N}__d={args.d_label}"
        f"__loss={args.loss}__act={args.activation}"
        f"__L={args.depth}__width={args.width}.json"
    )
    summary_json.write_text(
        json.dumps(
            {
                "settings": vars(args),
                "fit_row": fit_row,
                "reference_dir": str(reference_dir),
                "output_csv": str(output_csv),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {output_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
