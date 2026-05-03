from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jt577")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from modules.data import dataset_to_numpy_matrix, load_mnist, normalize_feature_matrix_unit_rms
from train_noisy_mlp_ensemble import CleanTensorDataset, MLP


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_ROOT = PROJECT_DIR / "trained_mlp_ensembles_featureDim"
DEFAULT_TRAINING_ROOT = PROJECT_DIR / "trained_mlp_ensembles_trainingSize"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "snr_sweep_eval"
MODEL_RE = re.compile(r"^model(\d+)\.pth$")
FOLDER_RE = re.compile(
    r"__N=(?P<n_train>\d+)__d=(?P<h>\d+)x(?P<w>\d+)__.*"
    r"__act=(?P<activation>[^_]+)__L=(?P<depth>\d+)__width=(?P<width>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate feature-dimension and training-size MLP ensemble sweeps on "
            "clean MNIST test data, cache per test-point/class SNR, and plot SNR "
            "versus sqrt(d) or sqrt(N)."
        )
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_FEATURE_ROOT,
        help=f"Feature-dimension sweep checkpoint root. Default: {DEFAULT_FEATURE_ROOT}",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=DEFAULT_TRAINING_ROOT,
        help=f"Training-size sweep checkpoint root. Default: {DEFAULT_TRAINING_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where cached datasets, summaries, and plots are written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="MNIST data root. If omitted, each checkpoint config's data_root is used.",
    )
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=["featureDim", "trainingSize"],
        default=["featureDim", "trainingSize"],
        help="Which sweeps to evaluate/plot. Default: featureDim trainingSize",
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        default=["erf"],
        help="Activation names to include. Default: erf",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="MNIST test batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Evaluation device. Default: auto",
    )
    parser.add_argument(
        "--std-ddof",
        type=int,
        default=1,
        help="Delta degrees of freedom for ensemble standard deviation. Default: 1.",
    )
    parser.add_argument(
        "--std-eps",
        type=float,
        default=1e-12,
        help="SNR is set to NaN where std <= this threshold. Default: 1e-12.",
    )
    parser.add_argument(
        "--max-models-per-group",
        type=int,
        default=None,
        help="Optional smoke-test limit per ensemble folder.",
    )
    parser.add_argument(
        "--max-test-points",
        type=int,
        default=None,
        help="Optional smoke-test limit on MNIST test examples per group.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=None,
        help="Optional plotting-only cap per sweep. The cached dataset still keeps every SNR point.",
    )
    parser.add_argument(
        "--plot-test-index",
        type=int,
        default=0,
        help="Fixed MNIST test example index for the selected-point SNR plot. Default: 0.",
    )
    parser.add_argument(
        "--plot-label",
        type=int,
        default=0,
        help="Fixed output neuron/class label for the selected-point SNR plot. Default: 0.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["selected", "all", "both"],
        default="selected",
        help=(
            "Plot only one fixed test point/output neuron, all cached SNR points, "
            "or both. Default: selected."
        ),
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Skip checkpoint evaluation and replot/resummarize from snr_points.npz.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Requested --device mps, but MPS is not available.")
    return device


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric_model_sort_key(path: Path) -> tuple[int, str]:
    match = MODEL_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def parse_folder_fallback(folder: Path) -> dict[str, Any]:
    match = FOLDER_RE.search(folder.name)
    if match is None:
        return {}
    h = int(match.group("h"))
    w = int(match.group("w"))
    return {
        "num_train": int(match.group("n_train")),
        "feature_dim": h * w,
        "config": {
            "train_size": int(match.group("n_train")),
            "img_hw": [h, w],
            "num_classes": 10,
            "data_root": "data",
            "activation": match.group("activation"),
            "depth": int(match.group("depth")),
            "width": int(match.group("width")),
            "weight_std": 1.0,
            "bias_std": 0.0,
        },
    }


def discover_checkpoint_groups(
    *,
    feature_root: Path,
    training_root: Path,
    sweeps: set[str],
    activations: set[str],
    max_models_per_group: int | None,
) -> list[dict[str, Any]]:
    roots = []
    if "featureDim" in sweeps:
        roots.append(("featureDim", feature_root))
    if "trainingSize" in sweeps:
        roots.append(("trainingSize", training_root))

    groups: list[dict[str, Any]] = []
    group_id = 0
    for sweep_kind, root in roots:
        if not root.exists():
            print(f"Skipping missing root: {root}", flush=True)
            continue

        for folder in sorted(path for path in root.iterdir() if path.is_dir()):
            metadata_path = folder / "metadata.json"
            metadata = load_json(metadata_path) if metadata_path.exists() else parse_folder_fallback(folder)
            cfg = dict(metadata.get("config", {}))
            activation = str(cfg.get("activation", "")).strip().lower()
            if activation not in activations:
                continue

            model_paths = sorted(folder.glob("model*.pth"), key=numeric_model_sort_key)
            if max_models_per_group is not None:
                model_paths = model_paths[: max(0, int(max_models_per_group))]
            if not model_paths:
                continue

            img_hw = tuple(int(value) for value in cfg.get("img_hw", (28, 28)))
            feature_dim = int(metadata.get("feature_dim", img_hw[0] * img_hw[1]))
            n_train = int(metadata.get("num_train", cfg.get("train_size") or 60_000))
            x_axis_name = "sqrt_d" if sweep_kind == "featureDim" else "sqrt_N"
            x_value = math.sqrt(float(feature_dim if sweep_kind == "featureDim" else n_train))

            groups.append(
                {
                    "group_id": group_id,
                    "sweep_kind": sweep_kind,
                    "folder": folder,
                    "metadata": metadata,
                    "config": cfg,
                    "activation": activation,
                    "model_paths": model_paths,
                    "train_size": n_train,
                    "feature_dim": feature_dim,
                    "sqrt_d": math.sqrt(float(feature_dim)),
                    "sqrt_N": math.sqrt(float(n_train)),
                    "x_axis_name": x_axis_name,
                    "x_value": x_value,
                }
            )
            group_id += 1

    return sorted(
        groups,
        key=lambda group: (
            group["sweep_kind"],
            group["activation"],
            group["x_value"],
            str(group["folder"]),
        ),
    )


def resolve_data_root(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override if override.is_absolute() else PROJECT_DIR / override

    data_root = Path(str(cfg.get("data_root", "data")))
    return data_root if data_root.is_absolute() else PROJECT_DIR / data_root


def make_test_loader_and_labels(
    cfg: dict[str, Any],
    *,
    data_root_override: Path | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_test_points: int | None,
) -> tuple[DataLoader, np.ndarray]:
    img_hw = tuple(int(value) for value in cfg.get("img_hw", (28, 28)))
    num_classes = int(cfg.get("num_classes", 10))
    data_root = resolve_data_root(cfg, data_root_override)

    _train_raw, test_raw = load_mnist(root=data_root, img_hw=img_hw)
    if max_test_points is not None:
        test_raw = Subset(test_raw, range(max(0, int(max_test_points))))

    x_test, _y_test_onehot, y_test = dataset_to_numpy_matrix(test_raw, num_classes)
    x_test = normalize_feature_matrix_unit_rms(x_test)

    loader = DataLoader(
        CleanTensorDataset(x_test, y_test),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )
    return loader, np.asarray(y_test, dtype=np.int64)


def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} did not load as a checkpoint dictionary.")
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    fallback_cfg: dict[str, Any],
    fallback_feature_dim: int,
) -> MLP:
    cfg = dict(fallback_cfg)
    cfg.update(checkpoint.get("config", {}))

    input_dim = int(checkpoint.get("input_dim", fallback_feature_dim))
    return MLP(
        input_dim=input_dim,
        num_classes=int(cfg.get("num_classes", 10)),
        depth=int(cfg["depth"]),
        width=int(cfg["width"]),
        activation=str(cfg["activation"]),
        weight_std=float(cfg.get("weight_std", 1.0)),
        bias_std=float(cfg.get("bias_std", 0.0)),
    )


def evaluate_group_snr(
    group: dict[str, Any],
    *,
    test_loader: DataLoader,
    y_test: np.ndarray,
    device: torch.device,
    std_ddof: int,
    std_eps: float,
) -> dict[str, np.ndarray | dict[str, Any]]:
    cfg = group["config"]
    num_test = int(y_test.shape[0])
    num_classes = int(cfg.get("num_classes", 10))
    mean = np.zeros((num_test, num_classes), dtype=np.float64)
    m2 = np.zeros((num_test, num_classes), dtype=np.float64)
    count = 0

    sample_x, _sample_y = test_loader.dataset[0]
    fallback_feature_dim = int(np.asarray(sample_x).reshape(-1).shape[0])

    for model_index, checkpoint_path in enumerate(group["model_paths"], start=1):
        checkpoint = torch_load_checkpoint(checkpoint_path)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"{checkpoint_path} does not contain a model_state_dict.")

        model = build_model_from_checkpoint(checkpoint, cfg, fallback_feature_dim)
        model.load_state_dict(state_dict)
        del checkpoint, state_dict
        model.to(device)
        model.eval()

        count += 1
        row_start = 0
        with torch.inference_mode():
            for x, _y in test_loader:
                batch_size = int(x.shape[0])
                row_stop = row_start + batch_size
                logits = model(x.to(device=device, non_blocking=True)).detach().cpu().numpy().astype(np.float64)
                delta = logits - mean[row_start:row_stop]
                mean[row_start:row_stop] += delta / float(count)
                m2[row_start:row_stop] += delta * (logits - mean[row_start:row_stop])
                row_start = row_stop

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if model_index % 25 == 0 or model_index == len(group["model_paths"]):
            print(
                f"  group {group['group_id']} {group['sweep_kind']} "
                f"x={group['x_value']:.4g}: evaluated {model_index}/{len(group['model_paths'])} models",
                flush=True,
            )

    variance = np.full_like(mean, np.nan)
    if count > int(std_ddof):
        variance = m2 / float(count - int(std_ddof))
    std = np.sqrt(np.maximum(variance, 0.0))
    snr = np.full_like(mean, np.nan)
    valid = std > float(std_eps)
    snr[valid] = mean[valid] / std[valid]

    return {
        "mean_output": mean,
        "std_output": std,
        "snr": snr,
        "metadata": {
            "num_models": count,
            "num_test": num_test,
            "num_classes": num_classes,
        },
    }


def flatten_group_result(group: dict[str, Any], y_test: np.ndarray, result: dict[str, Any]) -> dict[str, np.ndarray]:
    mean = np.asarray(result["mean_output"], dtype=np.float64)
    std = np.asarray(result["std_output"], dtype=np.float64)
    snr = np.asarray(result["snr"], dtype=np.float64)
    num_test, num_classes = mean.shape
    num_rows = int(num_test * num_classes)
    metadata = dict(result["metadata"])
    sweep_code = 0 if group["sweep_kind"] == "featureDim" else 1

    return {
        "group_id": np.full(num_rows, int(group["group_id"]), dtype=np.int32),
        "sweep_code": np.full(num_rows, sweep_code, dtype=np.int8),
        "train_size": np.full(num_rows, int(group["train_size"]), dtype=np.int32),
        "feature_dim": np.full(num_rows, int(group["feature_dim"]), dtype=np.int32),
        "sqrt_N": np.full(num_rows, float(group["sqrt_N"]), dtype=np.float64),
        "sqrt_d": np.full(num_rows, float(group["sqrt_d"]), dtype=np.float64),
        "x_value": np.full(num_rows, float(group["x_value"]), dtype=np.float64),
        "num_models": np.full(num_rows, int(metadata["num_models"]), dtype=np.int32),
        "test_index": np.repeat(np.arange(num_test, dtype=np.int32), num_classes),
        "label": np.tile(np.arange(num_classes, dtype=np.int16), num_test),
        "true_label": np.repeat(np.asarray(y_test, dtype=np.int16), num_classes),
        "mean_output": mean.reshape(-1),
        "std_output": std.reshape(-1),
        "snr": snr.reshape(-1),
    }


def concatenate_point_arrays(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise RuntimeError("No SNR point arrays were produced.")
    keys = list(chunks[0].keys())
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def save_npz_dataset(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def load_npz_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write_points_csv_gz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    fieldnames = [
        "sweep_kind",
        "group_id",
        "train_size",
        "feature_dim",
        "sqrt_N",
        "sqrt_d",
        "x_value",
        "num_models",
        "test_index",
        "label",
        "true_label",
        "mean_output",
        "std_output",
        "snr",
    ]
    sweep_names = np.where(arrays["sweep_code"] == 0, "featureDim", "trainingSize")

    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        num_rows = int(arrays["snr"].shape[0])
        for index in range(num_rows):
            writer.writerow(
                {
                    "sweep_kind": str(sweep_names[index]),
                    "group_id": int(arrays["group_id"][index]),
                    "train_size": int(arrays["train_size"][index]),
                    "feature_dim": int(arrays["feature_dim"][index]),
                    "sqrt_N": float(arrays["sqrt_N"][index]),
                    "sqrt_d": float(arrays["sqrt_d"][index]),
                    "x_value": float(arrays["x_value"][index]),
                    "num_models": int(arrays["num_models"][index]),
                    "test_index": int(arrays["test_index"][index]),
                    "label": int(arrays["label"][index]),
                    "true_label": int(arrays["true_label"][index]),
                    "mean_output": float(arrays["mean_output"][index]),
                    "std_output": float(arrays["std_output"][index]),
                    "snr": float(arrays["snr"][index]),
                }
            )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize_points(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for group_id in sorted(np.unique(arrays["group_id"]).astype(int).tolist()):
        mask = arrays["group_id"] == group_id
        snr = np.asarray(arrays["snr"][mask], dtype=np.float64)
        finite = np.isfinite(snr)
        finite_snr = snr[finite]
        if finite_snr.size == 0:
            finite_snr = np.asarray([np.nan], dtype=np.float64)

        sweep_kind = "featureDim" if int(arrays["sweep_code"][mask][0]) == 0 else "trainingSize"
        summaries.append(
            {
                "group_id": group_id,
                "sweep_kind": sweep_kind,
                "train_size": int(arrays["train_size"][mask][0]),
                "feature_dim": int(arrays["feature_dim"][mask][0]),
                "sqrt_N": float(arrays["sqrt_N"][mask][0]),
                "sqrt_d": float(arrays["sqrt_d"][mask][0]),
                "x_value": float(arrays["x_value"][mask][0]),
                "num_models": int(arrays["num_models"][mask][0]),
                "num_snr_points": int(mask.sum()),
                "num_finite_snr_points": int(finite.sum()),
                "mean_snr": float(np.nanmean(finite_snr)),
                "std_snr": float(np.nanstd(finite_snr, ddof=1)) if finite_snr.size > 1 else 0.0,
                "median_snr": float(np.nanmedian(finite_snr)),
                "q05_snr": float(np.nanquantile(finite_snr, 0.05)),
                "q25_snr": float(np.nanquantile(finite_snr, 0.25)),
                "q75_snr": float(np.nanquantile(finite_snr, 0.75)),
                "q95_snr": float(np.nanquantile(finite_snr, 0.95)),
            }
        )
    return sorted(summaries, key=lambda row: (row["sweep_kind"], row["x_value"]))


def downsample_for_plot(mask: np.ndarray, max_points: int | None, seed: int = 22334) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if max_points is None or indices.size <= int(max_points):
        return indices
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(indices, size=int(max_points), replace=False))


def plot_sweep(
    arrays: dict[str, np.ndarray],
    summaries: list[dict[str, Any]],
    *,
    sweep_kind: str,
    output_dir: Path,
    max_scatter_points: int | None,
) -> Path | None:
    sweep_code = 0 if sweep_kind == "featureDim" else 1
    mask = arrays["sweep_code"] == sweep_code
    if not np.any(mask):
        return None

    indices = downsample_for_plot(mask, max_scatter_points)
    x_label = "sqrt(d)" if sweep_kind == "featureDim" else "sqrt(N)"
    title = "Feature-dimension sweep: SNR vs sqrt(d)" if sweep_kind == "featureDim" else "Training-size sweep: SNR vs sqrt(N)"
    stem = "feature_dim_snr_vs_sqrt_d" if sweep_kind == "featureDim" else "training_size_snr_vs_sqrt_N"

    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    ax.scatter(
        arrays["x_value"][indices],
        arrays["snr"][indices],
        s=2.0,
        alpha=0.035,
        linewidths=0,
        color="#2f5f8f",
        rasterized=True,
        label="test point/class SNR",
    )

    sweep_summaries = [row for row in summaries if row["sweep_kind"] == sweep_kind]
    x = np.asarray([row["x_value"] for row in sweep_summaries], dtype=np.float64)
    median = np.asarray([row["median_snr"] for row in sweep_summaries], dtype=np.float64)
    q25 = np.asarray([row["q25_snr"] for row in sweep_summaries], dtype=np.float64)
    q75 = np.asarray([row["q75_snr"] for row in sweep_summaries], dtype=np.float64)
    mean = np.asarray([row["mean_snr"] for row in sweep_summaries], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    median = median[order]
    q25 = q25[order]
    q75 = q75[order]
    mean = mean[order]

    ax.plot(x, median, marker="o", color="#b23a48", linewidth=2.0, label="median SNR")
    ax.plot(x, mean, marker="s", color="#2f7d32", linewidth=1.5, label="mean SNR")
    ax.fill_between(x, q25, q75, color="#b23a48", alpha=0.16, label="25-75% SNR")

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xlabel(x_label)
    ax.set_ylabel("SNR = ensemble mean output / ensemble std output")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def plot_selected_point_sweep(
    arrays: dict[str, np.ndarray],
    *,
    sweep_kind: str,
    output_dir: Path,
    test_index: int,
    label: int,
) -> Path | None:
    sweep_code = 0 if sweep_kind == "featureDim" else 1
    mask = (
        (arrays["sweep_code"] == sweep_code)
        & (arrays["test_index"] == int(test_index))
        & (arrays["label"] == int(label))
    )
    if not np.any(mask):
        print(
            f"No cached SNR rows found for sweep={sweep_kind}, "
            f"test_index={test_index}, label={label}; skipping selected-point plot.",
            flush=True,
        )
        return None

    x = np.asarray(arrays["x_value"][mask], dtype=np.float64)
    snr = np.asarray(arrays["snr"][mask], dtype=np.float64)
    mean_output = np.asarray(arrays["mean_output"][mask], dtype=np.float64)
    std_output = np.asarray(arrays["std_output"][mask], dtype=np.float64)
    train_size = np.asarray(arrays["train_size"][mask], dtype=np.int64)
    feature_dim = np.asarray(arrays["feature_dim"][mask], dtype=np.int64)
    order = np.argsort(x)

    x = x[order]
    snr = snr[order]
    mean_output = mean_output[order]
    std_output = std_output[order]
    train_size = train_size[order]
    feature_dim = feature_dim[order]

    x_label = "sqrt(d)" if sweep_kind == "featureDim" else "sqrt(N)"
    title = (
        f"Fixed test point/output neuron SNR vs sqrt(d)"
        if sweep_kind == "featureDim"
        else f"Fixed test point/output neuron SNR vs sqrt(N)"
    )
    stem = (
        f"selected_test{int(test_index)}_label{int(label)}_snr_vs_sqrt_d"
        if sweep_kind == "featureDim"
        else f"selected_test{int(test_index)}_label{int(label)}_snr_vs_sqrt_N"
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.plot(x, snr, marker="o", linewidth=2.0, color="#2f5f8f")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xlabel(x_label)
    ax.set_ylabel("SNR = ensemble mean output / ensemble std output")
    ax.set_title(f"{title}\ntest_index={int(test_index)}, output neuron={int(label)}")
    ax.grid(True, alpha=0.25)

    for xi, yi, n_value, d_value in zip(x, snr, train_size, feature_dim):
        point_label = f"N={int(n_value)}, d={int(d_value)}"
        ax.annotate(point_label, (xi, yi), textcoords="offset points", xytext=(4, 5), fontsize=7)

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    selected_csv = output_dir / f"{stem}.csv"
    write_csv(
        selected_csv,
        [
            {
                "sweep_kind": sweep_kind,
                "test_index": int(test_index),
                "label": int(label),
                "x_value": float(x_value),
                "train_size": int(n_value),
                "feature_dim": int(d_value),
                "mean_output": float(mean_value),
                "std_output": float(std_value),
                "snr": float(snr_value),
            }
            for x_value, n_value, d_value, mean_value, std_value, snr_value in zip(
                x, train_size, feature_dim, mean_output, std_output, snr
            )
        ],
        [
            "sweep_kind",
            "test_index",
            "label",
            "x_value",
            "train_size",
            "feature_dim",
            "mean_output",
            "std_output",
            "snr",
        ],
    )
    return png_path


def evaluate_all(args: argparse.Namespace, device: torch.device) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    groups = discover_checkpoint_groups(
        feature_root=args.feature_root,
        training_root=args.training_root,
        sweeps=set(args.sweeps),
        activations={activation.strip().lower() for activation in args.activations},
        max_models_per_group=args.max_models_per_group,
    )
    if not groups:
        raise RuntimeError("No checkpoint groups were found for the requested sweeps/activations.")

    print(f"Found {len(groups)} ensemble groups.", flush=True)
    group_metadata = []
    loader_cache: dict[tuple[Any, ...], tuple[DataLoader, np.ndarray]] = {}
    point_chunks: list[dict[str, np.ndarray]] = []

    for group in groups:
        cfg = group["config"]
        loader_key = (
            tuple(int(value) for value in cfg.get("img_hw", (28, 28))),
            int(cfg.get("num_classes", 10)),
            str(resolve_data_root(cfg, args.data_root)),
            int(args.batch_size),
            int(args.num_workers),
            int(args.max_test_points) if args.max_test_points is not None else None,
            device.type,
        )
        if loader_key not in loader_cache:
            loader_cache[loader_key] = make_test_loader_and_labels(
                cfg,
                data_root_override=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                max_test_points=args.max_test_points,
            )

        test_loader, y_test = loader_cache[loader_key]
        print(
            f"Evaluating group {group['group_id']}: {group['sweep_kind']} "
            f"N={group['train_size']} d={group['feature_dim']} "
            f"x={group['x_value']:.4g} ({len(group['model_paths'])} models)",
            flush=True,
        )
        result = evaluate_group_snr(
            group,
            test_loader=test_loader,
            y_test=y_test,
            device=device,
            std_ddof=args.std_ddof,
            std_eps=args.std_eps,
        )
        point_chunks.append(flatten_group_result(group, y_test, result))
        group_metadata.append(
            {
                "group_id": int(group["group_id"]),
                "sweep_kind": group["sweep_kind"],
                "folder": str(group["folder"]),
                "activation": group["activation"],
                "train_size": int(group["train_size"]),
                "feature_dim": int(group["feature_dim"]),
                "sqrt_N": float(group["sqrt_N"]),
                "sqrt_d": float(group["sqrt_d"]),
                "x_axis_name": group["x_axis_name"],
                "x_value": float(group["x_value"]),
                "num_models": int(result["metadata"]["num_models"]),
                "num_test": int(result["metadata"]["num_test"]),
                "num_classes": int(result["metadata"]["num_classes"]),
            }
        )

    return concatenate_point_arrays(point_chunks), group_metadata


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points_npz = args.output_dir / "snr_points.npz"
    points_csv = args.output_dir / "snr_points.csv.gz"
    summary_csv = args.output_dir / "snr_summary_by_group.csv"
    summary_json = args.output_dir / "snr_summary_by_group.json"
    metadata_json = args.output_dir / "snr_eval_metadata.json"

    if args.reuse_results:
        if not points_npz.exists():
            raise FileNotFoundError(f"--reuse-results requested, but {points_npz} does not exist.")
        arrays = load_npz_dataset(points_npz)
        group_metadata: list[dict[str, Any]] = []
    else:
        print(f"Using device: {device}", flush=True)
        arrays, group_metadata = evaluate_all(args, device)
        save_npz_dataset(points_npz, arrays)
        write_points_csv_gz(points_csv, arrays)
        metadata_json.write_text(
            json.dumps(
                {
                    "feature_root": str(args.feature_root),
                    "training_root": str(args.training_root),
                    "output_dir": str(args.output_dir),
                    "device": str(device),
                    "std_ddof": int(args.std_ddof),
                    "std_eps": float(args.std_eps),
                    "config": vars(args),
                    "groups": group_metadata,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    summaries = summarize_points(arrays)
    write_csv(
        summary_csv,
        summaries,
        [
            "group_id",
            "sweep_kind",
            "train_size",
            "feature_dim",
            "sqrt_N",
            "sqrt_d",
            "x_value",
            "num_models",
            "num_snr_points",
            "num_finite_snr_points",
            "mean_snr",
            "std_snr",
            "median_snr",
            "q05_snr",
            "q25_snr",
            "q75_snr",
            "q95_snr",
        ],
    )
    summary_json.write_text(
        json.dumps(
            {
                "num_snr_points": int(arrays["snr"].shape[0]),
                "num_finite_snr_points": int(np.isfinite(arrays["snr"]).sum()),
                "summary": summaries,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    plot_paths = []
    for sweep_kind in args.sweeps:
        if args.plot_mode in {"selected", "both"}:
            selected_plot_path = plot_selected_point_sweep(
                arrays,
                sweep_kind=sweep_kind,
                output_dir=args.output_dir,
                test_index=args.plot_test_index,
                label=args.plot_label,
            )
            if selected_plot_path is not None:
                plot_paths.append(selected_plot_path)

        if args.plot_mode in {"all", "both"}:
            plot_path = plot_sweep(
                arrays,
                summaries,
                sweep_kind=sweep_kind,
                output_dir=args.output_dir,
                max_scatter_points=args.max_scatter_points,
            )
            if plot_path is not None:
                plot_paths.append(plot_path)

    print(f"Wrote cached NumPy dataset: {points_npz}", flush=True)
    if points_csv.exists():
        print(f"Wrote reusable CSV dataset: {points_csv}", flush=True)
    print(f"Wrote summary: {summary_csv}", flush=True)
    for plot_path in plot_paths:
        print(f"Wrote plot: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
