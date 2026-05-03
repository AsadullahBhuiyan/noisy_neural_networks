from __future__ import annotations

import argparse
import csv
import gc
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
import torch
from torch.utils.data import DataLoader

from modules.data import dataset_to_numpy_matrix, load_mnist, normalize_feature_matrix_unit_rms
from train_noisy_mlp_ensemble import CleanTensorDataset, MLP


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_ROOT = PROJECT_DIR / "trained_mlp_ensembles_activationComparison"
DEFAULT_OUTPUT_DIR = DEFAULT_CHECKPOINT_ROOT / "test_accuracy_eval"
MODEL_RE = re.compile(r"^model(\d+)\.pth$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every saved activation-comparison MLP on the clean MNIST "
            "test set and plot test accuracy versus noise p."
        )
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help=f"Folder containing ensemble subdirectories. Default: {DEFAULT_CHECKPOINT_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where CSV/JSON/plots are written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="MNIST data root. If omitted, each checkpoint config's data_root is used.",
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        default=["erf", "gelu", "tanh", "swish"],
        help="Activation names to include. Default: erf gelu",
    )
    parser.add_argument("--batch-size", type=int, default=2048, help="MNIST test batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Evaluation device. Default: auto",
    )
    parser.add_argument(
        "--max-models-per-group",
        type=int,
        default=None,
        help="Optional smoke-test limit per p/activation folder.",
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Skip checkpoint evaluation and replot from an existing per_model_test_accuracy.csv.",
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


def discover_checkpoint_groups(
    checkpoint_root: Path,
    activations: set[str],
    max_models_per_group: int | None,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for folder in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
        metadata_path = folder / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = load_json(metadata_path)
        cfg = metadata.get("config", {})
        activation = str(cfg.get("activation", "")).strip().lower()
        if activation not in activations:
            continue

        model_paths = sorted(folder.glob("model*.pth"), key=numeric_model_sort_key)
        if max_models_per_group is not None:
            model_paths = model_paths[: max(0, int(max_models_per_group))]
        if not model_paths:
            continue

        groups.append(
            {
                "folder": folder,
                "metadata": metadata,
                "config": cfg,
                "p": float(cfg["noise_probability"]),
                "activation": activation,
                "model_paths": model_paths,
            }
        )
    return sorted(groups, key=lambda group: (group["activation"], group["p"], str(group["folder"])))


def resolve_data_root(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override if override.is_absolute() else PROJECT_DIR / override

    data_root = Path(str(cfg.get("data_root", "data")))
    return data_root if data_root.is_absolute() else PROJECT_DIR / data_root


def make_test_loader(
    cfg: dict[str, Any],
    *,
    data_root_override: Path | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    img_hw = tuple(int(value) for value in cfg.get("img_hw", (28, 28)))
    num_classes = int(cfg.get("num_classes", 10))
    data_root = resolve_data_root(cfg, data_root_override)

    _train_raw, test_raw = load_mnist(root=data_root, img_hw=img_hw)
    x_test, _y_test_onehot, y_test = dataset_to_numpy_matrix(test_raw, num_classes)
    x_test = normalize_feature_matrix_unit_rms(x_test)

    return DataLoader(
        CleanTensorDataset(x_test, y_test),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )


def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} did not load as a checkpoint dictionary.")
    return checkpoint


def checkpoint_model_number(path: Path, checkpoint: dict[str, Any]) -> int:
    if "model_number" in checkpoint:
        return int(checkpoint["model_number"])
    match = MODEL_RE.match(path.name)
    return int(match.group(1)) if match else -1


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


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    fallback_cfg: dict[str, Any],
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"{checkpoint_path} does not contain a model_state_dict.")

    sample_x, _sample_y = test_loader.dataset[0]
    fallback_feature_dim = int(np.asarray(sample_x).reshape(-1).shape[0])
    model = build_model_from_checkpoint(checkpoint, fallback_cfg, fallback_feature_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_correct = 0
    total_seen = 0
    with torch.inference_mode():
        for x, y in test_loader:
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            logits = model(x)
            total_correct += int((torch.argmax(logits, dim=1) == y).sum().detach().cpu())
            total_seen += int(y.numel())

    accuracy = total_correct / max(1, total_seen)
    model_number = checkpoint_model_number(checkpoint_path, checkpoint)
    del model, checkpoint, state_dict
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_file": checkpoint_path.name,
        "model_number": model_number,
        "accuracy": accuracy,
        "num_test": total_seen,
    }


def evaluate_all(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    activations = {activation.strip().lower() for activation in args.activations}
    groups = discover_checkpoint_groups(args.checkpoint_root, activations, args.max_models_per_group)
    if not groups:
        raise RuntimeError(f"No checkpoint groups found under {args.checkpoint_root}.")

    loader_cache: dict[tuple[Any, ...], DataLoader] = {}
    records: list[dict[str, Any]] = []
    total_models = sum(len(group["model_paths"]) for group in groups)
    completed = 0
    print(f"Found {len(groups)} p/activation groups and {total_models} checkpoints.", flush=True)

    for group in groups:
        cfg = group["config"]
        loader_key = (
            tuple(int(value) for value in cfg.get("img_hw", (28, 28))),
            int(cfg.get("num_classes", 10)),
            str(resolve_data_root(cfg, args.data_root)),
            int(args.batch_size),
            int(args.num_workers),
            device.type,
        )
        if loader_key not in loader_cache:
            loader_cache[loader_key] = make_test_loader(
                cfg,
                data_root_override=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )

        test_loader = loader_cache[loader_key]
        print(
            f"Evaluating act={group['activation']} p={group['p']:g} "
            f"({len(group['model_paths'])} models)",
            flush=True,
        )
        for checkpoint_path in group["model_paths"]:
            result = evaluate_checkpoint(
                checkpoint_path,
                fallback_cfg=cfg,
                test_loader=test_loader,
                device=device,
            )
            completed += 1
            records.append(
                {
                    "activation": group["activation"],
                    "p": group["p"],
                    "model_number": result["model_number"],
                    "model_file": result["model_file"],
                    "checkpoint_path": str(checkpoint_path),
                    "accuracy": result["accuracy"],
                    "accuracy_percent": 100.0 * result["accuracy"],
                    "num_test": result["num_test"],
                }
            )
            if completed % 25 == 0 or completed == total_models:
                print(f"  evaluated {completed}/{total_models} checkpoints", flush=True)
    return records


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def read_per_model_csv(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["p"] = float(row["p"])
            row["model_number"] = int(row["model_number"])
            row["accuracy"] = float(row["accuracy"])
            row["accuracy_percent"] = float(row["accuracy_percent"])
            row["num_test"] = int(row["num_test"])
            records.append(row)
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for record in records:
        grouped[(str(record["activation"]), float(record["p"]))].append(float(record["accuracy"]))

    summaries: list[dict[str, Any]] = []
    for (activation, p), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        array = np.asarray(values, dtype=np.float64)
        std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        mean = float(np.mean(array))
        summaries.append(
            {
                "activation": activation,
                "p": p,
                "num_models": int(array.size),
                "mean_accuracy": mean,
                "std_accuracy": std,
                "mean_accuracy_percent": 100.0 * mean,
                "std_accuracy_percent": 100.0 * std,
            }
        )
    return summaries


def plot_summary(records: list[dict[str, Any]], summaries: list[dict[str, Any]], output_dir: Path) -> Path:
    colors = {"erf": "#2b6cb0", "gelu": "#c05621"}
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)

    activations = sorted({str(record["activation"]) for record in records})
    for activation in activations:
        act_summaries = [row for row in summaries if row["activation"] == activation]
        if not act_summaries:
            continue

        x = np.asarray([row["p"] for row in act_summaries], dtype=np.float64)
        y = np.asarray([row["mean_accuracy_percent"] for row in act_summaries], dtype=np.float64)
        yerr = np.asarray([row["std_accuracy_percent"] for row in act_summaries], dtype=np.float64)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        yerr = yerr[order]

        color = colors.get(activation, None)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            capsize=4.0,
            label=activation,
            color=color,
        )
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12 if color else 0.10)

        for p_value in x:
            points = [
                100.0 * float(record["accuracy"])
                for record in records
                if str(record["activation"]) == activation and float(record["p"]) == float(p_value)
            ]
            if points:
                jitter = np.linspace(-0.0015, 0.0015, num=len(points))
                ax.scatter(
                    np.full(len(points), p_value) + jitter,
                    points,
                    s=8,
                    color=color,
                    alpha=0.18,
                    linewidths=0,
                )

    ax.set_xlabel("noise p")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_title("Activation comparison: clean MNIST test accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(title="activation")

    png_path = output_dir / "activation_comparison_test_accuracy_vs_p.png"
    pdf_path = output_dir / "activation_comparison_test_accuracy_vs_p.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_model_csv = args.output_dir / "per_model_test_accuracy.csv"
    summary_csv = args.output_dir / "summary_test_accuracy_by_p_activation.csv"
    summary_json = args.output_dir / "summary_test_accuracy_by_p_activation.json"

    if args.reuse_results:
        if not per_model_csv.exists():
            raise FileNotFoundError(f"--reuse-results requested, but {per_model_csv} does not exist.")
        records = read_per_model_csv(per_model_csv)
    else:
        print(f"Using device: {device}", flush=True)
        records = evaluate_all(args, device)
        write_csv(
            per_model_csv,
            records,
            [
                "activation",
                "p",
                "model_number",
                "model_file",
                "checkpoint_path",
                "accuracy",
                "accuracy_percent",
                "num_test",
            ],
        )

    summaries = summarize(records)
    write_csv(
        summary_csv,
        summaries,
        [
            "activation",
            "p",
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
                "checkpoint_root": str(args.checkpoint_root),
                "output_dir": str(args.output_dir),
                "device": str(device),
                "num_models_evaluated": len(records),
                "summary": summaries,
                "config": vars(args),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    plot_path = plot_summary(records, summaries, args.output_dir)

    print(f"Wrote per-model results: {per_model_csv}", flush=True)
    print(f"Wrote summary: {summary_csv}", flush=True)
    print(f"Wrote plot: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
