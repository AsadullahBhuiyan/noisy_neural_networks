from __future__ import annotations

import csv
import gc
import json
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

from modules.data import load_mnist, normalize_feature_matrix_unit_rms
from train_noisy_mlp_ensemble import MLP


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_RE = re.compile(r"^model(\d+)\.pth$")
FOLDER_RE = re.compile(
    r"__N=(?P<n_train>\d+)__d=(?P<h>\d+)x(?P<w>\d+)__.*"
    r"__act=(?P<activation>[^_]+)__L=(?P<depth>\d+)__width=(?P<width>\d+)"
)

# Edit these values before running this script.
CHECKPOINT_DIR = Path(
    "/data2/jt577/2026/04.2026/NN_noise/trained_mlp_nested_ensembles_trainingSize/"
    "mlp_ensemble__additive_gaussian__p=0.98__N=16000__d=28x28__loss=mse__act=erf__L=3__width=2048__noiseK=100__initM=10__models=1000"
)
TEST_INDEX = 0
LABEL = 7
OUTPUT_DIR = "/data2/jt577/2026/04.2026/NN_noise/trained_mlp_nested_ensembles_trainingSize"  # None writes to CHECKPOINT_DIR / "single_point_output_eval".
DATA_ROOT: Path | None = None  # None uses the data_root saved in metadata.json.
DEVICE = "auto"  # "auto", "cuda", "cpu", or "mps".
MAX_MODELS: int | None = None  # Set to a small integer for a quick test.
BINS = 50


def resolve_path(path_value: str | Path, *, base: Path = Path.cwd()) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested DEVICE='cuda', but CUDA is not available.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Requested DEVICE='mps', but MPS is not available.")
    return device


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric_model_sort_key(path: Path) -> tuple[int, str]:
    match = MODEL_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def model_number_from_path(path: Path) -> int:
    match = MODEL_RE.match(path.name)
    return int(match.group(1)) if match else -1


def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} did not load as a checkpoint dictionary.")
    return checkpoint


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


def load_group_config(checkpoint_dir: Path, first_model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = checkpoint_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else parse_folder_fallback(checkpoint_dir)
    cfg = dict(metadata.get("config", {}))

    if not cfg:
        checkpoint = torch_load_checkpoint(first_model_path)
        cfg.update(checkpoint.get("config", {}))
        metadata = {"config": cfg}

    required = ["depth", "width", "activation"]
    missing = [name for name in required if name not in cfg]
    if missing:
        raise KeyError(f"Could not infer required config fields from metadata/checkpoint: {missing}")

    cfg.setdefault("num_classes", 10)
    cfg.setdefault("img_hw", (28, 28))
    cfg.setdefault("data_root", "data")
    cfg.setdefault("weight_std", 1.0)
    cfg.setdefault("bias_std", 0.0)
    return cfg, metadata


def resolve_data_root(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return resolve_path(override, base=PROJECT_DIR)
    data_root = Path(str(cfg.get("data_root", "data")))
    return data_root if data_root.is_absolute() else PROJECT_DIR / data_root


def load_single_test_vector(cfg: dict[str, Any], data_root_override: Path | None, test_index: int) -> tuple[torch.Tensor, int]:
    img_hw = tuple(int(value) for value in cfg.get("img_hw", (28, 28)))
    data_root = resolve_data_root(cfg, data_root_override)
    _train_raw, test_raw = load_mnist(root=data_root, img_hw=img_hw)

    if test_index < 0 or test_index >= len(test_raw):
        raise IndexError(f"test-index {test_index} is outside the MNIST test set of size {len(test_raw)}.")

    image, true_label = test_raw[int(test_index)]
    x = np.asarray(image, dtype=np.float32).reshape(1, -1)
    x = normalize_feature_matrix_unit_rms(x)
    return torch.as_tensor(x, dtype=torch.float32), int(true_label)


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


def evaluate_outputs(
    model_paths: list[Path],
    *,
    cfg: dict[str, Any],
    x: torch.Tensor,
    label: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    num_classes = int(cfg.get("num_classes", 10))
    if label < 0 or label >= num_classes:
        raise ValueError(f"label must be in [0, {num_classes - 1}], received {label}.")

    x_device = x.to(device=device)
    records: list[dict[str, Any]] = []
    fallback_feature_dim = int(x.reshape(-1).shape[0])

    for index, model_path in enumerate(model_paths, start=1):
        checkpoint = torch_load_checkpoint(model_path)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"{model_path} does not contain a model_state_dict.")

        model = build_model_from_checkpoint(checkpoint, cfg, fallback_feature_dim)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        model_number = int(checkpoint.get("model_number", model_number_from_path(model_path)))
        del checkpoint, state_dict

        with torch.inference_mode():
            logits = model(x_device).detach().cpu().numpy().reshape(-1)

        records.append(
            {
                "model_number": model_number,
                "model_file": model_path.name,
                "checkpoint_path": str(model_path),
                "output": float(logits[int(label)]),
                "predicted_label": int(np.argmax(logits)),
                **{f"logit_{class_id}": float(logits[class_id]) for class_id in range(num_classes)},
            }
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if index % 25 == 0 or index == len(model_paths):
            print(f"  evaluated {index}/{len(model_paths)} models", flush=True)

    return records


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def summarize_outputs(outputs: np.ndarray) -> dict[str, float | int]:
    outputs = np.asarray(outputs, dtype=np.float64)
    return {
        "num_models": int(outputs.size),
        "mean": float(np.mean(outputs)),
        "std": float(np.std(outputs, ddof=1)) if outputs.size > 1 else 0.0,
        "median": float(np.median(outputs)),
        "min": float(np.min(outputs)),
        "max": float(np.max(outputs)),
        "q05": float(np.quantile(outputs, 0.05)),
        "q25": float(np.quantile(outputs, 0.25)),
        "q75": float(np.quantile(outputs, 0.75)),
        "q95": float(np.quantile(outputs, 0.95)),
    }


def plot_distribution(
    outputs: np.ndarray,
    *,
    summary: dict[str, float | int],
    checkpoint_dir: Path,
    test_index: int,
    true_label: int,
    label: int,
    output_dir: Path,
    bins: int,
) -> Path:
    mean = float(summary["mean"])
    std = float(summary["std"])

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.hist(outputs, bins=int(bins), density=False, alpha=0.78, color="#3f6f8f", edgecolor="white", linewidth=0.6)
    ax.axvline(mean, color="#b23a48", linewidth=2.0, label=f"mean = {mean:.6g}")
    ax.axvline(mean - std, color="#2f7d32", linestyle="--", linewidth=1.4, label=f"mean +/- std, std = {std:.6g}")
    ax.axvline(mean + std, color="#2f7d32", linestyle="--", linewidth=1.4)
    ax.set_xlabel(f"network output for neuron {int(label)}")
    ax.set_ylabel("model count")
    ax.set_title(
        "Single test point ensemble output distribution\n"
        f"{checkpoint_dir.name}, test_index={int(test_index)}, true_label={int(true_label)}, neuron={int(label)}"
    )
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)

    png_path = output_dir / f"test{int(test_index)}_label{int(label)}_output_distribution.png"
    pdf_path = output_dir / f"test{int(test_index)}_label{int(label)}_output_distribution.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def main() -> None:
    checkpoint_dir = resolve_path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint folder does not exist: {checkpoint_dir}")

    output_dir = OUTPUT_DIR
    if output_dir is None:
        output_dir = checkpoint_dir / "single_point_output_eval"
    else:
        output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = sorted(checkpoint_dir.glob("model*.pth"), key=numeric_model_sort_key)
    if MAX_MODELS is not None:
        model_paths = model_paths[: max(0, int(MAX_MODELS))]
    if not model_paths:
        raise RuntimeError(f"No model*.pth files found in {checkpoint_dir}")

    cfg, metadata = load_group_config(checkpoint_dir, model_paths[0])
    x, true_label = load_single_test_vector(cfg, DATA_ROOT, int(TEST_INDEX))
    device = choose_device(DEVICE)

    print(f"Using device: {device}", flush=True)
    print(
        f"Evaluating {len(model_paths)} models on test_index={TEST_INDEX}, "
        f"true_label={true_label}, output neuron={LABEL}",
        flush=True,
    )
    records = evaluate_outputs(
        model_paths,
        cfg=cfg,
        x=x,
        label=int(LABEL),
        device=device,
    )

    outputs = np.asarray([record["output"] for record in records], dtype=np.float64)
    summary = summarize_outputs(outputs)
    summary.update(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "test_index": int(TEST_INDEX),
            "true_label": int(true_label),
            "label": int(LABEL),
            "device": str(device),
            "config": cfg,
            "metadata_num_train": metadata.get("num_train"),
            "metadata_feature_dim": metadata.get("feature_dim"),
        }
    )

    num_classes = int(cfg.get("num_classes", 10))
    output_csv = output_dir / f"test{int(TEST_INDEX)}_label{int(LABEL)}_per_model_outputs.csv"
    fieldnames = [
        "model_number",
        "model_file",
        "checkpoint_path",
        "output",
        "predicted_label",
        *[f"logit_{class_id}" for class_id in range(num_classes)],
    ]
    write_csv(output_csv, records, fieldnames)

    summary_json = output_dir / f"test{int(TEST_INDEX)}_label{int(LABEL)}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    plot_path = plot_distribution(
        outputs,
        summary=summary,
        checkpoint_dir=checkpoint_dir,
        test_index=int(TEST_INDEX),
        true_label=true_label,
        label=int(LABEL),
        output_dir=output_dir,
        bins=int(BINS),
    )

    print(f"mean={float(summary['mean']):.8g}", flush=True)
    print(f"std={float(summary['std']):.8g}", flush=True)
    print(f"Wrote per-model outputs: {output_csv}", flush=True)
    print(f"Wrote summary: {summary_json}", flush=True)
    print(f"Wrote plot: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
