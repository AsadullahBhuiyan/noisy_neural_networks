from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from custom_ntk import evaluate_clean_ntk
from modules.data import DatasetSubset, dataset_to_numpy_matrix, load_mnist, normalize_feature_matrix_unit_rms
from modules.metrics import acc_from_logits, mse_onehot_from_logits
from train_noisy_mlp_ensemble import MLP, _balanced_indices, _choose_device, _set_global_seed


################################################################################
# USER INPUTS
#
# This script trains an ensemble of clean MLPs per width on MNIST, computes one
# clean NTK prediction for the same data, and compares the ensemble-mean test
# logits with the NTK test logits using scatter plots and correlation summaries.
################################################################################

SEED = 22334
NUM_CLASSES = 10

TRAIN_SIZE = 8000          # Use None for the full clean MNIST train split.
TEST_LIMIT = None          # Use None for the full clean MNIST test split.
IMG_HW = (28, 28)

WIDTHS = [2048]
NUM_MODELS_PER_WIDTH = 100
DEPTH = 2
ACTIVATION = "erf"         # erf, gelu, tanh, sin, softplus, relu
WEIGHT_STD = 1.0
BIAS_STD = 0.0

LOSS = "mse"     # cross_entropy or mse
EPOCHS = 30
BATCH_SIZE = 250
LEARNING_RATE = 5e-4
OPTIMIZER = "adam"        # adam, adamw, or sgd
WEIGHT_DECAY = 0.0
NUM_WORKERS = 0
EVAL_BATCH_SIZE = 1024

NTK_DIAG_REG = 1e-12
NTK_DEVICE = None          # None, "cpu", or "cuda"

CENTER_CLASSWISE_LOGITS = False
ALLOW_CPU = False

DATA_ROOT = "data"
OUTPUT_ROOT = "outputs/clean_mlp_vs_ntk_widths"

################################################################################
# END USER INPUTS
################################################################################


@dataclass
class WidthSweepConfig:
    seed: int = 22334
    num_classes: int = 10
    train_size: int | None = 4000
    test_limit: int | None = None
    img_hw: tuple[int, int] = (28, 28)
    widths: list[int] | tuple[int, ...] = (128, 512, 2048)
    num_models_per_width: int = 1
    depth: int = 3
    activation: str = "erf"
    weight_std: float = 1.0
    bias_std: float = 0.0
    loss: str = "cross_entropy"
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    num_workers: int = 0
    eval_batch_size: int = 1024
    ntk_diag_reg: float = 1e-12
    ntk_device: str | None = None
    center_classwise_logits: bool = True
    allow_cpu: bool = False
    data_root: str = "data"
    output_root: str = "outputs/clean_mlp_vs_ntk_widths"


@dataclass
class PreparedCleanData:
    x_train: np.ndarray
    y_train: np.ndarray
    y_train_onehot: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_test_onehot: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


def build_config_from_user_inputs() -> WidthSweepConfig:
    widths = [int(width) for width in WIDTHS]
    if not widths:
        raise ValueError("WIDTHS must contain at least one width.")
    if any(width < 1 for width in widths):
        raise ValueError("All WIDTHS entries must be positive.")
    if NUM_MODELS_PER_WIDTH < 1:
        raise ValueError("NUM_MODELS_PER_WIDTH must be at least 1.")
    if EPOCHS < 1:
        raise ValueError("EPOCHS must be at least 1.")
    if BATCH_SIZE < 1 or EVAL_BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE and EVAL_BATCH_SIZE must be at least 1.")
    if LOSS.strip().lower() not in {"cross_entropy", "mse"}:
        raise ValueError("LOSS must be 'cross_entropy' or 'mse'.")

    return WidthSweepConfig(
        seed=int(SEED),
        num_classes=int(NUM_CLASSES),
        train_size=None if TRAIN_SIZE is None else int(TRAIN_SIZE),
        test_limit=None if TEST_LIMIT is None else int(TEST_LIMIT),
        img_hw=(int(IMG_HW[0]), int(IMG_HW[1])),
        widths=widths,
        num_models_per_width=int(NUM_MODELS_PER_WIDTH),
        depth=int(DEPTH),
        activation=str(ACTIVATION),
        weight_std=float(WEIGHT_STD),
        bias_std=float(BIAS_STD),
        loss=str(LOSS),
        epochs=int(EPOCHS),
        batch_size=int(BATCH_SIZE),
        lr=float(LEARNING_RATE),
        optimizer=str(OPTIMIZER),
        weight_decay=float(WEIGHT_DECAY),
        num_workers=int(NUM_WORKERS),
        eval_batch_size=int(EVAL_BATCH_SIZE),
        ntk_diag_reg=float(NTK_DIAG_REG),
        ntk_device=None if NTK_DEVICE is None else str(NTK_DEVICE),
        center_classwise_logits=bool(CENTER_CLASSWISE_LOGITS),
        allow_cpu=bool(ALLOW_CPU),
        data_root=str(DATA_ROOT),
        output_root=str(OUTPUT_ROOT),
    )


def _resolve_root(root: str, project_dir: Path) -> Path:
    path = Path(root)
    return path if path.is_absolute() else project_dir / path


def _normalize_activation_name(name: str) -> str:
    return str(name).strip().lower()


def _maybe_center_logits(values: np.ndarray, *, enabled: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if not enabled:
        return arr
    return arr - np.mean(arr, axis=-1, keepdims=True)


def _comparison_summary(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | None]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    diff = cand - ref
    corr = None
    if ref.size >= 2 and float(np.std(ref)) > 0.0 and float(np.std(cand)) > 0.0:
        corr = float(np.corrcoef(ref, cand)[0, 1])
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "correlation": corr,
    }


def _activation_functions_for_custom_ntk(name: str):
    normalized = _normalize_activation_name(name)
    if normalized in {"erf", "relu"}:
        return None, None
    if normalized == "gelu":
        def act(z):
            return torch.nn.functional.gelu(z, approximate="none")

        def act_p(z):
            phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            return cdf + z * phi

        return act, act_p
    if normalized == "tanh":
        def act(z):
            return torch.tanh(z)

        def act_p(z):
            t = torch.tanh(z)
            return 1.0 - t * t

        return act, act_p
    if normalized == "sin":
        return torch.sin, torch.cos
    if normalized == "softplus":
        def act(z):
            return torch.nn.functional.softplus(z)

        def act_p(z):
            return torch.sigmoid(z)

        return act, act_p
    raise ValueError(
        f"Unsupported activation {name!r}. Expected erf, gelu, tanh, sin, softplus, or relu."
    )


def _resolve_ntk_device(cfg: WidthSweepConfig, runtime_device: torch.device) -> torch.device | None:
    if cfg.ntk_device is not None:
        return torch.device(cfg.ntk_device)
    if runtime_device.type == "cuda":
        return runtime_device
    return torch.device("cpu")


def prepare_clean_data(cfg: WidthSweepConfig, project_dir: Path) -> PreparedCleanData:
    data_root = _resolve_root(cfg.data_root, project_dir)
    train_raw, test_raw = load_mnist(root=data_root, img_hw=cfg.img_hw)

    if cfg.train_size is None:
        train_indices = np.arange(len(train_raw), dtype=np.int64)
    else:
        train_indices = _balanced_indices(
            np.asarray(train_raw.targets, dtype=np.int64),
            train_size=int(cfg.train_size),
            num_classes=int(cfg.num_classes),
            seed=int(cfg.seed) + 1,
        )

    if cfg.test_limit is None or int(cfg.test_limit) >= len(test_raw):
        test_indices = np.arange(len(test_raw), dtype=np.int64)
    else:
        test_indices = np.arange(int(cfg.test_limit), dtype=np.int64)

    train_dataset = DatasetSubset(train_raw, train_indices)
    test_dataset = DatasetSubset(test_raw, test_indices)

    x_train, y_train_onehot, y_train = dataset_to_numpy_matrix(train_dataset, cfg.num_classes)
    x_test, y_test_onehot, y_test = dataset_to_numpy_matrix(test_dataset, cfg.num_classes)
    x_train = normalize_feature_matrix_unit_rms(x_train)
    x_test = normalize_feature_matrix_unit_rms(x_test)

    return PreparedCleanData(
        x_train=x_train,
        y_train=y_train,
        y_train_onehot=y_train_onehot,
        x_test=x_test,
        y_test=y_test,
        y_test_onehot=y_test_onehot,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
    )


def compute_clean_ntk_logits(
    cfg: WidthSweepConfig,
    data: PreparedCleanData,
    *,
    device: torch.device | None,
) -> dict[str, object]:
    act_fn, act_p_fn = _activation_functions_for_custom_ntk(cfg.activation)
    start = time.perf_counter()
    result = evaluate_clean_ntk(
        X_train=data.x_train,
        y_train=data.y_train_onehot,
        X_test=data.x_test,
        depth=int(cfg.depth),
        act=str(cfg.activation),
        act_fn=act_fn,
        act_p_fn=act_p_fn,
        Cb=float(cfg.bias_std) ** 2,
        Cw=float(cfg.weight_std) ** 2,
        diag_reg=float(cfg.ntk_diag_reg),
        device=device,
        dtype=torch.float64,
    )
    test_logits = result.test_logits.detach().cpu().numpy().astype(np.float64, copy=False)
    train_logits = result.train_logits.detach().cpu().numpy().astype(np.float64, copy=False)
    elapsed = time.perf_counter() - start

    return {
        "train_logits": train_logits,
        "test_logits": test_logits,
        "train_acc": float(acc_from_logits(train_logits, data.y_train)),
        "test_acc": float(acc_from_logits(test_logits, data.y_test)),
        "train_mse": float(mse_onehot_from_logits(train_logits, data.y_train, cfg.num_classes)),
        "test_mse": float(mse_onehot_from_logits(test_logits, data.y_test, cfg.num_classes)),
        "elapsed_seconds": elapsed,
    }


def _make_optimizer(cfg: WidthSweepConfig, model: nn.Module):
    normalized = cfg.optimizer.strip().lower()
    if normalized == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if normalized == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if normalized == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    raise ValueError("OPTIMIZER must be one of: adam, adamw, sgd.")


def train_clean_mlp_for_width(
    cfg: WidthSweepConfig,
    data: PreparedCleanData,
    *,
    width: int,
    model_number: int,
    device: torch.device,
) -> dict[str, object]:
    model_seed = int(cfg.seed) + 1_000_003 * int(model_number) + 97 * int(width)
    _set_global_seed(model_seed)

    x_train_tensor = torch.as_tensor(data.x_train, dtype=torch.float32)
    y_train_tensor = torch.as_tensor(data.y_train, dtype=torch.long)
    y_train_onehot_tensor = torch.as_tensor(data.y_train_onehot, dtype=torch.float32)
    x_test_tensor = torch.as_tensor(data.x_test, dtype=torch.float32)
    y_test_tensor = torch.as_tensor(data.y_test, dtype=torch.long)
    y_test_onehot_tensor = torch.as_tensor(data.y_test_onehot, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor, y_train_onehot_tensor),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(model_seed + 17),
    )
    test_loader = DataLoader(
        TensorDataset(x_test_tensor, y_test_tensor, y_test_onehot_tensor),
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = MLP(
        input_dim=int(data.x_train.shape[1]),
        num_classes=int(cfg.num_classes),
        depth=int(cfg.depth),
        width=int(width),
        activation=str(cfg.activation),
        weight_std=float(cfg.weight_std),
        bias_std=float(cfg.bias_std),
    ).to(device=device, dtype=torch.float32)

    optimizer = _make_optimizer(cfg, model)
    normalized_loss = cfg.loss.strip().lower()
    if normalized_loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    history = []
    start = time.perf_counter()
    for epoch_index in range(int(cfg.epochs)):
        epoch_number = epoch_index + 1
        model.train()
        total_loss = 0.0
        total_seen = 0
        total_correct = 0

        for x_batch, y_batch, y_onehot_batch in train_loader:
            x_batch = x_batch.to(device=device, non_blocking=True)
            y_batch = y_batch.to(device=device, non_blocking=True)
            y_onehot_batch = y_onehot_batch.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            if normalized_loss == "cross_entropy":
                loss = criterion(logits, y_batch)
            else:
                loss = criterion(logits, y_onehot_batch)
            loss.backward()
            optimizer.step()

            batch_size = int(y_batch.numel())
            total_seen += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((torch.argmax(logits.detach(), dim=1) == y_batch).sum().detach().cpu())

        train_loss = total_loss / max(1, total_seen)
        train_acc = total_correct / max(1, total_seen)

        if epoch_number == 1 or epoch_number == int(cfg.epochs) or epoch_number % 20 == 0:
            print(
                f"width={width} model={model_number}/{cfg.num_models_per_width} "
                f"epoch {epoch_number}/{cfg.epochs} "
                f"train_loss={train_loss:.6f} train_acc={100.0 * train_acc:.2f}%",
                flush=True,
            )

        history.append(
            {
                "epoch": epoch_number,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
            }
        )

    model.eval()
    logits_batches = []
    with torch.inference_mode():
        for x_batch, _y_batch, _y_onehot_batch in test_loader:
            x_batch = x_batch.to(device=device, non_blocking=True)
            logits_batches.append(model(x_batch).detach().cpu())

    test_logits = torch.cat(logits_batches, dim=0).numpy().astype(np.float64, copy=False)
    train_elapsed = time.perf_counter() - start

    return {
        "width": int(width),
        "model_number": int(model_number),
        "model_seed": int(model_seed),
        "test_logits": test_logits,
        "test_acc": float(acc_from_logits(test_logits, data.y_test)),
        "test_mse": float(mse_onehot_from_logits(test_logits, data.y_test, cfg.num_classes)),
        "elapsed_seconds": train_elapsed,
        "history": history,
    }


def train_clean_mlp_ensemble_for_width(
    cfg: WidthSweepConfig,
    data: PreparedCleanData,
    *,
    width: int,
    device: torch.device,
) -> dict[str, object]:
    logits_sum = np.zeros((data.x_test.shape[0], int(cfg.num_classes)), dtype=np.float64)
    logits_sq_sum = np.zeros_like(logits_sum)
    model_summaries = []
    histories = []
    elapsed_total = 0.0

    for model_number in range(1, int(cfg.num_models_per_width) + 1):
        print(
            f"Training clean MLP width={width}, ensemble member "
            f"{model_number}/{cfg.num_models_per_width} on device {device}...",
            flush=True,
        )
        model_result = train_clean_mlp_for_width(
            cfg,
            data,
            width=int(width),
            model_number=model_number,
            device=device,
        )
        logits = np.asarray(model_result["test_logits"], dtype=np.float64)
        logits_sum += logits
        logits_sq_sum += logits * logits
        elapsed_total += float(model_result["elapsed_seconds"])
        model_summaries.append(
            {
                "model_number": int(model_number),
                "model_seed": int(model_result["model_seed"]),
                "test_acc": float(model_result["test_acc"]),
                "test_mse": float(model_result["test_mse"]),
                "elapsed_seconds": float(model_result["elapsed_seconds"]),
                "final_train_loss": float(model_result["history"][-1]["train_loss"]),
                "final_train_accuracy": float(model_result["history"][-1]["train_accuracy"]),
            }
        )
        histories.append(
            {
                "model_number": int(model_number),
                "model_seed": int(model_result["model_seed"]),
                "history": model_result["history"],
            }
        )
        print(
            f"width={width} model={model_number}/{cfg.num_models_per_width} done: "
            f"test_acc={100.0 * model_result['test_acc']:.2f}% "
            f"test_mse={model_result['test_mse']:.6e}",
            flush=True,
        )

    num_models = int(cfg.num_models_per_width)
    mean_test_logits = logits_sum / float(num_models)
    variance_test_logits = logits_sq_sum / float(num_models) - mean_test_logits * mean_test_logits
    variance_test_logits = np.maximum(variance_test_logits, 0.0)
    ddof1_variance_test_logits = (
        variance_test_logits * float(num_models) / float(num_models - 1)
        if num_models > 1
        else np.zeros_like(variance_test_logits)
    )

    per_model_acc = np.asarray([item["test_acc"] for item in model_summaries], dtype=np.float64)
    per_model_mse = np.asarray([item["test_mse"] for item in model_summaries], dtype=np.float64)

    return {
        "width": int(width),
        "num_models": num_models,
        "mean_test_logits": mean_test_logits,
        "variance_test_logits": variance_test_logits,
        "std_test_logits": np.sqrt(ddof1_variance_test_logits),
        "ensemble_mean_test_acc": float(acc_from_logits(mean_test_logits, data.y_test)),
        "ensemble_mean_test_mse": float(mse_onehot_from_logits(mean_test_logits, data.y_test, cfg.num_classes)),
        "per_model_test_acc_mean": float(np.mean(per_model_acc)),
        "per_model_test_acc_std": float(np.std(per_model_acc, ddof=1)) if num_models > 1 else 0.0,
        "per_model_test_mse_mean": float(np.mean(per_model_mse)),
        "per_model_test_mse_std": float(np.std(per_model_mse, ddof=1)) if num_models > 1 else 0.0,
        "elapsed_seconds": float(elapsed_total),
        "model_summaries": model_summaries,
        "histories": histories,
    }


def _prepare_matplotlib_cache(output_dir: Path) -> None:
    matplotlib_dir = output_dir / ".matplotlib"
    cache_dir = output_dir / ".cache"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)


def _save_scatter_plot(
    *,
    output_dir: Path,
    width: int,
    ntk_logits: np.ndarray,
    mlp_logits: np.ndarray,
    summary: dict[str, float | None],
    centered: bool,
) -> str:
    _prepare_matplotlib_cache(output_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_flat = np.asarray(ntk_logits, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(mlp_logits, dtype=np.float64).reshape(-1)
    lower = float(min(np.min(x_flat), np.min(y_flat)))
    upper = float(max(np.max(x_flat), np.max(y_flat)))

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(x_flat, y_flat, s=6, alpha=0.18, linewidths=0.0)
    ax.plot([lower, upper], [lower, upper], color="crimson", linestyle="--", linewidth=1.5)
    ax.set_title(f"Width {width}: MLP Ensemble Mean vs NTK Test Logits", fontsize=19)
    label_suffix = " (class-centered)" if centered else ""
    ax.set_xlabel(f"NTK test logits{label_suffix}", fontsize=20)
    ax.set_ylabel(f"MLP ensemble-mean test logits{label_suffix}", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.97,
        (
            f"corr={summary['correlation']}\n"
            f"MAE={summary['mae']:.3e}\n"
            f"RMSE={summary['rmse']:.3e}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
        fontsize=14
    )

    plot_path = output_dir / f"scatter_width={int(width)}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path.name


def _save_width_summary_plot(output_dir: Path, width_results: list[dict[str, object]]) -> str:
    _prepare_matplotlib_cache(output_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = [int(item["width"]) for item in width_results]
    corrs = [float(item["comparison"]["correlation"]) if item["comparison"]["correlation"] is not None else np.nan for item in width_results]
    maes = [float(item["comparison"]["mae"]) for item in width_results]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].plot(widths, corrs, marker="o")
    axes[0].set_title("Width vs Correlation")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Correlation")
    axes[0].grid(alpha=0.25)

    axes[1].plot(widths, maes, marker="o")
    axes[1].set_title("Width vs MAE")
    axes[1].set_xlabel("Width")
    axes[1].set_ylabel("MAE")
    axes[1].grid(alpha=0.25)

    path = output_dir / "width_summary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def save_outputs(
    cfg: WidthSweepConfig,
    data: PreparedCleanData,
    ntk_result: dict[str, object],
    width_results: list[dict[str, object]],
    *,
    project_dir: Path,
) -> Path:
    output_root = _resolve_root(cfg.output_root, project_dir)
    widths_label = "-".join(str(width) for width in cfg.widths)
    output_dir = output_root / (
        f"clean_mlp_vs_ntk__d={cfg.img_hw[0]}x{cfg.img_hw[1]}__N={data.x_train.shape[0]}"
        f"__act={cfg.activation}__L={cfg.depth}__widths={widths_label}"
        f"__models={cfg.num_models_per_width}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    centered_ntk_logits = _maybe_center_logits(ntk_result["test_logits"], enabled=cfg.center_classwise_logits)
    scatter_files = {}
    for item in width_results:
        scatter_files[str(item["width"])] = _save_scatter_plot(
            output_dir=output_dir,
            width=int(item["width"]),
            ntk_logits=centered_ntk_logits,
            mlp_logits=np.asarray(item["mean_test_logits_comparison"], dtype=np.float64),
            summary=item["comparison"],
            centered=bool(cfg.center_classwise_logits),
        )

    width_summary_plot = _save_width_summary_plot(output_dir, width_results)

    summary = {
        "config": asdict(cfg),
        "dataset": {
            "num_train": int(data.x_train.shape[0]),
            "num_test": int(data.x_test.shape[0]),
            "feature_dim": int(data.x_train.shape[1]),
            "train_indices": data.train_indices.tolist(),
            "test_indices": data.test_indices.tolist(),
        },
        "ntk": {
            "train_acc": float(ntk_result["train_acc"]),
            "test_acc": float(ntk_result["test_acc"]),
            "train_mse": float(ntk_result["train_mse"]),
            "test_mse": float(ntk_result["test_mse"]),
            "elapsed_seconds": float(ntk_result["elapsed_seconds"]),
        },
        "width_results": [
            {
                "width": int(item["width"]),
                "num_models": int(item["num_models"]),
                "ensemble_mean_test_acc": float(item["ensemble_mean_test_acc"]),
                "ensemble_mean_test_mse": float(item["ensemble_mean_test_mse"]),
                "per_model_test_acc_mean": float(item["per_model_test_acc_mean"]),
                "per_model_test_acc_std": float(item["per_model_test_acc_std"]),
                "per_model_test_mse_mean": float(item["per_model_test_mse_mean"]),
                "per_model_test_mse_std": float(item["per_model_test_mse_std"]),
                "elapsed_seconds": float(item["elapsed_seconds"]),
                "comparison": item["comparison"],
                "scatter_plot": scatter_files[str(item["width"])],
                "model_summaries": item["model_summaries"],
                "histories": item["histories"],
            }
            for item in width_results
        ],
        "center_classwise_logits": bool(cfg.center_classwise_logits),
        "artifacts": {
            "width_summary_plot": width_summary_plot,
            "scatter_plots": scatter_files,
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    savez_payload = {
        "x_train": data.x_train,
        "y_train": data.y_train,
        "y_train_onehot": data.y_train_onehot,
        "x_test": data.x_test,
        "y_test": data.y_test,
        "y_test_onehot": data.y_test_onehot,
        "ntk_test_logits_raw": np.asarray(ntk_result["test_logits"], dtype=np.float64),
        "ntk_test_logits_comparison": centered_ntk_logits,
    }
    for item in width_results:
        width = int(item["width"])
        savez_payload[f"mlp_ensemble_mean_test_logits_width={width}_raw"] = np.asarray(
            item["mean_test_logits_raw"],
            dtype=np.float64,
        )
        savez_payload[f"mlp_ensemble_mean_test_logits_width={width}_comparison"] = np.asarray(
            item["mean_test_logits_comparison"],
            dtype=np.float64,
        )
        savez_payload[f"mlp_ensemble_std_test_logits_width={width}"] = np.asarray(
            item["std_test_logits"],
            dtype=np.float64,
        )

    np.savez_compressed(output_dir / "predictions.npz", **savez_payload)
    return output_dir


def main() -> None:
    cfg = build_config_from_user_inputs()
    project_dir = Path(__file__).resolve().parent
    runtime_device = _choose_device(cfg.allow_cpu)
    ntk_device = _resolve_ntk_device(cfg, runtime_device)

    print("Preparing clean MNIST train/test data...", flush=True)
    data = prepare_clean_data(cfg, project_dir)
    print(
        f"Prepared clean dataset with {data.x_train.shape[0]} train and {data.x_test.shape[0]} test points.",
        flush=True,
    )

    print(f"Computing clean NTK once on device {ntk_device}...", flush=True)
    ntk_result = compute_clean_ntk_logits(cfg, data, device=ntk_device)
    print(
        f"NTK done: train_acc={100.0 * ntk_result['train_acc']:.2f}% "
        f"test_acc={100.0 * ntk_result['test_acc']:.2f}% "
        f"elapsed={ntk_result['elapsed_seconds']:.2f}s",
        flush=True,
    )

    centered_ntk_logits = _maybe_center_logits(ntk_result["test_logits"], enabled=cfg.center_classwise_logits)
    width_results = []
    for width in cfg.widths:
        print(
            f"Training clean MLP ensemble with width={width}, "
            f"num_models={cfg.num_models_per_width} on device {runtime_device}...",
            flush=True,
        )
        mlp_result = train_clean_mlp_ensemble_for_width(cfg, data, width=int(width), device=runtime_device)
        centered_mean_mlp_logits = _maybe_center_logits(
            mlp_result["mean_test_logits"],
            enabled=cfg.center_classwise_logits,
        )
        comparison = _comparison_summary(centered_ntk_logits, centered_mean_mlp_logits)
        width_results.append(
            {
                "width": int(width),
                "num_models": mlp_result["num_models"],
                "ensemble_mean_test_acc": mlp_result["ensemble_mean_test_acc"],
                "ensemble_mean_test_mse": mlp_result["ensemble_mean_test_mse"],
                "per_model_test_acc_mean": mlp_result["per_model_test_acc_mean"],
                "per_model_test_acc_std": mlp_result["per_model_test_acc_std"],
                "per_model_test_mse_mean": mlp_result["per_model_test_mse_mean"],
                "per_model_test_mse_std": mlp_result["per_model_test_mse_std"],
                "elapsed_seconds": mlp_result["elapsed_seconds"],
                "comparison": comparison,
                "model_summaries": mlp_result["model_summaries"],
                "histories": mlp_result["histories"],
                "mean_test_logits_raw": np.asarray(mlp_result["mean_test_logits"], dtype=np.float64),
                "mean_test_logits_comparison": centered_mean_mlp_logits,
                "std_test_logits": np.asarray(mlp_result["std_test_logits"], dtype=np.float64),
            }
        )
        print(
            f"width={width} ensemble done: "
            f"mean-logit test_acc={100.0 * mlp_result['ensemble_mean_test_acc']:.2f}% "
            f"per-model acc={100.0 * mlp_result['per_model_test_acc_mean']:.2f}% "
            f"+/-{100.0 * mlp_result['per_model_test_acc_std']:.2f}% "
            f"corr={comparison['correlation']} mae={comparison['mae']:.6e}",
            flush=True,
        )

    output_dir = save_outputs(cfg, data, ntk_result, width_results, project_dir=project_dir)
    print(f"Saved clean width-sweep comparison to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
