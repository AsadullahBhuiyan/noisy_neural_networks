from __future__ import annotations

import csv
import gzip
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from modules.data import (
    DatasetSubset,
    dataset_to_numpy_matrix,
    load_image_classification_dataset,
    normalize_dataset_name,
    normalize_feature_matrix_zero_mean_unit_rms,
)


################################################################################
# USER INPUTS
#
# Edit only this block for normal use. The script trains
# NUM_NOISE_DATASETS * NUM_MODELS_PER_NOISE_DATASET independent MLPs. For each
# noisy training dataset, it trains several independent network initializations
# on that same fixed noisy data instance.
################################################################################

NUM_MODELS = 1
NUM_NOISE_DATASETS = 1
NUM_MODELS_PER_NOISE_DATASET = 1
SEED = 22334
NUM_CLASSES = 10
DATASET = "mnist"  # mnist, fashion_mnist, or kmnist

# Use None for the full training set. Use an integer, e.g. 540, for a
# balanced subset with approximately equal class counts.
TRAIN_SIZE = 10000

# Image size. Use (28, 28) for ordinary data, (100, 100) for upsampled data.
IMG_HW = (28, 28)

DATA_ROOT = "data"
OUTPUT_ROOT = "trained_mlp_ensembles"

# Noise options: "replacement" or "additive_gaussian".
# replacement: replace each feature with normalized-scale uniform noise with probability p.
# additive_gaussian: form (1 - p) * data + p * gaussian noise.
NOISE_TYPE = "additive_gaussian"
NOISE_PROBABILITY = 0.9

# Architecture.
ACTIVATION = "erf"  # erf, gelu, tanh, sin, softplus, relu
DEPTH = 3
WIDTH = 512
WEIGHT_STD = 1.0
BIAS_STD = 0.0

# Training.
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
OPTIMIZER = "adamw"  # adam, adamw, or sgd
WEIGHT_DECAY = 0.0
LOSS_NAME = "xent"  # xent or mse
NUM_WORKERS = 0
EVAL_BATCH_SIZE = 1024
SAVE_EVERY = 1
SAVE_TEST_OUTPUTS = True
TEST_OUTPUTS_FILENAME = "test_outputs.csv.gz"

# Device / memory behavior.
ALLOW_CPU = False
MATERIALIZE_NOISY_TRAIN = False

################################################################################
# END USER INPUTS
################################################################################


@dataclass
class EnsembleTrainConfig:
    num_models: int = 1000
    num_noise_datasets: int = 1
    num_models_per_noise_dataset: int = 1
    seed: int = 22334
    num_classes: int = 10
    dataset: str = "mnist"
    train_size: int | None = None
    img_hw: tuple[int, int] = (28, 28)
    data_root: str = "data"
    output_root: str = "trained_mlp_ensembles"

    noise_type: str = "additive_gaussian"
    noise_probability: float = 0.98

    activation: str = "gelu"
    depth: int = 4
    width: int = 2048
    weight_std: float = 1.0
    bias_std: float = 0.0

    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    loss_name: str = "xent"
    num_workers: int = 0
    eval_batch_size: int = 1024
    save_every: int = 1
    save_test_outputs: bool = True
    test_outputs_filename: str = "test_outputs.csv.gz"
    allow_cpu: bool = False
    materialize_noisy_train: bool = False


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _choose_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        "No GPU backend was found. Set ALLOW_CPU = True in the USER INPUTS block "
        "if you intentionally want to train on CPU."
    )


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor] | nn.Module:
    normalized = name.strip().lower()
    if normalized == "erf":
        return ErfActivation()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "sin":
        return SinActivation()
    if normalized == "softplus":
        return nn.Softplus()
    if normalized == "relu":
        return nn.ReLU()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(
        f"Unsupported activation {name!r}. Expected erf, gelu, tanh, sin, softplus, silu, swish, leaky_relu, or relu."
    )


class ErfActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.erf(x)


class SinActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class MLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        depth: int,
        width: int,
        activation: str,
        weight_std: float,
        bias_std: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")

        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(max(0, int(depth) - 1)):
            linear = nn.Linear(in_dim, int(width))
            _init_linear(linear, weight_std=weight_std, bias_std=bias_std)
            layers.append(linear)
            layers.append(_activation(activation))
            in_dim = int(width)

        out = nn.Linear(in_dim, int(num_classes))
        _init_linear(out, weight_std=weight_std, bias_std=bias_std)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _init_linear(linear: nn.Linear, *, weight_std: float, bias_std: float) -> None:
    fan_in = int(linear.weight.shape[1])
    nn.init.normal_(linear.weight, mean=0.0, std=float(weight_std) / math.sqrt(float(fan_in)))
    if bias_std == 0.0:
        nn.init.zeros_(linear.bias)
    else:
        nn.init.normal_(linear.bias, mean=0.0, std=float(bias_std))


def _balanced_indices(labels: np.ndarray, train_size: int, num_classes: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if train_size >= labels.shape[0]:
        return np.arange(labels.shape[0], dtype=np.int64)
    if train_size < num_classes:
        raise ValueError("train_size must be at least num_classes for balanced subsampling.")

    rng = np.random.default_rng(int(seed))
    per_class = int(train_size) // int(num_classes)
    remainder = int(train_size) - per_class * int(num_classes)
    chosen = []
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        take = per_class + (1 if class_id < remainder else 0)
        if take > class_indices.size:
            raise ValueError(
                f"Requested {take} examples for class {class_id}, but only "
                f"{class_indices.size} are available."
            )
        chosen.append(rng.choice(class_indices, size=take, replace=False))
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return indices.astype(np.int64)


def _load_clean_dataset_arrays(cfg: EnsembleTrainConfig, project_dir: Path):
    train_raw, test_raw = load_image_classification_dataset(
        root=_resolve_data_root(cfg, project_dir),
        img_hw=cfg.img_hw,
        dataset_name=cfg.dataset,
    )

    if cfg.train_size is None:
        train_dataset = train_raw
        train_indices = np.arange(len(train_raw), dtype=np.int64)
    else:
        train_indices = _balanced_indices(
            np.asarray(train_raw.targets, dtype=np.int64),
            train_size=int(cfg.train_size),
            num_classes=int(cfg.num_classes),
            seed=int(cfg.seed) + 1,
        )
        train_dataset = DatasetSubset(train_raw, train_indices)

    x_train, _y_train_onehot, y_train = dataset_to_numpy_matrix(train_dataset, cfg.num_classes)
    x_test, _y_test_onehot, y_test = dataset_to_numpy_matrix(test_raw, cfg.num_classes)
    return x_train, y_train, x_test, y_test, train_indices


def _resolve_data_root(cfg: EnsembleTrainConfig, project_dir: Path) -> Path:
    path = Path(cfg.data_root)
    return path if path.is_absolute() else project_dir / path


def _resolve_output_root(cfg: EnsembleTrainConfig, project_dir: Path) -> Path:
    path = Path(cfg.output_root)
    return path if path.is_absolute() else project_dir / path


def _folder_name(cfg: EnsembleTrainConfig, n_train: int, feature_dim: int) -> str:
    h, w = cfg.img_hw
    p_str = f"{cfg.noise_probability:g}"
    loss_name = _normalize_loss_name(cfg.loss_name)
    dataset = normalize_dataset_name(cfg.dataset)
    return (
        f"mlp_ensemble__dataset={dataset}__{cfg.noise_type}__p={p_str}"
        f"__N={n_train}__d={h}x{w}"
        f"__loss={loss_name}__act={cfg.activation}__L={cfg.depth}__width={cfg.width}"
        f"__noiseK={cfg.num_noise_datasets}__initM={cfg.num_models_per_noise_dataset}"
        f"__models={cfg.num_models}"
    )


def _normalize_loss_name(loss_name: str) -> str:
    normalized = str(loss_name).strip().lower()
    if normalized in {"xent", "cross_entropy", "crossentropy", "ce"}:
        return "xent"
    if normalized in {"mse", "l2"}:
        return "mse"
    raise ValueError("loss_name must be one of: xent, mse.")


def _loss_targets(logits: torch.Tensor, y: torch.Tensor, cfg: EnsembleTrainConfig) -> torch.Tensor:
    return F.one_hot(y, num_classes=int(cfg.num_classes)).to(device=logits.device, dtype=logits.dtype)


def _loss_value(
    logits: torch.Tensor,
    y: torch.Tensor,
    cfg: EnsembleTrainConfig,
    *,
    reduction: str,
) -> torch.Tensor:
    loss_name = _normalize_loss_name(cfg.loss_name)
    if loss_name == "xent":
        return F.cross_entropy(logits, y, reduction=reduction)
    targets = _loss_targets(logits, y, cfg)
    return F.mse_loss(logits, targets, reduction=reduction)


def _loss_denominator(loss_name: str, total_seen: int, num_classes: int) -> int:
    if loss_name == "mse":
        return max(1, int(total_seen) * int(num_classes))
    return max(1, int(total_seen))


class DeterministicNoisyDataset(Dataset):
    """
    Fixed noisy training set without storing the noisy copy.

    For a given model seed, index, and clean input, __getitem__ always returns
    the same noisy vector. Different model seeds produce independent noisy
    training instances.
    """

    def __init__(
        self,
        x_clean: np.ndarray,
        y: np.ndarray,
        *,
        noise_type: str,
        noise_probability: float,
        seed: int,
    ) -> None:
        self.x_clean = np.asarray(x_clean, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.noise_type = str(noise_type).strip().lower()
        self.noise_probability = float(noise_probability)
        self.seed = int(seed)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        index = int(index)
        x = self.x_clean[index]
        rng = np.random.default_rng(self.seed + index)

        p = self.noise_probability
        if p <= 0.0:
            noisy = x.copy()
        elif self.noise_type in {"replacement", "uniform"}:
            mask = rng.random(size=x.shape) < p
            noise = rng.uniform(
                low=-math.sqrt(3.0),
                high=math.sqrt(3.0),
                size=x.shape,
            ).astype(np.float32)
            noisy = x.copy()
            noisy[mask] = noise[mask]
        elif self.noise_type in {"additive_gaussian", "gaussian", "gaussian_additive"}:
            gaussian = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float32)
            noisy = (1.0 - p) * x + p * gaussian
        else:
            raise ValueError(
                "noise_type must be 'replacement' or 'additive_gaussian'. "
                f"Received {self.noise_type!r}."
            )

        return torch.from_numpy(np.asarray(noisy, dtype=np.float32)), torch.tensor(self.y[index], dtype=torch.long)


class MaterializedNoisyDataset(Dataset):
    def __init__(
        self,
        x_clean: np.ndarray,
        y: np.ndarray,
        *,
        noise_type: str,
        noise_probability: float,
        seed: int,
    ) -> None:
        self.x_noisy = _make_noisy_array(
            x_clean,
            noise_type=noise_type,
            noise_probability=noise_probability,
            seed=seed,
        )
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        index = int(index)
        return torch.from_numpy(self.x_noisy[index]), torch.tensor(self.y[index], dtype=torch.long)


def _make_noisy_array(
    x_clean: np.ndarray,
    *,
    noise_type: str,
    noise_probability: float,
    seed: int,
) -> np.ndarray:
    x_clean = np.asarray(x_clean, dtype=np.float32)
    noise_type = str(noise_type).strip().lower()
    p = float(noise_probability)
    if p <= 0.0:
        return x_clean.copy()
    rng = np.random.default_rng(int(seed))
    if noise_type in {"replacement", "uniform"}:
        mask = rng.random(size=x_clean.shape) < p
        noise = rng.uniform(
            low=-math.sqrt(3.0),
            high=math.sqrt(3.0),
            size=x_clean.shape,
        ).astype(np.float32)
        x_noisy = x_clean.copy()
        x_noisy[mask] = noise[mask]
        return x_noisy
    if noise_type in {"additive_gaussian", "gaussian", "gaussian_additive"}:
        gaussian = rng.normal(loc=0.0, scale=1.0, size=x_clean.shape).astype(np.float32)
        return (1.0 - p) * x_clean + p * gaussian
    raise ValueError(f"Unknown noise_type={noise_type!r}.")


def _make_train_dataset(cfg: EnsembleTrainConfig, x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Dataset:
    if cfg.materialize_noisy_train:
        return MaterializedNoisyDataset(
            x_train,
            y_train,
            noise_type=cfg.noise_type,
            noise_probability=cfg.noise_probability,
            seed=seed,
        )
    return DeterministicNoisyDataset(
        x_train,
        y_train,
        noise_type=cfg.noise_type,
        noise_probability=cfg.noise_probability,
        seed=seed,
    )


class CleanTensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return self.x[int(index)], self.y[int(index)]


def _make_optimizer(cfg: EnsembleTrainConfig, model: nn.Module):
    normalized = cfg.optimizer.strip().lower()
    if normalized == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if normalized == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if normalized == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    raise ValueError("optimizer must be one of: adam, adamw, sgd.")


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: EnsembleTrainConfig,
) -> dict[str, float]:
    model.eval()
    loss_name = _normalize_loss_name(cfg.loss_name)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            logits = model(x)
            loss = _loss_value(logits, y, cfg, reduction="sum")
            total_loss += float(loss.detach().cpu())
            total_correct += int((torch.argmax(logits, dim=1) == y).sum().detach().cpu())
            total_seen += int(y.numel())
    return {
        "loss": total_loss / _loss_denominator(loss_name, total_seen, cfg.num_classes),
        "accuracy": total_correct / max(1, total_seen),
    }


def _test_output_fieldnames(num_classes: int) -> list[str]:
    return [
        "model_number",
        "model_index",
        "noise_dataset_number",
        "noise_dataset_index",
        "init_number",
        "init_index",
        "model_seed",
        "noise_seed",
        "train_size",
        "feature_dim",
        "test_index",
        "true_label",
        "predicted_label",
        *[f"logit_{class_id}" for class_id in range(int(num_classes))],
    ]


def _open_text_maybe_gzip(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def _initialize_test_outputs_csv(path: Path, num_classes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text_maybe_gzip(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=_test_output_fieldnames(num_classes))
        writer.writeheader()


def _append_model_test_outputs_csv(
    path: Path,
    *,
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    model_number: int,
    model_index: int,
    noise_dataset_number: int,
    noise_dataset_index: int,
    init_number: int,
    init_index: int,
    model_seed: int,
    noise_seed: int,
    train_size: int,
    feature_dim: int,
) -> None:
    fieldnames = _test_output_fieldnames(num_classes)
    model.eval()
    test_index = 0
    with _open_text_maybe_gzip(path, "at") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        with torch.inference_mode():
            for x, y in test_loader:
                x = x.to(device=device, non_blocking=True)
                logits = model(x).detach().cpu().numpy().astype(np.float64, copy=False)
                labels = y.detach().cpu().numpy().astype(np.int64, copy=False)
                predicted = np.argmax(logits, axis=1).astype(np.int64, copy=False)
                for row_index in range(int(logits.shape[0])):
                    row = {
                        "model_number": int(model_number),
                        "model_index": int(model_index),
                        "noise_dataset_number": int(noise_dataset_number),
                        "noise_dataset_index": int(noise_dataset_index),
                        "init_number": int(init_number),
                        "init_index": int(init_index),
                        "model_seed": int(model_seed),
                        "noise_seed": int(noise_seed),
                        "train_size": int(train_size),
                        "feature_dim": int(feature_dim),
                        "test_index": int(test_index),
                        "true_label": int(labels[row_index]),
                        "predicted_label": int(predicted[row_index]),
                    }
                    for class_id in range(int(num_classes)):
                        row[f"logit_{class_id}"] = float(logits[row_index, class_id])
                    writer.writerow(row)
                    test_index += 1


def _normalize_ensemble_config(cfg: EnsembleTrainConfig) -> EnsembleTrainConfig:
    payload = asdict(cfg)
    payload["dataset"] = normalize_dataset_name(payload.get("dataset", "mnist"))
    payload["loss_name"] = _normalize_loss_name(cfg.loss_name)

    num_models = int(payload["num_models"])
    num_noise_datasets = int(payload.get("num_noise_datasets", 1))
    num_models_per_noise_dataset = int(payload.get("num_models_per_noise_dataset", 1))
    if num_models < 1:
        raise ValueError("num_models must be at least 1.")
    if num_noise_datasets < 1:
        raise ValueError("num_noise_datasets must be at least 1.")
    if num_models_per_noise_dataset < 1:
        raise ValueError("num_models_per_noise_dataset must be at least 1.")

    if num_noise_datasets == 1 and num_models_per_noise_dataset == 1 and num_models != 1:
        num_noise_datasets = num_models
        num_models_per_noise_dataset = 1

    expected_num_models = num_noise_datasets * num_models_per_noise_dataset
    if num_models != expected_num_models:
        print(
            f"Adjusting num_models from {num_models} to "
            f"num_noise_datasets * num_models_per_noise_dataset = {expected_num_models}.",
            flush=True,
        )
        num_models = expected_num_models

    payload["num_models"] = int(num_models)
    payload["num_noise_datasets"] = int(num_noise_datasets)
    payload["num_models_per_noise_dataset"] = int(num_models_per_noise_dataset)
    return EnsembleTrainConfig(**payload)


def _iter_model_specs(cfg: EnsembleTrainConfig):
    model_index = 0
    for noise_dataset_index in range(int(cfg.num_noise_datasets)):
        noise_dataset_number = noise_dataset_index + 1
        noise_seed = int(cfg.seed) + 111 + noise_dataset_index
        for init_index in range(int(cfg.num_models_per_noise_dataset)):
            init_number = init_index + 1
            model_number = model_index + 1
            model_seed = int(cfg.seed) + 10_000 * model_number
            yield {
                "model_index": model_index,
                "model_number": model_number,
                "noise_dataset_index": noise_dataset_index,
                "noise_dataset_number": noise_dataset_number,
                "init_index": init_index,
                "init_number": init_number,
                "noise_seed": noise_seed,
                "model_seed": model_seed,
            }
            model_index += 1


def train_one_model(
    *,
    cfg: EnsembleTrainConfig,
    model_index: int,
    model_number: int,
    noise_dataset_index: int,
    noise_dataset_number: int,
    init_index: int,
    init_number: int,
    model_seed: int,
    noise_seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    test_outputs_path: Path | None,
) -> dict[str, object]:
    _set_global_seed(model_seed)

    train_dataset = _make_train_dataset(cfg, x_train, y_train, seed=noise_seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(model_seed + 77)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
        generator=loader_generator,
    )

    model = MLP(
        input_dim=int(x_train.shape[1]),
        num_classes=int(cfg.num_classes),
        depth=int(cfg.depth),
        width=int(cfg.width),
        activation=str(cfg.activation),
        weight_std=float(cfg.weight_std),
        bias_std=float(cfg.bias_std),
    ).to(device)

    optimizer = _make_optimizer(cfg, model)
    loss_name = _normalize_loss_name(cfg.loss_name)

    history = []
    start_time = time.perf_counter()
    for epoch_index in range(int(cfg.epochs)):
        epoch_number = epoch_index + 1
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for x, y in train_loader:
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = _loss_value(logits, y, cfg, reduction="mean")
            loss.backward()
            optimizer.step()

            batch_size = int(y.numel())
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((torch.argmax(logits.detach(), dim=1) == y).sum().detach().cpu())
            total_seen += batch_size

        train_metrics = {
            "loss": total_loss / max(1, total_seen),
            "accuracy": total_correct / max(1, total_seen),
        }
        test_metrics = _evaluate(model, test_loader, device, cfg)
        history.append(
            {
                "epoch": epoch_number,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )
        print(
            f"model {model_number}/{cfg.num_models} "
            f"(noise {noise_dataset_number}/{cfg.num_noise_datasets}, "
            f"init {init_number}/{cfg.num_models_per_noise_dataset}) "
            f"epoch {epoch_number}/{cfg.epochs} "
            f"loss={loss_name} "
            f"train_loss={train_metrics['loss']:.6f} train_acc={100.0 * train_metrics['accuracy']:.2f}% "
            f"test_loss={test_metrics['loss']:.4f} test_acc={100.0 * test_metrics['accuracy']:.2f}%",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    final_metrics = history[-1] if history else {}
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "model_index": model_index,
        "model_number": model_number,
        "noise_dataset_index": int(noise_dataset_index),
        "noise_dataset_number": int(noise_dataset_number),
        "init_index": int(init_index),
        "init_number": int(init_number),
        "model_seed": model_seed,
        "noise_seed": noise_seed,
        "input_dim": int(x_train.shape[1]),
        "num_train": int(x_train.shape[0]),
        "history": history,
        "final_metrics": final_metrics,
        "elapsed_seconds": elapsed,
    }
    model_path = output_dir / f"model{model_number}.pth"
    torch.save(checkpoint, model_path)
    if test_outputs_path is not None:
        _append_model_test_outputs_csv(
            test_outputs_path,
            model=model,
            test_loader=test_loader,
            device=device,
            num_classes=int(cfg.num_classes),
            model_number=model_number,
            model_index=model_index,
            noise_dataset_number=noise_dataset_number,
            noise_dataset_index=noise_dataset_index,
            init_number=init_number,
            init_index=init_index,
            model_seed=model_seed,
            noise_seed=noise_seed,
            train_size=int(x_train.shape[0]),
            feature_dim=int(x_train.shape[1]),
        )
    return {
        "model_file": model_path.name,
        "model_number": model_number,
        "noise_dataset_index": int(noise_dataset_index),
        "noise_dataset_number": int(noise_dataset_number),
        "init_index": int(init_index),
        "init_number": int(init_number),
        "model_seed": model_seed,
        "noise_seed": noise_seed,
        "elapsed_seconds": elapsed,
        "final_metrics": final_metrics,
    }


def train_ensemble(cfg: EnsembleTrainConfig, project_dir: Path) -> Path:
    cfg = _normalize_ensemble_config(cfg)
    _set_global_seed(cfg.seed)
    device = _choose_device(cfg.allow_cpu)
    print(f"Using device: {device}", flush=True)
    print(
        f"Nested ensemble layout: {cfg.num_noise_datasets} noisy training datasets x "
        f"{cfg.num_models_per_noise_dataset} initializations = {cfg.num_models} models.",
        flush=True,
    )
    print(f"Loading and preprocessing clean {cfg.dataset} train/test arrays...", flush=True)
    x_train, y_train, x_test, y_test, train_indices = _load_clean_dataset_arrays(cfg, project_dir)
    x_train = normalize_feature_matrix_zero_mean_unit_rms(x_train)
    x_test = normalize_feature_matrix_zero_mean_unit_rms(x_test)

    output_root = _resolve_output_root(cfg, project_dir)
    output_dir = output_root / _folder_name(cfg, n_train=x_train.shape[0], feature_dim=x_train.shape[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    test_loader = DataLoader(
        CleanTensorDataset(x_test, y_test),
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    metadata = {
        "config": asdict(cfg),
        "device": str(device),
        "dataset": cfg.dataset,
        "num_train": int(x_train.shape[0]),
        "num_test": int(x_test.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "train_indices": np.asarray(train_indices, dtype=np.int64).tolist(),
        "test_is_clean": True,
        "normalization": (
            "Each train/test example is mean-centered across features and normalized to centered unit RMS. "
            "Replacement noise swaps each feature with normalized-scale uniform noise with probability p. "
            "Additive Gaussian noise uses (1 - p) * clean + p * noise. "
            "Noisy training examples are not normalized again."
        ),
        "loss_name": cfg.loss_name,
        "mse_targets": "One-hot label vectors are used when loss_name='mse'.",
        "nested_ensemble_layout": {
            "num_noise_datasets": int(cfg.num_noise_datasets),
            "num_models_per_noise_dataset": int(cfg.num_models_per_noise_dataset),
            "num_models": int(cfg.num_models),
            "meaning": (
                "For each noisy training dataset, multiple independently initialized "
                "networks are trained on the same deterministic noisy dataset."
            ),
        },
        "checkpoint_format": (
            "Each modelN.pth contains model_state_dict, config, seeds, nested "
            "noise/init indices, history, and metrics."
        ),
    }
    if cfg.save_test_outputs:
        metadata["test_outputs_csv"] = str(output_dir / cfg.test_outputs_filename)
        metadata["test_outputs_rows"] = f"One row per trained model and clean {cfg.dataset} test example."
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    test_outputs_path = output_dir / cfg.test_outputs_filename if cfg.save_test_outputs else None
    if test_outputs_path is not None:
        _initialize_test_outputs_csv(test_outputs_path, int(cfg.num_classes))

    summaries = []
    for spec in _iter_model_specs(cfg):
        summary = train_one_model(
            cfg=cfg,
            model_index=int(spec["model_index"]),
            model_number=int(spec["model_number"]),
            noise_dataset_index=int(spec["noise_dataset_index"]),
            noise_dataset_number=int(spec["noise_dataset_number"]),
            init_index=int(spec["init_index"]),
            init_number=int(spec["init_number"]),
            model_seed=int(spec["model_seed"]),
            noise_seed=int(spec["noise_seed"]),
            x_train=x_train,
            y_train=y_train,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            test_outputs_path=test_outputs_path,
        )
        summaries.append(summary)
        model_index = int(spec["model_index"])
        if model_index % max(1, int(cfg.save_every)) == 0 or model_index + 1 == int(cfg.num_models):
            (output_dir / "training_summary.json").write_text(
                json.dumps({"models": summaries}, indent=2),
                encoding="utf-8",
            )

    return output_dir


def build_config_from_user_inputs() -> EnsembleTrainConfig:
    num_noise_datasets = int(NUM_NOISE_DATASETS)
    num_models_per_noise_dataset = int(NUM_MODELS_PER_NOISE_DATASET)
    if num_noise_datasets == 1 and num_models_per_noise_dataset == 1 and int(NUM_MODELS) != 1:
        num_noise_datasets = int(NUM_MODELS)
        num_models_per_noise_dataset = 1

    return EnsembleTrainConfig(
        num_models=num_noise_datasets * num_models_per_noise_dataset,
        num_noise_datasets=num_noise_datasets,
        num_models_per_noise_dataset=num_models_per_noise_dataset,
        seed=SEED,
        num_classes=NUM_CLASSES,
        dataset=DATASET,
        train_size=TRAIN_SIZE,
        img_hw=IMG_HW,
        data_root=DATA_ROOT,
        output_root=OUTPUT_ROOT,
        noise_type=NOISE_TYPE,
        noise_probability=NOISE_PROBABILITY,
        activation=ACTIVATION,
        depth=DEPTH,
        width=WIDTH,
        weight_std=WEIGHT_STD,
        bias_std=BIAS_STD,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        optimizer=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        loss_name=LOSS_NAME,
        num_workers=NUM_WORKERS,
        eval_batch_size=EVAL_BATCH_SIZE,
        save_every=SAVE_EVERY,
        save_test_outputs=SAVE_TEST_OUTPUTS,
        test_outputs_filename=TEST_OUTPUTS_FILENAME,
        allow_cpu=ALLOW_CPU,
        materialize_noisy_train=MATERIALIZE_NOISY_TRAIN,
    )


def main() -> None:
    cfg = build_config_from_user_inputs()
    project_dir = Path(__file__).resolve().parent
    output_dir = train_ensemble(cfg, project_dir)
    print(f"Saved ensemble checkpoints to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
