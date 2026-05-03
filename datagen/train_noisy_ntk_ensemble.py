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

from custom_ntk import ActivationMoments, build_clean_first_layer_terms
from modules.data import (
    DatasetSubset,
    dataset_to_numpy_matrix,
    load_mnist,
    normalize_feature_matrix_unit_rms,
)


################################################################################
# USER INPUTS
#
# Edit only this block for normal use. This script trains exact infinite-width
# NTK kernel regressors on NUM_NOISE_DATASETS independently noised MNIST training
# sets, evaluates each one on a clean MNIST test subset, and stores all class
# logits for plotting later.
################################################################################

NUM_NOISE_DATASETS = 100
SEED = 22334
NUM_CLASSES = 10

# Use None for the full MNIST training set. Use an integer, e.g. 1000, for a
# balanced subset with approximately equal class counts.
TRAIN_SIZE = 400

# Use None for the full MNIST test set. Use an integer for a clean test subset.
TEST_SIZE = 10
TEST_SUBSET_MODE = "balanced"  # balanced, random, or first

# Image size. Use (28, 28) for ordinary MNIST, (100, 100) for upsampled MNIST.
IMG_HW = (28, 28)

DATA_ROOT = "data"
OUTPUT_ROOT = "trained_ntk_ensembles"

# Noise options: "replacement" or "additive_gaussian".
NOISE_TYPE = "additive_gaussian"
NOISE_PROBABILITY = 0.95

# Match the MLP script's default deterministic dataset behavior with
# "per_example". Use "single_rng" to materialize noise from one RNG stream.
NOISE_GENERATION_MODE = "per_example"  # per_example or single_rng

# NTK architecture. WEIGHT_STD and BIAS_STD are converted to Cw and Cb by
# squaring, matching weights ~ N(0, WEIGHT_STD^2 / fan_in) and biases ~ N(0, BIAS_STD^2).
ACTIVATION = "erf"  # erf, relu, gelu, tanh, sin, softplus, silu, swish, leaky_relu
DEPTH = 3
WEIGHT_STD = 1.0
BIAS_STD = 0.0
DIAG_REG = 1e-8

# Generic activations use Gauss-Hermite quadrature. Closed-form erf/relu ignore
# MOMENT_QUADRATURE_N.
MOMENT_QUADRATURE_N = 40
MOMENT_BATCH_SIZE = 4096

# Device / memory behavior.
ALLOW_CPU = False
DTYPE = "float64"  # float64 is safest for the kernel solve; float32 uses less memory.
ALLOW_TF32 = True
TEST_BATCH_SIZE = 512

# Output behavior.
SAVE_CSV = True
SAVE_NPZ = True
SAVE_TRAIN_OUTPUTS = False
TEST_OUTPUTS_FILENAME = "test_outputs.csv.gz"
ALL_OUTPUTS_FILENAME = "all_test_outputs.npz"
SAVE_EVERY = 1

################################################################################
# END USER INPUTS
################################################################################


@dataclass
class NTKEnsembleConfig:
    num_noise_datasets: int = 1
    seed: int = 22334
    num_classes: int = 10
    train_size: int | None = 1000
    test_size: int | None = 1000
    test_subset_mode: str = "balanced"
    img_hw: tuple[int, int] = (28, 28)
    data_root: str = "data"
    output_root: str = "trained_ntk_ensembles"

    noise_type: str = "additive_gaussian"
    noise_probability: float = 0.9
    noise_generation_mode: str = "per_example"

    activation: str = "erf"
    depth: int = 3
    weight_std: float = 1.0
    bias_std: float = 0.0
    diag_reg: float = 1e-8
    moment_quadrature_n: int = 40
    moment_batch_size: int = 4096

    allow_cpu: bool = False
    dtype: str = "float64"
    allow_tf32: bool = True
    test_batch_size: int = 512

    save_csv: bool = True
    save_npz: bool = True
    save_train_outputs: bool = False
    test_outputs_filename: str = "test_outputs.csv.gz"
    all_outputs_filename: str = "all_test_outputs.npz"
    save_every: int = 1


@dataclass
class TrainedNTKState:
    train_solution: torch.Tensor
    train_diag_layers: list[torch.Tensor]
    train_logits: torch.Tensor | None
    train_seconds: float
    solve_diag_reg: float


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
        "if you intentionally want to run the NTK solve on CPU."
    )


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized in {"float64", "double", "fp64"}:
        return torch.float64
    if normalized in {"float32", "single", "fp32"}:
        return torch.float32
    raise ValueError("DTYPE must be one of: float64, float32.")


def _ntk_constants(cfg: NTKEnsembleConfig) -> tuple[float, float]:
    return float(cfg.bias_std) ** 2, float(cfg.weight_std) ** 2


def _resolve_data_root(cfg: NTKEnsembleConfig, project_dir: Path) -> Path:
    path = Path(cfg.data_root)
    return path if path.is_absolute() else project_dir / path


def _resolve_output_root(cfg: NTKEnsembleConfig, project_dir: Path) -> Path:
    path = Path(cfg.output_root)
    return path if path.is_absolute() else project_dir / path


def _balanced_indices(labels: np.ndarray, subset_size: int, num_classes: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if subset_size >= labels.shape[0]:
        return np.arange(labels.shape[0], dtype=np.int64)
    if subset_size < num_classes:
        raise ValueError("subset_size must be at least num_classes for balanced subsampling.")

    rng = np.random.default_rng(int(seed))
    per_class = int(subset_size) // int(num_classes)
    remainder = int(subset_size) - per_class * int(num_classes)
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


def _subset_indices(
    labels: np.ndarray,
    subset_size: int | None,
    *,
    num_classes: int,
    seed: int,
    mode: str,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if subset_size is None or int(subset_size) >= labels.shape[0]:
        return np.arange(labels.shape[0], dtype=np.int64)

    size = int(subset_size)
    normalized = str(mode).strip().lower()
    if normalized == "balanced":
        return _balanced_indices(labels, size, num_classes=num_classes, seed=seed)
    if normalized == "random":
        rng = np.random.default_rng(int(seed))
        return rng.choice(np.arange(labels.shape[0]), size=size, replace=False).astype(np.int64)
    if normalized == "first":
        return np.arange(size, dtype=np.int64)
    raise ValueError("TEST_SUBSET_MODE must be one of: balanced, random, first.")


def _load_clean_mnist_arrays(cfg: NTKEnsembleConfig, project_dir: Path):
    train_raw, test_raw = load_mnist(root=_resolve_data_root(cfg, project_dir), img_hw=cfg.img_hw)

    if cfg.train_size is None:
        train_indices = np.arange(len(train_raw), dtype=np.int64)
    else:
        train_indices = _balanced_indices(
            np.asarray(train_raw.targets, dtype=np.int64),
            subset_size=int(cfg.train_size),
            num_classes=int(cfg.num_classes),
            seed=int(cfg.seed) + 1,
        )

    test_indices = _subset_indices(
        np.asarray(test_raw.targets, dtype=np.int64),
        cfg.test_size,
        num_classes=int(cfg.num_classes),
        seed=int(cfg.seed) + 2,
        mode=cfg.test_subset_mode,
    )

    train_dataset = DatasetSubset(train_raw, train_indices)
    test_dataset = DatasetSubset(test_raw, test_indices)
    x_train, _y_train_onehot, y_train = dataset_to_numpy_matrix(train_dataset, cfg.num_classes)
    x_test, _y_test_onehot, y_test = dataset_to_numpy_matrix(test_dataset, cfg.num_classes)

    # Keep this explicit to mirror train_noisy_mlp_ensemble.py. It is idempotent
    # because dataset_to_numpy_matrix also normalizes each row to unit RMS.
    x_train = normalize_feature_matrix_unit_rms(x_train)
    x_test = normalize_feature_matrix_unit_rms(x_test)
    return x_train, y_train, x_test, y_test, train_indices, test_indices


def _make_noisy_vector(
    x: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_type: str,
    noise_probability: float,
) -> np.ndarray:
    p = float(noise_probability)
    if p <= 0.0:
        return np.asarray(x, dtype=np.float32).copy()

    normalized = str(noise_type).strip().lower()
    if normalized == "replacement":
        mask = rng.random(size=x.shape) < p
        replacements = rng.random(size=x.shape).astype(np.float32)
        noisy = np.asarray(x, dtype=np.float32).copy()
        noisy[mask] = replacements[mask]
        return noisy

    if normalized in {"additive_gaussian", "gaussian", "gaussian_additive"}:
        epsilon = 1.0 - p
        gaussian = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float32)
        return epsilon * np.asarray(x, dtype=np.float32) + (1.0 - epsilon) * gaussian

    raise ValueError("NOISE_TYPE must be 'replacement' or 'additive_gaussian'.")


def _make_noisy_array(
    x_clean: np.ndarray,
    *,
    noise_type: str,
    noise_probability: float,
    seed: int,
    generation_mode: str,
) -> np.ndarray:
    x_clean = np.asarray(x_clean, dtype=np.float32)
    mode = str(generation_mode).strip().lower()

    if mode == "single_rng":
        rng = np.random.default_rng(int(seed))
        if float(noise_probability) <= 0.0:
            return x_clean.copy()
        normalized = str(noise_type).strip().lower()
        if normalized == "replacement":
            mask = rng.random(size=x_clean.shape) < float(noise_probability)
            replacements = rng.random(size=x_clean.shape).astype(np.float32)
            x_noisy = x_clean.copy()
            x_noisy[mask] = replacements[mask]
            return x_noisy
        if normalized in {"additive_gaussian", "gaussian", "gaussian_additive"}:
            epsilon = 1.0 - float(noise_probability)
            gaussian = rng.normal(loc=0.0, scale=1.0, size=x_clean.shape).astype(np.float32)
            return epsilon * x_clean + (1.0 - epsilon) * gaussian
        raise ValueError("NOISE_TYPE must be 'replacement' or 'additive_gaussian'.")

    if mode == "per_example":
        x_noisy = np.empty_like(x_clean, dtype=np.float32)
        for index in range(int(x_clean.shape[0])):
            rng = np.random.default_rng(int(seed) + index)
            x_noisy[index] = _make_noisy_vector(
                x_clean[index],
                rng=rng,
                noise_type=noise_type,
                noise_probability=noise_probability,
            )
        return x_noisy

    raise ValueError("NOISE_GENERATION_MODE must be one of: per_example, single_rng.")


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    out = np.zeros((labels.shape[0], int(num_classes)), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def _activation_functions(
    name: str,
) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, Callable[[torch.Tensor], torch.Tensor] | None]:
    normalized = str(name).strip().lower()
    if normalized in {"erf", "relu"}:
        return None, None
    if normalized == "gelu":
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
        return (
            lambda x: torch.nn.functional.gelu(x),
            lambda x: 0.5 * (1.0 + torch.erf(x * inv_sqrt2)) + x * torch.exp(-0.5 * x * x) * inv_sqrt2pi,
        )
    if normalized == "tanh":
        return torch.tanh, lambda x: 1.0 - torch.tanh(x) ** 2
    if normalized == "sin":
        return torch.sin, torch.cos
    if normalized == "softplus":
        return torch.nn.functional.softplus, torch.sigmoid
    if normalized in {"silu", "swish"}:
        return (
            torch.nn.functional.silu,
            lambda x: torch.sigmoid(x) * (1.0 + x * (1.0 - torch.sigmoid(x))),
        )
    if normalized == "leaky_relu":
        return (
            lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01),
            lambda x: torch.where(x >= 0.0, torch.ones_like(x), torch.full_like(x, 0.01)),
        )
    raise ValueError(
        f"Unsupported activation {name!r}. Expected erf, relu, gelu, tanh, sin, "
        "softplus, silu, swish, or leaky_relu."
    )


def _make_moments(
    cfg: NTKEnsembleConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ActivationMoments:
    act_fn, act_p_fn = _activation_functions(cfg.activation)
    return ActivationMoments(
        act=cfg.activation,
        act_fn=act_fn,
        act_p_fn=act_p_fn,
        quadrature_n=int(cfg.moment_quadrature_n),
        batch_size=int(cfg.moment_batch_size),
        device=device,
        dtype=dtype,
    )


def _train_exact_ntk(
    *,
    x_train_noisy,
    y_train_onehot: np.ndarray,
    cfg: NTKEnsembleConfig,
    moments: ActivationMoments,
    device: torch.device,
    dtype: torch.dtype,
) -> TrainedNTKState:
    start = time.perf_counter()
    cb, cw = _ntk_constants(cfg)

    first = build_clean_first_layer_terms(
        X_train=x_train_noisy,
        X_test=x_train_noisy[:1],
        Cb=cb,
        Cw=cw,
        device=device,
        dtype=dtype,
    )
    train_k = first.train_train
    train_theta = train_k.clone()
    train_diag_layers = [torch.diag(train_k).clone()]

    for _layer_number in range(2, int(cfg.depth) + 1):
        train_var = torch.diag(train_k)
        train_q1 = train_var[:, None]
        train_q2 = train_var[None, :]
        train_eaa = moments.nngp(train_q1, train_q2, train_k)
        train_epp = moments.sigma_dot(train_q1, train_q2, train_k)

        next_train_k = cb + cw * train_eaa
        next_train_theta = cb + cw * (train_eaa + train_epp * train_theta)

        train_k = next_train_k
        train_theta = next_train_theta
        train_diag_layers.append(torch.diag(train_k).clone())
        del train_eaa, train_epp

    y_train = torch.as_tensor(y_train_onehot, device=device, dtype=dtype)
    identity = torch.eye(train_theta.shape[0], device=device, dtype=dtype)
    solve_diag_reg = float(cfg.diag_reg)
    train_solution = None
    for attempt in range(6):
        try:
            train_solution = torch.linalg.solve(train_theta + solve_diag_reg * identity, y_train)
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise
            if attempt == 5:
                raise
            solve_diag_reg = max(10.0 * solve_diag_reg, 1e-12)
            print(f"Kernel solve failed; retrying with DIAG_REG={solve_diag_reg:g}", flush=True)
    if train_solution is None:
        raise RuntimeError("Kernel solve did not produce a solution.")
    train_logits = train_theta @ train_solution if bool(cfg.save_train_outputs) else None

    return TrainedNTKState(
        train_solution=train_solution,
        train_diag_layers=train_diag_layers,
        train_logits=train_logits,
        train_seconds=time.perf_counter() - start,
        solve_diag_reg=solve_diag_reg,
    )


def _predict_exact_ntk(
    *,
    x_train_noisy,
    x_test: np.ndarray,
    state: TrainedNTKState,
    cfg: NTKEnsembleConfig,
    moments: ActivationMoments,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    cb, cw = _ntk_constants(cfg)
    batch_size = max(1, int(cfg.test_batch_size))
    logits_chunks = []

    for start in range(0, int(x_test.shape[0]), batch_size):
        end = min(start + batch_size, int(x_test.shape[0]))
        first = build_clean_first_layer_terms(
            X_train=x_train_noisy,
            X_test=x_test[start:end],
            Cb=cb,
            Cw=cw,
            device=device,
            dtype=dtype,
        )

        test_diag = first.test_diag
        test_k = first.test_train
        test_theta = test_k.clone()

        for layer_number in range(2, int(cfg.depth) + 1):
            train_var = state.train_diag_layers[layer_number - 2]
            test_q1 = test_diag[:, None]
            test_q2 = train_var[None, :]
            test_eaa = moments.nngp(test_q1, test_q2, test_k)
            test_epp = moments.sigma_dot(test_q1, test_q2, test_k)

            next_test_k = cb + cw * test_eaa
            next_test_theta = cb + cw * (test_eaa + test_epp * test_theta)
            next_test_diag = cb + cw * moments.nngp(test_diag, test_diag, test_diag)

            test_diag = next_test_diag
            test_k = next_test_k
            test_theta = next_test_theta

        logits = test_theta @ state.train_solution
        logits_chunks.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))

    return np.concatenate(logits_chunks, axis=0)


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return 0.0
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def _folder_name(cfg: NTKEnsembleConfig, *, n_train: int, n_test: int, feature_dim: int) -> str:
    h, w = cfg.img_hw
    p_str = f"{cfg.noise_probability:g}"
    weight_str = f"{cfg.weight_std:g}"
    bias_str = f"{cfg.bias_std:g}"
    reg_str = f"{cfg.diag_reg:g}"
    return (
        f"ntk_ensemble__{cfg.noise_type}__p={p_str}"
        f"__N={n_train}__M={n_test}__d={h}x{w}"
        f"__act={cfg.activation}__L={cfg.depth}"
        f"__wstd={weight_str}__bstd={bias_str}__reg={reg_str}"
        f"__noiseK={cfg.num_noise_datasets}"
    )


def _test_output_fieldnames(num_classes: int) -> list[str]:
    return [
        "ntk_number",
        "ntk_index",
        "noise_dataset_number",
        "noise_dataset_index",
        "noise_seed",
        "train_size",
        "feature_dim",
        "test_subset_index",
        "mnist_test_index",
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


def _append_test_outputs_csv(
    path: Path,
    *,
    logits: np.ndarray,
    y_test: np.ndarray,
    test_indices: np.ndarray,
    num_classes: int,
    ntk_number: int,
    ntk_index: int,
    noise_seed: int,
    train_size: int,
    feature_dim: int,
) -> None:
    fieldnames = _test_output_fieldnames(num_classes)
    predicted = np.argmax(logits, axis=1).astype(np.int64, copy=False)

    with _open_text_maybe_gzip(path, "at") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for row_index in range(int(logits.shape[0])):
            row = {
                "ntk_number": int(ntk_number),
                "ntk_index": int(ntk_index),
                "noise_dataset_number": int(ntk_number),
                "noise_dataset_index": int(ntk_index),
                "noise_seed": int(noise_seed),
                "train_size": int(train_size),
                "feature_dim": int(feature_dim),
                "test_subset_index": int(row_index),
                "mnist_test_index": int(test_indices[row_index]),
                "true_label": int(y_test[row_index]),
                "predicted_label": int(predicted[row_index]),
            }
            for class_id in range(int(num_classes)):
                row[f"logit_{class_id}"] = float(logits[row_index, class_id])
            writer.writerow(row)


def run_ntk_ensemble(cfg: NTKEnsembleConfig, project_dir: Path) -> Path:
    if int(cfg.num_noise_datasets) < 1:
        raise ValueError("NUM_NOISE_DATASETS must be at least 1.")
    if int(cfg.depth) < 1:
        raise ValueError("DEPTH must be at least 1.")

    _set_global_seed(cfg.seed)
    device = _choose_device(cfg.allow_cpu)
    dtype = _torch_dtype(cfg.dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)

    print(f"Using device: {device} ({dtype})", flush=True)
    print("Loading and preprocessing clean MNIST train/test arrays...", flush=True)
    x_train, y_train, x_test, y_test, train_indices, test_indices = _load_clean_mnist_arrays(cfg, project_dir)
    y_train_onehot = _one_hot(y_train, int(cfg.num_classes))

    output_root = _resolve_output_root(cfg, project_dir)
    output_dir = output_root / _folder_name(
        cfg,
        n_train=int(x_train.shape[0]),
        n_test=int(x_test.shape[0]),
        feature_dim=int(x_train.shape[1]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cb, cw = _ntk_constants(cfg)
    metadata = {
        "config": asdict(cfg),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "num_train": int(x_train.shape[0]),
        "num_test": int(x_test.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "train_indices": np.asarray(train_indices, dtype=np.int64).tolist(),
        "test_indices": np.asarray(test_indices, dtype=np.int64).tolist(),
        "test_is_clean": True,
        "normalization": "Each clean train/test example is normalized to unit RMS before training noise is applied.",
        "target_format": "One-hot MNIST labels, solved by exact NTK kernel regression.",
        "ntk_constants": {"Cb": cb, "Cw": cw},
        "noise_seed_rule": "noise_seed = SEED + 111 + noise_dataset_index",
        "test_outputs_csv": str(output_dir / cfg.test_outputs_filename) if cfg.save_csv else None,
        "all_outputs_npz": str(output_dir / cfg.all_outputs_filename) if cfg.save_npz else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    test_outputs_path = output_dir / cfg.test_outputs_filename if cfg.save_csv else None
    if test_outputs_path is not None:
        _initialize_test_outputs_csv(test_outputs_path, int(cfg.num_classes))

    moments = _make_moments(cfg, device=device, dtype=dtype)
    summaries = []
    all_logits = []
    all_noise_seeds = []

    for noise_dataset_index in range(int(cfg.num_noise_datasets)):
        noise_dataset_number = noise_dataset_index + 1
        noise_seed = int(cfg.seed) + 111 + noise_dataset_index
        print(
            f"NTK {noise_dataset_number}/{cfg.num_noise_datasets}: "
            f"building noisy training set with seed {noise_seed}...",
            flush=True,
        )
        x_train_noisy = _make_noisy_array(
            x_train,
            noise_type=cfg.noise_type,
            noise_probability=float(cfg.noise_probability),
            seed=noise_seed,
            generation_mode=cfg.noise_generation_mode,
        )
        x_train_noisy_device = torch.as_tensor(x_train_noisy, device=device, dtype=dtype)

        print(f"NTK {noise_dataset_number}/{cfg.num_noise_datasets}: solving train kernel...", flush=True)
        state = _train_exact_ntk(
            x_train_noisy=x_train_noisy_device,
            y_train_onehot=y_train_onehot,
            cfg=cfg,
            moments=moments,
            device=device,
            dtype=dtype,
        )

        print(f"NTK {noise_dataset_number}/{cfg.num_noise_datasets}: evaluating clean test subset...", flush=True)
        eval_start = time.perf_counter()
        test_logits = _predict_exact_ntk(
            x_train_noisy=x_train_noisy_device,
            x_test=x_test,
            state=state,
            cfg=cfg,
            moments=moments,
            device=device,
            dtype=dtype,
        )
        eval_seconds = time.perf_counter() - eval_start

        train_accuracy = None
        if state.train_logits is not None:
            train_accuracy = _accuracy(state.train_logits.detach().cpu().numpy(), y_train)
        test_accuracy = _accuracy(test_logits, y_test)

        if test_outputs_path is not None:
            _append_test_outputs_csv(
                test_outputs_path,
                logits=test_logits,
                y_test=y_test,
                test_indices=test_indices,
                num_classes=int(cfg.num_classes),
                ntk_number=noise_dataset_number,
                ntk_index=noise_dataset_index,
                noise_seed=noise_seed,
                train_size=int(x_train.shape[0]),
                feature_dim=int(x_train.shape[1]),
            )

        run_payload = {
            "logits": test_logits,
            "y_test": np.asarray(y_test, dtype=np.int64),
            "test_indices": np.asarray(test_indices, dtype=np.int64),
            "noise_seed": np.asarray(noise_seed, dtype=np.int64),
            "train_indices": np.asarray(train_indices, dtype=np.int64),
        }
        if state.train_logits is not None:
            run_payload["train_logits"] = state.train_logits.detach().cpu().numpy().astype(np.float64, copy=False)
            run_payload["y_train"] = np.asarray(y_train, dtype=np.int64)

        if cfg.save_npz:
            np.savez_compressed(output_dir / f"ntk{noise_dataset_number}_outputs.npz", **run_payload)
            all_logits.append(test_logits)
            all_noise_seeds.append(noise_seed)

        summary = {
            "ntk_number": int(noise_dataset_number),
            "ntk_index": int(noise_dataset_index),
            "noise_seed": int(noise_seed),
            "train_seconds": float(state.train_seconds),
            "eval_seconds": float(eval_seconds),
            "solve_diag_reg": float(state.solve_diag_reg),
            "train_accuracy": train_accuracy,
            "test_accuracy": float(test_accuracy),
        }
        summaries.append(summary)
        print(
            f"NTK {noise_dataset_number}/{cfg.num_noise_datasets} done: "
            f"test_acc={100.0 * test_accuracy:.2f}% "
            f"train_seconds={state.train_seconds:.2f} eval_seconds={eval_seconds:.2f}",
            flush=True,
        )

        if noise_dataset_index % max(1, int(cfg.save_every)) == 0 or noise_dataset_number == int(cfg.num_noise_datasets):
            (output_dir / "ntk_summary.json").write_text(
                json.dumps({"ntks": summaries}, indent=2),
                encoding="utf-8",
            )

        del state, x_train_noisy_device
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if cfg.save_npz and all_logits:
        np.savez_compressed(
            output_dir / cfg.all_outputs_filename,
            logits=np.stack(all_logits, axis=0),
            y_test=np.asarray(y_test, dtype=np.int64),
            test_indices=np.asarray(test_indices, dtype=np.int64),
            noise_seeds=np.asarray(all_noise_seeds, dtype=np.int64),
            train_indices=np.asarray(train_indices, dtype=np.int64),
        )

    return output_dir


def build_config_from_user_inputs() -> NTKEnsembleConfig:
    return NTKEnsembleConfig(
        num_noise_datasets=NUM_NOISE_DATASETS,
        seed=SEED,
        num_classes=NUM_CLASSES,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        test_subset_mode=TEST_SUBSET_MODE,
        img_hw=IMG_HW,
        data_root=DATA_ROOT,
        output_root=OUTPUT_ROOT,
        noise_type=NOISE_TYPE,
        noise_probability=NOISE_PROBABILITY,
        noise_generation_mode=NOISE_GENERATION_MODE,
        activation=ACTIVATION,
        depth=DEPTH,
        weight_std=WEIGHT_STD,
        bias_std=BIAS_STD,
        diag_reg=DIAG_REG,
        moment_quadrature_n=MOMENT_QUADRATURE_N,
        moment_batch_size=MOMENT_BATCH_SIZE,
        allow_cpu=ALLOW_CPU,
        dtype=DTYPE,
        allow_tf32=ALLOW_TF32,
        test_batch_size=TEST_BATCH_SIZE,
        save_csv=SAVE_CSV,
        save_npz=SAVE_NPZ,
        save_train_outputs=SAVE_TRAIN_OUTPUTS,
        test_outputs_filename=TEST_OUTPUTS_FILENAME,
        all_outputs_filename=ALL_OUTPUTS_FILENAME,
        save_every=SAVE_EVERY,
    )


def main() -> None:
    cfg = build_config_from_user_inputs()
    project_dir = Path(__file__).resolve().parent
    output_dir = run_ntk_ensemble(cfg, project_dir)
    print(f"Saved NTK outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
