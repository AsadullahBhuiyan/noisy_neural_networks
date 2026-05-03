from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from modules.data import DatasetSubset, dataset_to_numpy_matrix, load_mnist, normalize_feature_matrix_unit_rms
from modules.metrics import acc_from_logits
from noise_models import (
    EpsilonCorrectionConfig,
    PredictionCorrection,
    PureNoiseConfig,
    PureNoisePipeline,
    compute_noise_combination_coeffs,
    noise_variance,
    precompute_first_layer_train_stats,
)
from train_noisy_mlp_ensemble import MLP


################################################################################
# USER INPUTS
#
# Edit this block for normal use. The script matches one saved MLP ensemble in
# trained_mlp_ensembles, evaluates its saved checkpoints on the clean MNIST test
# set, and compares the empirical mean / variance of the logits against the
# first-order perturbative analytical model.
################################################################################

LIST_SETTINGS = False

NOISE_PROBABILITY = 0.99
TRAIN_SIZE = 1000
IMG_HW = (128, 128)
ACTIVATION = "erf"

NOISE_TYPE = "additive_gaussian"
DEPTH = None
WIDTH = None
NUM_MODELS_IN_FOLDER = None

DATA_ROOT = "data"
ENSEMBLE_ROOT = "trained_mlp_ensembles"
OUTPUT_ROOT = "outputs/noisy_mlp_ensemble_comparisons"

BATCH_SIZE = 1024
MAX_MODELS = None
TEST_LIMIT = None
ALLOW_CPU = False
CENTER_CLASSWISE_LOGITS = True

################################################################################
# END USER INPUTS
################################################################################


@dataclass
class CompareConfig:
    noise_probability: float
    train_size: int
    img_hw: tuple[int, int]
    activation: str
    noise_type: str = "additive_gaussian"
    depth: int | None = None
    width: int | None = None
    num_models: int | None = None
    data_root: str = "data"
    ensemble_root: str = "trained_mlp_ensembles"
    output_root: str = "outputs/noisy_mlp_ensemble_comparisons"
    batch_size: int = 1024
    max_models: int | None = None
    test_limit: int | None = None
    allow_cpu: bool = False


@dataclass
class PreparedComparisonData:
    x_train_clean: np.ndarray
    y_train_onehot: np.ndarray
    y_train: np.ndarray
    x_test_clean: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    num_classes: int
    test_is_full_mnist: bool


def _parse_img_hw(value: str) -> tuple[int, int]:
    normalized = value.lower().strip().replace(" ", "")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid image size {value!r}. Expected format like '28x28'."
        )
    try:
        height = int(parts[0])
        width = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Invalid image size {value!r}. Expected integers like '28x28'."
        ) from exc
    if height < 1 or width < 1:
        raise ValueError("Image height and width must be positive.")
    return height, width


def build_config_from_user_inputs() -> CompareConfig:
    if BATCH_SIZE < 1:
        raise ValueError("--batch-size must be at least 1.")
    if MAX_MODELS is not None and MAX_MODELS < 1:
        raise ValueError("--max-models must be at least 1 when provided.")
    if TEST_LIMIT is not None and TEST_LIMIT < 1:
        raise ValueError("--test-limit must be at least 1 when provided.")

    return CompareConfig(
        noise_probability=float(NOISE_PROBABILITY),
        train_size=int(TRAIN_SIZE),
        img_hw=tuple(_parse_img_hw(IMG_HW) if isinstance(IMG_HW, str) else IMG_HW),
        activation=str(ACTIVATION),
        noise_type=str(NOISE_TYPE),
        depth=DEPTH,
        width=WIDTH,
        num_models=NUM_MODELS_IN_FOLDER,
        data_root=str(DATA_ROOT),
        ensemble_root=str(ENSEMBLE_ROOT),
        output_root=str(OUTPUT_ROOT),
        batch_size=int(BATCH_SIZE),
        max_models=MAX_MODELS,
        test_limit=TEST_LIMIT,
        allow_cpu=bool(ALLOW_CPU),
    )


def _resolve_root(root: str, project_dir: Path) -> Path:
    path = Path(root)
    return path if path.is_absolute() else project_dir / path


def _choose_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        "No GPU backend was found. Re-run with --allow-cpu if you want to compare on CPU."
    )


def _comparison_device_for_analytics(runtime_device: torch.device) -> torch.device:
    if runtime_device.type == "cuda":
        return runtime_device
    return torch.device("cpu")


def _center_classwise_logits(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr - np.mean(arr, axis=-1, keepdims=True)


def _maybe_center_classwise_logits(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if not CENTER_CLASSWISE_LOGITS:
        return arr
    return _center_classwise_logits(arr)


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _float_matches(a: float, b: float, *, atol: float = 1e-12, rtol: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def _normalized_noise_name(name: str) -> str:
    return str(name).strip().lower()


def _normalized_activation_name(name: str) -> str:
    return str(name).strip().lower()


def _iter_saved_ensembles(ensemble_root: Path):
    for metadata_path in sorted(ensemble_root.glob("mlp_ensemble__*/metadata.json")):
        ensemble_dir = metadata_path.parent
        summary_path = ensemble_dir / "training_summary.json"
        if not summary_path.exists():
            continue
        try:
            metadata = _load_json(metadata_path)
            summary = _load_json(summary_path)
        except Exception:
            continue
        yield {
            "ensemble_dir": ensemble_dir,
            "metadata": metadata,
            "summary": summary,
        }


def _available_setting_string(record: dict[str, object]) -> str:
    metadata = record["metadata"]
    cfg = metadata["config"]
    img_hw = tuple(cfg["img_hw"])
    summary = record["summary"]
    completed_models = len(summary.get("models", []))
    return (
        f"p={cfg['noise_probability']:g}, N={metadata['num_train']}, d={img_hw[0]}x{img_hw[1]}, "
        f"act={cfg['activation']}, noise={cfg['noise_type']}, depth={cfg['depth']}, "
        f"width={cfg['width']}, models={completed_models}, dir={record['ensemble_dir'].name}"
    )


def _record_matches(record: dict[str, object], cfg: CompareConfig) -> bool:
    metadata = record["metadata"]
    saved_cfg = metadata["config"]
    img_hw = tuple(int(v) for v in saved_cfg["img_hw"])

    if _normalized_noise_name(saved_cfg["noise_type"]) != _normalized_noise_name(cfg.noise_type):
        return False
    if not _float_matches(saved_cfg["noise_probability"], cfg.noise_probability):
        return False
    if int(metadata["num_train"]) != int(cfg.train_size):
        return False
    if img_hw != tuple(cfg.img_hw):
        return False
    if _normalized_activation_name(saved_cfg["activation"]) != _normalized_activation_name(cfg.activation):
        return False
    if cfg.depth is not None and int(saved_cfg["depth"]) != int(cfg.depth):
        return False
    if cfg.width is not None and int(saved_cfg["width"]) != int(cfg.width):
        return False
    if cfg.num_models is not None and int(saved_cfg["num_models"]) != int(cfg.num_models):
        return False
    return True


def _select_saved_ensemble(cfg: CompareConfig, ensemble_root: Path) -> dict[str, object]:
    matches = [record for record in _iter_saved_ensembles(ensemble_root) if _record_matches(record, cfg)]

    if not matches:
        available = [_available_setting_string(record) for record in _iter_saved_ensembles(ensemble_root)]
        message = [
            "No saved ensemble matched the requested setting.",
            f"Requested: p={cfg.noise_probability:g}, N={cfg.train_size}, "
            f"d={cfg.img_hw[0]}x{cfg.img_hw[1]}, act={cfg.activation}, noise={cfg.noise_type}",
        ]
        if available:
            message.append("Available settings:")
            message.extend(f"  - {item}" for item in available)
        raise FileNotFoundError("\n".join(message))

    if len(matches) > 1:
        choices = "\n".join(f"  - {_available_setting_string(record)}" for record in matches)
        raise RuntimeError(
            "The requested setting matched multiple saved ensembles. Add --depth, --width, "
            "or --num-models to disambiguate.\n"
            f"{choices}"
        )

    return matches[0]


def _print_available_settings(ensemble_root: Path) -> None:
    found = False
    for record in _iter_saved_ensembles(ensemble_root):
        found = True
        print(_available_setting_string(record), flush=True)
    if not found:
        print(f"No saved ensembles were found under {ensemble_root}.", flush=True)


def _load_exact_train_and_test_data(
    record: dict[str, object],
    *,
    data_root: Path,
    test_limit: int | None,
) -> PreparedComparisonData:
    metadata = record["metadata"]
    saved_cfg = metadata["config"]
    img_hw = tuple(int(v) for v in saved_cfg["img_hw"])
    num_classes = int(saved_cfg["num_classes"])

    train_raw, test_raw = load_mnist(root=data_root, img_hw=img_hw)
    train_indices = np.asarray(metadata["train_indices"], dtype=np.int64)
    train_dataset = DatasetSubset(train_raw, train_indices)

    x_train_clean, y_train_onehot, y_train = dataset_to_numpy_matrix(train_dataset, num_classes)
    x_test_clean, _y_test_onehot, y_test = dataset_to_numpy_matrix(test_raw, num_classes)

    x_train_clean = normalize_feature_matrix_unit_rms(x_train_clean)
    x_test_clean = normalize_feature_matrix_unit_rms(x_test_clean)

    test_is_full_mnist = True
    if test_limit is not None and test_limit < x_test_clean.shape[0]:
        x_test_clean = x_test_clean[:test_limit]
        y_test = y_test[:test_limit]
        test_is_full_mnist = False

    return PreparedComparisonData(
        x_train_clean=x_train_clean,
        y_train_onehot=y_train_onehot,
        y_train=y_train,
        x_test_clean=x_test_clean,
        y_test=y_test,
        train_indices=train_indices,
        num_classes=num_classes,
        test_is_full_mnist=test_is_full_mnist,
    )


def _activation_bundle(act_name: str):
    normalized = act_name.strip().lower()

    if normalized == "erf":
        def act(z):
            return torch.erf(z)

        def act_p(z):
            return (2.0 / math.sqrt(math.pi)) * torch.exp(-(z * z))

        def act_pp(z):
            return -(4.0 / math.sqrt(math.pi)) * z * torch.exp(-(z * z))

        def act_ppp(z):
            return (4.0 / math.sqrt(math.pi)) * ((2.0 * z * z) - 1.0) * torch.exp(-(z * z))

        def act_pppp(z):
            return (8.0 / math.sqrt(math.pi)) * z * (3.0 - 2.0 * z * z) * torch.exp(-(z * z))

        return act, act_p, act_pp, act_ppp, act_pppp

    if normalized == "gelu":
        def _phi(z):
            return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

        def act(z):
            return torch.nn.functional.gelu(z, approximate="none")

        def act_p(z):
            cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            return cdf + z * _phi(z)

        def act_pp(z):
            return (2.0 - z * z) * _phi(z)

        def act_ppp(z):
            return (z**3 - 4.0 * z) * _phi(z)

        def act_pppp(z):
            return (-z**4 + 7.0 * z * z - 4.0) * _phi(z)

        return act, act_p, act_pp, act_ppp, act_pppp

    if normalized == "tanh":
        def act(z):
            return torch.tanh(z)

        def act_p(z):
            t = torch.tanh(z)
            return 1.0 - t * t

        def act_pp(z):
            t = torch.tanh(z)
            return -2.0 * t * (1.0 - t * t)

        def act_ppp(z):
            t = torch.tanh(z)
            s = 1.0 - t * t
            return -2.0 * s * s + 4.0 * t * t * s

        def act_pppp(z):
            t = torch.tanh(z)
            s = 1.0 - t * t
            return (16.0 * t - 24.0 * t**3) * s

        return act, act_p, act_pp, act_ppp, act_pppp

    if normalized == "sin":
        def act(z):
            return torch.sin(z)

        def act_p(z):
            return torch.cos(z)

        def act_pp(z):
            return -torch.sin(z)

        def act_ppp(z):
            return -torch.cos(z)

        def act_pppp(z):
            return torch.sin(z)

        return act, act_p, act_pp, act_ppp, act_pppp

    if normalized == "softplus":
        def act(z):
            return torch.nn.functional.softplus(z)

        def act_p(z):
            return torch.sigmoid(z)

        def act_pp(z):
            s = torch.sigmoid(z)
            return s * (1.0 - s)

        def act_ppp(z):
            s = torch.sigmoid(z)
            return s * (1.0 - s) * (1.0 - 2.0 * s)

        def act_pppp(z):
            s = torch.sigmoid(z)
            return s * (1.0 - s) * (1.0 - 6.0 * s + 6.0 * s * s)

        return act, act_p, act_pp, act_ppp, act_pppp

    raise ValueError(
        f"Unsupported activation {act_name!r}. "
        "Expected one of: 'erf', 'gelu', 'tanh', 'sin', 'softplus'."
    )


def _build_pure_noise_pipeline(
    *,
    saved_cfg: dict[str, object],
    x_train_clean_tensor: torch.Tensor,
    x_test_tensor: torch.Tensor,
    act,
    act_p,
    act_pp,
    act_ppp,
    act_pppp,
    device: torch.device,
) -> tuple[PureNoiseConfig, PureNoisePipeline]:
    pure_cfg = PureNoiseConfig(
        noise_model=str(saved_cfg["noise_type"]),
        x_test=x_test_tensor,
        X_train=x_train_clean_tensor,
        Cb=float(saved_cfg["bias_std"]) ** 2,
        Cw=float(saved_cfg["weight_std"]) ** 2,
        act=act,
        n_layers=int(saved_cfg["depth"]),
        gh_1d_n=80,
        gh_2d_n=60,
        device=device,
        dtype=torch.float64,
    )
    pipe = PureNoisePipeline(
        cfg=pure_cfg,
        act_p=act_p,
        act_pp=act_pp,
        act_ppp=act_ppp,
        act_pppp=act_pppp,
        n_classes=int(saved_cfg["num_classes"]),
    )
    return pure_cfg, pipe


def _torch_load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def compute_saved_mlp_ensemble_statistics(
    cfg: CompareConfig,
    record: dict[str, object],
    base_data: PreparedComparisonData,
    device: torch.device,
) -> dict[str, object]:
    metadata = record["metadata"]
    saved_cfg = metadata["config"]
    summary = record["summary"]
    model_records = list(summary.get("models", []))
    if not model_records:
        raise RuntimeError(f"No completed models were recorded in {record['ensemble_dir']}.")

    if cfg.max_models is not None:
        model_records = model_records[: int(cfg.max_models)]
    num_models = len(model_records)

    x_test_tensor = torch.as_tensor(base_data.x_test_clean, dtype=torch.float32)
    y_test_tensor = torch.as_tensor(base_data.y_test, dtype=torch.long)
    test_loader = DataLoader(
        TensorDataset(x_test_tensor, y_test_tensor),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = MLP(
        input_dim=int(base_data.x_test_clean.shape[1]),
        num_classes=int(saved_cfg["num_classes"]),
        depth=int(saved_cfg["depth"]),
        width=int(saved_cfg["width"]),
        activation=str(saved_cfg["activation"]),
        weight_std=float(saved_cfg["weight_std"]),
        bias_std=float(saved_cfg["bias_std"]),
    ).to(device=device, dtype=torch.float32)
    model.eval()

    logits_sum = np.zeros((base_data.x_test_clean.shape[0], base_data.num_classes), dtype=np.float64)
    logits_sq_sum = np.zeros_like(logits_sum)
    centered_logits_sum = np.zeros_like(logits_sum)
    test_accuracy_runs = np.empty((num_models,), dtype=np.float64)
    test_error_percent_runs = np.empty((num_models,), dtype=np.float64)

    overall_start = time.perf_counter()
    with torch.inference_mode():
        for model_index, model_record in enumerate(model_records):
            model_number = int(model_record["model_number"])
            model_path = record["ensemble_dir"] / str(model_record["model_file"])
            if not model_path.exists():
                raise FileNotFoundError(f"Missing checkpoint referenced by summary: {model_path}")

            checkpoint = _torch_load_checkpoint(model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            logits_batches = []
            model_dtype = next(model.parameters()).dtype
            for x_batch, _y_batch in test_loader:
                x_batch = x_batch.to(device=device, dtype=model_dtype, non_blocking=True)
                logits = model(x_batch)
                logits_batches.append(logits.detach().cpu())

            logits = torch.cat(logits_batches, dim=0).numpy().astype(np.float64, copy=False)
            centered_logits = _maybe_center_classwise_logits(logits)
            logits_sum += logits
            logits_sq_sum += logits * logits
            centered_logits_sum += centered_logits

            acc = acc_from_logits(logits, base_data.y_test)
            test_accuracy_runs[model_index] = float(acc)
            test_error_percent_runs[model_index] = 100.0 * (1.0 - float(acc))

            run_number = model_index + 1
            if run_number == 1 or run_number == num_models or run_number % 25 == 0:
                elapsed = time.perf_counter() - overall_start
                avg_time = elapsed / run_number
                eta = avg_time * (num_models - run_number)
                print(
                    f"Evaluated saved model {run_number}/{num_models} "
                    f"(checkpoint model{model_number}.pth, elapsed {elapsed:.2f}s, "
                    f"estimated remaining {eta:.2f}s).",
                    flush=True,
                )

    mean_ensemble_logits = logits_sum / float(num_models)
    variance_ensemble_logits = logits_sq_sum / float(num_models) - mean_ensemble_logits * mean_ensemble_logits
    variance_ensemble_logits = np.maximum(variance_ensemble_logits, 0.0)
    centered_mean_ensemble_logits = centered_logits_sum / float(num_models)

    return {
        "mean_mlp_ensemble": mean_ensemble_logits,
        "mean_mlp_ensemble_centered": centered_mean_ensemble_logits,
        "variance_mlp_ensemble": variance_ensemble_logits,
        "test_accuracy_runs": test_accuracy_runs,
        "test_error_percent_runs": test_error_percent_runs,
        "test_accuracy_mean": float(np.mean(test_accuracy_runs)),
        "test_accuracy_std": float(np.std(test_accuracy_runs, ddof=0)),
        "test_error_percent_mean": float(np.mean(test_error_percent_runs)),
        "test_error_percent_std": float(np.std(test_error_percent_runs, ddof=0)),
        "ensemble_mean_accuracy": float(acc_from_logits(mean_ensemble_logits, base_data.y_test)),
        "evaluated_num_models": int(num_models),
    }


def compute_first_order_analytical_statistics(
    record: dict[str, object],
    base_data: PreparedComparisonData,
    device: torch.device,
) -> dict[str, object]:
    metadata = record["metadata"]
    saved_cfg = metadata["config"]
    noise_type = _normalized_noise_name(saved_cfg["noise_type"])
    if noise_type not in {"additive_gaussian", "replacement"}:
        raise ValueError(
            "compare_noisy.py supports analytical statistics only for "
            "noise_type='additive_gaussian' or 'replacement'."
        )

    act, act_p, act_pp, act_ppp, act_pppp = _activation_bundle(str(saved_cfg["activation"]))
    epsilon = 1.0 - float(saved_cfg["noise_probability"])
    num_tests = int(base_data.x_test_clean.shape[0])
    num_classes = int(saved_cfg["num_classes"])

    mean_first_order = np.empty((num_tests, num_classes), dtype=np.float64)
    variance_first_order = np.empty((num_tests, num_classes), dtype=np.float64)

    y_train_tensor = torch.as_tensor(base_data.y_train_onehot, dtype=torch.float64, device=device)
    x_train_clean_tensor = torch.as_tensor(base_data.x_train_clean, dtype=torch.float64, device=device)
    train_stats = precompute_first_layer_train_stats(
        X_train=x_train_clean_tensor,
        y_train=y_train_tensor,
        num_classes=num_classes,
    )

    for test_position in range(num_tests):
        x_test_tensor = torch.as_tensor(base_data.x_test_clean[test_position], dtype=torch.float64, device=device)
        pure_cfg, pipe = _build_pure_noise_pipeline(
            saved_cfg=saved_cfg,
            x_train_clean_tensor=x_train_clean_tensor,
            x_test_tensor=x_test_tensor,
            act=act,
            act_p=act_p,
            act_pp=act_pp,
            act_ppp=act_ppp,
            act_pppp=act_pppp,
            device=device,
        )
        eps_cfg = EpsilonCorrectionConfig(
            eps=epsilon,
            y_train=y_train_tensor,
            num_classes=num_classes,
            device=device,
            dtype=torch.float64,
        )
        correction = PredictionCorrection(
            pipe,
            pure_cfg,
            eps_cfg,
            train_stats=train_stats,
        )

        pure_mean_scalar = float(pipe.prediction.output_mean.detach().cpu().item())
        mean_first_order[test_position] = (
            pure_mean_scalar
            + correction.output_mean_first_order_correction.detach().cpu().numpy()
        )

        coeffs = compute_noise_combination_coeffs(pipe)
        variance_scalar = float(
            noise_variance(coeffs, pipe, pure_cfg, num_classes=num_classes).detach().cpu().item()
        )
        variance_first_order[test_position] = variance_scalar

        run_number = test_position + 1
        if run_number == 1 or run_number == num_tests or run_number % 250 == 0:
            print(
                f"Computed first-order analytical statistics for test point "
                f"{run_number}/{num_tests}.",
                flush=True,
            )

    first_order_accuracy = acc_from_logits(mean_first_order, base_data.y_test)
    centered_mean_first_order = _maybe_center_classwise_logits(mean_first_order)
    return {
        "epsilon": epsilon,
        "mean_first_order_analytical": mean_first_order,
        "mean_first_order_analytical_centered": centered_mean_first_order,
        "variance_first_order_analytical": variance_first_order,
        "first_order_accuracy": float(first_order_accuracy),
        "first_order_error_percent": 100.0 * (1.0 - float(first_order_accuracy)),
    }


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x_flat = np.asarray(x, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_flat.size < 2:
        return None
    if float(np.std(x_flat)) == 0.0 or float(np.std(y_flat)) == 0.0:
        return None
    return float(np.corrcoef(x_flat, y_flat)[0, 1])


def _comparison_summary(full: np.ndarray, analytical: np.ndarray) -> dict[str, float | None]:
    full_arr = np.asarray(full, dtype=np.float64)
    analytical_arr = np.asarray(analytical, dtype=np.float64)
    diff = full_arr - analytical_arr
    positive_mask = analytical_arr > 0.0
    mean_ratio = None
    if np.any(positive_mask):
        mean_ratio = float(np.mean(full_arr[positive_mask] / analytical_arr[positive_mask]))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "correlation": _correlation(full, analytical),
        "mean_full_over_analytical_ratio_for_positive_analytical": mean_ratio,
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
    file_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    summary: dict[str, float | None],
) -> str:
    _prepare_matplotlib_cache(output_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_flat = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y_values, dtype=np.float64).reshape(-1)
    lower = float(min(np.min(x_flat), np.min(y_flat)))
    upper = float(max(np.max(x_flat), np.max(y_flat)))

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(x_flat, y_flat, s=6, alpha=0.18, linewidths=0.0)
    ax.plot([lower, upper], [lower, upper], color="crimson", linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.97,
        (
            f"MAE={summary['mae']:.3e}\n"
            f"RMSE={summary['rmse']:.3e}\n"
            f"corr={summary['correlation']}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )

    plot_path = output_dir / file_name
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path.name


def save_database_json(
    *,
    project_dir: Path,
    compare_cfg: CompareConfig,
    record: dict[str, object],
    base_data: PreparedComparisonData,
    ensemble_stats: dict[str, object],
    analytical: dict[str, object],
) -> Path:
    metadata = record["metadata"]
    saved_cfg = metadata["config"]
    ensemble_dir = record["ensemble_dir"]

    output_root = _resolve_root(compare_cfg.output_root, project_dir)
    output_dir = output_root / ensemble_dir.name
    if compare_cfg.max_models is not None:
        output_dir = output_dir / f"subset__models={int(compare_cfg.max_models)}"
    if compare_cfg.test_limit is not None:
        output_dir = output_dir / f"subset__test={int(compare_cfg.test_limit)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_ensemble_raw = np.asarray(ensemble_stats["mean_mlp_ensemble"], dtype=np.float64)
    mean_ensemble = np.asarray(ensemble_stats["mean_mlp_ensemble_centered"], dtype=np.float64)
    variance_ensemble = np.asarray(ensemble_stats["variance_mlp_ensemble"], dtype=np.float64)
    mean_first_order_raw = np.asarray(analytical["mean_first_order_analytical"], dtype=np.float64)
    mean_first_order = np.asarray(analytical["mean_first_order_analytical_centered"], dtype=np.float64)
    variance_first_order = np.asarray(analytical["variance_first_order_analytical"], dtype=np.float64)

    mean_summary = _comparison_summary(mean_ensemble, mean_first_order)
    variance_summary = _comparison_summary(variance_ensemble, variance_first_order)

    mean_plot = _save_scatter_plot(
        output_dir=output_dir,
        file_name="mean_scatter.png",
        x_values=mean_ensemble,
        y_values=mean_first_order,
        title="Saved MLP Ensemble Mean vs First-Order Mean",
        x_label="Saved ensemble mean logits (class-centered)",
        y_label="First-order analytical mean logits (class-centered)",
        summary=mean_summary,
    )
    variance_plot = _save_scatter_plot(
        output_dir=output_dir,
        file_name="variance_scatter.png",
        x_values=variance_ensemble,
        y_values=variance_first_order,
        title="Saved MLP Ensemble Variance vs First-Order Variance",
        x_label="Saved ensemble variance",
        y_label="First-order analytical variance",
        summary=variance_summary,
    )

    database = {
        "schema_version": 1,
        "description": (
            "Per-test-point, per-class saved-noisy-MLP-ensemble comparison against "
            "the first-order perturbative analytical model."
        ),
        "requested_config": asdict(compare_cfg),
        "matched_ensemble_dir": str(ensemble_dir),
        "saved_ensemble_config": saved_cfg,
        "classwise_logit_centering": {
            "enabled_for_mean_comparison": bool(CENTER_CLASSWISE_LOGITS),
            "description": (
                "For each test point, subtract the mean across classes from the "
                "logit vector before comparing ensemble and analytical mean predictions."
            ),
            "variance_note": (
                "Variance arrays remain the existing marginal class-wise variances. "
                "Exact class-centered analytical variances would require the full "
                "class-class covariance, which is not available in the current formula."
            ),
        },
        "dataset": {
            "num_train": int(base_data.x_train_clean.shape[0]),
            "num_test": int(base_data.x_test_clean.shape[0]),
            "feature_dim": int(base_data.x_train_clean.shape[1]),
            "train_indices": base_data.train_indices.tolist(),
            "test_is_full_mnist": bool(base_data.test_is_full_mnist),
        },
        "axes": {
            "test_index": list(range(int(base_data.x_test_clean.shape[0]))),
            "class": list(range(int(base_data.num_classes))),
        },
        "true_labels": np.asarray(base_data.y_test, dtype=np.int64).tolist(),
        "mean_mlp_ensemble_raw": mean_ensemble_raw.tolist(),
        "mean_mlp_ensemble": mean_ensemble.tolist(),
        "mean_first_order_analytical_raw": mean_first_order_raw.tolist(),
        "mean_first_order_analytical": mean_first_order.tolist(),
        "variance_mlp_ensemble": variance_ensemble.tolist(),
        "variance_first_order_analytical": variance_first_order.tolist(),
        "summary": {
            "saved_model_test_accuracy_mean": float(ensemble_stats["test_accuracy_mean"]),
            "saved_model_test_accuracy_std": float(ensemble_stats["test_accuracy_std"]),
            "saved_model_test_error_percent_mean": float(ensemble_stats["test_error_percent_mean"]),
            "saved_model_test_error_percent_std": float(ensemble_stats["test_error_percent_std"]),
            "saved_ensemble_mean_accuracy": float(ensemble_stats["ensemble_mean_accuracy"]),
            "first_order_test_accuracy": float(analytical["first_order_accuracy"]),
            "first_order_test_error_percent": float(analytical["first_order_error_percent"]),
            "evaluated_num_models": int(ensemble_stats["evaluated_num_models"]),
            "epsilon": float(analytical["epsilon"]),
            "mean_comparison": mean_summary,
            "variance_comparison": variance_summary,
        },
        "artifacts": {
            "mean_scatter_plot": mean_plot,
            "variance_scatter_plot": variance_plot,
        },
    }

    database_path = output_dir / "comparison_database.json"
    database_path.write_text(json.dumps(database, indent=2), encoding="utf-8")
    return database_path


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    ensemble_root = _resolve_root(str(ENSEMBLE_ROOT), project_dir)

    if LIST_SETTINGS:
        _print_available_settings(ensemble_root)
        return

    compare_cfg = build_config_from_user_inputs()
    record = _select_saved_ensemble(compare_cfg, ensemble_root)
    metadata = record["metadata"]
    saved_cfg = metadata["config"]

    print(f"Matched saved ensemble: {record['ensemble_dir']}", flush=True)
    print("Loading exact clean train subset and clean MNIST test set...", flush=True)
    data_root = _resolve_root(compare_cfg.data_root, project_dir)
    base_data = _load_exact_train_and_test_data(
        record,
        data_root=data_root,
        test_limit=compare_cfg.test_limit,
    )

    runtime_device = _choose_device(compare_cfg.allow_cpu)
    analytical_device = _comparison_device_for_analytics(runtime_device)
    print(f"Using runtime device: {runtime_device}", flush=True)
    print(f"Using analytical device: {analytical_device}", flush=True)
    print(
        f"Comparing p={saved_cfg['noise_probability']:g}, N={metadata['num_train']}, "
        f"d={saved_cfg['img_hw'][0]}x{saved_cfg['img_hw'][1]}, act={saved_cfg['activation']}.",
        flush=True,
    )

    print("Evaluating saved MLP ensemble checkpoints...", flush=True)
    ensemble_stats = compute_saved_mlp_ensemble_statistics(
        compare_cfg,
        record,
        base_data,
        runtime_device,
    )

    print("Computing first-order analytical mean and variance...", flush=True)
    analytical = compute_first_order_analytical_statistics(
        record,
        base_data,
        analytical_device,
    )

    database_path = save_database_json(
        project_dir=project_dir,
        compare_cfg=compare_cfg,
        record=record,
        base_data=base_data,
        ensemble_stats=ensemble_stats,
        analytical=analytical,
    )

    mean_summary = _comparison_summary(
        np.asarray(ensemble_stats["mean_mlp_ensemble_centered"], dtype=np.float64),
        np.asarray(analytical["mean_first_order_analytical_centered"], dtype=np.float64),
    )
    variance_summary = _comparison_summary(
        np.asarray(ensemble_stats["variance_mlp_ensemble"], dtype=np.float64),
        np.asarray(analytical["variance_first_order_analytical"], dtype=np.float64),
    )

    print(f"Saved comparison database JSON: {database_path}", flush=True)
    print(
        "Mean comparison: "
        f"MAE={mean_summary['mae']:.6e}, RMSE={mean_summary['rmse']:.6e}, "
        f"corr={mean_summary['correlation']}",
        flush=True,
    )
    print(
        "Variance comparison: "
        f"MAE={variance_summary['mae']:.6e}, RMSE={variance_summary['rmse']:.6e}, "
        f"corr={variance_summary['correlation']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
