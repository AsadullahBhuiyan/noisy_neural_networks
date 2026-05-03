from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from compare_noisy import (
    _activation_bundle,
    _build_pure_noise_pipeline,
    _comparison_device_for_analytics,
    _resolve_root,
)
from compare_test_outputs_to_analytical import (
    _choose_device,
    _load_exact_data,
    _load_metadata,
    _select_test_indices,
)
from modules.metrics import acc_from_logits
from noise_models import (
    EpsilonCorrectionConfig,
    PredictionCorrection,
    precompute_first_layer_train_stats,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_OUTPUTS_PATH = PROJECT_DIR / (
    "trained_mlp_nested_ensembles_trainingSize/"
    "mlp_ensemble__additive_gaussian__p=0.95__N=16000__d=28x28__loss=mse__"
    "act=erf__L=3__width=2048__noiseK=100__initM=10__models=1000/"
    "test_outputs.csv.gz"
)
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "outputs/analytical_perturbative_test_accuracy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute MNIST test accuracy using only the first-order analytical "
            "perturbative output logits reconstructed from a saved ensemble metadata file."
        )
    )
    parser.add_argument(
        "--test-outputs-path",
        type=Path,
        default=DEFAULT_TEST_OUTPUTS_PATH,
        help=(
            "Path to test_outputs.csv.gz. The CSV is not read; its neighboring metadata.json "
            f"is used to recover the training subset/config. Default: {DEFAULT_TEST_OUTPUTS_PATH}"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="MNIST data root, relative to this script unless absolute. Default: data",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to save the summary JSON. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit this to evaluate all MNIST test examples.",
    )
    parser.add_argument(
        "--test-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit MNIST test indices to evaluate instead of the full test set.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every this many test examples. Use 0 to disable periodic progress.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution if no CUDA/MPS backend is available.",
    )
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="Also save analytical logits/predictions/labels to an .npz next to the summary JSON.",
    )
    return parser.parse_args()


def _resolved_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def compute_first_order_analytical_logits(
    *,
    metadata: dict[str, object],
    data,
    device: torch.device,
    progress_every: int,
) -> np.ndarray:
    saved_cfg = metadata["config"]
    noise_type = str(saved_cfg["noise_type"]).strip().lower()
    if noise_type not in {"additive_gaussian", "replacement"}:
        raise ValueError(
            "Analytical logits are currently supported only for "
            "noise_type='additive_gaussian' or 'replacement'."
        )

    act, act_p, act_pp, act_ppp, act_pppp = _activation_bundle(str(saved_cfg["activation"]))
    epsilon = 1.0 - float(saved_cfg["noise_probability"])
    num_tests = int(data.x_test_clean.shape[0])
    num_classes = int(saved_cfg["num_classes"])

    logits = np.empty((num_tests, num_classes), dtype=np.float64)

    y_train_tensor = torch.as_tensor(data.y_train_onehot, dtype=torch.float64, device=device)
    x_train_clean_tensor = torch.as_tensor(data.x_train_clean, dtype=torch.float64, device=device)
    train_stats = precompute_first_layer_train_stats(
        X_train=x_train_clean_tensor,
        y_train=y_train_tensor,
        num_classes=num_classes,
    )

    start = time.perf_counter()
    with torch.no_grad():
        for test_position in range(num_tests):
            x_test_tensor = torch.as_tensor(data.x_test_clean[test_position], dtype=torch.float64, device=device)
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
            logits[test_position] = (
                pure_mean_scalar
                + correction.output_mean_first_order_correction.detach().cpu().numpy()
            )

            run_number = test_position + 1
            if run_number == 1 or run_number == num_tests or (
                int(progress_every) > 0 and run_number % int(progress_every) == 0
            ):
                elapsed = time.perf_counter() - start
                avg = elapsed / run_number
                eta = avg * (num_tests - run_number)
                print(
                    f"Computed analytical logits for test point {run_number}/{num_tests} "
                    f"(elapsed {elapsed:.1f}s, estimated remaining {eta:.1f}s).",
                    flush=True,
                )

    return logits


def main() -> None:
    args = parse_args()

    test_outputs_path = _resolved_path(args.test_outputs_path, PROJECT_DIR)
    if not test_outputs_path.exists():
        raise FileNotFoundError(f"TEST_OUTPUTS_PATH does not exist: {test_outputs_path}")

    ensemble_dir, metadata = _load_metadata(test_outputs_path)
    saved_cfg = metadata["config"]
    num_full_test = int(metadata.get("num_test", 10_000))
    selected_test_indices = _select_test_indices(
        num_available=num_full_test,
        test_limit=args.test_limit,
        test_indices=args.test_indices,
    )
    data_root = _resolve_root(str(args.data_root), PROJECT_DIR)

    print(f"Using metadata from: {ensemble_dir / 'metadata.json'}", flush=True)
    print(
        f"Analytical model: p={saved_cfg['noise_probability']:g}, "
        f"N={metadata['num_train']}, d={saved_cfg['img_hw'][0]}x{saved_cfg['img_hw'][1]}, "
        f"act={saved_cfg['activation']}, depth={saved_cfg['depth']}, width={saved_cfg['width']}.",
        flush=True,
    )
    print(f"Evaluating {selected_test_indices.size} MNIST test examples.", flush=True)

    data = _load_exact_data(
        metadata=metadata,
        data_root=data_root,
        selected_test_indices=selected_test_indices,
    )

    runtime_device = _choose_device(bool(args.allow_cpu))
    analytical_device = _comparison_device_for_analytics(runtime_device)
    print(f"Using analytical device: {analytical_device}", flush=True)

    logits = compute_first_order_analytical_logits(
        metadata=metadata,
        data=data,
        device=analytical_device,
        progress_every=int(args.progress_every),
    )
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    correct = predictions == data.y_test
    accuracy = float(acc_from_logits(logits, data.y_test))

    output_path = _resolved_path(args.output_path, PROJECT_DIR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "description": (
            "MNIST test accuracy from classifying with only the first-order analytical "
            "perturbative output logits."
        ),
        "test_outputs_path": str(test_outputs_path),
        "metadata_path": str(ensemble_dir / "metadata.json"),
        "data_root": str(data_root),
        "saved_ensemble_config": saved_cfg,
        "num_train": int(data.x_train_clean.shape[0]),
        "num_test_evaluated": int(data.x_test_clean.shape[0]),
        "selected_test_indices": data.selected_test_indices.tolist(),
        "correct_count": int(np.sum(correct)),
        "accuracy": accuracy,
        "error_percent": 100.0 * (1.0 - accuracy),
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_logits:
        logits_path = output_path.with_suffix(".npz")
        np.savez_compressed(
            logits_path,
            selected_test_indices=data.selected_test_indices,
            y_test=data.y_test,
            analytical_logits=logits,
            analytical_predictions=predictions,
            correct=correct,
        )
        summary["logits_npz"] = str(logits_path)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Analytical perturbative test accuracy: {accuracy:.8f}", flush=True)
    print(f"Correct: {int(np.sum(correct))}/{int(correct.size)}", flush=True)
    print(f"Error percent: {100.0 * (1.0 - accuracy):.4f}", flush=True)
    print(f"Saved summary: {output_path}", flush=True)


if __name__ == "__main__":
    main()
