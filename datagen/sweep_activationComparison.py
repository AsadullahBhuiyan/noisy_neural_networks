from __future__ import annotations

import itertools
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from train_noisy_mlp_ensemble import EnsembleTrainConfig, train_ensemble


################################################################################
# USER INPUTS
#
# This script launches many calls to train_noisy_mlp_ensemble.train_ensemble.
# It trains one full ensemble for every Cartesian-product combination below.
#
# Important: this can get very large. For example,
#   4 train sizes x 3 image sizes x 3 activations x 3 noise probabilities
# with NUM_MODELS_PER_COMBO = 1000 gives 108,000 trained networks.
################################################################################

DRY_RUN = False
SKIP_COMPLETED = True
STOP_ON_ERROR = True

NUM_MODELS_PER_COMBO = 50
SEED = 22334
NUM_CLASSES = 10

TRAIN_SIZES = [16000]
IMG_HWS = [(28, 28)]
NOISE_TYPES = ["additive_gaussian"]
NOISE_PROBABILITIES = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]

ACTIVATIONS = ["tanh", "swish"]
DEPTHS = [3]
WIDTHS = [2048]
WEIGHT_STDS = [1.0]
BIAS_STDS = [0.0]

EPOCHS = 25
BATCH_SIZE = 2000
LEARNING_RATE = 1e-4
OPTIMIZER = "adam"  # adam, adamw, or sgd
WEIGHT_DECAY = 0.0
LOSS_NAMES = ["mse"]  # choose from xent or mse
NUM_WORKERS = 0
EVAL_BATCH_SIZE = 1024
SAVE_EVERY = 1

DATA_ROOT = "data"
OUTPUT_ROOT = "trained_mlp_ensembles_activationComparison"
ALLOW_CPU = False
MATERIALIZE_NOISY_TRAIN = False

################################################################################
# END USER INPUTS
################################################################################


MNIST_FULL_TRAIN_SIZE = 60_000


def _expected_n_train(train_size: int | None) -> int:
    if train_size is None:
        return MNIST_FULL_TRAIN_SIZE
    return min(int(train_size), MNIST_FULL_TRAIN_SIZE)


def _expected_output_dir(cfg: EnsembleTrainConfig, project_dir: Path) -> Path:
    from train_noisy_mlp_ensemble import _folder_name

    output_root = Path(cfg.output_root)
    if not output_root.is_absolute():
        output_root = project_dir / output_root
    n_train = _expected_n_train(cfg.train_size)
    feature_dim = int(cfg.img_hw[0]) * int(cfg.img_hw[1])
    return output_root / _folder_name(cfg, n_train=n_train, feature_dim=feature_dim)


def _completed_model_count(output_dir: Path) -> int:
    summary_path = output_dir / "training_summary.json"
    if not summary_path.exists():
        return 0
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        return len(summary.get("models", []))
    except Exception:
        return 0


def _iter_configs():
    for (
        train_size,
        img_hw,
        noise_type,
        noise_probability,
        activation,
        depth,
        width,
        weight_std,
        bias_std,
        loss_name,
    ) in itertools.product(
        TRAIN_SIZES,
        IMG_HWS,
        NOISE_TYPES,
        NOISE_PROBABILITIES,
        ACTIVATIONS,
        DEPTHS,
        WIDTHS,
        WEIGHT_STDS,
        BIAS_STDS,
        LOSS_NAMES,
    ):
        yield EnsembleTrainConfig(
            num_models=NUM_MODELS_PER_COMBO,
            seed=SEED,
            num_classes=NUM_CLASSES,
            train_size=train_size,
            img_hw=img_hw,
            data_root=DATA_ROOT,
            output_root=OUTPUT_ROOT,
            noise_type=noise_type,
            noise_probability=noise_probability,
            activation=activation,
            depth=depth,
            width=width,
            weight_std=weight_std,
            bias_std=bias_std,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            optimizer=OPTIMIZER,
            weight_decay=WEIGHT_DECAY,
            loss_name=loss_name,
            num_workers=NUM_WORKERS,
            eval_batch_size=EVAL_BATCH_SIZE,
            save_every=SAVE_EVERY,
            allow_cpu=ALLOW_CPU,
            materialize_noisy_train=MATERIALIZE_NOISY_TRAIN,
        )


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    configs = list(_iter_configs())
    total = len(configs)

    sweep_start = time.perf_counter()
    print(f"Prepared {total} ensemble-training jobs.", flush=True)
    print(f"DRY_RUN = {DRY_RUN}", flush=True)

    sweep_records = []
    for job_index, cfg in enumerate(configs, start=1):
        output_dir = _expected_output_dir(cfg, project_dir)
        completed = _completed_model_count(output_dir)
        record = {
            "job_index": job_index,
            "num_jobs": total,
            "config": asdict(cfg),
            "output_dir": str(output_dir),
            "completed_model_count_before": completed,
            "status": "pending",
        }

        label = (
            f"[{job_index}/{total}] "
            f"N={cfg.train_size}, d={cfg.img_hw[0]}x{cfg.img_hw[1]}, "
            f"noise={cfg.noise_type}, p={cfg.noise_probability:g}, "
            f"loss={cfg.loss_name}, act={cfg.activation}, L={cfg.depth}, width={cfg.width}"
        )
        print(label, flush=True)
        print(f"  output: {output_dir}", flush=True)

        if SKIP_COMPLETED and completed >= int(cfg.num_models):
            record["status"] = "skipped_completed"
            sweep_records.append(record)
            print(f"  skipped: already has {completed}/{cfg.num_models} models", flush=True)
            continue

        if completed > 0 and completed < int(cfg.num_models):
            message = (
                f"Partial run exists with {completed}/{cfg.num_models} completed models. "
                "The single-ensemble trainer currently starts from model1, so this script "
                "will not overwrite a partial run automatically."
            )
            record["status"] = "partial_existing"
            record["error"] = message
            sweep_records.append(record)
            print(f"  blocked: {message}", flush=True)
            if STOP_ON_ERROR:
                break
            continue

        if DRY_RUN:
            record["status"] = "dry_run"
            sweep_records.append(record)
            continue

        try:
            job_start = time.perf_counter()
            actual_output_dir = train_ensemble(cfg, project_dir)
            record["status"] = "completed"
            record["actual_output_dir"] = str(actual_output_dir)
            record["elapsed_seconds"] = time.perf_counter() - job_start
            sweep_records.append(record)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
            record["traceback"] = traceback.format_exc()
            sweep_records.append(record)
            print(record["traceback"], flush=True)
            if STOP_ON_ERROR:
                break

        sweep_log_path = project_dir / OUTPUT_ROOT / "sweep_log.json"
        sweep_log_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_log_path.write_text(
            json.dumps(
                {
                    "dry_run": DRY_RUN,
                    "elapsed_seconds": time.perf_counter() - sweep_start,
                    "records": sweep_records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    sweep_log_path = project_dir / OUTPUT_ROOT / "sweep_log.json"
    sweep_log_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_log_path.write_text(
        json.dumps(
            {
                "dry_run": DRY_RUN,
                "elapsed_seconds": time.perf_counter() - sweep_start,
                "records": sweep_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved sweep log to {sweep_log_path}", flush=True)


if __name__ == "__main__":
    main()
