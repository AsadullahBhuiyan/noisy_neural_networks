from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


################################################################################
# USER INPUTS
#
# Edit only this block for normal use.
################################################################################

# Point this at one output directory created by train_noisy_ntk_ensemble.py.
# If RESULTS_DIR is None, the newest directory inside OUTPUT_ROOT is used.
RESULTS_DIR = "/data2/jt577/2026/04.2026/NN_noise/trained_ntk_ensembles/ntk_ensemble__additive_gaussian__p=0.95__N=400__M=10__d=28x28__act=erf__L=3__wstd=1__bstd=0__reg=1e-08__noiseK=100"
OUTPUT_ROOT = "trained_ntk_ensembles"

# Which clean test point and output label/logit to plot.
# TEST_POINT_INDEX is the row inside the saved test subset, not necessarily the
# original MNIST test-set index. Set TEST_POINT_IS_MNIST_INDEX=True to select by
# original MNIST test index instead.
TEST_POINT_INDEX = 0
TEST_POINT_IS_MNIST_INDEX = False
LABEL = 7

# Histogram / output.
BINS = "auto"  # "auto" or an integer like 30
DENSITY = False
TITLE = None
OUTPUT_FILENAME = None  # If None, a descriptive filename is used.
OUTPUT_DPI = 200

################################################################################
# END USER INPUTS
################################################################################


def _project_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_results_dir() -> Path:
    if RESULTS_DIR is not None:
        path = Path(RESULTS_DIR)
        return path if path.is_absolute() else _project_dir() / path

    root = Path(OUTPUT_ROOT)
    root = root if root.is_absolute() else _project_dir() / root
    if not root.exists():
        raise FileNotFoundError(f"OUTPUT_ROOT does not exist: {root}")

    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found inside {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _natural_run_key(path: Path) -> tuple[int, str]:
    match = re.search(r"ntk(\d+)_outputs\.npz$", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def _load_outputs(results_dir: Path, metadata: dict) -> dict[str, np.ndarray]:
    combined_name = "all_test_outputs.npz"
    if metadata.get("config", {}).get("all_outputs_filename"):
        combined_name = str(metadata["config"]["all_outputs_filename"])

    combined_path = results_dir / combined_name
    if combined_path.exists():
        with np.load(combined_path) as data:
            return {key: data[key].copy() for key in data.files}

    per_run_paths = sorted(results_dir.glob("ntk*_outputs.npz"), key=_natural_run_key)
    if not per_run_paths:
        raise FileNotFoundError(
            f"Could not find all_test_outputs.npz or ntk*_outputs.npz inside {results_dir}"
        )

    logits = []
    noise_seeds = []
    y_test = None
    test_indices = None
    train_indices = None
    for path in per_run_paths:
        with np.load(path) as data:
            logits.append(np.asarray(data["logits"], dtype=np.float64))
            noise_seeds.append(int(np.asarray(data["noise_seed"]).reshape(())))
            if y_test is None:
                y_test = np.asarray(data["y_test"], dtype=np.int64)
            if test_indices is None:
                test_indices = np.asarray(data["test_indices"], dtype=np.int64)
            if train_indices is None and "train_indices" in data.files:
                train_indices = np.asarray(data["train_indices"], dtype=np.int64)

    payload = {
        "logits": np.stack(logits, axis=0),
        "noise_seeds": np.asarray(noise_seeds, dtype=np.int64),
        "y_test": y_test,
        "test_indices": test_indices,
    }
    if train_indices is not None:
        payload["train_indices"] = train_indices
    return payload


def _load_metadata(results_dir: Path) -> dict:
    path = results_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _select_test_row(test_indices: np.ndarray, point_index: int, *, is_mnist_index: bool) -> int:
    if not is_mnist_index:
        row = int(point_index)
        if row < 0 or row >= int(test_indices.shape[0]):
            raise IndexError(f"TEST_POINT_INDEX={row} is outside the saved test subset.")
        return row

    matches = np.where(np.asarray(test_indices, dtype=np.int64) == int(point_index))[0]
    if matches.size == 0:
        raise ValueError(f"MNIST test index {point_index} is not present in this run's saved test subset.")
    return int(matches[0])


def _make_output_path(results_dir: Path, *, test_row: int, mnist_index: int, label: int) -> Path:
    if OUTPUT_FILENAME is not None:
        path = Path(OUTPUT_FILENAME)
        return path if path.is_absolute() else results_dir / path
    return results_dir / f"ntk_hist__testrow={test_row}__mnist={mnist_index}__label={label}.png"


def plot_histogram() -> Path:
    results_dir = _resolve_results_dir()
    metadata = _load_metadata(results_dir)
    payload = _load_outputs(results_dir, metadata)

    logits = np.asarray(payload["logits"], dtype=np.float64)
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape (ensemble, test, class), got {logits.shape}.")

    test_indices = np.asarray(payload["test_indices"], dtype=np.int64)
    y_test = np.asarray(payload["y_test"], dtype=np.int64)
    test_row = _select_test_row(
        test_indices,
        int(TEST_POINT_INDEX),
        is_mnist_index=bool(TEST_POINT_IS_MNIST_INDEX),
    )

    label = int(LABEL)
    if label < 0 or label >= int(logits.shape[2]):
        raise IndexError(f"LABEL={label} is outside the saved class dimension {logits.shape[2]}.")

    values = logits[:, test_row, label]
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.shape[0] > 1 else 0.0
    true_label = int(y_test[test_row])
    mnist_index = int(test_indices[test_row])

    output_path = _make_output_path(results_dir, test_row=test_row, mnist_index=mnist_index, label=label)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bins = BINS
    if isinstance(bins, str) and bins.strip().lower() != "auto":
        bins = int(bins)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.hist(
        values,
        bins=bins,
        density=bool(DENSITY),
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )
    ax.axvline(mean, color="#c44e52", linewidth=2.0, label=f"mean = {mean:.6g}")
    ax.axvline(mean - std, color="#55a868", linewidth=1.6, linestyle="--", label=f"std = {std:.6g}")
    ax.axvline(mean + std, color="#55a868", linewidth=1.6, linestyle="--")

    if TITLE is None:
        run_name = results_dir.name
        title = f"NTK ensemble logits for test row {test_row}, label {label}"
        if len(run_name) <= 80:
            title = f"{title}\n{run_name}"
    else:
        title = str(TITLE)

    ylabel = "Density" if bool(DENSITY) else "Count"
    ax.set_title(title)
    ax.set_xlabel(f"Logit for label {label}")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)

    detail = (
        f"ensemble size: {values.shape[0]}\n"
        f"test row: {test_row}\n"
        f"MNIST test index: {mnist_index}\n"
        f"true label: {true_label}\n"
        f"mean: {mean:.8g}\n"
        f"std: {std:.8g}"
    )
    if metadata.get("config"):
        config = metadata["config"]
        detail += (
            f"\nnoise: {config.get('noise_type')} p={config.get('noise_probability')}"
            f"\ntrain size: {metadata.get('num_train')}"
        )
    ax.text(
        0.98,
        0.96,
        detail,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )

    fig.savefig(output_path, dpi=int(OUTPUT_DPI))
    plt.close(fig)

    stats_path = output_path.with_suffix(".json")
    stats = {
        "results_dir": str(results_dir),
        "plot_file": str(output_path),
        "ensemble_size": int(values.shape[0]),
        "test_subset_row": int(test_row),
        "mnist_test_index": int(mnist_index),
        "true_label": int(true_label),
        "label": int(label),
        "mean": mean,
        "std": std,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Saved histogram to {output_path}", flush=True)
    print(f"Saved stats to {stats_path}", flush=True)
    print(f"mean={mean:.8g} std={std:.8g}", flush=True)
    return output_path


if __name__ == "__main__":
    plot_histogram()
