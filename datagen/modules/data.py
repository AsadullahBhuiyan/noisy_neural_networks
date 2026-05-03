from __future__ import annotations

import gzip
import math
import random as pyrandom
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .config import ExperimentConfig


def set_seed(seed: int) -> None:
    pyrandom.seed(seed)
    np.random.seed(seed)


class MNISTArrayDataset:
    def __init__(self, images: np.ndarray, targets: np.ndarray, img_hw: Tuple[int, int]) -> None:
        self.images = np.asarray(images, dtype=np.uint8)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.img_hw = (int(img_hw[0]), int(img_hw[1]))

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        image = self.images[int(index)]
        height, width = self.img_hw

        if image.shape != (height, width):
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((width, height), resample=Image.Resampling.BILINEAR)
            image_array = np.asarray(pil_image, dtype=np.float32)
        else:
            image_array = image.astype(np.float32)

        image_array = image_array / 255.0
        return image_array[None, :, :], int(self.targets[int(index)])


class DatasetSubset:
    def __init__(self, dataset, indices: np.ndarray) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    @property
    def targets(self) -> np.ndarray:
        base_targets = _targets_from_dataset(self.dataset)
        return base_targets[self.indices]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int):
        return self.dataset[int(self.indices[int(index)])]


def _resolve_mnist_file(raw_dir: Path, name: str) -> Path:
    raw_path = raw_dir / name
    if raw_path.exists():
        return raw_path

    gz_path = raw_dir / f"{name}.gz"
    if gz_path.exists():
        return gz_path

    raise FileNotFoundError(f"Could not find {raw_path} or {gz_path}.")


def _read_binary_file(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()

    with open(path, "rb") as handle:
        return handle.read()


def _read_idx_images(path: Path) -> np.ndarray:
    data = _read_binary_file(path)
    magic = int.from_bytes(data[0:4], byteorder="big")
    if magic != 2051:
        raise ValueError(f"{path} is not a valid IDX image file.")

    num_images = int.from_bytes(data[4:8], byteorder="big")
    rows = int.from_bytes(data[8:12], byteorder="big")
    cols = int.from_bytes(data[12:16], byteorder="big")
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    return images.reshape(num_images, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    data = _read_binary_file(path)
    magic = int.from_bytes(data[0:4], byteorder="big")
    if magic != 2049:
        raise ValueError(f"{path} is not a valid IDX label file.")

    num_labels = int.from_bytes(data[4:8], byteorder="big")
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels.reshape(num_labels).astype(np.int64)


def load_mnist(root: Path, img_hw: Tuple[int, int]) -> tuple[MNISTArrayDataset, MNISTArrayDataset]:
    raw_dir = Path(root) / "MNIST" / "raw"

    train_images = _read_idx_images(_resolve_mnist_file(raw_dir, "train-images-idx3-ubyte"))
    train_labels = _read_idx_labels(_resolve_mnist_file(raw_dir, "train-labels-idx1-ubyte"))
    test_images = _read_idx_images(_resolve_mnist_file(raw_dir, "t10k-images-idx3-ubyte"))
    test_labels = _read_idx_labels(_resolve_mnist_file(raw_dir, "t10k-labels-idx1-ubyte"))

    return (
        MNISTArrayDataset(train_images, train_labels, img_hw=img_hw),
        MNISTArrayDataset(test_images, test_labels, img_hw=img_hw),
    )


def _targets_from_dataset(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_targets = _targets_from_dataset(dataset.dataset)
        subset_indices = np.asarray(dataset.indices, dtype=np.int64)
        return base_targets[subset_indices]

    return np.asarray([int(dataset[index][1]) for index in range(len(dataset))], dtype=np.int64)


def make_balanced_multiclass_dataset(
    dataset,
    fraction: float,
    seed: int,
    num_classes: int = 10,
) -> tuple[DatasetSubset, Dict[str, object]]:
    targets = _targets_from_dataset(dataset)
    clipped_fraction = float(np.clip(float(fraction), 0.0, 1.0))
    rng = np.random.default_rng(int(seed))

    class_indices = []
    class_counts = []
    for class_id in range(num_classes):
        current_indices = np.where(targets == class_id)[0]
        class_indices.append(current_indices)
        class_counts.append(int(current_indices.size))

    min_class_count = int(min(class_counts))
    keep_per_class = max(1, int(math.floor(clipped_fraction * min_class_count)))

    chosen_indices = []
    for class_id in range(num_classes):
        permuted = rng.permutation(class_indices[class_id])[:min_class_count]
        chosen_indices.append(permuted[:keep_per_class])

    all_indices = np.concatenate(chosen_indices, axis=0)
    all_indices = rng.permutation(all_indices)

    balanced_subset = DatasetSubset(dataset, all_indices)
    info = {
        "num_classes": int(num_classes),
        "orig_counts": {str(class_id): int(class_counts[class_id]) for class_id in range(num_classes)},
        "balanced_per_class": int(keep_per_class),
        "balanced_total": int(num_classes * keep_per_class),
        "fraction_used": float(clipped_fraction),
        "n_per_base_min_over_classes": int(min_class_count),
    }
    return balanced_subset, info


def dataset_to_numpy_matrix(dataset, num_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_x, _ = dataset[0]
    sample_x = np.asarray(sample_x, dtype=np.float32)
    _, height, width = sample_x.shape

    num_examples = len(dataset)
    flat_dim = int(height) * int(width)

    inputs = np.empty((num_examples, flat_dim), dtype=np.float32)
    labels = np.empty((num_examples,), dtype=np.int64)
    labels_onehot = np.zeros((num_examples, num_classes), dtype=np.float32)

    for index in range(num_examples):
        image, label = dataset[index]
        flat_image = np.asarray(image, dtype=np.float32).reshape(-1)
        class_id = int(label)
        inputs[index] = flat_image
        labels[index] = class_id
        labels_onehot[index, class_id] = 1.0

    inputs = normalize_feature_matrix_unit_rms(inputs)
    return inputs, labels_onehot, labels


def normalize_feature_matrix_unit_rms(inputs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each feature vector x so that
        (1 / d) * sum_i x_i^2 = 1.

    Zero vectors are left unchanged.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    rms_sq = np.mean(inputs * inputs, axis=1, keepdims=True)
    scale = np.sqrt(np.maximum(rms_sq, np.float32(eps)))
    normalized = inputs / scale
    zero_mask = rms_sq <= np.float32(eps)
    if np.any(zero_mask):
        normalized[zero_mask[:, 0]] = inputs[zero_mask[:, 0]]
    return normalized


def corrupt_replacement_noise(inputs: np.ndarray, p: float, seed: int) -> np.ndarray:
    if p <= 0.0:
        return inputs.copy()

    rng = np.random.default_rng(int(seed))
    mask = rng.random(size=inputs.shape) < float(p)
    replacements = rng.random(size=inputs.shape).astype(np.float32)

    noisy_inputs = inputs.copy()
    noisy_inputs[mask] = replacements[mask]
    return noisy_inputs


def corrupt_additive_gaussian_noise(inputs: np.ndarray, p: float, seed: int) -> np.ndarray:
    if p <= 0.0:
        return inputs.copy()

    epsilon = 1.0 - float(p)
    rng = np.random.default_rng(int(seed))
    gaussian_noise = rng.normal(loc=0.0, scale=1.0, size=inputs.shape).astype(np.float32)
    return (epsilon * inputs) + ((1.0 - epsilon) * gaussian_noise)


def apply_input_noise(
    inputs: np.ndarray,
    noise_probability: float,
    noise_type: str,
    seed: int,
) -> np.ndarray:
    normalized_noise_type = str(noise_type).strip().lower()

    if normalized_noise_type == "replacement":
        return corrupt_replacement_noise(inputs, p=noise_probability, seed=seed)

    if normalized_noise_type in {"additive_gaussian", "gaussian", "gaussian_additive"}:
        return corrupt_additive_gaussian_noise(inputs, p=noise_probability, seed=seed)

    raise ValueError(
        "cfg.noise_type must be 'replacement' or 'additive_gaussian'. "
        f"Received {noise_type!r}."
    )


@dataclass
class BasePreparedDataset:
    x_train_clean: np.ndarray
    y_train_onehot: np.ndarray
    y_train: np.ndarray
    x_test_clean: np.ndarray
    y_test_onehot: np.ndarray
    y_test: np.ndarray
    train_info: Dict[str, object]
    test_info: Dict[str, object]


@dataclass
class PreparedDataset(BasePreparedDataset):
    x_train_noisy: np.ndarray


def prepare_clean_train_and_clean_test(cfg: ExperimentConfig, project_dir: Path) -> BasePreparedDataset:
    set_seed(cfg.seed)

    train_raw, test_raw = load_mnist(root=cfg.resolve_data_root(project_dir), img_hw=cfg.img_hw)
    train_clean, train_info = make_balanced_multiclass_dataset(
        train_raw,
        fraction=cfg.train_fraction,
        seed=cfg.seed + 1,
        num_classes=cfg.num_classes,
    )
    test_clean, test_info = make_balanced_multiclass_dataset(
        test_raw,
        fraction=cfg.test_fraction,
        seed=cfg.seed + 2,
        num_classes=cfg.num_classes,
    )

    x_train_clean, y_train_onehot, y_train = dataset_to_numpy_matrix(train_clean, cfg.num_classes)
    x_test_clean, y_test_onehot, y_test = dataset_to_numpy_matrix(test_clean, cfg.num_classes)

    return BasePreparedDataset(
        x_train_clean=x_train_clean,
        y_train_onehot=y_train_onehot,
        y_train=y_train,
        x_test_clean=x_test_clean,
        y_test_onehot=y_test_onehot,
        y_test=y_test,
        train_info=train_info,
        test_info=test_info,
    )


def add_noise_to_training_data(
    base_data: BasePreparedDataset,
    cfg: ExperimentConfig,
    noise_seed: int,
) -> PreparedDataset:
    x_train_noisy = apply_input_noise(
        base_data.x_train_clean,
        noise_probability=cfg.noise_probability,
        noise_type=cfg.noise_type,
        seed=noise_seed,
    )

    return PreparedDataset(
        x_train_clean=base_data.x_train_clean,
        x_train_noisy=x_train_noisy,
        y_train_onehot=base_data.y_train_onehot,
        y_train=base_data.y_train,
        x_test_clean=base_data.x_test_clean,
        y_test_onehot=base_data.y_test_onehot,
        y_test=base_data.y_test,
        train_info=base_data.train_info,
        test_info=base_data.test_info,
    )


def prepare_noisy_train_and_clean_test(
    cfg: ExperimentConfig,
    project_dir: Path,
    noise_seed: int | None = None,
) -> PreparedDataset:
    base_data = prepare_clean_train_and_clean_test(cfg, project_dir=project_dir)
    resolved_noise_seed = cfg.seed + 111 if noise_seed is None else int(noise_seed)
    return add_noise_to_training_data(base_data, cfg, noise_seed=resolved_noise_seed)
