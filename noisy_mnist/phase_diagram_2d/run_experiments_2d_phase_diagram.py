#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

def configure_runtime() -> Path:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    root_str = str(root)
    if root_str not in os.sys.path:
        os.sys.path.insert(0, root_str)
    return root


PROJECT_ROOT = configure_runtime()

cpu_max = 20
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


DEFAULT_P_VALUES = np.linspace(0.90, 0.995, 30).tolist()
BALANCED_TRAIN_PER_CLASS = 5412
BALANCED_TEST_PER_CLASS = 892
BALANCED_TRAIN_TOTAL = 10 * BALANCED_TRAIN_PER_CLASS
BALANCED_TEST_TOTAL = 10 * BALANCED_TEST_PER_CLASS
DEFAULT_BATCH_SIZES = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    BALANCED_TRAIN_TOTAL,
]
DEFAULT_TRAIN_SAMPLES_PER_CLASS_VALUES = [500, 1000, 2000, 4000, BALANCED_TRAIN_PER_CLASS]
DEFAULT_SEEDS = list(range(5))
DEFAULT_WIDTH = 512
DEFAULT_REFERENCE_EPOCHS = 20
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_CHANCE_ACCURACY = 0.1
PHASE_THRESHOLDS = (0.10, 0.20)
SCALING_BATCHES = [64, 256, 1024]
COMPARISON_BATCHES = [128, 256, 1024, 4096, 16384, BALANCED_TRAIN_TOTAL]
BALANCED_DATASET_TAG = "mnist_balanced_perclass_noreplace_v3"

GLOBAL_STATE: dict[str, Any] = {}


# User config
RUN_P_VALUES = list(DEFAULT_P_VALUES)
RUN_BATCH_SIZES = list(DEFAULT_BATCH_SIZES)
RUN_TRAIN_SAMPLES_PER_CLASS_VALUES = list(DEFAULT_TRAIN_SAMPLES_PER_CLASS_VALUES)
RUN_SEEDS = list(DEFAULT_SEEDS)
RUN_WIDTH = DEFAULT_WIDTH
RUN_REFERENCE_EPOCHS = DEFAULT_REFERENCE_EPOCHS
RUN_LEARNING_RATE = DEFAULT_LEARNING_RATE
RUN_TRAIN_LOG_INTERVAL = 100
RUN_TEST_LOG_INTERVAL = 500
RUN_PROGRESS_STEP_INTERVAL = 1000
RUN_MAX_WORKERS = cpu_max
RUN_MAX_IN_FLIGHT = max(1, 9 * cpu_max // 10)
RUN_DATA_WORKERS = 0
RUN_CPU_THREADS_PER_WORKER = 1
RUN_CPU_CORE_START = 81
RUN_OUTPUT_ROOT = str(PROJECT_ROOT / "phase_diagram_2d")
RUN_DATA_ROOT = "data"
RUN_SEED = 1234
RUN_SUBSET_SEED = 2024
RUN_SUFFIX: str | None = None
RUN_OVERWRITE = False
RUN_USE_CUDA = False
RUN_SKIP_RUN = False
RUN_SKIP_PLOTS = False


@dataclass(frozen=True)
class RunSpec:
    p: float
    batch_size: int
    train_samples_per_class: int
    seed: int


@dataclass
class ExperimentConfig:
    p_values: list[float]
    batch_sizes: list[int]
    train_samples_per_class_values: list[int]
    seeds: list[int]
    width: int
    reference_epochs: int
    learning_rate: float
    train_log_interval: int
    test_log_interval: int
    progress_step_interval: int
    max_workers: int
    max_in_flight: int
    data_workers: int
    cpu_threads_per_worker: int
    cpu_cores: list[int]
    output_root: str
    data_root: str
    seed: int
    subset_seed: int
    suffix: str | None
    overwrite: bool
    use_cuda: bool
    skip_run: bool
    skip_plots: bool
    smoke_test: bool


class SaltAndPepperDataset(Dataset):
    def __init__(self, base: Dataset, p: float, seed: int) -> None:
        self.base = base
        self.p = float(p)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.base[idx]
        if self.p <= 0.0:
            return x, int(y)
        mask = torch.rand(x.shape, generator=self.generator, dtype=x.dtype) < self.p
        replacement = torch.rand(x.shape, generator=self.generator, dtype=x.dtype)
        return torch.where(mask, replacement, x), int(y)


class SingleHiddenMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, width),
            nn.ReLU(),
            nn.Linear(width, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_runtime_log(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def apply_cpu_affinity(cores: list[int] | None) -> None:
    if not cores:
        return
    try:
        os.sched_setaffinity(0, set(cores))
        print(f"[{timestamp()}] Pinned process to CPU cores: {sorted(set(cores))}", flush=True)
    except AttributeError:
        print("CPU affinity is not supported on this platform.", flush=True)
    except OSError as exc:
        print(f"Failed to set CPU affinity to {cores}: {exc}", flush=True)


def set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def total_train_size(train_samples_per_class: int) -> int:
    return 10 * int(train_samples_per_class)


def per_class_from_total(total_count: int) -> int:
    return int(total_count) // 10


def format_train_size_label(total_count: int) -> str:
    return f"n_c = {per_class_from_total(total_count)} (N = {int(total_count)})"


def normalize_train_samples_per_class(value: int) -> int:
    return max(1, min(int(value), BALANCED_TRAIN_PER_CLASS))


def build_cache_key(
    p_values: list[float],
    batch_sizes: list[int],
    train_samples_per_class_values: list[int],
    seeds: list[int],
    width: int,
    reference_epochs: int,
    learning_rate: float,
    train_log_interval: int,
    test_log_interval: int,
) -> str:
    p_min, p_max = min(p_values), max(p_values)
    b_min, b_max = min(batch_sizes), max(batch_sizes)
    tpc_min, tpc_max = min(train_samples_per_class_values), max(train_samples_per_class_values)
    return (
        f"phase2d_{BALANCED_DATASET_TAG}_r{len(seeds)}_e{reference_epochs}_"
        f"p{p_min:.4f}-{p_max:.4f}_"
        f"b{b_min}-{b_max}_"
        f"tpc{tpc_min}-{tpc_max}_"
        f"w{width}_lr{learning_rate:.4g}_"
        f"logtr{train_log_interval}_logte{test_log_interval}"
    )


def format_p(p: float) -> str:
    return f"{p:.4f}"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
        suffix=".tmp",
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_json_path(runs_dir: Path, spec: RunSpec) -> Path:
    return runs_dir / (
        f"p{format_p(spec.p)}_B{spec.batch_size}_tpc{spec.train_samples_per_class}_seed{spec.seed}.json"
    )


def mnist_transform() -> transforms.Compose:
    return transforms.ToTensor()


def get_worker_train_dataset(data_root: str) -> Dataset:
    cached = GLOBAL_STATE.get("train_dataset")
    if cached is not None:
        return cached
    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=mnist_transform())
    GLOBAL_STATE["train_dataset"] = dataset
    GLOBAL_STATE["train_targets"] = torch.as_tensor(dataset.targets, dtype=torch.long)
    return dataset


def get_worker_test_dataset(data_root: str) -> Dataset:
    cached = GLOBAL_STATE.get("test_dataset")
    if cached is not None:
        return cached
    dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=mnist_transform())
    GLOBAL_STATE["test_dataset"] = dataset
    GLOBAL_STATE["test_targets"] = torch.as_tensor(dataset.targets, dtype=torch.long)
    return dataset


def balanced_indices(
    targets: torch.Tensor,
    requested_per_class_count: int,
    seed: int,
    cache_key: str,
) -> list[int]:
    cached = GLOBAL_STATE.get(cache_key)
    if cached is not None:
        return cached
    available_per_class = [
        int(torch.count_nonzero(targets == digit).item())
        for digit in range(10)
    ]
    max_per_class = min(available_per_class)
    per_class = min(int(requested_per_class_count), max_per_class)
    if per_class <= 0:
        raise ValueError(f"Requested per-class count must be positive; got {requested_per_class_count}.")
    generator = torch.Generator().manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    for digit in range(10):
        idx = torch.nonzero(targets == digit, as_tuple=False).view(-1)
        if idx.numel() == 0:
            raise ValueError(f"No examples found for digit {digit}.")
        perm = torch.randperm(idx.numel(), generator=generator)
        selected.append(idx[perm[:per_class]])
    merged = torch.cat(selected)
    merged = merged[torch.randperm(merged.numel(), generator=generator)]
    indices = merged.tolist()
    GLOBAL_STATE[cache_key] = indices
    return indices


def balanced_train_indices(data_root: str, train_samples_per_class: int, subset_seed: int) -> list[int]:
    cache_key = f"train_indices_balanced_tpc{train_samples_per_class}_{subset_seed}"
    train_dataset = get_worker_train_dataset(data_root)
    _ = train_dataset
    targets = GLOBAL_STATE["train_targets"]
    return balanced_indices(
        targets,
        requested_per_class_count=train_samples_per_class,
        seed=subset_seed + train_samples_per_class,
        cache_key=cache_key,
    )


def balanced_test_indices(data_root: str, test_samples_per_class: int, subset_seed: int) -> list[int]:
    cache_key = f"test_indices_balanced_tpc{test_samples_per_class}_{subset_seed}"
    test_dataset = get_worker_test_dataset(data_root)
    _ = test_dataset
    targets = GLOBAL_STATE["test_targets"]
    return balanced_indices(
        targets,
        requested_per_class_count=test_samples_per_class,
        seed=subset_seed + 10_000 + test_samples_per_class,
        cache_key=cache_key,
    )


def balanced_subset_indices(data_root: str, train_samples_per_class: int, subset_seed: int) -> list[int]:
    cache_key = f"indices_tpc{train_samples_per_class}_{subset_seed}"
    cached = GLOBAL_STATE.get(cache_key)
    if cached is not None:
        return cached

    indices = balanced_train_indices(data_root, train_samples_per_class, subset_seed)
    GLOBAL_STATE[cache_key] = indices
    return indices


def build_train_subset(data_root: str, train_samples_per_class: int, subset_seed: int) -> Dataset:
    train_dataset = get_worker_train_dataset(data_root)
    return Subset(train_dataset, balanced_subset_indices(data_root, train_samples_per_class, subset_seed))


def build_test_subset(data_root: str, subset_seed: int) -> Dataset:
    test_dataset = get_worker_test_dataset(data_root)
    indices = balanced_test_indices(data_root, BALANCED_TEST_PER_CLASS, subset_seed)
    return Subset(test_dataset, indices)


def compute_effective_batch_size(requested_batch_size: int, n_train: int) -> int:
    return min(int(requested_batch_size), int(n_train))


def compute_total_samples_seen(n_train: int, reference_epochs: int) -> int:
    return int(n_train) * int(reference_epochs)


def compute_num_steps(total_samples_seen: int, effective_batch_size: int) -> int:
    return int(math.ceil(total_samples_seen / effective_batch_size))


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_examples += x.size(0)
    return total_loss / max(1, total_examples), total_correct / max(1, total_examples)


def make_run_payload(
    spec: RunSpec,
    *,
    width: int,
    requested_train_samples_per_class: int,
    actual_train_samples_per_class: int,
    actual_n: int,
    actual_test_samples_per_class: int,
    actual_test_n: int,
    effective_batch_size: int,
    total_samples_seen: int,
    num_steps: int,
    final_test_accuracy: float,
    final_test_loss: float,
    final_train_loss: float,
    train_loss_curve: list[list[float]],
    test_acc_curve: list[list[float]],
) -> dict[str, Any]:
    return {
        "p": float(spec.p),
        "batch_size": int(spec.batch_size),
        "effective_batch_size": int(effective_batch_size),
        "requested_train_samples_per_class": int(requested_train_samples_per_class),
        "train_samples_per_class": int(actual_train_samples_per_class),
        "requested_N": int(total_train_size(requested_train_samples_per_class)),
        "N": int(actual_n),
        "test_samples_per_class": int(actual_test_samples_per_class),
        "test_N": int(actual_test_n),
        "seed": int(spec.seed),
        "width": int(width),
        "total_samples_seen": int(total_samples_seen),
        "num_steps": int(num_steps),
        "final_test_accuracy": float(final_test_accuracy),
        "final_test_loss": float(final_test_loss),
        "final_train_loss": float(final_train_loss),
        "train_loss_curve": train_loss_curve,
        "test_acc_curve": test_acc_curve,
    }


def should_log(step: int, total_steps: int, interval: int) -> bool:
    interval = max(1, int(interval))
    return step == total_steps or step % interval == 0


def _init_worker(cpu_cores: list[int], cfg_dict: dict[str, Any]) -> None:
    apply_cpu_affinity(cpu_cores)
    try:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
    except Exception:
        pass
    cfg = ExperimentConfig(**cfg_dict)
    GLOBAL_STATE.clear()
    get_worker_train_dataset(cfg.data_root)
    get_worker_test_dataset(cfg.data_root)


def run_single_spec(spec: RunSpec, cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = ExperimentConfig(**cfg_dict)
    device = choose_device(cfg.use_cuda)
    set_seed(spec.seed, cfg.use_cuda)

    train_subset = build_train_subset(cfg.data_root, spec.train_samples_per_class, cfg.subset_seed)
    test_dataset = build_test_subset(cfg.data_root, cfg.subset_seed)
    actual_n_train = len(train_subset)
    actual_n_test = len(test_dataset)
    actual_train_samples_per_class = per_class_from_total(actual_n_train)
    actual_test_samples_per_class = per_class_from_total(actual_n_test)

    effective_batch_size = compute_effective_batch_size(spec.batch_size, actual_n_train)
    total_samples_seen = compute_total_samples_seen(actual_n_train, cfg.reference_epochs)
    total_steps = compute_num_steps(total_samples_seen, effective_batch_size)
    steps_per_epoch = compute_num_steps(actual_n_train, effective_batch_size)
    test_interval = max(1, min(cfg.test_log_interval, steps_per_epoch))
    progress_interval = max(1, cfg.progress_step_interval)

    noise_seed = spec.seed * 1000 + 17
    shuffle_seed = spec.seed * 1000 + 31

    train_dataset = SaltAndPepperDataset(train_subset, p=spec.p, seed=noise_seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=cfg.data_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(shuffle_seed),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(1024, len(test_dataset)),
        shuffle=False,
        num_workers=cfg.data_workers,
        pin_memory=pin_memory,
    )

    model = SingleHiddenMLP(cfg.width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    train_iter = iter(train_loader)
    step = 0
    samples_seen = 0
    loss_sum = 0.0
    loss_weight = 0
    train_loss_curve: list[list[float]] = []
    test_acc_curve: list[list[float]] = []

    start = time.time()
    print(
        f"[{timestamp()}] start: p={spec.p:.4f} B={spec.batch_size} "
        f"requested_tpc={spec.train_samples_per_class} actual_tpc={actual_train_samples_per_class} "
        f"actual_N={actual_n_train} seed={spec.seed} effective_B={effective_batch_size} steps={total_steps}",
        flush=True,
    )
    step_pbar = None
    if cfg.max_workers <= 1:
        step_pbar = tqdm(
            total=total_steps,
            desc=f"steps p={spec.p:.4f} B={spec.batch_size} tpc={actual_train_samples_per_class}",
            unit="step",
            leave=False,
        )

    while samples_seen < total_samples_seen:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        remaining = total_samples_seen - samples_seen
        if x.size(0) > remaining:
            x = x[:remaining]
            y = y[:remaining]

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        batch_size_now = x.size(0)
        step += 1
        samples_seen += batch_size_now
        loss_sum += loss.item() * batch_size_now
        loss_weight += batch_size_now
        if step_pbar is not None:
            step_pbar.update(1)
            if should_log(step, total_steps, cfg.train_log_interval):
                step_pbar.set_postfix_str(f"loss={loss.item():.4f}")

        if should_log(step, total_steps, cfg.train_log_interval):
            train_loss_curve.append([int(step), float(loss_sum / max(1, loss_weight))])
            loss_sum = 0.0
            loss_weight = 0

        if should_log(step, total_steps, test_interval):
            _, test_acc = evaluate_clean(model, test_loader, device)
            test_acc_curve.append([int(step), float(test_acc)])

        if should_log(step, total_steps, progress_interval):
            elapsed = time.time() - start
            print(
                f"[{timestamp()}] progress: p={spec.p:.4f} B={spec.batch_size} "
                f"tpc={spec.train_samples_per_class} seed={spec.seed} "
                f"step={step}/{total_steps} samples={samples_seen}/{total_samples_seen} elapsed={elapsed:.1f}s",
                flush=True,
            )

    final_test_loss, final_test_accuracy = evaluate_clean(model, test_loader, device)
    final_train_loss = float(train_loss_curve[-1][1]) if train_loss_curve else float("nan")
    if step_pbar is not None:
        step_pbar.close()
    if not test_acc_curve or int(test_acc_curve[-1][0]) != total_steps:
        test_acc_curve.append([int(total_steps), float(final_test_accuracy)])

    print(
        f"[{timestamp()}] done: p={spec.p:.4f} B={spec.batch_size} "
        f"requested_tpc={spec.train_samples_per_class} actual_tpc={actual_train_samples_per_class} "
        f"actual_N={actual_n_train} seed={spec.seed} "
        f"acc={final_test_accuracy:.4f} loss={final_test_loss:.4f}",
        flush=True,
    )
    return make_run_payload(
        spec,
        width=cfg.width,
        requested_train_samples_per_class=spec.train_samples_per_class,
        actual_train_samples_per_class=actual_train_samples_per_class,
        actual_n=actual_n_train,
        actual_test_samples_per_class=actual_test_samples_per_class,
        actual_test_n=actual_n_test,
        effective_batch_size=effective_batch_size,
        total_samples_seen=total_samples_seen,
        num_steps=total_steps,
        final_test_accuracy=final_test_accuracy,
        final_test_loss=final_test_loss,
        final_train_loss=final_train_loss,
        train_loss_curve=train_loss_curve,
        test_acc_curve=test_acc_curve,
    )


def build_summary_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in run_rows:
        rows.append(
            {
                "p": float(row["p"]),
                "batch_size": int(row["batch_size"]),
                "train_samples_per_class": int(row.get("train_samples_per_class", per_class_from_total(row["N"]))),
                "requested_train_samples_per_class": int(
                    row.get("requested_train_samples_per_class", per_class_from_total(row.get("requested_N", row["N"])))
                ),
                "N": int(row["N"]),
                "requested_N": int(row.get("requested_N", row["N"])),
                "test_samples_per_class": int(row.get("test_samples_per_class", per_class_from_total(row.get("test_N", 0)))),
                "test_N": int(row.get("test_N", 0)),
                "seed": int(row["seed"]),
                "width": int(row["width"]),
                "final_test_accuracy": float(row["final_test_accuracy"]),
                "final_test_loss": float(row["final_test_loss"]),
                "final_train_loss": float(row.get("final_train_loss", float("nan"))),
            }
        )
    rows.sort(key=lambda item: (item["N"], item["batch_size"], item["p"], item["seed"]))
    return rows


def build_per_run_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in run_rows:
        rows.append(
            {
                "p": float(row["p"]),
                "batch_size": int(row["batch_size"]),
                "effective_batch_size": int(row["effective_batch_size"]),
                "train_samples_per_class": int(row.get("train_samples_per_class", per_class_from_total(row["N"]))),
                "requested_train_samples_per_class": int(
                    row.get("requested_train_samples_per_class", per_class_from_total(row.get("requested_N", row["N"])))
                ),
                "N": int(row["N"]),
                "requested_N": int(row.get("requested_N", row["N"])),
                "test_samples_per_class": int(row.get("test_samples_per_class", per_class_from_total(row.get("test_N", 0)))),
                "test_N": int(row.get("test_N", 0)),
                "seed": int(row["seed"]),
                "width": int(row["width"]),
                "total_samples_seen": int(row["total_samples_seen"]),
                "num_steps": int(row["num_steps"]),
                "final_test_accuracy": float(row["final_test_accuracy"]),
                "final_test_loss": float(row["final_test_loss"]),
                "final_train_loss": float(row.get("final_train_loss", float("nan"))),
            }
        )
    rows.sort(key=lambda item: (item["N"], item["batch_size"], item["p"], item["seed"]))
    return rows


def load_all_run_rows(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []
    rows = [load_json(path) for path in sorted(runs_dir.glob("*.json"))]
    rows.sort(key=lambda row: (row["N"], row["batch_size"], row["p"], row["seed"]))
    return rows


def aggregate_by_mean(run_rows: list[dict[str, Any]]) -> dict[tuple[int, int, float], dict[str, float]]:
    grouped: dict[tuple[int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        key = (int(row["N"]), int(row["batch_size"]), float(row["p"]))
        grouped[key].append(row)

    out: dict[tuple[int, int, float], dict[str, float]] = {}
    for key, rows in grouped.items():
        accs = [float(row["final_test_accuracy"]) for row in rows]
        test_losses = [float(row["final_test_loss"]) for row in rows]
        train_losses = [float(row.get("final_train_loss", float("nan"))) for row in rows]
        valid_train_losses = [value for value in train_losses if np.isfinite(value)]
        out[key] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "stderr_accuracy": float(np.std(accs) / math.sqrt(max(1, len(rows)))),
            "mean_loss": float(np.mean(test_losses)),
            "mean_test_loss": float(np.mean(test_losses)),
            "mean_train_loss": float(np.mean(valid_train_losses)) if valid_train_losses else float("nan"),
            "count": float(len(rows)),
        }
    return out


def enforce_nonincreasing(values: list[float]) -> list[float]:
    if not values:
        return []
    out = [float(values[0])]
    for value in values[1:]:
        out.append(min(out[-1], float(value)))
    return out


def estimate_crossing(ps: list[float], accs: list[float], threshold: float) -> float | None:
    if len(ps) < 2:
        return None
    monotone = enforce_nonincreasing(accs)
    for idx in range(1, len(ps)):
        prev_acc = monotone[idx - 1]
        curr_acc = monotone[idx]
        if prev_acc >= threshold >= curr_acc:
            p0 = float(ps[idx - 1])
            p1 = float(ps[idx])
            if abs(curr_acc - prev_acc) < 1e-12:
                return p1
            frac = (threshold - prev_acc) / (curr_acc - prev_acc)
            return float(p0 + frac * (p1 - p0))
    return None


def linear_edges(values: list[float]) -> np.ndarray:
    vals = np.asarray(sorted(values), dtype=float)
    if vals.size == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5], dtype=float)
    mids = 0.5 * (vals[:-1] + vals[1:])
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = mids
    edges[0] = vals[0] - (mids[0] - vals[0])
    edges[-1] = vals[-1] + (vals[-1] - mids[-1])
    return edges


def log_edges(values: list[int]) -> np.ndarray:
    vals = np.asarray(sorted(values), dtype=float)
    if vals.size == 1:
        return np.array([vals[0] / 2.0, vals[0] * 2.0], dtype=float)
    mids = np.sqrt(vals[:-1] * vals[1:])
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = mids
    edges[0] = vals[0] ** 2 / mids[0]
    edges[-1] = vals[-1] ** 2 / mids[-1]
    return edges


def build_heatmap_grid(
    aggregated: dict[tuple[int, int, float], dict[str, float]],
    n_value: int,
    batch_sizes: list[int],
    p_values: list[float],
    metric_key: str = "mean_accuracy",
) -> np.ndarray:
    grid = np.full((len(batch_sizes), len(p_values)), np.nan, dtype=float)
    for i, batch_size in enumerate(batch_sizes):
        for j, p in enumerate(p_values):
            stats = aggregated.get((int(n_value), int(batch_size), float(p)))
            if stats is not None:
                value = stats.get(metric_key, float("nan"))
                if np.isfinite(value):
                    grid[i, j] = float(value)
    return grid


def add_accuracy_heatmap(
    ax: plt.Axes,
    batch_sizes: list[int],
    p_values: list[float],
    grid: np.ndarray,
    title: str,
    contour_threshold: float,
) -> Any:
    norm = TwoSlopeNorm(vmin=0.0, vcenter=DEFAULT_CHANCE_ACCURACY, vmax=1.0)
    mesh = ax.pcolormesh(
        linear_edges(p_values),
        log_edges(batch_sizes),
        grid,
        shading="auto",
        cmap="coolwarm",
        norm=norm,
    )
    if np.isfinite(grid).sum() >= 4:
        p_mesh, b_mesh = np.meshgrid(p_values, batch_sizes)
        try:
            ax.contour(p_mesh, b_mesh, grid, levels=[contour_threshold], colors="black", linewidths=1.1)
        except ValueError:
            pass
    ax.set_title(title)
    ax.set_xlabel("corruption probability p")
    ax.set_ylabel("batch size")
    ax.set_yscale("log")
    ax.set_yticks(batch_sizes)
    ax.set_yticklabels([str(batch_size) for batch_size in batch_sizes])
    return mesh


def add_scalar_heatmap(
    ax: plt.Axes,
    batch_sizes: list[int],
    p_values: list[float],
    grid: np.ndarray,
    title: str,
    *,
    cmap: str = "viridis",
) -> Any:
    finite = grid[np.isfinite(grid)]
    if finite.size:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12
    else:
        vmin, vmax = 0.0, 1.0
    mesh = ax.pcolormesh(
        linear_edges(p_values),
        log_edges(batch_sizes),
        grid,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("corruption probability p")
    ax.set_ylabel("batch size")
    ax.set_yscale("log")
    ax.set_yticks(batch_sizes)
    ax.set_yticklabels([str(batch_size) for batch_size in batch_sizes])
    return mesh


def fit_phase_boundary(batch_sizes: list[int], p_critical: list[float]) -> dict[str, float] | None:
    valid = [
        (float(batch_size), float(pc))
        for batch_size, pc in zip(batch_sizes, p_critical)
        if np.isfinite(pc) and 0.0 < 1.0 - pc < 1.0
    ]
    if len(valid) < 2:
        return None
    x = np.log([batch_size for batch_size, _ in valid])
    y = np.log([1.0 - pc for _, pc in valid])
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "a": float(np.exp(intercept)),
        "beta": float(-slope),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def scaling_collapse_score(scaled_x: np.ndarray, y: np.ndarray, bins: int = 30) -> float:
    if scaled_x.size < 6:
        return float("inf")
    grid = np.linspace(float(scaled_x.min()), float(scaled_x.max()), bins + 1)
    score = 0.0
    used = 0
    for idx in range(bins):
        left = grid[idx]
        right = grid[idx + 1]
        mask = (scaled_x >= left) & (scaled_x <= right if idx == bins - 1 else scaled_x < right)
        if np.count_nonzero(mask) < 2:
            continue
        score += float(np.var(y[mask]))
        used += 1
    if used == 0:
        return float("inf")
    return score / used


def find_best_nu(ps: np.ndarray, ns: np.ndarray, accs: np.ndarray, p_critical: float) -> float:
    best_nu = 1.0
    best_score = float("inf")
    for nu in np.linspace(0.2, 5.0, 97):
        scaled_x = (ps - p_critical) * np.power(ns, 1.0 / nu)
        score = scaling_collapse_score(scaled_x, accs)
        if score < best_score:
            best_score = score
            best_nu = float(nu)
    return best_nu


def interpolate_accuracy_at_pc(points: list[tuple[float, float]], p_c: float) -> float:
    if len(points) < 2:
        return float("nan")
    ps = np.asarray([point[0] for point in points], dtype=float)
    accs = np.asarray([point[1] for point in points], dtype=float)
    if p_c < float(ps.min()) or p_c > float(ps.max()):
        return float("nan")
    return float(np.interp(p_c, ps, accs))


def find_best_pc_nu(
    aggregated: dict[tuple[int, int, float], dict[str, float]],
    *,
    batch_size: int,
    n_values: list[int],
    p_values: list[float],
    acc_window: tuple[float, float] = (0.12, 0.78),
    p_grid_size: int = 121,
    nu_grid_size: int = 97,
) -> dict[str, float] | None:
    rows = []
    for n_value in n_values:
        for p in p_values:
            stats = aggregated.get((int(n_value), int(batch_size), float(p)))
            if stats is not None:
                rows.append((float(p), float(n_value), float(stats["mean_accuracy"])))
    if len(rows) < 8 or len({row[1] for row in rows}) < 2 or len({row[0] for row in rows}) < 4:
        return None

    lo, hi = acc_window
    windowed = [row for row in rows if lo <= row[2] <= hi]
    work = windowed if len(windowed) >= 8 else rows
    ps = np.asarray([row[0] for row in work], dtype=float)
    ns = np.asarray([row[1] for row in work], dtype=float)
    accs = np.asarray([row[2] for row in work], dtype=float)

    p_min = float(np.min(ps)) + 1e-4
    p_max = float(np.max(ps)) - 1e-4
    if not p_max > p_min:
        return None

    best = None
    best_score = float("inf")
    for p_c in np.linspace(p_min, p_max, p_grid_size):
        for nu in np.linspace(0.2, 5.0, nu_grid_size):
            scaled_x = (ps - p_c) * np.power(ns, 1.0 / nu)
            score = scaling_collapse_score(scaled_x, accs)
            if score < best_score:
                best_score = score
                best = {
                    "p_c": float(p_c),
                    "nu": float(nu),
                    "score": float(score),
                    "n_points": float(len(work)),
                }
    return best


def find_best_full_collapse(
    aggregated: dict[tuple[int, int, float], dict[str, float]],
    *,
    batch_size: int,
    n_values: list[int],
    p_values: list[float],
    acc_window: tuple[float, float] = (0.12, 0.78),
    p_grid_size: int = 41,
    nu_grid_size: int = 49,
    kappa_grid_size: int = 81,
) -> dict[str, float] | None:
    base = find_best_pc_nu(
        aggregated,
        batch_size=batch_size,
        n_values=n_values,
        p_values=p_values,
        acc_window=acc_window,
        p_grid_size=max(31, p_grid_size),
        nu_grid_size=max(31, nu_grid_size),
    )
    if base is None:
        return None

    rows = []
    for n_value in n_values:
        for p in p_values:
            stats = aggregated.get((int(n_value), int(batch_size), float(p)))
            if stats is not None:
                rows.append((float(p), float(n_value), float(stats["mean_accuracy"])))
    lo, hi = acc_window
    windowed = [row for row in rows if lo <= row[2] <= hi]
    work = windowed if len(windowed) >= 8 else rows
    if len(work) < 8:
        return None

    ps = np.asarray([row[0] for row in work], dtype=float)
    ns = np.asarray([row[1] for row in work], dtype=float)
    accs = np.asarray([row[2] for row in work], dtype=float)

    p_min_data = float(np.min(ps)) + 1e-4
    p_max_data = float(np.max(ps)) - 1e-4
    p_half_width = min(0.02, 0.5 * max(1e-4, p_max_data - p_min_data))
    p_min = max(p_min_data, float(base["p_c"]) - p_half_width)
    p_max = min(p_max_data, float(base["p_c"]) + p_half_width)
    if not p_max > p_min:
        p_min, p_max = p_min_data, p_max_data

    nu_min = max(0.2, 0.5 * float(base["nu"]))
    nu_max = min(8.0, 2.0 * float(base["nu"]))
    if not nu_max > nu_min:
        nu_min, nu_max = 0.2, 5.0

    best = None
    best_score = float("inf")
    p_candidates = np.linspace(p_min, p_max, p_grid_size)
    nu_candidates = np.linspace(nu_min, nu_max, nu_grid_size)
    kappa_candidates = np.linspace(-2.0, 2.0, kappa_grid_size)
    points_by_n: dict[int, list[tuple[float, float]]] = {}
    for n_value in n_values:
        curve = []
        for p in p_values:
            stats = aggregated.get((int(n_value), int(batch_size), float(p)))
            if stats is not None:
                curve.append((float(p), float(stats["mean_accuracy"])))
        if curve:
            points_by_n[int(n_value)] = curve

    for p_c in p_candidates:
        ac_values = [
            interpolate_accuracy_at_pc(points_by_n[n_value], float(p_c))
            for n_value in sorted(points_by_n)
        ]
        ac_values = [value for value in ac_values if np.isfinite(value)]
        if not ac_values:
            continue
        a_c = float(np.mean(ac_values))
        for nu in nu_candidates:
            scaled_x = (ps - p_c) * np.power(ns, 1.0 / nu)
            for kappa in kappa_candidates:
                scaled_y = (accs - a_c) * np.power(ns, kappa)
                score = scaling_collapse_score(scaled_x, scaled_y)
                if score < best_score:
                    best_score = score
                    best = {
                        "p_c": float(p_c),
                        "nu": float(nu),
                        "A_c": float(a_c),
                        "kappa": float(kappa),
                        "score": float(score),
                        "n_points": float(len(work)),
                    }
    return best


def import_fssa_with_compat():
    if not hasattr(np, "int"):
        np.int = int
    try:
        import scipy.optimize.optimize as _opt_mod

        _opt_mod = importlib.reload(_opt_mod)
        from scipy.optimize import _optimize as _opt

        for name in ("OptimizeResult", "_status_message", "wrap_function"):
            if not hasattr(_opt_mod, name) and hasattr(_opt, name):
                setattr(_opt_mod, name, getattr(_opt, name))
        if not hasattr(_opt_mod, "wrap_function"):
            def _wrap_function(function, args):
                ncalls = [0]

                def function_wrapper(x):
                    ncalls[0] += 1
                    return function(x, *args)

                return ncalls, function_wrapper

            _opt_mod.wrap_function = _wrap_function
    except Exception:
        pass
    import fssa

    return fssa


def find_pyfssa_collapse_zeta0_batch(
    aggregated: dict[tuple[int, int, float], dict[str, float]],
    *,
    batch_size: int,
    n_values: list[int],
    p_values: list[float],
    rho_grid_size: int = 40,
    nu_grid_size: int = 60,
) -> tuple[dict[str, Any] | None, Any]:
    present_n = []
    common_p: set[float] | None = None
    for n_value in n_values:
        pset = {
            float(p)
            for p in p_values
            if (int(n_value), int(batch_size), float(p)) in aggregated
        }
        if len(pset) >= 2:
            present_n.append(int(n_value))
            common_p = pset if common_p is None else (common_p & pset)
    if len(present_n) < 2:
        return {"error": f"need >=2 N values for batch_size={batch_size}"}, None
    ps = sorted(common_p) if common_p else []
    if len(ps) < 2:
        return {"error": "need >=2 common p values across N"}, None

    L = np.array(present_n, dtype=float)
    rho = np.array(ps, dtype=float)
    a = np.zeros((len(L), len(rho)))
    da = np.zeros_like(a)
    for i, n_value in enumerate(present_n):
        for j, p in enumerate(ps):
            stats = aggregated[(int(n_value), int(batch_size), float(p))]
            a[i, j] = float(stats["mean_accuracy"])
            da[i, j] = float(stats.get("stderr_accuracy", 0.0))
    min_nonzero = float(da[da > 0].min()) if np.any(da > 0) else 1e-6
    da = np.where(np.isfinite(da), da, min_nonzero)
    da = np.clip(da, 1e-6, None)

    try:
        fssa = import_fssa_with_compat()
    except Exception as exc:
        return {"error": f"fssa import failed: {exc}"}, None

    zeta_fixed = 0.0
    rho_grid = np.linspace(float(rho.min()) + 1e-3, float(rho.max()) - 1e-3, rho_grid_size)
    nu_grid = np.linspace(0.2, 10.0, nu_grid_size)
    best = (float("inf"), float("nan"), float("nan"))
    for rho_c in rho_grid:
        for nu in nu_grid:
            scaled = fssa.scaledata(L, rho, a, da, rho_c, nu, zeta_fixed)
            score = fssa.quality(scaled.x, scaled.y, scaled.dy)
            if score < best[0]:
                best = (float(score), float(rho_c), float(nu))
    s_star, rho_c_star, nu_star = best
    if not np.isfinite(s_star):
        return {"error": "failed to find finite pyfssa quality"}, None
    scaled = fssa.scaledata(L, rho, a, da, rho_c_star, nu_star, zeta_fixed)
    return {
        "rho_c": float(rho_c_star),
        "nu": float(nu_star),
        "zeta": float(zeta_fixed),
        "S": float(s_star),
        "n_values": [int(v) for v in present_n],
    }, scaled


def pick_representative_runs(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated = aggregate_by_mean(run_rows)
    candidates = []
    for (n_value, batch_size, p), stats in aggregated.items():
        if n_value != BALANCED_TRAIN_TOTAL:
            continue
        candidates.append((int(batch_size), float(p), float(stats["mean_accuracy"])))

    above = sorted(
        [item for item in candidates if item[2] > PHASE_THRESHOLDS[0]],
        key=lambda item: abs(item[2] - PHASE_THRESHOLDS[0]),
    )
    below = sorted(
        [item for item in candidates if item[2] <= PHASE_THRESHOLDS[0]],
        key=lambda item: abs(item[2] - PHASE_THRESHOLDS[0]),
    )

    selected = []
    for candidate in above[:1] + below[:1]:
        if candidate not in selected:
            selected.append(candidate)
    for candidate in [item for item in above if item[0] == BALANCED_TRAIN_TOTAL][:1]:
        if candidate not in selected:
            selected.append(candidate)
    for candidate in [item for item in below if item[0] == BALANCED_TRAIN_TOTAL][:1]:
        if candidate not in selected:
            selected.append(candidate)

    out = []
    for batch_size, p, mean_acc in selected:
        matching = [
            row
            for row in run_rows
            if int(row["N"]) == BALANCED_TRAIN_TOTAL and int(row["batch_size"]) == batch_size and abs(float(row["p"]) - p) < 1e-12
        ]
        if not matching:
            continue
        out.append(min(matching, key=lambda row: abs(float(row["final_test_accuracy"]) - mean_acc)))
    return out


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")


def add_figure_page(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_p_by_batch_pages(
    pdf: PdfPages,
    aggregated: dict[tuple[int, int, float], dict[str, float]],
    *,
    batch_sizes: list[int],
    p_values: list[float],
    n_values: list[int],
    metric_key: str,
    y_label: str,
    page_title: str,
) -> None:
    plots_per_page = 6
    for start in range(0, len(batch_sizes), plots_per_page):
        page_batch_sizes = batch_sizes[start : start + plots_per_page]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), squeeze=False)
        axes_flat = axes.ravel()
        for ax, batch_size in zip(axes_flat, page_batch_sizes):
            for n_value in n_values:
                ps_present = []
                accs = []
                for p in p_values:
                    stats = aggregated.get((int(n_value), int(batch_size), float(p)))
                    if stats is not None:
                        value = stats.get(metric_key, float("nan"))
                        if np.isfinite(value):
                            ps_present.append(float(p))
                            accs.append(float(value))
                if ps_present:
                    ax.plot(ps_present, accs, marker="o", label=format_train_size_label(n_value))
            ax.set_title(f"B = {batch_size}")
            ax.set_xlabel("p")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            if ax.lines:
                ax.legend(fontsize=8)
        for ax in axes_flat[len(page_batch_sizes) :]:
            ax.set_axis_off()
        fig.suptitle(page_title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        add_figure_page(pdf, fig)


def write_plots(cache_dir: Path, run_rows: list[dict[str, Any]]) -> None:
    if not run_rows:
        return

    figures_dir = cache_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    combined_pdf_path = figures_dir / f"phase_diagram_characterizations_{cache_dir.name}.pdf"
    aggregated = aggregate_by_mean(run_rows)
    batch_sizes = sorted({int(row["batch_size"]) for row in run_rows})
    p_values = sorted({float(row["p"]) for row in run_rows})
    n_values = sorted({int(row["N"]) for row in run_rows})
    with PdfPages(combined_pdf_path) as pdf:
        grid_max_n = build_heatmap_grid(
            aggregated,
            BALANCED_TRAIN_TOTAL,
            batch_sizes,
            p_values,
            metric_key="mean_accuracy",
        )
        if np.isfinite(grid_max_n).any():
            fig, ax = plt.subplots(figsize=(9, 5.5))
            mesh = add_accuracy_heatmap(
                ax,
                batch_sizes,
                p_values,
                grid_max_n,
                f"MNIST phase diagram ({format_train_size_label(BALANCED_TRAIN_TOTAL)})",
                PHASE_THRESHOLDS[0],
            )
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label("mean clean test accuracy")
            fig.tight_layout()
            add_figure_page(pdf, fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        annotation_lines = []
        for threshold in PHASE_THRESHOLDS:
            pcs = []
            for batch_size in batch_sizes:
                ps_present = []
                accs = []
                for p in p_values:
                    stats = aggregated.get((BALANCED_TRAIN_TOTAL, int(batch_size), float(p)))
                    if stats is not None:
                        ps_present.append(float(p))
                        accs.append(float(stats["mean_accuracy"]))
                pc = estimate_crossing(ps_present, accs, threshold)
                pcs.append(float("nan") if pc is None else float(pc))
            ax.plot(batch_sizes, pcs, marker="o", label=f"threshold={threshold:.2f}")
            fit = fit_phase_boundary(batch_sizes, pcs)
            if fit is not None:
                batch_grid = np.geomspace(min(batch_sizes), max(batch_sizes), 300)
                ax.plot(batch_grid, 1.0 - fit["a"] * np.power(batch_grid, -fit["beta"]), linestyle="--", alpha=0.8)
                annotation_lines.append(
                    f"thr={threshold:.2f}: a={fit['a']:.4g}, beta={fit['beta']:.4f}, R^2={fit['r2']:.4f}"
                )
        ax.set_xscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel("estimated critical p_c")
        ax.set_title(f"Phase boundary at {format_train_size_label(BALANCED_TRAIN_TOTAL)}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        if annotation_lines:
            ax.text(
                0.02,
                0.02,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            )
        fig.tight_layout()
        add_figure_page(pdf, fig)

        fig, axes = plt.subplots(1, len(n_values), figsize=(4.5 * len(n_values), 4.5), squeeze=False)
        fig.subplots_adjust(left=0.06, right=0.92, bottom=0.14, top=0.88, wspace=0.30)
        mesh = None
        for idx, n_value in enumerate(n_values):
            ax = axes[0, idx]
            grid = build_heatmap_grid(
                aggregated,
                n_value,
                batch_sizes,
                p_values,
                metric_key="mean_accuracy",
            )
            if not np.isfinite(grid).any():
                ax.set_axis_off()
                ax.set_title(f"N = {n_value} (no data)")
                continue
            mesh = add_accuracy_heatmap(
                ax,
                batch_sizes,
                p_values,
                grid,
                format_train_size_label(n_value),
                PHASE_THRESHOLDS[0],
            )
        if mesh is not None:
            cax = fig.add_axes([0.935, 0.18, 0.008, 0.64])
            cbar = fig.colorbar(mesh, cax=cax)
            cbar.set_label("mean clean test accuracy")
        add_figure_page(pdf, fig)

        plot_metric_vs_p_by_batch_pages(
            pdf,
            aggregated,
            batch_sizes=batch_sizes,
            p_values=p_values,
            n_values=n_values,
            metric_key="mean_accuracy",
            y_label="mean clean test accuracy",
            page_title="Accuracy vs p by Dataset Size at Fixed Batch Size",
        )

        grid_max_n = build_heatmap_grid(
            aggregated,
            BALANCED_TRAIN_TOTAL,
            batch_sizes,
            p_values,
            metric_key="mean_test_loss",
        )
        if np.isfinite(grid_max_n).any():
            fig, ax = plt.subplots(figsize=(9, 5.5))
            mesh = add_scalar_heatmap(
                ax,
                batch_sizes,
                p_values,
                grid_max_n,
                f"Test-loss phase diagram ({format_train_size_label(BALANCED_TRAIN_TOTAL)})",
            )
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label("mean clean test loss")
            fig.tight_layout()
            add_figure_page(pdf, fig)

        fig, axes = plt.subplots(1, len(n_values), figsize=(4.5 * len(n_values), 4.5), squeeze=False)
        fig.subplots_adjust(left=0.06, right=0.92, bottom=0.14, top=0.88, wspace=0.30)
        mesh = None
        for idx, n_value in enumerate(n_values):
            ax = axes[0, idx]
            grid = build_heatmap_grid(
                aggregated,
                n_value,
                batch_sizes,
                p_values,
                metric_key="mean_test_loss",
            )
            if not np.isfinite(grid).any():
                ax.set_axis_off()
                ax.set_title(f"N = {n_value} (no data)")
                continue
            mesh = add_scalar_heatmap(ax, batch_sizes, p_values, grid, format_train_size_label(n_value))
        if mesh is not None:
            cax = fig.add_axes([0.935, 0.18, 0.008, 0.64])
            cbar = fig.colorbar(mesh, cax=cax)
            cbar.set_label("mean clean test loss")
        add_figure_page(pdf, fig)

        plot_metric_vs_p_by_batch_pages(
            pdf,
            aggregated,
            batch_sizes=batch_sizes,
            p_values=p_values,
            n_values=n_values,
            metric_key="mean_test_loss",
            y_label="mean clean test loss",
            page_title="Test Loss vs p by Dataset Size at Fixed Batch Size",
        )

        grid_max_n = build_heatmap_grid(
            aggregated,
            BALANCED_TRAIN_TOTAL,
            batch_sizes,
            p_values,
            metric_key="mean_train_loss",
        )
        if np.isfinite(grid_max_n).any():
            fig, ax = plt.subplots(figsize=(9, 5.5))
            mesh = add_scalar_heatmap(
                ax,
                batch_sizes,
                p_values,
                grid_max_n,
                f"Train-loss phase diagram ({format_train_size_label(BALANCED_TRAIN_TOTAL)})",
            )
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label("mean final train loss")
            fig.tight_layout()
            add_figure_page(pdf, fig)

        fig, axes = plt.subplots(1, len(n_values), figsize=(4.5 * len(n_values), 4.5), squeeze=False)
        fig.subplots_adjust(left=0.06, right=0.92, bottom=0.14, top=0.88, wspace=0.30)
        mesh = None
        for idx, n_value in enumerate(n_values):
            ax = axes[0, idx]
            grid = build_heatmap_grid(
                aggregated,
                n_value,
                batch_sizes,
                p_values,
                metric_key="mean_train_loss",
            )
            if not np.isfinite(grid).any():
                ax.set_axis_off()
                ax.set_title(f"N = {n_value} (no data)")
                continue
            mesh = add_scalar_heatmap(ax, batch_sizes, p_values, grid, format_train_size_label(n_value))
        if mesh is not None:
            cax = fig.add_axes([0.935, 0.18, 0.008, 0.64])
            cbar = fig.colorbar(mesh, cax=cax)
            cbar.set_label("mean final train loss")
        add_figure_page(pdf, fig)

        plot_metric_vs_p_by_batch_pages(
            pdf,
            aggregated,
            batch_sizes=batch_sizes,
            p_values=p_values,
            n_values=n_values,
            metric_key="mean_train_loss",
            y_label="mean final train loss",
            page_title="Train Loss vs p by Dataset Size at Fixed Batch Size",
        )


def run_priority_key(spec: RunSpec) -> tuple[int, int, int, float, int]:
    total_n = total_train_size(spec.train_samples_per_class)
    if spec.train_samples_per_class == BALANCED_TRAIN_PER_CLASS:
        tier = 0
    elif spec.batch_size == BALANCED_TRAIN_TOTAL:
        tier = 1
    else:
        tier = 2
    return (tier, -total_n, spec.batch_size, spec.p, spec.seed)


def build_specs(cfg: ExperimentConfig) -> list[RunSpec]:
    specs = [
        RunSpec(
            p=float(p),
            batch_size=int(batch_size),
            train_samples_per_class=int(train_samples_per_class),
            seed=int(seed),
        )
        for train_samples_per_class in cfg.train_samples_per_class_values
        for batch_size in cfg.batch_sizes
        for p in cfg.p_values
        for seed in cfg.seeds
    ]
    specs.sort(key=run_priority_key)
    return specs


def format_run_status(spec: RunSpec, *, queued: int | None = None, active: int | None = None) -> str:
    parts = [
        f"p={spec.p:.4f}",
        f"B={spec.batch_size}",
        f"n_c={spec.train_samples_per_class}",
    ]
    if active is not None:
        parts.append(f"active={active}")
    if queued is not None:
        parts.append(f"queued={queued}")
    return " ".join(parts)


def write_metadata(cache_dir: Path, cfg: ExperimentConfig, cache_key: str, total_runs: int) -> None:
    atomic_write_json(
        cache_dir / "config.json",
        {
            "timestamp": timestamp(),
            "cache_key": cache_key,
            "config": asdict(cfg),
            "total_requested_runs": int(total_runs),
        },
    )


def execute_runs(cache_dir: Path, cfg: ExperimentConfig, cache_key: str) -> None:
    runs_dir = cache_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(cfg)
    pending = []
    for spec in specs:
        if cfg.overwrite or not run_json_path(runs_dir, spec).exists():
            pending.append(spec)

    print(
        f"[{timestamp()}] Launching {len(specs)} runs with max_workers={cfg.max_workers} "
        f"(pending={len(pending)}, max_in_flight={cfg.max_in_flight})"
        + (f" (suffix={cfg.suffix})" if cfg.suffix else ""),
        flush=True,
    )
    if not pending:
        return

    cfg_dict = asdict(cfg)
    if cfg.max_workers <= 1:
        _init_worker(cfg.cpu_cores, cfg_dict)
        pbar = tqdm(total=len(pending), desc="Runs", unit="run")
        for idx, spec in enumerate(pending, start=1):
            print(
                f"[{timestamp()}] dispatch {idx}/{len(pending)} p={spec.p:.4f} "
                f"B={spec.batch_size} tpc={spec.train_samples_per_class} seed={spec.seed}",
                flush=True,
            )
            payload = run_single_spec(spec, cfg_dict)
            atomic_write_json(run_json_path(runs_dir, spec), payload)
            pbar.update(1)
            pbar.set_postfix_str(format_run_status(spec))
        pbar.close()
        return

    queue = deque(pending)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=cfg.max_workers,
        initializer=_init_worker,
        initargs=(cfg.cpu_cores, cfg_dict),
    ) as executor:
        in_flight: dict[Any, RunSpec] = {}
        pbar = tqdm(total=len(pending), desc="Runs", unit="run")
        while queue or in_flight:
            while queue and len(in_flight) < cfg.max_in_flight:
                spec = queue.popleft()
                future = executor.submit(run_single_spec, spec, cfg_dict)
                in_flight[future] = spec
                pbar.set_postfix_str(
                    format_run_status(spec, active=len(in_flight), queued=len(queue))
                )
            for future in as_completed(in_flight):
                spec = in_flight.pop(future)
                payload = future.result()
                atomic_write_json(run_json_path(runs_dir, spec), payload)
                completed += 1
                pbar.update(1)
                pbar.set_postfix_str(format_run_status(spec, active=len(in_flight), queued=len(queue)))
                print(
                    f"[{timestamp()}] completed {completed}/{len(pending)} "
                    f"p={spec.p:.4f} B={spec.batch_size} tpc={spec.train_samples_per_class} seed={spec.seed}",
                    flush=True,
                )
                break
        pbar.close()


def build_config() -> ExperimentConfig:
    train_samples_per_class_values = [
        normalize_train_samples_per_class(value)
        for value in RUN_TRAIN_SAMPLES_PER_CLASS_VALUES
    ]
    max_workers = max(1, int(RUN_MAX_WORKERS))
    max_in_flight = max(1, int(RUN_MAX_IN_FLIGHT))
    cpu_core_start = int(RUN_CPU_CORE_START)
    cpu_cores = list(range(cpu_core_start, cpu_core_start + max_workers))

    return ExperimentConfig(
        p_values=[float(value) for value in RUN_P_VALUES],
        batch_sizes=[int(value) for value in RUN_BATCH_SIZES],
        train_samples_per_class_values=train_samples_per_class_values,
        seeds=[int(value) for value in RUN_SEEDS],
        width=int(RUN_WIDTH),
        reference_epochs=int(RUN_REFERENCE_EPOCHS),
        learning_rate=float(RUN_LEARNING_RATE),
        train_log_interval=int(RUN_TRAIN_LOG_INTERVAL),
        test_log_interval=int(RUN_TEST_LOG_INTERVAL),
        progress_step_interval=int(RUN_PROGRESS_STEP_INTERVAL),
        max_workers=max_workers,
        max_in_flight=max_in_flight,
        data_workers=int(RUN_DATA_WORKERS),
        cpu_threads_per_worker=int(RUN_CPU_THREADS_PER_WORKER),
        cpu_cores=cpu_cores,
        output_root=str(RUN_OUTPUT_ROOT),
        data_root=str(RUN_DATA_ROOT),
        seed=int(RUN_SEED),
        subset_seed=int(RUN_SUBSET_SEED),
        suffix=RUN_SUFFIX,
        overwrite=bool(RUN_OVERWRITE),
        use_cuda=bool(RUN_USE_CUDA),
        skip_run=bool(RUN_SKIP_RUN),
        skip_plots=bool(RUN_SKIP_PLOTS),
        smoke_test=False,
    )


def main() -> None:
    start_time = datetime.now()
    cfg = build_config()

    if cfg.use_cuda and cfg.max_workers > 1:
        raise ValueError("use_cuda=True only supports max_workers=1 for now.")

    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.cpu_threads_per_worker))
    os.environ.setdefault("MKL_NUM_THREADS", str(cfg.cpu_threads_per_worker))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cfg.cpu_threads_per_worker))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cfg.cpu_threads_per_worker))

    apply_cpu_affinity(cfg.cpu_cores)
    random.seed(cfg.seed)

    cache_key = build_cache_key(
        p_values=cfg.p_values,
        batch_sizes=cfg.batch_sizes,
        train_samples_per_class_values=cfg.train_samples_per_class_values,
        seeds=cfg.seeds,
        width=cfg.width,
        reference_epochs=cfg.reference_epochs,
        learning_rate=cfg.learning_rate,
        train_log_interval=cfg.train_log_interval,
        test_log_interval=cfg.test_log_interval,
    )
    suffix_tag = f"_{cfg.suffix}" if cfg.suffix else ""
    output_root = Path(cfg.output_root).resolve()
    cache_dir = output_root / f"{cache_key}{suffix_tag}"
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "data").mkdir(parents=True, exist_ok=True)
    (cache_dir / "figures").mkdir(parents=True, exist_ok=True)
    (cache_dir / "runs").mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_root / "latest.json",
        {
            "timestamp": timestamp(),
            "cache_key": cache_key,
            "suffix": cfg.suffix,
            "cache_dir": str(cache_dir),
        },
    )

    write_metadata(cache_dir, cfg, cache_key, len(build_specs(cfg)))

    print(f"[{timestamp()}] output cache dir: {cache_dir}", flush=True)
    print(
        f"[{timestamp()}] |p|={len(cfg.p_values)} |B|={len(cfg.batch_sizes)} "
        f"|n_c|={len(cfg.train_samples_per_class_values)} |seeds|={len(cfg.seeds)}",
        flush=True,
    )

    if not cfg.skip_run:
        execute_runs(cache_dir, cfg, cache_key)

    run_rows = load_all_run_rows(cache_dir / "runs")
    if run_rows:
        write_csv(cache_dir / "summary.csv", build_summary_rows(run_rows))
        write_csv(cache_dir / "data" / f"results_per_run_{cache_key}{suffix_tag}.csv", build_per_run_rows(run_rows))
        write_csv(cache_dir / "data" / f"results_summary_{cache_key}{suffix_tag}.csv", build_summary_rows(run_rows))
        print(f"[{timestamp()}] wrote summary data for {len(run_rows)} runs", flush=True)
    else:
        print(f"[{timestamp()}] no run JSON files found under {cache_dir / 'runs'}", flush=True)

    if not cfg.skip_plots and run_rows:
        write_plots(cache_dir, run_rows)
        print(f"[{timestamp()}] wrote figures under {cache_dir / 'figures'}", flush=True)

    elapsed = datetime.now() - start_time
    runtime_log_path = cache_dir / "data" / f"runtime_log_{Path(__file__).stem}.csv"
    append_runtime_log(
        runtime_log_path,
        {
            "timestamp_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": int(elapsed.total_seconds()),
            "script": Path(__file__).name,
            "cache_key": cache_key,
            "output_dir": str(cache_dir),
            "total_runs_recorded": len(run_rows),
        },
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Elapsed: {elapsed}", flush=True)


if __name__ == "__main__":
    main()
