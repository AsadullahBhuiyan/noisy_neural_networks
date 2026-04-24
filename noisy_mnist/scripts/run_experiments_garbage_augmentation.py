from _path_setup import configure_runtime
PROJECT_ROOT = configure_runtime()

import os
import sys

# Set this once and it applies both before imports and in runtime config.
cpu_max = 20
CPU_CORES_DEFAULT = list(range(30, 30+cpu_max))


def _apply_cpu_affinity_early(cores: list[int] | None) -> None:
    if not cores:
        return
    try:
        os.sched_setaffinity(0, set(cores))
    except Exception:
        # Keep startup robust on platforms without sched_setaffinity.
        pass


if CPU_CORES_DEFAULT:
    # Keep BLAS/OpenMP thread pools inside the pinned core set.
    _n_threads = str(len(set(CPU_CORES_DEFAULT)))
    os.environ.setdefault("OMP_NUM_THREADS", _n_threads)
    os.environ.setdefault("MKL_NUM_THREADS", _n_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", _n_threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", _n_threads)
_apply_cpu_affinity_early(CPU_CORES_DEFAULT)

import csv
import math
import random
import statistics
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets
from tqdm import tqdm

# Allow running from either repo root or scripts/ directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.nnet_models import apply_corruption, build_model, evaluate, train_one_epoch


class TensorWithIntLabels(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.x[idx], int(self.y[idx].item())


def _collate_xy(batch: list[tuple[torch.Tensor, int | torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for x, y in batch:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        xs.append(x)
        if isinstance(y, torch.Tensor):
            ys.append(int(y.item()))
        else:
            ys.append(int(y))
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_runtime_log(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def apply_cpu_affinity(cores: list[int] | None) -> None:
    if not cores:
        return
    try:
        os.sched_setaffinity(0, set(cores))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Pinned process to CPU cores: {sorted(set(cores))}")
    except AttributeError:
        print("CPU affinity is not supported on this platform.")
    except OSError as exc:
        print(f"Failed to set CPU affinity to {cores}: {exc}")


def _safe_to_tensor(img) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    return x


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _stderr_from_values(values: list[float]) -> float:
    if len(values) <= 1:
        return float("inf")
    return statistics.pstdev(values) / math.sqrt(len(values))


def _build_noisy_addition(
    noise_source: Dataset,
    n_noise: int,
    seed: int,
    corruption_p: float,
) -> Dataset:
    if n_noise <= 0:
        return TensorWithIntLabels(
            torch.empty(0, 1, 28, 28),
            torch.empty(0, dtype=torch.long),
        )

    _set_seed(seed)
    source_idx = torch.randint(low=0, high=len(noise_source), size=(n_noise,))

    xs = []
    ys = []
    for idx in source_idx.tolist():
        x, y = noise_source[idx]
        xs.append(x)
        ys.append(y)

    x_clean = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    # Tunable replacement-channel intensity.
    x_noisy = apply_corruption(x_clean, mode="replacement", p=corruption_p, sigma=0.0)
    return TensorWithIntLabels(x_noisy, y)


def _train_once(
    noise_source: Dataset,
    test_subset: Subset,
    clean_subset: Subset,
    noise_ratio: float,
    corruption_p: float,
    clean_train_size: int,
    seed: int,
    activation: str,
    width: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    loss_type: str,
    num_workers: int,
    use_cuda: bool,
    show_epoch_pbar: bool,
) -> dict:
    _set_seed(seed)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    n_noise = int(round(noise_ratio * clean_train_size))
    noisy_addition = _build_noisy_addition(
        noise_source,
        n_noise=n_noise,
        seed=seed + 9973,
        corruption_p=corruption_p,
    )
    if n_noise > 0:
        train_set = ConcatDataset([clean_subset, noisy_addition])
    else:
        train_set = clean_subset

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=_collate_xy,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=_collate_xy,
    )

    model = build_model("mlp", activation=activation, mlp_hidden_sizes=[width]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    last_train_loss = math.nan
    for _ in tqdm(
        range(epochs),
        desc=f"epochs (ratio={noise_ratio:.3f})",
        unit="epoch",
        leave=False,
        disable=not show_epoch_pbar or epochs <= 1,
    ):
        last_train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_type)
    test_loss, test_acc = evaluate(model, test_loader, device, loss_type)

    return {
        "activation": activation,
        "model_type": "mlp",
        "loss_type": loss_type,
        "noise_ratio": noise_ratio,
        "corruption_p": corruption_p,
        "clean_train_size": clean_train_size,
        "noisy_train_size": n_noise,
        "total_train_size": clean_train_size + n_noise,
        "seed": seed,
        "train_loss": last_train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }


def _train_once_worker(
    use_train_pool_split_fraction: bool,
    clean_indices: list[int],
    test_indices: list[int],
    noise_source_indices: list[int],
    noise_ratio: float,
    corruption_p: float,
    clean_train_size: int,
    seed: int,
    activation: str,
    width: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    loss_type: str,
    num_workers: int,
    use_cuda: bool,
    show_epoch_pbar: bool,
) -> dict:
    train_base = datasets.MNIST(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=True,
        download=True,
        transform=_safe_to_tensor,
    )
    if use_train_pool_split_fraction:
        test_base = train_base
    else:
        test_base = datasets.MNIST(
            root=os.path.join(PROJECT_ROOT, "data"),
            train=False,
            download=True,
            transform=_safe_to_tensor,
        )

    clean_subset = Subset(train_base, clean_indices)
    test_subset = Subset(test_base, test_indices)
    noise_source = Subset(train_base, noise_source_indices)

    return _train_once(
        noise_source=noise_source,
        test_subset=test_subset,
        clean_subset=clean_subset,
        noise_ratio=noise_ratio,
        corruption_p=corruption_p,
        clean_train_size=clean_train_size,
        seed=seed,
        activation=activation,
        width=width,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss_type=loss_type,
        num_workers=num_workers,
        use_cuda=use_cuda,
        show_epoch_pbar=show_epoch_pbar,
    )


def _init_worker(cpu_cores: list[int] | None, cpu_threads_per_worker: int) -> None:
    apply_cpu_affinity(cpu_cores)
    try:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(max(1, cpu_threads_per_worker))
    except Exception:
        pass


def summarize_runs(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        key = (float(row["corruption_p"]), float(row["noise_ratio"]))
        grouped.setdefault(key, []).append(row)

    out = []
    for corruption_p, ratio in sorted(grouped.keys()):
        items = grouped[(corruption_p, ratio)]
        accs = [float(r["test_accuracy"]) for r in items]
        test_losses = [float(r["test_loss"]) for r in items]
        train_losses = [float(r["train_loss"]) for r in items]
        n_noise = int(items[0]["noisy_train_size"])
        clean_size = int(items[0]["clean_train_size"])
        is_clean_benchmark = any(bool(r.get("is_clean_benchmark_once", False)) for r in items)
        repeats = 0 if is_clean_benchmark else len(items)
        out.append(
            {
                "corruption_p": corruption_p,
                "noise_ratio": ratio,
                "clean_train_size": clean_size,
                "noisy_train_size": n_noise,
                "total_train_size": clean_size + n_noise,
                "repeats": repeats,
                "mean_test_accuracy": statistics.mean(accs),
                "std_test_accuracy": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "stderr_test_accuracy": (
                    statistics.pstdev(accs) / math.sqrt(len(accs)) if len(accs) > 1 else 0.0
                ),
                "mean_test_loss": statistics.mean(test_losses),
                "mean_train_loss": statistics.mean(train_losses),
            }
        )
    return out


def write_summary_pdf(path: str, summary_rows: list[dict], metadata: dict) -> None:
    if not summary_rows:
        return
    by_p: dict[float, list[dict]] = {}
    for row in summary_rows:
        by_p.setdefault(float(row["corruption_p"]), []).append(row)

    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for corruption_p in sorted(by_p):
            items = sorted(by_p[corruption_p], key=lambda r: float(r["noise_ratio"]))
            x = [float(r["noise_ratio"]) for r in items]
            means = [float(r["mean_test_accuracy"]) for r in items]
            errs = [float(r["stderr_test_accuracy"]) for r in items]
            errs_plot = [0.0 if abs(xi) < 1e-12 else max(ei, 1e-6) for xi, ei in zip(x, errs)]
            ax.errorbar(
                x,
                means,
                yerr=errs_plot,
                marker=".",
                linewidth=1.5,
                capsize=3,
                label=f"p={corruption_p:.2f}",
            )
        ax.set_title("Garbage augmentation: mean test accuracy vs noisy-addition ratio")
        ax.set_xlabel("noisy_size / clean_size")
        ax.set_ylabel("mean test accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for corruption_p in sorted(by_p):
            items = sorted(by_p[corruption_p], key=lambda r: float(r["noise_ratio"]))
            x = [float(r["noise_ratio"]) for r in items]
            stds = [float(r["std_test_accuracy"]) for r in items]
            ax.plot(x, stds, marker=".", linewidth=1.5, label=f"p={corruption_p:.2f}")
        ax.set_title("Garbage augmentation: std test accuracy vs noisy-addition ratio")
        ax.set_xlabel("noisy_size / clean_size")
        ax.set_ylabel("std test accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for corruption_p in sorted(by_p):
            items = sorted(by_p[corruption_p], key=lambda r: float(r["noise_ratio"]))
            x = [float(r["noise_ratio"]) for r in items]
            mean_test_losses = [float(r["mean_test_loss"]) for r in items]
            ax.plot(x, mean_test_losses, marker=".", linewidth=1.5, label=f"p={corruption_p:.2f}")
        ax.set_title("Garbage augmentation: mean test loss vs noisy-addition ratio")
        ax.set_xlabel("noisy_size / clean_size")
        ax.set_ylabel("mean test loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for corruption_p in sorted(by_p):
            items = sorted(by_p[corruption_p], key=lambda r: float(r["noise_ratio"]))
            x = [float(r["noise_ratio"]) for r in items]
            mean_train_losses = [float(r["mean_train_loss"]) for r in items]
            ax.plot(x, mean_train_losses, marker=".", linewidth=1.5, label=f"p={corruption_p:.2f}")
        ax.set_title("Garbage augmentation: mean train loss vs noisy-addition ratio")
        ax.set_xlabel("noisy_size / clean_size")
        ax.set_ylabel("mean train loss (final epoch)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        meta_lines = [f"{k}: {v}" for k, v in metadata.items()]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.95, "Run Metadata", fontsize=14, fontweight="bold", va="top")
        fig.text(0.05, 0.9, "\n".join(meta_lines), fontsize=10, va="top")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    start_time = datetime.now()

    # Manual configuration.
    activation = "relu"
    width = 128
    clean_train_size = 30_000
    use_train_pool_split_fraction = False
    test_fraction = 1.0 / 2.0
    test_size = 10_000
    noise_size_ratios = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32]
    corruption_ps = [0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
    epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "cross_entropy"
    num_workers = 0
    cpu_threads_per_worker = 1
    use_cuda = False
    cpu_cores = CPU_CORES_DEFAULT  # Example: [0, 1, 2, 3]
    show_epoch_pbar = True
    split_seed = 1234
    seed = 1234

    min_repeats = 10
    max_repeats = 25
    stderr_target = 1e-2
    max_workers = cpu_max
    max_in_flight = max(1, 9 * max_workers // 10)
    if max_in_flight < max_workers:
        max_in_flight = max_workers
    if use_cuda and max_workers > 1:
        raise ValueError("use_cuda=True only supports max_workers=1.")
    if any((p < 0.0 or p > 1.0) for p in corruption_ps):
        raise ValueError("All corruption_ps must be in [0, 1].")

    output_data_dir = os.path.join("results", "data")
    output_fig_dir = os.path.join("results", "figures")
    suffix = "garbage_augmentation_mlp_relu"

    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_fig_dir, exist_ok=True)
    apply_cpu_affinity(cpu_cores)
    torch.set_num_threads(max(1, cpu_threads_per_worker))
    torch.set_num_interop_threads(1)

    _set_seed(split_seed)
    train_base = datasets.MNIST(root="./data", train=True, download=True, transform=_safe_to_tensor)
    if use_train_pool_split_fraction:
        if not (0.0 < test_fraction < 1.0):
            raise ValueError("test_fraction must be in (0, 1) when use_train_pool_split_fraction=True.")
        perm = torch.randperm(len(train_base), generator=torch.Generator().manual_seed(split_seed))
        n_test_from_train = max(1, int(round(test_fraction * len(train_base))))
        n_clean_pool = len(train_base) - n_test_from_train
        if clean_train_size > n_clean_pool:
            raise ValueError(
                f"clean_train_size={clean_train_size} exceeds available clean pool={n_clean_pool} "
                f"after reserving test split with test_fraction={test_fraction:.3f}."
            )
        clean_indices = perm[:n_clean_pool].tolist()
        test_indices = perm[n_clean_pool:].tolist()
        clean_subset = Subset(train_base, clean_indices[:clean_train_size])
        test_subset = Subset(train_base, test_indices)
        effective_test_size = len(test_subset)
        split_tag = f"tf{test_fraction:.3f}_src-train"
    else:
        test_base = datasets.MNIST(root="./data", train=False, download=True, transform=_safe_to_tensor)
        clean_train_size = max(1, min(clean_train_size, len(train_base)))
        test_size = max(1, min(test_size, len(test_base)))
        train_perm = torch.randperm(len(train_base), generator=torch.Generator().manual_seed(split_seed))
        test_perm = torch.randperm(len(test_base), generator=torch.Generator().manual_seed(split_seed + 1))
        clean_subset = Subset(train_base, train_perm[:clean_train_size].tolist())
        test_subset = Subset(test_base, test_perm[:test_size].tolist())
        effective_test_size = len(test_subset)
        split_tag = f"nte{effective_test_size}"
    clean_indices = list(clean_subset.indices)  # type: ignore[attr-defined]
    test_indices = list(test_subset.indices)  # type: ignore[attr-defined]
    noise_source_indices = list(clean_indices)

    cache_key = (
        f"mlpga_rmax{max_repeats}_rmin{min_repeats}_se{stderr_target:.0e}_"
        f"e{epochs}_nclean{clean_train_size}_{split_tag}_"
        f"ratio{min(noise_size_ratios):.2f}-{max(noise_size_ratios):.2f}_"
        f"cp{min(corruption_ps):.3f}-{max(corruption_ps):.3f}_"
        f"w{width}_d1_loss-{loss_type}"
    )
    suffix_tag = f"_{suffix}" if suffix else ""

    all_rows: list[dict] = []
    random.seed(seed)
    nonzero_ratios = [float(r) for r in noise_size_ratios if abs(float(r)) > 1e-12]
    p_outer_pbar = tqdm(corruption_ps, desc=f"corruption p sweep{suffix_tag}", unit="p")

    for corruption_p in p_outer_pbar:
        p_outer_pbar.set_postfix_str(f"p={corruption_p:.3f}")
        ratio_pbar = tqdm(
            noise_size_ratios,
            desc=f"noise ratio sweep p={corruption_p:.3f}{suffix_tag}",
            unit="ratio",
            leave=False,
        )
        for ratio in ratio_pbar:
            if abs(float(ratio)) < 1e-12:
                # Clean benchmark once per p; report repeats=0 in summary by convention.
                run_seed = split_seed
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{timestamp}] start: clean benchmark p={corruption_p:.3f} "
                    f"ratio=0.000 seed={run_seed}"
                )
                row = _train_once(
                    noise_source=clean_subset,
                    test_subset=test_subset,
                    clean_subset=clean_subset,
                    noise_ratio=0.0,
                    corruption_p=corruption_p,
                    clean_train_size=clean_train_size,
                    seed=run_seed,
                    activation=activation,
                    width=width,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    loss_type=loss_type,
                    num_workers=num_workers,
                    use_cuda=use_cuda,
                    show_epoch_pbar=show_epoch_pbar,
                )
                row["is_clean_benchmark_once"] = True
                all_rows.append(row)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{timestamp}] done: clean benchmark p={corruption_p:.3f} ratio=0.000 "
                    f"acc={row['test_accuracy']:.4f}"
                )
                ratio_pbar.set_postfix_str("ratio=0.000 repeats=0 stderr=0.0000")
        ratio_pbar.close()

        if nonzero_ratios:
            per_ratio_rows: dict[float, list[dict]] = {ratio: [] for ratio in nonzero_ratios}
            queue = deque(nonzero_ratios)
            in_flight: dict = {}
            run_start_times: dict = {}
            total_submitted = 0
            pbar = tqdm(
                total=0,
                desc=f"nonzero ratio runs p={corruption_p:.3f}{suffix_tag}",
                unit="run",
                leave=False,
            )

            def _needs_more(ratio: float) -> bool:
                rows = per_ratio_rows[ratio]
                n = len(rows)
                if n < min_repeats:
                    return True
                if n >= max_repeats:
                    return False
                accs = [float(r["test_accuracy"]) for r in rows]
                return _stderr_from_values(accs) > stderr_target

            def _submit_ratio(executor: ProcessPoolExecutor, ratio: float) -> None:
                nonlocal total_submitted
                run_seed = random.randint(1, 1_000_000_000)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{timestamp}] start: p={corruption_p:.3f} ratio={ratio:.3f} "
                    f"repeat={len(per_ratio_rows[ratio]) + 1} seed={run_seed}"
                )
                fut = executor.submit(
                    _train_once_worker,
                    use_train_pool_split_fraction=use_train_pool_split_fraction,
                    clean_indices=clean_indices,
                    test_indices=test_indices,
                    noise_source_indices=noise_source_indices,
                    noise_ratio=ratio,
                    corruption_p=corruption_p,
                    clean_train_size=clean_train_size,
                    seed=run_seed,
                    activation=activation,
                    width=width,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    loss_type=loss_type,
                    num_workers=num_workers,
                    use_cuda=use_cuda,
                    show_epoch_pbar=show_epoch_pbar,
                )
                in_flight[fut] = (ratio, run_seed)
                run_start_times[fut] = datetime.now()
                total_submitted += 1
                pbar.total = total_submitted
                pbar.refresh()

            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(cpu_cores, cpu_threads_per_worker),
            ) as executor:
                while queue or in_flight:
                    while queue and len(in_flight) < max_in_flight:
                        _submit_ratio(executor, queue.popleft())

                    for fut in as_completed(in_flight):
                        ratio, run_seed = in_flight[fut]
                        row = fut.result()
                        row["seed"] = run_seed
                        all_rows.append(row)
                        per_ratio_rows[ratio].append(row)

                        accs = [float(r["test_accuracy"]) for r in per_ratio_rows[ratio]]
                        stderr_now = _stderr_from_values(accs)
                        elapsed = datetime.now() - run_start_times.get(fut, datetime.now())
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(
                            f"[{timestamp}] done: p={corruption_p:.3f} ratio={ratio:.3f} "
                            f"repeats={len(per_ratio_rows[ratio])} acc={row['test_accuracy']:.4f} "
                            f"stderr={stderr_now:.4f} elapsed={elapsed}"
                        )
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"p={corruption_p:.3f} ratio={ratio:.3f} "
                            f"repeats={len(per_ratio_rows[ratio])} stderr={stderr_now:.4f}"
                        )

                        if _needs_more(ratio):
                            queue.append(ratio)

                        del in_flight[fut]
                        run_start_times.pop(fut, None)
                        break

            pbar.close()

    p_outer_pbar.close()

    summary_rows = summarize_runs(all_rows)
    per_run_path = os.path.join(output_data_dir, f"results_per_run_{cache_key}{suffix_tag}.csv")
    summary_path = os.path.join(output_data_dir, f"results_summary_{cache_key}{suffix_tag}.csv")
    pdf_path = os.path.join(output_fig_dir, f"accuracy_vs_{cache_key}{suffix_tag}.pdf")
    write_csv(per_run_path, all_rows)
    write_csv(summary_path, summary_rows)

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "activation": activation,
        "width": width,
        "depth": 1,
        "clean_train_size": clean_train_size,
        "corruption_ps": corruption_ps,
        "use_train_pool_split_fraction": use_train_pool_split_fraction,
        "test_fraction": test_fraction if use_train_pool_split_fraction else None,
        "test_size": effective_test_size,
        "noise_size_ratios": noise_size_ratios,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "stderr_target": stderr_target,
        "min_repeats": min_repeats,
        "max_repeats": max_repeats,
        "cpu_cores": cpu_cores,
        "cpu_threads_per_worker": cpu_threads_per_worker,
        "max_workers": max_workers,
        "max_in_flight": max_in_flight,
        "total_runs": len(all_rows),
    }
    write_summary_pdf(pdf_path, summary_rows, metadata)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Saved:")
    print(f"  {per_run_path}")
    print(f"  {summary_path}")
    print(f"  {pdf_path}")

    elapsed = datetime.now() - start_time
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    runtime_log_path = os.path.join(output_data_dir, f"runtime_log_{script_name}.csv")
    append_runtime_log(
        runtime_log_path,
        {
            "timestamp_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": int(elapsed.total_seconds()),
            "script": script_name,
            "cache_key": cache_key,
            "total_runs": len(all_rows),
        },
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Elapsed: {elapsed}")


if __name__ == "__main__":
    main()
