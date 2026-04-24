#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import random
from collections import deque
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

import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import run_experiments_2d_phase_diagram as base


DEFAULT_WIDTHS = [64, 128, 256, 512, 1024]
DEFAULT_REFERENCE_WIDTH = 512
DEFAULT_FIXED_TRAIN_SAMPLES_PER_CLASS = base.BALANCED_TRAIN_PER_CLASS


# User config
RUN_P_VALUES = list(base.DEFAULT_P_VALUES)
RUN_BATCH_SIZES = list(base.DEFAULT_BATCH_SIZES)
RUN_WIDTHS = list(DEFAULT_WIDTHS)
RUN_SEEDS = list(base.DEFAULT_SEEDS)
RUN_FIXED_TRAIN_SAMPLES_PER_CLASS = DEFAULT_FIXED_TRAIN_SAMPLES_PER_CLASS
RUN_REFERENCE_EPOCHS = base.DEFAULT_REFERENCE_EPOCHS
RUN_LEARNING_RATE = base.DEFAULT_LEARNING_RATE
RUN_TRAIN_LOG_INTERVAL = 100
RUN_TEST_LOG_INTERVAL = 500
RUN_PROGRESS_STEP_INTERVAL = 1000
RUN_MAX_WORKERS = base.cpu_max
RUN_MAX_IN_FLIGHT = max(1, 9 * base.cpu_max // 10)
RUN_DATA_WORKERS = 0
RUN_CPU_THREADS_PER_WORKER = 1
RUN_CPU_CORE_START = 67
RUN_OUTPUT_ROOT = str(PROJECT_ROOT / "phase_diagram_2d" / "width_sweep")
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
    width: int
    seed: int


@dataclass
class ExperimentConfig:
    p_values: list[float]
    batch_sizes: list[int]
    widths: list[int]
    seeds: list[int]
    fixed_train_samples_per_class: int
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


def build_cache_key(
    p_values: list[float],
    batch_sizes: list[int],
    widths: list[int],
    seeds: list[int],
    fixed_train_samples_per_class: int,
    reference_epochs: int,
    learning_rate: float,
    train_log_interval: int,
    test_log_interval: int,
) -> str:
    p_min, p_max = min(p_values), max(p_values)
    b_min, b_max = min(batch_sizes), max(batch_sizes)
    w_min, w_max = min(widths), max(widths)
    return (
        f"phase2d_width_{base.BALANCED_DATASET_TAG}_r{len(seeds)}_e{reference_epochs}_"
        f"p{p_min:.4f}-{p_max:.4f}_"
        f"b{b_min}-{b_max}_"
        f"w{w_min}-{w_max}_"
        f"tpc{fixed_train_samples_per_class}_lr{learning_rate:.4g}_"
        f"logtr{train_log_interval}_logte{test_log_interval}"
    )


def run_json_path(runs_dir: Path, spec: RunSpec) -> Path:
    return runs_dir / f"p{spec.p:.4f}_B{spec.batch_size}_W{spec.width}_seed{spec.seed}.json"


def make_run_payload(
    spec: RunSpec,
    *,
    fixed_train_samples_per_class: int,
    actual_train_samples_per_class: int,
    actual_test_samples_per_class: int,
    actual_n: int,
    actual_test_n: int,
    effective_batch_size: int,
    total_samples_seen: int,
    num_steps: int,
    final_test_accuracy: float,
    final_test_loss: float,
    train_loss_curve: list[list[float]],
    test_acc_curve: list[list[float]],
) -> dict[str, Any]:
    return {
        "p": float(spec.p),
        "batch_size": int(spec.batch_size),
        "effective_batch_size": int(effective_batch_size),
        "width": int(spec.width),
        "requested_train_samples_per_class": int(fixed_train_samples_per_class),
        "train_samples_per_class": int(actual_train_samples_per_class),
        "N": int(actual_n),
        "requested_N": int(base.total_train_size(fixed_train_samples_per_class)),
        "test_samples_per_class": int(actual_test_samples_per_class),
        "test_N": int(actual_test_n),
        "seed": int(spec.seed),
        "total_samples_seen": int(total_samples_seen),
        "num_steps": int(num_steps),
        "final_test_accuracy": float(final_test_accuracy),
        "final_test_loss": float(final_test_loss),
        "train_loss_curve": train_loss_curve,
        "test_acc_curve": test_acc_curve,
    }


def _init_worker(cpu_cores: list[int], cfg_dict: dict[str, Any]) -> None:
    base.apply_cpu_affinity(cpu_cores)
    try:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
    except Exception:
        pass
    cfg = ExperimentConfig(**cfg_dict)
    base.GLOBAL_STATE.clear()
    base.get_worker_train_dataset(cfg.data_root)
    base.get_worker_test_dataset(cfg.data_root)


def run_single_spec(spec: RunSpec, cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = ExperimentConfig(**cfg_dict)
    device = base.choose_device(cfg.use_cuda)
    base.set_seed(spec.seed, cfg.use_cuda)

    train_subset = base.build_train_subset(cfg.data_root, cfg.fixed_train_samples_per_class, cfg.subset_seed)
    test_dataset = base.build_test_subset(cfg.data_root, cfg.subset_seed)
    actual_n_train = len(train_subset)
    actual_n_test = len(test_dataset)
    actual_train_samples_per_class = base.per_class_from_total(actual_n_train)
    actual_test_samples_per_class = base.per_class_from_total(actual_n_test)

    effective_batch_size = base.compute_effective_batch_size(spec.batch_size, actual_n_train)
    total_samples_seen = base.compute_total_samples_seen(actual_n_train, cfg.reference_epochs)
    total_steps = base.compute_num_steps(total_samples_seen, effective_batch_size)
    steps_per_epoch = base.compute_num_steps(actual_n_train, effective_batch_size)
    test_interval = max(1, min(cfg.test_log_interval, steps_per_epoch))
    progress_interval = max(1, cfg.progress_step_interval)

    noise_seed = spec.seed * 1000 + 17
    shuffle_seed = spec.seed * 1000 + 31

    train_dataset = base.SaltAndPepperDataset(train_subset, p=spec.p, seed=noise_seed)
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

    model = base.SingleHiddenMLP(spec.width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    train_iter = iter(train_loader)
    step = 0
    samples_seen = 0
    loss_sum = 0.0
    loss_weight = 0
    train_loss_curve: list[list[float]] = []
    test_acc_curve: list[list[float]] = []

    print(
        f"[{base.timestamp()}] start: p={spec.p:.4f} B={spec.batch_size} W={spec.width} "
        f"tpc={actual_train_samples_per_class} N={actual_n_train} "
        f"seed={spec.seed} effective_B={effective_batch_size} steps={total_steps}",
        flush=True,
    )

    step_pbar = None
    if cfg.max_workers <= 1:
        step_pbar = tqdm(
            total=total_steps,
            desc=f"steps p={spec.p:.4f} B={spec.batch_size} W={spec.width} tpc={actual_train_samples_per_class}",
            unit="step",
            leave=False,
        )

    started_at = datetime.now()
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
            if base.should_log(step, total_steps, cfg.train_log_interval):
                step_pbar.set_postfix_str(f"loss={loss.item():.4f}")

        if base.should_log(step, total_steps, cfg.train_log_interval):
            train_loss_curve.append([int(step), float(loss_sum / max(1, loss_weight))])
            loss_sum = 0.0
            loss_weight = 0

        if base.should_log(step, total_steps, test_interval):
            _, test_acc = base.evaluate_clean(model, test_loader, device)
            test_acc_curve.append([int(step), float(test_acc)])

        if base.should_log(step, total_steps, progress_interval):
            elapsed = datetime.now() - started_at
            print(
                f"[{base.timestamp()}] progress: p={spec.p:.4f} B={spec.batch_size} W={spec.width} "
                f"step={step}/{total_steps} samples={samples_seen}/{total_samples_seen} elapsed={elapsed}",
                flush=True,
            )

    final_test_loss, final_test_accuracy = base.evaluate_clean(model, test_loader, device)
    if step_pbar is not None:
        step_pbar.close()
    if not test_acc_curve or int(test_acc_curve[-1][0]) != total_steps:
        test_acc_curve.append([int(total_steps), float(final_test_accuracy)])

    print(
        f"[{base.timestamp()}] done: p={spec.p:.4f} B={spec.batch_size} W={spec.width} "
        f"acc={final_test_accuracy:.4f} loss={final_test_loss:.4f}",
        flush=True,
    )
    return make_run_payload(
        spec,
        fixed_train_samples_per_class=cfg.fixed_train_samples_per_class,
        actual_train_samples_per_class=actual_train_samples_per_class,
        actual_test_samples_per_class=actual_test_samples_per_class,
        actual_n=actual_n_train,
        actual_test_n=actual_n_test,
        effective_batch_size=effective_batch_size,
        total_samples_seen=total_samples_seen,
        num_steps=total_steps,
        final_test_accuracy=final_test_accuracy,
        final_test_loss=final_test_loss,
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
                "width": int(row["width"]),
                "train_samples_per_class": int(row.get("train_samples_per_class", base.per_class_from_total(row["N"]))),
                "requested_train_samples_per_class": int(
                    row.get(
                        "requested_train_samples_per_class",
                        base.per_class_from_total(row.get("requested_N", row["N"])),
                    )
                ),
                "N": int(row["N"]),
                "requested_N": int(row.get("requested_N", row["N"])),
                "test_samples_per_class": int(
                    row.get("test_samples_per_class", base.per_class_from_total(row.get("test_N", 0)))
                ),
                "test_N": int(row.get("test_N", 0)),
                "seed": int(row["seed"]),
                "final_test_accuracy": float(row["final_test_accuracy"]),
                "final_test_loss": float(row["final_test_loss"]),
            }
        )
    rows.sort(key=lambda item: (item["width"], item["batch_size"], item["p"], item["seed"]))
    return rows


def build_per_run_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in run_rows:
        rows.append(
            {
                "p": float(row["p"]),
                "batch_size": int(row["batch_size"]),
                "effective_batch_size": int(row["effective_batch_size"]),
                "width": int(row["width"]),
                "train_samples_per_class": int(row.get("train_samples_per_class", base.per_class_from_total(row["N"]))),
                "requested_train_samples_per_class": int(
                    row.get(
                        "requested_train_samples_per_class",
                        base.per_class_from_total(row.get("requested_N", row["N"])),
                    )
                ),
                "N": int(row["N"]),
                "requested_N": int(row.get("requested_N", row["N"])),
                "test_samples_per_class": int(
                    row.get("test_samples_per_class", base.per_class_from_total(row.get("test_N", 0)))
                ),
                "test_N": int(row.get("test_N", 0)),
                "seed": int(row["seed"]),
                "total_samples_seen": int(row["total_samples_seen"]),
                "num_steps": int(row["num_steps"]),
                "final_test_accuracy": float(row["final_test_accuracy"]),
                "final_test_loss": float(row["final_test_loss"]),
            }
        )
    rows.sort(key=lambda item: (item["width"], item["batch_size"], item["p"], item["seed"]))
    return rows


def load_all_run_rows(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []
    rows = [base.load_json(path) for path in sorted(runs_dir.glob("*.json"))]
    rows.sort(key=lambda row: (row["width"], row["batch_size"], row["p"], row["seed"]))
    return rows


def aggregate_by_mean(run_rows: list[dict[str, Any]]) -> dict[tuple[int, int, float], dict[str, float]]:
    grouped: dict[tuple[int, int, float], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (int(row["width"]), int(row["batch_size"]), float(row["p"]))
        grouped.setdefault(key, []).append(row)

    out: dict[tuple[int, int, float], dict[str, float]] = {}
    for key, rows in grouped.items():
        accs = [float(row["final_test_accuracy"]) for row in rows]
        losses = [float(row["final_test_loss"]) for row in rows]
        out[key] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "stderr_accuracy": float(np.std(accs) / math.sqrt(max(1, len(rows)))),
            "mean_loss": float(np.mean(losses)),
            "count": float(len(rows)),
        }
    return out


def build_heatmap_grid(
    aggregated: dict[tuple[int, int, float], dict[str, float]],
    width: int,
    batch_sizes: list[int],
    p_values: list[float],
) -> np.ndarray:
    grid = np.full((len(batch_sizes), len(p_values)), np.nan, dtype=float)
    for i, batch_size in enumerate(batch_sizes):
        for j, p in enumerate(p_values):
            stats = aggregated.get((int(width), int(batch_size), float(p)))
            if stats is not None:
                grid[i, j] = float(stats["mean_accuracy"])
    return grid


def transform_aggregated_for_scaling(
    aggregated: dict[tuple[int, int, float], dict[str, float]]
) -> dict[tuple[int, int, float], dict[str, float]]:
    transformed: dict[tuple[int, int, float], dict[str, float]] = {}
    for (width, batch_size, p), stats in aggregated.items():
        transformed[(int(width), int(batch_size), float(p))] = dict(stats)
    return transformed


def pick_representative_runs(run_rows: list[dict[str, Any]], reference_width: int) -> list[dict[str, Any]]:
    aggregated = aggregate_by_mean(run_rows)
    candidates = []
    for (width, batch_size, p), stats in aggregated.items():
        if int(width) != int(reference_width):
            continue
        candidates.append((int(batch_size), float(p), float(stats["mean_accuracy"])))

    above = sorted(
        [item for item in candidates if item[2] > base.PHASE_THRESHOLDS[0]],
        key=lambda item: abs(item[2] - base.PHASE_THRESHOLDS[0]),
    )
    below = sorted(
        [item for item in candidates if item[2] <= base.PHASE_THRESHOLDS[0]],
        key=lambda item: abs(item[2] - base.PHASE_THRESHOLDS[0]),
    )
    selected = []
    for candidate in above[:1] + below[:1]:
        if candidate not in selected:
            selected.append(candidate)

    out = []
    for batch_size, p, mean_acc in selected:
        matching = [
            row
            for row in run_rows
            if int(row["width"]) == int(reference_width)
            and int(row["batch_size"]) == int(batch_size)
            and abs(float(row["p"]) - float(p)) < 1e-12
        ]
        if not matching:
            continue
        out.append(min(matching, key=lambda row: abs(float(row["final_test_accuracy"]) - mean_acc)))
    return out


def write_plots(cache_dir: Path, run_rows: list[dict[str, Any]], cfg: ExperimentConfig) -> None:
    if not run_rows:
        return

    figures_dir = cache_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    aggregated = aggregate_by_mean(run_rows)
    scaling_aggregated = transform_aggregated_for_scaling(aggregated)
    widths = sorted({int(row["width"]) for row in run_rows})
    batch_sizes = sorted({int(row["batch_size"]) for row in run_rows})
    p_values = sorted({float(row["p"]) for row in run_rows})
    reference_width = DEFAULT_REFERENCE_WIDTH if DEFAULT_REFERENCE_WIDTH in widths else widths[len(widths) // 2]
    fixed_total_n = base.total_train_size(cfg.fixed_train_samples_per_class)

    grid_ref = build_heatmap_grid(aggregated, reference_width, batch_sizes, p_values)
    if np.isfinite(grid_ref).any():
        fig, ax = base.plt.subplots(figsize=(9, 5.5))
        mesh = base.add_heatmap(
            ax,
            batch_sizes,
            p_values,
            grid_ref,
            f"MNIST phase diagram (width = {reference_width}, {base.format_train_size_label(fixed_total_n)})",
            base.PHASE_THRESHOLDS[0],
        )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("mean clean test accuracy")
        fig.tight_layout()
        base.save_plot(fig, figures_dir / f"heatmap_W{reference_width}.pdf")
        base.plt.close(fig)

    fig, axes = base.plt.subplots(1, len(widths), figsize=(4.5 * len(widths), 4.5), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.14, top=0.88, wspace=0.30)
    mesh = None
    for idx, width in enumerate(widths):
        ax = axes[0, idx]
        grid = build_heatmap_grid(aggregated, width, batch_sizes, p_values)
        if not np.isfinite(grid).any():
            ax.set_axis_off()
            ax.set_title(f"W = {width} (no data)")
            continue
        mesh = base.add_heatmap(ax, batch_sizes, p_values, grid, f"W = {width}", base.PHASE_THRESHOLDS[0])
    if mesh is not None:
        cax = fig.add_axes([0.935, 0.18, 0.008, 0.64])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("mean clean test accuracy")
    base.save_plot(fig, figures_dir / "heatmap_by_width.pdf")
    base.plt.close(fig)

    with PdfPages(figures_dir / "phase_boundary_by_width.pdf") as pdf:
        for threshold in base.PHASE_THRESHOLDS:
            fig, ax = base.plt.subplots(figsize=(8, 5))
            annotation_lines = []
            for width in widths:
                pcs = []
                for batch_size in batch_sizes:
                    ps_present = []
                    accs = []
                    for p in p_values:
                        stats = aggregated.get((int(width), int(batch_size), float(p)))
                        if stats is not None:
                            ps_present.append(float(p))
                            accs.append(float(stats["mean_accuracy"]))
                    pc = base.estimate_crossing(ps_present, accs, threshold)
                    pcs.append(float("nan") if pc is None else float(pc))
                ax.plot(batch_sizes, pcs, marker="o", label=f"W={width}")
                fit = base.fit_phase_boundary(batch_sizes, pcs)
                if fit is not None:
                    batch_grid = np.geomspace(min(batch_sizes), max(batch_sizes), 300)
                    ax.plot(batch_grid, 1.0 - fit["a"] * np.power(batch_grid, -fit["beta"]), linestyle="--", alpha=0.8)
                    annotation_lines.append(
                        f"W={width}: a={fit['a']:.4g}, beta={fit['beta']:.4f}, R^2={fit['r2']:.4f}"
                    )
            ax.set_xscale("log")
            ax.set_xlabel("batch size")
            ax.set_ylabel(f"estimated critical p_c at acc={threshold:.2f}")
            ax.set_title(f"Phase boundary by width ({base.format_train_size_label(fixed_total_n)})")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            if annotation_lines:
                ax.text(
                    0.02,
                    0.02,
                    "\n".join(annotation_lines),
                    transform=ax.transAxes,
                    fontsize=8,
                    va="bottom",
                    ha="left",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
                )
            fig.tight_layout()
            pdf.savefig(fig)
            base.plt.close(fig)

    scaling_batches = [batch_size for batch_size in base.SCALING_BATCHES if batch_size in batch_sizes]
    if scaling_batches:
        with PdfPages(figures_dir / "width_scaling.pdf") as pdf:
            fig, axes = base.plt.subplots(1, len(scaling_batches), figsize=(5 * len(scaling_batches), 4), squeeze=False)
            for idx, batch_size in enumerate(scaling_batches):
                ax = axes[0, idx]
                for width in widths:
                    ps_present = []
                    accs = []
                    for p in p_values:
                        stats = aggregated.get((int(width), int(batch_size), float(p)))
                        if stats is not None:
                            ps_present.append(float(p))
                            accs.append(float(stats["mean_accuracy"]))
                    if ps_present:
                        ax.plot(ps_present, accs, marker="o", label=f"W={width}")
                ax.set_title(f"B = {batch_size}")
                ax.set_xlabel("p")
                ax.set_ylabel("mean clean test accuracy")
                ax.grid(True, alpha=0.3)
                ax.legend()
            fig.tight_layout()
            base.save_plot(fig, figures_dir / "width_scaling_curves.pdf")
            pdf.savefig(fig)
            base.plt.close(fig)

            fig, axes = base.plt.subplots(1, len(scaling_batches), figsize=(5 * len(scaling_batches), 4), squeeze=False)
            for idx, batch_size in enumerate(scaling_batches):
                ax = axes[0, idx]
                fit = base.find_best_full_collapse(
                    scaling_aggregated,
                    batch_size=batch_size,
                    n_values=widths,
                    p_values=p_values,
                )
                if fit is None:
                    ax.set_axis_off()
                    ax.set_title(f"B = {batch_size} (insufficient data)")
                    continue
                p_critical = float(fit["p_c"])
                nu = float(fit["nu"])
                a_critical = float(fit["A_c"])
                kappa = float(fit["kappa"])
                for width in widths:
                    curve = [
                        (float(p), float(scaling_aggregated[(int(width), int(batch_size), float(p))]["mean_accuracy"]))
                        for p in p_values
                        if (int(width), int(batch_size), float(p)) in scaling_aggregated
                    ]
                    if not curve:
                        continue
                    ps_np = np.asarray([item[0] for item in curve], dtype=float)
                    accs_np = np.asarray([item[1] for item in curve], dtype=float)
                    scaled_x = (ps_np - p_critical) * np.power(float(width), 1.0 / nu)
                    scaled_y = (accs_np - a_critical) * np.power(float(width), kappa)
                    order = np.argsort(scaled_x)
                    ax.plot(scaled_x[order], scaled_y[order], marker="o", label=f"W={width}")
                ax.set_title(
                    f"B = {batch_size}, p_c={p_critical:.4f}, nu={nu:.3f}, "
                    f"A_c={a_critical:.3f}, kappa={kappa:.3f}"
                )
                ax.set_xlabel(r"$(p-p_c)W^{1/\nu}$")
                ax.set_ylabel(r"$(A-A_c)W^{\kappa}$")
                ax.grid(True, alpha=0.3)
                ax.legend()
            fig.tight_layout()
            base.save_plot(fig, figures_dir / "width_collapse.pdf")
            pdf.savefig(fig)
            base.plt.close(fig)

            fig, axes = base.plt.subplots(1, len(scaling_batches), figsize=(5 * len(scaling_batches), 4), squeeze=False)
            for idx, batch_size in enumerate(scaling_batches):
                ax = axes[0, idx]
                result, scaled = base.find_pyfssa_collapse_zeta0_batch(
                    scaling_aggregated,
                    batch_size=batch_size,
                    n_values=widths,
                    p_values=p_values,
                )
                if scaled is None:
                    ax.set_axis_off()
                    ax.set_title(f"B = {batch_size} ({result.get('error', 'pyfssa failed')})")
                    continue
                for i, width in enumerate(result["n_values"]):
                    ax.plot(scaled.x[i], scaled.y[i], marker="o", linestyle="-", label=f"W={width}")
                ax.set_title(
                    f"B = {batch_size}, p_c={result['rho_c']:.4f}, nu={result['nu']:.3f}, S={result['S']:.4g}"
                )
                ax.set_xlabel(r"$(p-p_c)W^{1/\nu}$")
                ax.set_ylabel(r"$A(p)-A(p_c)$")
                ax.grid(True, alpha=0.3)
                ax.legend()
            fig.tight_layout()
            base.save_plot(fig, figures_dir / "width_collapse_pyfssa.pdf")
            pdf.savefig(fig)
            base.plt.close(fig)

    fig, ax = base.plt.subplots(figsize=(8, 5))
    for batch_size in [64, 256, 1024, 4096, 16384, fixed_total_n]:
        if batch_size not in batch_sizes:
            continue
        ps_present = []
        accs = []
        for p in p_values:
            stats = aggregated.get((int(reference_width), int(batch_size), float(p)))
            if stats is not None:
                ps_present.append(float(p))
                accs.append(float(stats["mean_accuracy"]))
        if ps_present:
            label = "full batch" if batch_size == fixed_total_n else f"B={batch_size}"
            ax.plot(ps_present, accs, marker="o", label=label)
    ax.axhline(base.PHASE_THRESHOLDS[0], color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_title(f"Full-batch vs minibatch (W = {reference_width}, {base.format_train_size_label(fixed_total_n)})")
    ax.set_xlabel("p")
    ax.set_ylabel("mean clean test accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    base.save_plot(fig, figures_dir / f"full_batch_vs_minibatch_W{reference_width}.pdf")
    base.plt.close(fig)

    chosen_runs = pick_representative_runs(run_rows, reference_width)
    if chosen_runs:
        fig, axes = base.plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for row in chosen_runs:
            label = (
                f"p={row['p']:.4f}, B={row['batch_size']}, W={row['width']}, "
                f"seed={row['seed']}, acc={row['final_test_accuracy']:.3f}"
            )
            train_curve = np.asarray(row["train_loss_curve"], dtype=float)
            test_curve = np.asarray(row["test_acc_curve"], dtype=float)
            if train_curve.size:
                axes[0].plot(train_curve[:, 0], train_curve[:, 1], label=label)
            if test_curve.size:
                axes[1].plot(test_curve[:, 0], test_curve[:, 1], label=label)
        axes[0].set_title(f"Training dynamics near the phase boundary (W = {reference_width})")
        axes[0].set_ylabel("train loss")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("gradient step")
        axes[1].set_ylabel("clean test accuracy")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        base.save_plot(fig, figures_dir / f"training_dynamics_W{reference_width}.pdf")
        base.plt.close(fig)


def run_priority_key(spec: RunSpec, fixed_total_n: int) -> tuple[int, int, int, float, int]:
    if spec.width == DEFAULT_REFERENCE_WIDTH:
        tier = 0
    elif spec.batch_size >= fixed_total_n:
        tier = 1
    else:
        tier = 2
    return (tier, spec.width, spec.batch_size, spec.p, spec.seed)


def build_specs(cfg: ExperimentConfig) -> list[RunSpec]:
    fixed_total_n = base.total_train_size(cfg.fixed_train_samples_per_class)
    specs = [
        RunSpec(p=float(p), batch_size=int(batch_size), width=int(width), seed=int(seed))
        for width in cfg.widths
        for batch_size in cfg.batch_sizes
        for p in cfg.p_values
        for seed in cfg.seeds
    ]
    specs.sort(key=lambda spec: run_priority_key(spec, fixed_total_n))
    return specs


def format_run_status(
    spec: RunSpec,
    *,
    fixed_train_samples_per_class: int,
    queued: int | None = None,
    active: int | None = None,
) -> str:
    parts = [
        f"p={spec.p:.4f}",
        f"B={spec.batch_size}",
        f"W={spec.width}",
        f"n_c={fixed_train_samples_per_class}",
    ]
    if active is not None:
        parts.append(f"active={active}")
    if queued is not None:
        parts.append(f"queued={queued}")
    return " ".join(parts)


def write_metadata(cache_dir: Path, cfg: ExperimentConfig, cache_key: str, total_runs: int) -> None:
    base.atomic_write_json(
        cache_dir / "config.json",
        {
            "timestamp": base.timestamp(),
            "cache_key": cache_key,
            "config": asdict(cfg),
            "total_requested_runs": int(total_runs),
        },
    )


def execute_runs(cache_dir: Path, cfg: ExperimentConfig, cache_key: str) -> None:
    runs_dir = cache_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(cfg)
    pending = [spec for spec in specs if cfg.overwrite or not run_json_path(runs_dir, spec).exists()]

    print(
        f"[{base.timestamp()}] Launching {len(specs)} runs with max_workers={cfg.max_workers} "
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
                f"[{base.timestamp()}] dispatch {idx}/{len(pending)} p={spec.p:.4f} "
                f"B={spec.batch_size} W={spec.width} seed={spec.seed}",
                flush=True,
            )
            payload = run_single_spec(spec, cfg_dict)
            base.atomic_write_json(run_json_path(runs_dir, spec), payload)
            pbar.update(1)
            pbar.set_postfix_str(
                format_run_status(spec, fixed_train_samples_per_class=cfg.fixed_train_samples_per_class)
            )
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
                    format_run_status(
                        spec,
                        fixed_train_samples_per_class=cfg.fixed_train_samples_per_class,
                        active=len(in_flight),
                        queued=len(queue),
                    )
                )
            for future in as_completed(in_flight):
                spec = in_flight.pop(future)
                payload = future.result()
                base.atomic_write_json(run_json_path(runs_dir, spec), payload)
                completed += 1
                pbar.update(1)
                pbar.set_postfix_str(
                    format_run_status(
                        spec,
                        fixed_train_samples_per_class=cfg.fixed_train_samples_per_class,
                        active=len(in_flight),
                        queued=len(queue),
                    )
                )
                print(
                    f"[{base.timestamp()}] completed {completed}/{len(pending)} "
                    f"p={spec.p:.4f} B={spec.batch_size} W={spec.width} seed={spec.seed}",
                    flush=True,
                )
                break
        pbar.close()


def build_config() -> ExperimentConfig:
    fixed_train_samples_per_class = base.normalize_train_samples_per_class(RUN_FIXED_TRAIN_SAMPLES_PER_CLASS)
    max_workers = max(1, int(RUN_MAX_WORKERS))
    max_in_flight = max(1, int(RUN_MAX_IN_FLIGHT))
    cpu_core_start = int(RUN_CPU_CORE_START)
    cpu_cores = list(range(cpu_core_start, cpu_core_start + max_workers))

    return ExperimentConfig(
        p_values=[float(value) for value in RUN_P_VALUES],
        batch_sizes=[int(value) for value in RUN_BATCH_SIZES],
        widths=[int(value) for value in RUN_WIDTHS],
        seeds=[int(value) for value in RUN_SEEDS],
        fixed_train_samples_per_class=fixed_train_samples_per_class,
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

    base.apply_cpu_affinity(cfg.cpu_cores)
    random.seed(cfg.seed)

    cache_key = build_cache_key(
        p_values=cfg.p_values,
        batch_sizes=cfg.batch_sizes,
        widths=cfg.widths,
        seeds=cfg.seeds,
        fixed_train_samples_per_class=cfg.fixed_train_samples_per_class,
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
    base.atomic_write_json(
        output_root / "latest.json",
        {
            "timestamp": base.timestamp(),
            "cache_key": cache_key,
            "suffix": cfg.suffix,
            "cache_dir": str(cache_dir),
        },
    )

    write_metadata(cache_dir, cfg, cache_key, len(build_specs(cfg)))

    print(f"[{base.timestamp()}] output cache dir: {cache_dir}", flush=True)
    print(
        f"[{base.timestamp()}] |p|={len(cfg.p_values)} |B|={len(cfg.batch_sizes)} "
        f"|W|={len(cfg.widths)} |seeds|={len(cfg.seeds)} "
        f"n_c={cfg.fixed_train_samples_per_class} N={base.total_train_size(cfg.fixed_train_samples_per_class)}",
        flush=True,
    )

    if not cfg.skip_run:
        execute_runs(cache_dir, cfg, cache_key)

    run_rows = load_all_run_rows(cache_dir / "runs")
    if run_rows:
        base.write_csv(cache_dir / "summary.csv", build_summary_rows(run_rows))
        base.write_csv(cache_dir / "data" / f"results_per_run_{cache_key}{suffix_tag}.csv", build_per_run_rows(run_rows))
        base.write_csv(cache_dir / "data" / f"results_summary_{cache_key}{suffix_tag}.csv", build_summary_rows(run_rows))
        print(f"[{base.timestamp()}] wrote summary data for {len(run_rows)} runs", flush=True)
    else:
        print(f"[{base.timestamp()}] no run JSON files found under {cache_dir / 'runs'}", flush=True)

    if not cfg.skip_plots and run_rows:
        write_plots(cache_dir, run_rows, cfg)
        print(f"[{base.timestamp()}] wrote figures under {cache_dir / 'figures'}", flush=True)

    elapsed = datetime.now() - start_time
    runtime_log_path = cache_dir / "data" / f"runtime_log_{Path(__file__).stem}.csv"
    base.append_runtime_log(
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
