#!/usr/bin/env python3
"""
Balanced multiclass MNIST corruption sweep, using the training/eval pipeline from
run_experiments_noisy_training_data_ntrain_sweep.py (nnet_models).

Functionality matches run_nn_multiclass.py:
- Balanced per-digit train/test subsets
- Corruption per-pixel replacement with probability p
- Multiple realizations per p
- Per-run + summary CSVs
- Summary PDF plots (mean/stderr vs p)
"""

from __future__ import annotations

from _path_setup import configure_runtime
PROJECT_ROOT = configure_runtime()

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import csv
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.nnet_models import (
    CorruptedDataset,
    build_model,
    compute_loss,
    evaluate,
    get_device,
    set_seed,
    train_one_epoch,
)


def init_like_ntk(module: torch.nn.Module, w_std: float = 1.0, b_std: float = 1.0) -> None:
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            fan_in = m.weight.shape[1]
            with torch.no_grad():
                m.weight.normal_(mean=0.0, std=w_std / math.sqrt(fan_in))
                if m.bias is not None:
                    m.bias.normal_(mean=0.0, std=b_std)


def _targets_from_dataset(ds: Dataset) -> torch.Tensor:
    if hasattr(ds, "targets"):
        t = ds.targets
        if isinstance(t, list):
            t = torch.tensor(t, dtype=torch.long)
        return t.to(dtype=torch.long)
    if isinstance(ds, Subset):
        base_t = _targets_from_dataset(ds.dataset)
        idx = torch.as_tensor(ds.indices, dtype=torch.long)
        return base_t[idx]
    ys = [int(ds[i][1]) for i in range(len(ds))]
    return torch.tensor(ys, dtype=torch.long)


def make_balanced_multiclass_subset(
    ds: Dataset,
    fraction: float,
    seed: int,
    num_classes: int = 10,
) -> Tuple[Dataset, Dict]:
    t = _targets_from_dataset(ds)
    frac = max(0.0, min(1.0, float(fraction)))

    idx_by_c: list[torch.Tensor] = []
    counts: list[int] = []
    for c in range(num_classes):
        idx_c = torch.nonzero(t == c, as_tuple=False).view(-1)
        idx_by_c.append(idx_c)
        counts.append(int(idx_c.numel()))

    n_per_base = min(counts)
    n_keep = max(1, int(math.floor(frac * n_per_base)))

    g = torch.Generator().manual_seed(int(seed))
    chosen = []
    for c in range(num_classes):
        idx_c = idx_by_c[c]
        perm = idx_c[torch.randperm(idx_c.numel(), generator=g)][:n_per_base]
        chosen.append(perm[:n_keep])

    all_idx = torch.cat(chosen, dim=0)
    all_idx = all_idx[torch.randperm(all_idx.numel(), generator=g)]
    balanced = Subset(ds, all_idx.tolist())

    info = {
        "num_classes": int(num_classes),
        "orig_counts": {str(c): int(counts[c]) for c in range(num_classes)},
        "balanced_per_class": int(n_keep),
        "balanced_total": int(num_classes * n_keep),
        "fraction_used": float(frac),
        "n_per_base_min_over_classes": int(n_per_base),
    }
    return balanced, info


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def summarize(per_run_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, float], list[dict]] = {}
    for row in per_run_rows:
        key = (row["model_type"], row["activation"], float(row["p"]))
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (model_type, activation, p), rows in groups.items():
        test_acc = [float(r["test_accuracy"]) for r in rows]
        test_loss = [float(r["test_loss"]) for r in rows]
        train_acc = [float(r["train_accuracy"]) for r in rows]
        train_loss = [float(r["train_loss"]) for r in rows]
        runs = len(rows)
        std_test_acc = _std(test_acc)
        std_test_loss = _std(test_loss)
        std_train_acc = _std(train_acc)
        std_train_loss = _std(train_loss)

        summary_rows.append(
            {
                "model_type": model_type,
                "activation": activation,
                "p": float(p),
                "runs": int(runs),
                "mean_test_accuracy": _mean(test_acc),
                "std_test_accuracy": std_test_acc,
                "mean_test_loss": _mean(test_loss),
                "std_test_loss": std_test_loss,
                "mean_train_accuracy": _mean(train_acc),
                "std_train_accuracy": std_train_acc,
                "mean_train_loss": _mean(train_loss),
                "std_train_loss": std_train_loss,
                "stderr_test_accuracy": std_test_acc / math.sqrt(runs) if runs else 0.0,
                "stderr_test_loss": std_test_loss / math.sqrt(runs) if runs else 0.0,
                "stderr_train_accuracy": std_train_acc / math.sqrt(runs) if runs else 0.0,
                "stderr_train_loss": std_train_loss / math.sqrt(runs) if runs else 0.0,
            }
        )

    summary_rows.sort(key=lambda r: (r["activation"], r["p"]))
    return summary_rows


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_pdf(path: str, summary_rows: list[dict], metadata: dict) -> None:
    if not summary_rows:
        return

    with PdfPages(path) as pdf:
        model_types = sorted({r["model_type"] for r in summary_rows})
        for model_type in model_types:
            sub = [r for r in summary_rows if r["model_type"] == model_type]
            activations = sorted({r["activation"] for r in sub})
            n = len(activations)
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                d = sorted([r for r in sub if r["activation"] == act], key=lambda r: float(r["p"]))
                if not d:
                    continue
                ax.errorbar(
                    [float(r["p"]) for r in d],
                    [float(r["mean_test_accuracy"]) for r in d],
                    yerr=[float(r["stderr_test_accuracy"]) for r in d],
                    marker=".",
                    label=f"{act}",
                )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel("p")
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean test accuracy")
            axes[0].legend(fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                d = sorted([r for r in sub if r["activation"] == act], key=lambda r: float(r["p"]))
                if not d:
                    continue
                ax.plot(
                    [float(r["p"]) for r in d],
                    [float(r["std_test_accuracy"]) for r in d],
                    marker=".",
                )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel("p")
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("std test accuracy")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                d = sorted([r for r in sub if r["activation"] == act], key=lambda r: float(r["p"]))
                if not d:
                    continue
                ax.errorbar(
                    [float(r["p"]) for r in d],
                    [float(r["mean_test_loss"]) for r in d],
                    yerr=[float(r["stderr_test_loss"]) for r in d],
                    marker=".",
                )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel("p")
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean test loss")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                d = sorted([r for r in sub if r["activation"] == act], key=lambda r: float(r["p"]))
                if not d:
                    continue
                ax.errorbar(
                    [float(r["p"]) for r in d],
                    [float(r["mean_train_loss"]) for r in d],
                    yerr=[float(r["stderr_train_loss"]) for r in d],
                    marker=".",
                )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel("p")
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean train loss")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        meta_lines = [f"{key}: {value}" for key, value in metadata.items()]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.95, "Run Metadata", fontsize=14, fontweight="bold", va="top")
        fig.text(0.05, 0.9, "\n".join(meta_lines), fontsize=10, va="top")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def build_cache_key(
    realizations: int,
    epochs: int,
    train_fraction: float,
    test_fraction: float,
    ps: list[float],
    hidden: int,
    mlp_depth: int,
    loss_type: str,
    w_std: float,
    b_std: float,
) -> str:
    p_min, p_max = min(ps), max(ps)
    strength_tag = f"p{p_min:.2f}-{p_max:.2f}"
    return (
        f"mlpns_mnist_balanced_r{realizations}_e{epochs}_"
        f"tf{train_fraction:.3f}_te{test_fraction:.3f}_"
        f"{strength_tag}_w{hidden}_d{mlp_depth}_loss-{loss_type}_wstd{w_std:.3f}_bstd{b_std:.3f}"
    )


def main() -> None:
    # Manual configuration (edit these values directly).
    activations = ["relu"]
    model_types = ["mlp"]
    corruption_mode = "replacement"
    ps = np.linspace(0.9, 1.0, 11).tolist()
    realizations = 1

    hidden = 256
    mlp_depth = 1
    epochs = 50
    batch_size = 1024
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "cross_entropy"  # "cross_entropy" or "quadratic"
    w_std = 1.0/2
    b_std = 0

    train_fraction = 1.0
    test_fraction = 1.0
    num_classes = 10

    num_workers = 2
    use_cuda = False
    seed = 1234

    output_dir = os.path.join("results", "data")
    suffix = "multiclass_balanced"

    if loss_type not in ("cross_entropy", "quadratic"):
        raise ValueError("loss_type must be 'cross_entropy' or 'quadratic'.")
    if not ps:
        raise ValueError("ps must be non-empty.")

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)

    def _safe_to_tensor(img) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        return torch.clamp(x, 0.0, 1.0)

    train_raw = datasets.MNIST(root="./data", train=True, download=True, transform=_safe_to_tensor)
    test_raw = datasets.MNIST(root="./data", train=False, download=True, transform=_safe_to_tensor)

    train_ds, train_info = make_balanced_multiclass_subset(
        train_raw, fraction=train_fraction, seed=seed + 1, num_classes=num_classes
    )
    test_ds, test_info = make_balanced_multiclass_subset(
        test_raw, fraction=test_fraction, seed=seed + 2, num_classes=num_classes
    )

    per_run_rows: list[dict] = []
    total_runs = len(ps) * realizations * len(activations) * len(model_types)
    pbar = tqdm(total=total_runs, desc="multiclass sweep", unit="run")

    for activation in activations:
        for model_type in model_types:
            for p in ps:
                for r in range(realizations):
                    realization_seed = seed + 10_000 * r + int(1_000_000 * float(p))
                    set_seed(realization_seed)
                    device = get_device(use_cuda)

                    train_noisy = CorruptedDataset(train_ds, p=float(p), sigma=0.0, mode=corruption_mode)
                    pin = bool(use_cuda and torch.cuda.is_available())
                    train_loader = DataLoader(
                        train_noisy,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=pin,
                    )
                    test_loader = DataLoader(
                        test_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin,
                    )

                    model = build_model(
                        model_type=model_type,
                        activation=activation,
                        mlp_hidden_sizes=[hidden] * mlp_depth,
                    ).to(device)
                    init_like_ntk(model, w_std=w_std, b_std=b_std)
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay,
                    )

                    last_test_loss = math.nan
                    last_test_acc = math.nan
                    epoch_pbar = tqdm(
                        range(epochs),
                        desc=f"epochs (act={activation}, p={p:.3f})",
                        unit="epoch",
                        leave=False,
                        disable=epochs <= 1,
                    )
                    for _ in epoch_pbar:
                        train_one_epoch(model, train_loader, optimizer, device, loss_type)
                        last_test_loss, last_test_acc = evaluate(model, test_loader, device, loss_type)
                        epoch_pbar.set_postfix_str(
                            f"test_loss={last_test_loss:.4f} test_acc={last_test_acc:.4f}"
                        )

                    train_loss, train_acc = evaluate(model, train_loader, device, loss_type)
                    test_loss = last_test_loss
                    test_acc = last_test_acc

                    per_run_rows.append(
                        {
                            "activation": activation,
                            "model_type": model_type,
                            "corruption_mode": corruption_mode,
                            "p": float(p),
                            "sigma": 0.0,
                            "realization": r,
                            "seed": realization_seed,
                            "epochs": epochs,
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "batch_size": batch_size,
                            "train_fraction": float(train_fraction),
                            "test_fraction": float(test_fraction),
                            "train_balanced_total": int(train_info["balanced_total"]),
                            "test_balanced_total": int(test_info["balanced_total"]),
                            "hidden": int(hidden),
                            "mlp_depth": int(mlp_depth),
                            "loss_type": loss_type,
                            "w_std": float(w_std),
                            "b_std": float(b_std),
                            "train_loss": float(train_loss),
                            "train_accuracy": float(train_acc),
                            "test_loss": float(test_loss),
                            "test_accuracy": float(test_acc),
                        }
                    )
                    pbar.set_postfix_str(f"p={p:.3f} acc={test_acc:.4f} loss={test_loss:.4f}")
                    pbar.update(1)

    pbar.close()

    summary_rows = summarize(per_run_rows)

    cache_key = build_cache_key(
        realizations=realizations,
        epochs=epochs,
        train_fraction=train_fraction,
        test_fraction=test_fraction,
        ps=ps,
        hidden=hidden,
        mlp_depth=mlp_depth,
        loss_type=loss_type,
        w_std=w_std,
        b_std=b_std,
    )
    suffix_tag = f"_{suffix}" if suffix else ""

    per_run_path = os.path.join(output_dir, f"results_per_run_{cache_key}{suffix_tag}.csv")
    summary_path = os.path.join(output_dir, f"results_summary_{cache_key}{suffix_tag}.csv")
    pdf_path = os.path.join("results", "figures", f"accuracy_vs_{cache_key}{suffix_tag}.pdf")
    config_path = os.path.join(output_dir, f"run_config_{cache_key}{suffix_tag}.json")

    write_csv(per_run_rows, per_run_path)
    write_csv(summary_rows, summary_path)

    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cache_key": cache_key,
        "activations": activations,
        "model_types": model_types,
        "corruption_mode": corruption_mode,
        "ps": ps,
        "realizations": realizations,
        "hidden": hidden,
        "mlp_depth": mlp_depth,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "w_std": w_std,
        "b_std": b_std,
        "train_fraction": train_fraction,
        "test_fraction": test_fraction,
        "train_info": train_info,
        "test_info": test_info,
        "num_workers": num_workers,
        "use_cuda": use_cuda,
        "seed": seed,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    write_summary_pdf(pdf_path, summary_rows, metadata)

    print("\nSaved:")
    print(f"  {per_run_path}")
    print(f"  {summary_path}")
    print(f"  {pdf_path}")
    print(f"  {config_path}")


if __name__ == "__main__":
    main()
