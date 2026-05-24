#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from src.nnet_models import apply_corruption, build_model, compute_loss, evaluate
except ModuleNotFoundError:
    from nnet_models import apply_corruption, build_model, compute_loss, evaluate


class FullyCorruptedDataset(Dataset):
    def __init__(self, base: Dataset, p: float, sigma: float, mode: str) -> None:
        self.base = base
        self.p = p
        self.sigma = sigma
        self.mode = mode

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.base[idx]
        return apply_corruption(x, self.mode, self.p, self.sigma), y


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(raw: str, default: list[float]) -> list[float]:
    if not raw.strip():
        return default
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_dataset(base: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return base
    max_samples = max(1, min(len(base), max_samples))
    return torch.utils.data.Subset(base, list(range(max_samples)))


def build_mnist(max_train_samples: int | None, max_test_samples: int | None) -> tuple[Dataset, Dataset]:
    transform = transforms.ToTensor()
    train_base = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_base = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return limit_dataset(train_base, max_train_samples), limit_dataset(test_base, max_test_samples)


def tensorize_dataset(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for x, y in dataset:
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def make_fixed_replacement_test_loaders(
    test_base: Dataset,
    p_test_values: list[float],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[float, DataLoader]:
    test_x, test_y = tensorize_dataset(test_base)
    loaders: dict[float, DataLoader] = {}
    for i, p_test in enumerate(p_test_values):
        generator = torch.Generator().manual_seed(seed + i)
        if p_test <= 0.0:
            corrupted = test_x.clone()
        else:
            mask = torch.rand(test_x.shape, generator=generator) < p_test
            replacement = torch.rand(test_x.shape, generator=generator)
            corrupted = torch.where(mask, replacement, test_x)
        loaders[p_test] = DataLoader(
            TensorDataset(corrupted, test_y),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders


def train_one(
    train_base: Dataset,
    test_loaders: dict[float, DataLoader],
    p_train: float,
    args: argparse.Namespace,
    run_seed: int,
    run_id: int,
) -> list[dict]:
    set_seed(run_seed)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    train_set = FullyCorruptedDataset(train_base, p=p_train, sigma=0.0, mode="replacement")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    model = build_model("mlp", "relu", mlp_hidden_sizes=[args.mlp_width] * args.mlp_depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for _ in tqdm(range(args.epochs), desc=f"p_train={p_train:.3f}", unit="epoch", leave=False):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = compute_loss(logits, y, "cross_entropy")
            loss.backward()
            optimizer.step()

    rows: list[dict] = []
    for p_test, loader in test_loaders.items():
        test_loss, test_acc = evaluate(model, loader, device, "cross_entropy")
        rows.append(
            {
                "p_train": p_train,
                "p_test": p_test,
                "run_id": run_id,
                "seed": run_seed,
                "epochs": args.epochs,
                "mlp_depth": args.mlp_depth,
                "mlp_width": args.mlp_width,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on fully replacement-corrupted MNIST and test on fixed p_test corruptions.")
    parser.add_argument("--p-train-values", type=str, default="", help="Comma-separated p_train values; default np.linspace(0,1,21).")
    parser.add_argument("--p-test-values", type=str, default="", help="Comma-separated p_test values; default np.linspace(0,1,81).")
    parser.add_argument("--run-offset", type=int, default=0)
    parser.add_argument("--run-stride", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=20, help="Independent training repeats per p_train value.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--mlp-depth", type=int, default=3)
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--test-corruption-seed", type=int, default=4321)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--results-path", type=str, required=True)
    args = parser.parse_args()

    p_train_values = parse_float_list(args.p_train_values, np.linspace(0.0, 1.0, 21).tolist())
    p_test_values = parse_float_list(args.p_test_values, np.linspace(0.0, 1.0, 81).tolist())
    if args.run_stride < 1:
        raise ValueError("run_stride must be >= 1")
    if args.run_offset < 0 or args.run_offset >= args.run_stride:
        raise ValueError("run_offset must satisfy 0 <= run_offset < run_stride")

    all_jobs = [
        (p_train, run_id)
        for p_train in p_train_values
        for run_id in range(args.num_runs)
    ]
    selected_jobs = [job for i, job in enumerate(all_jobs) if i % args.run_stride == args.run_offset]
    train_base, test_base = build_mnist(args.max_train_samples, args.max_test_samples)
    test_loaders = make_fixed_replacement_test_loaders(
        test_base,
        p_test_values=p_test_values,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.test_corruption_seed,
    )

    all_rows: list[dict] = []
    for p_train, run_id in selected_jobs:
        run_seed = args.seed + int(round(p_train * 1_000_000)) + run_id
        all_rows.extend(train_one(train_base, test_loaders, p_train, args, run_seed=run_seed, run_id=run_id))

    write_csv(Path(args.results_path), all_rows)
    print(f"Saved {len(all_rows)} rows to {Path(args.results_path).resolve()}")


if __name__ == "__main__":
    main()
