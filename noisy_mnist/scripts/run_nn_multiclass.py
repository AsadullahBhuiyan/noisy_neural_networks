#!/usr/bin/env python3
"""
mnist_corruption_1hidden_nn_multiclass.py

Train a simple 1-hidden-layer NN on *noisy* MNIST (all 10 digits),
evaluate on *clean* test accuracy.

Features:
- Uses ALL digits 0..9 (multiclass)
- Builds *balanced* train/test sets: equal # per digit
- Choose fraction of the balanced per-digit pool for train/test
- Corruption: per-pixel, with prob p replace with Uniform[0,1]
- Multiple independent noise realizations (different corruption seeds)
- Loss: "xent" (CrossEntropy on logits) OR "mse" (MSE on softmax probs vs one-hot)
- Writes:
    results_per_run.csv
    results_summary.csv
"""

from __future__ import annotations

from _path_setup import configure_runtime
PROJECT_ROOT = configure_runtime()

import os
import math
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


# -------------------------
# Repro / device helpers
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(use_cuda: bool) -> torch.device:
    return torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")


# -------------------------
# Corruption (replacement)
# -------------------------
class CorruptedDataset(Dataset):
    """
    Replacement noise:
      for each pixel independently, with prob p replace with Uniform[0,1]
    """
    def __init__(self, base: Dataset, p: float, seed: int) -> None:
        self.base = base
        self.p = float(p)
        self.gen = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]  # x: [1,28,28] in [0,1], y in {0..9}
        if self.p <= 0.0:
            return x, y
        mask = (torch.rand(x.numel(), generator=self.gen, dtype=x.dtype).view_as(x) < self.p)
        repl = torch.rand(x.numel(), generator=self.gen, dtype=x.dtype).view_as(x)
        x_corr = torch.where(mask, repl, x)
        return x_corr, int(y)


# -------------------------
# MNIST helpers
# -------------------------
def make_mnist_root(root: str = "./data") -> Tuple[Dataset, Dataset]:
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    return train, test


def _targets_from_dataset(ds: Dataset) -> torch.Tensor:
    # Works for torchvision MNIST and Subset(MNIST)
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


def make_balanced_multiclass_dataset(
    ds: Dataset,
    fraction: float,
    seed: int,
    num_classes: int = 10,
) -> Tuple[Dataset, Dict]:
    """
    Build a balanced dataset with equal counts per class (0..num_classes-1).

    Procedure:
    - For each class c, take min_c count across classes (n_per_base)
    - Then keep n_keep = floor(fraction * n_per_base) from each class
    - Shuffle final indices

    Returns:
      balanced_ds (Subset)
      info dict
    """
    t = _targets_from_dataset(ds)
    frac = float(fraction)
    frac = max(0.0, min(1.0, frac))

    # gather indices per class
    idx_by_c = []
    counts = []
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

def init_like_ntk(module: nn.Module, w_std: float = 1.0, b_std: float = 1.0) -> None:
    """
    Match Neural Tangents stax.Dense parameterization:
      W_ij ~ N(0, w_std^2 / fan_in)
      b_i  ~ N(0, b_std^2)
    where w_std = sqrt(w_var), b_std = sqrt(b_var).
    """

    for m in module.modules():
        if isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            with torch.no_grad():
                m.weight.normal_(mean=0.0, std=w_std / math.sqrt(fan_in))
                if m.bias is not None:
                    m.bias.normal_(mean=0.0, std=b_std)

class OneHiddenNN(nn.Module):
    """
    Flatten -> (Linear -> act) x nlayers -> Linear(num_classes)
    """
    def __init__(self, hidden: int = 256, nlayers: int = 1, act: str = "relu", num_classes: int = 10):
        super().__init__()

        act = act.lower()
        if act == "tanh":
            self.act = torch.tanh
        elif act == "gelu":
            self.act = F.gelu
        else:
            self.act = F.relu

        self.layers = nn.ModuleList()
        in_dim = 28 * 28

        # first hidden layer
        self.layers.append(nn.Linear(in_dim, hidden))
        # additional hidden layers
        for _ in range(nlayers - 1):
            self.layers.append(nn.Linear(hidden, hidden))

        self.final_layer = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = self.act(layer(x))
        return self.final_layer(x)  # [B,10]



# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def eval_clean(model: nn.Module, loader: DataLoader, device: torch.device, loss_mode: str, num_classes: int) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()  # [B]

        logits = model(x)

        if loss_mode == "xent":
            loss = F.cross_entropy(logits, y)
        else:
            probs = torch.softmax(logits, dim=1)
            y_oh = F.one_hot(y, num_classes=num_classes).to(dtype=probs.dtype)
            loss = F.mse_loss(probs, y_oh)

        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)

    return {
        "loss": float(loss_sum / max(1, total)),
        "acc": float(correct / max(1, total)),
    }


def train_one_run(
    train_clean: Dataset,
    test_clean: Dataset,
    *,
    p: float,
    realization_seed: int,
    cfg: "Config",
) -> Dict:
    device = get_device(cfg.use_cuda)
    set_seed(realization_seed)

    train_noisy = CorruptedDataset(train_clean, p=float(p), seed=realization_seed + 111)

    pin = bool(cfg.use_cuda and torch.cuda.is_available())
    train_loader = DataLoader(
        train_noisy, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_clean, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin
    )

    model = OneHiddenNN(hidden=cfg.hidden, nlayers=cfg.nlayers, act=cfg.activation, num_classes=cfg.num_classes).to(device)
    init_like_ntk(model, w_std=cfg.w_std, b_std=cfg.b_std)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"    Training p={p:.3f}:", end="\n")
    for _epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x)

            if cfg.loss_mode == "xent":
                loss = F.cross_entropy(logits, y)
            else:
                probs = torch.softmax(logits, dim=1)
                y_oh = F.one_hot(y, num_classes=cfg.num_classes).to(dtype=probs.dtype)
                loss = F.mse_loss(probs, y_oh)
            
            print(f"        epoch={_epoch+1}/{cfg.epochs} loss={loss.item():.4f}", end="\r")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    test_metrics = eval_clean(model, test_loader, device=device, loss_mode=cfg.loss_mode, num_classes=cfg.num_classes)
    train_metrics = eval_clean(model, train_loader, device=device, loss_mode=cfg.loss_mode, num_classes=cfg.num_classes)

    return {
        "p": float(p),
        "realization": int(cfg.realization_idx),
        "seed": int(realization_seed),
        "device": str(device),
        "loss_mode": cfg.loss_mode,
        "num_classes": int(cfg.num_classes),
        "hidden": int(cfg.hidden),
        "activation": cfg.activation,
        "epochs": int(cfg.epochs),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "batch_size": int(cfg.batch_size),
        "train_loss": float(train_metrics["loss"]),
        "train_acc": float(train_metrics["acc"]),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
    }


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    g = per_run.groupby(["p"])
    out = g.agg(
        runs=("test_acc", "count"),
        mean_test_acc=("test_acc", "mean"),
        std_test_acc=("test_acc", "std"),
        mean_test_loss=("test_loss", "mean"),
        std_test_loss=("test_loss", "std"),
        mean_train_acc=("train_acc", "mean"),
        std_train_acc=("train_acc", "std"),
    ).reset_index()
    out["stderr_test_acc"] = out["std_test_acc"] / np.sqrt(out["runs"].clip(lower=1))
    out["stderr_test_loss"] = out["std_test_loss"] / np.sqrt(out["runs"].clip(lower=1))
    return out.sort_values("p")


# -------------------------
# Config + main
# -------------------------
@dataclass
class Config:
    # data
    num_classes: int = 10
    train_fraction: float = 0.2
    test_fraction: float = 1.0

    # corruption
    p_list: List[float] = None
    realizations: int = 5

    # model / training
    loss_mode: str = "xent"     # "xent" or "mse"
    hidden: int = 256
    nlayers: int = 1
    activation: str = "relu"    # "relu" / "tanh" / "gelu"
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    w_std: float = 1.0
    b_std: float = 1.0

    # loader / compute
    batch_size: int = 256
    num_workers: int = 2
    use_cuda: bool = True

    # misc
    seed: int = 1234
    outdir: str = "mnist_corruption_1hidden_nn_multiclass_results"

    # internal per-run
    realization_idx: int = 0


def main() -> None:
    # =========================
    # EDIT SETTINGS HERE
    # =========================
    cfg = Config(
        train_fraction=0.4,
        test_fraction=1.0,
        p_list=np.linspace(0.9, 1.0, 11).to_list(),
        realizations=1,
        loss_mode="xent",     # "xent" or "mse"
        hidden=64,
        nlayers=1,
        activation="relu",
        epochs=20,
        lr=1e-3,
        weight_decay=0,
        w_std=1.0,
        b_std=1.0,
        batch_size=128,
        num_workers=2,
        use_cuda=True,
        seed=1234,
        outdir="mnist_corruption_1hidden_nn_multiclass_results",
    )

    if cfg.p_list is None or len(cfg.p_list) == 0:
        raise ValueError("cfg.p_list must be non-empty.")
    if cfg.loss_mode not in ("xent", "mse"):
        raise ValueError("loss_mode must be 'xent' or 'mse'.")

    outdir = Path(cfg.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = make_mnist_root()

    # Balanced train/test (equal per digit)
    train_ds, train_info = make_balanced_multiclass_dataset(
        train_raw, fraction=cfg.train_fraction, seed=cfg.seed + 1, num_classes=cfg.num_classes
    )
    test_ds, test_info = make_balanced_multiclass_dataset(
        test_raw, fraction=cfg.test_fraction, seed=cfg.seed + 2, num_classes=cfg.num_classes
    )

    print("Train info:", train_info)
    print("Test  info:", test_info)
    print(f"Training on NOISY train, evaluating on CLEAN test. loss_mode={cfg.loss_mode}")

    per_run_rows: List[Dict] = []
    for p in cfg.p_list:
        for r in range(cfg.realizations):
            cfg.realization_idx = r
            realization_seed = cfg.seed + 10_000 * r + int(1_000_000 * float(p))

            row = train_one_run(
                train_clean=train_ds,
                test_clean=test_ds,
                p=float(p),
                realization_seed=realization_seed,
                cfg=cfg,
            )
            row.update({
                "train_fraction": float(cfg.train_fraction),
                "test_fraction": float(cfg.test_fraction),
                "train_balanced_total": int(train_info["balanced_total"]),
                "test_balanced_total": int(test_info["balanced_total"]),
            })
            per_run_rows.append(row)
            print(f"p={p:.3f} r={r:02d} -> test_acc={row['test_acc']:.4f} test_loss={row['test_loss']:.4f}")

    per_run = pd.DataFrame(per_run_rows)
    summary = summarize(per_run)

    per_run_path = outdir / "results_per_run.csv"
    summary_path = outdir / "results_summary.csv"
    config_path = outdir / "run_config.json"

    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(config_path, "w") as f:
        json.dump(
            {"config": asdict(cfg), "train_info": train_info, "test_info": test_info},
            f,
            indent=2
        )

    print("\nSaved:")
    print(" ", per_run_path)
    print(" ", summary_path)
    print(" ", config_path)


if __name__ == "__main__":
    main()
