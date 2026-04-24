from _path_setup import configure_runtime
PROJECT_ROOT = configure_runtime()

import os
cpu_max = 10

import csv
import random
import statistics
import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from src.nnet_models import TrainConfig, train_and_evaluate


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, float, float, int], list[dict]] = {}
    for row in rows:
        key = (
            row["activation"],
            row["model_type"],
            row["corruption_mode"],
            float(row["p"]),
            float(row["sigma"]),
            int(row["batch_size"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (activation, model_type, corruption_mode, p, sigma, batch_size), items in grouped.items():
        accs = [float(r["test_accuracy"]) for r in items]
        losses = [float(r["test_loss"]) for r in items]
        train_losses = [float(r.get("train_loss", 0.0)) for r in items]
        summary_rows.append(
            {
                "activation": activation,
                "model_type": model_type,
                "corruption_mode": corruption_mode,
                "p": p,
                "sigma": sigma,
                "batch_size": batch_size,
                "repeats": len(items),
                "mean_test_accuracy": statistics.mean(accs),
                "std_test_accuracy": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "stderr_test_accuracy": (statistics.pstdev(accs) / (len(items) ** 0.5))
                if len(items) > 1
                else 0.0,
                "mean_test_loss": statistics.mean(losses),
                "std_test_loss": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
                "stderr_test_loss": (statistics.pstdev(losses) / (len(items) ** 0.5))
                if len(items) > 1
                else 0.0,
                "mean_train_loss": statistics.mean(train_losses),
                "std_train_loss": statistics.pstdev(train_losses) if len(train_losses) > 1 else 0.0,
                "stderr_train_loss": (statistics.pstdev(train_losses) / (len(items) ** 0.5))
                if len(items) > 1
                else 0.0,
            }
        )
    summary_rows.sort(
        key=lambda r: (
            r["model_type"],
            r["activation"],
            r["corruption_mode"],
            r["batch_size"],
            r["p"],
            r["sigma"],
        )
    )
    return summary_rows


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


def _init_worker(cores: list[int] | None) -> None:
    apply_cpu_affinity(cores)
    try:
        import torch

        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
    except Exception:
        pass


def build_cache_key(
    max_repeats: int,
    epochs: int,
    dynamical_epoch: bool,
    epoch_per_batch_ratio: float,
    test_fraction: float,
    exact_sample_counts: bool,
    exact_train_samples: int | None,
    exact_test_samples: int | None,
    loss_type: str,
    corruption_mode: str,
    ps: list[float],
    sigmas: list[float],
    batch_sizes: list[int],
    width: int,
    mlp_depth: int,
) -> str:
    if corruption_mode == "replacement":
        p_min, p_max = min(ps), max(ps)
        strength_tag = f"p{p_min:.2f}-{p_max:.2f}"
    else:
        s_min, s_max = min(sigmas), max(sigmas)
        strength_tag = f"s{s_min:.2f}-{s_max:.2f}"
    b_min, b_max = min(batch_sizes), max(batch_sizes)
    if exact_sample_counts:
        split_tag = f"ntr{exact_train_samples}_nte{exact_test_samples}"
    else:
        split_tag = f"tf{test_fraction:.3f}"
    if dynamical_epoch:
        epoch_tag = f"edyn{epoch_per_batch_ratio:.6g}"
    else:
        epoch_tag = f"e{epochs}"
    return (
        f"mlpbs_rmax{max_repeats}_{epoch_tag}_{split_tag}_"
        f"{strength_tag}_b{b_min}-{b_max}_w{width}_d{mlp_depth}_loss-{loss_type}"
    )


def _stderr_from_stats(stats: dict) -> float:
    n = stats["n"]
    if n <= 1:
        return float("inf")
    return math.sqrt(stats["m2"] / (n - 1)) / math.sqrt(n)


def _update_running_stats(stats: dict, value: float) -> None:
    stats["n"] += 1
    delta = value - stats["mean"]
    stats["mean"] += delta / stats["n"]
    delta2 = value - stats["mean"]
    stats["m2"] += delta * delta2


def write_summary_pdf(
    path: str,
    summary_rows: list[dict],
    metadata: dict,
) -> None:
    if not summary_rows:
        return
    model_types = sorted({r["model_type"] for r in summary_rows})
    sweep_label = "sigma" if any(r["corruption_mode"] == "additive" for r in summary_rows) else "p"

    with PdfPages(path) as pdf:
        for model_type in model_types:
            sub = [r for r in summary_rows if r["model_type"] == model_type]
            activations = sorted({r["activation"] for r in sub})
            batch_sizes = sorted({int(r["batch_size"]) for r in sub})
            corruption_modes = {r["corruption_mode"] for r in sub}
            sweep_label = "sigma" if len(corruption_modes) == 1 and "additive" in corruption_modes else "p"
            for act in activations:
                act_rows = [r for r in sub if r["activation"] == act]
                if not act_rows:
                    continue

                # Mean test accuracy
                fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
                for b in batch_sizes:
                    d = sorted(
                        [r for r in act_rows if int(r["batch_size"]) == b],
                        key=lambda r: float(r[sweep_label]),
                    )
                    ax.errorbar(
                        [float(r[sweep_label]) for r in d],
                        [float(r["mean_test_accuracy"]) for r in d],
                        yerr=[float(r["stderr_test_accuracy"]) for r in d],
                        marker=".",
                        label=f"b={b}",
                    )
                ax.set_title(f"{model_type} / {act} — mean test accuracy")
                ax.set_xlabel(sweep_label)
                ax.set_ylabel("mean test accuracy")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                # Std test accuracy
                fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
                for b in batch_sizes:
                    d = sorted(
                        [r for r in act_rows if int(r["batch_size"]) == b],
                        key=lambda r: float(r[sweep_label]),
                    )
                    ax.plot(
                        [float(r[sweep_label]) for r in d],
                        [float(r["std_test_accuracy"]) for r in d],
                        marker=".",
                        label=f"b={b}",
                    )
                ax.set_title(f"{model_type} / {act} — std test accuracy")
                ax.set_xlabel(sweep_label)
                ax.set_ylabel("std test accuracy")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                # Mean test loss
                fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
                for b in batch_sizes:
                    d = sorted(
                        [r for r in act_rows if int(r["batch_size"]) == b],
                        key=lambda r: float(r[sweep_label]),
                    )
                    ax.errorbar(
                        [float(r[sweep_label]) for r in d],
                        [float(r["mean_test_loss"]) for r in d],
                        yerr=[float(r["stderr_test_loss"]) for r in d],
                        marker=".",
                        label=f"b={b}",
                    )
                ax.set_title(f"{model_type} / {act} — mean test loss")
                ax.set_xlabel(sweep_label)
                ax.set_ylabel("mean test loss")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                # Mean train loss
                fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
                for b in batch_sizes:
                    d = sorted(
                        [r for r in act_rows if int(r["batch_size"]) == b],
                        key=lambda r: float(r[sweep_label]),
                    )
                    ax.errorbar(
                        [float(r[sweep_label]) for r in d],
                        [float(r["mean_train_loss"]) for r in d],
                        yerr=[float(r["stderr_train_loss"]) for r in d],
                        marker=".",
                        label=f"b={b}",
                    )
                ax.set_title(f"{model_type} / {act} — mean train loss")
                ax.set_xlabel(sweep_label)
                ax.set_ylabel("mean train loss")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
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


def main() -> None:
    start_time = datetime.now()
    # Manual configuration (edit these values directly).
    activations = ["relu"]
    model_types = ["mlp"]  # MLP only
    corruption_mode = "replacement"  # options: "replacement", "additive"
    ps = np.linspace(0.9, 1.0, 11)
    sigmas = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    batch_sizes = [int(x) for x in (128 * np.array([1, 4, 16, 32]))]
    width = 64
    mlp_depth = 1
    epochs = 20
    dynamical_epoch = True
    epoch_per_batch_ratio = 20/128  # Only used if dynamical_epoch=True. Keeps epochs / batch_size approximately fixed across the sweep.
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "cross_entropy"  # options: "cross_entropy", "quadratic"

    max_repeats = 10
    min_repeats = 10
    stderr_target = 1e-2

    max_workers = cpu_max
    max_in_flight = max(1, 9*cpu_max//10)
    data_workers = 0
    cpu_threads_per_worker = 1
    cpu_cores = list(range(60,60+cpu_max))
    brightness_scale = 1.0
    custom_split = False
    test_fraction = 0.5
    split_seed = 1234
    split_source = "train"
    exact_sample_counts = False
    exact_train_samples = None
    exact_test_samples = None
    output_dir = os.path.join("results", "data")
    suffix = "batchsize_sweep_relu_only_fixed_epoch_batchsize_ratio"
    seed = 1234
    max_train_samples = None
    use_cuda = False

    if use_cuda and max_workers > 1:
        raise ValueError("use_cuda=True only supports max_workers=1 for now.")
    if exact_sample_counts and (not exact_train_samples or not exact_test_samples):
        raise ValueError(
            "Set exact_train_samples and exact_test_samples to positive integers when exact_sample_counts=True."
        )
    if dynamical_epoch and epoch_per_batch_ratio <= 0:
        raise ValueError("epoch_per_batch_ratio must be positive when dynamical_epoch=True.")

    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads_per_worker))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads_per_worker))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads_per_worker))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads_per_worker))

    apply_cpu_affinity(cpu_cores)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)

    sweep_values = ps if corruption_mode == "replacement" else sigmas
    sweep_label = "p" if corruption_mode == "replacement" else "sigma"

    group_keys: list[tuple[str, str, int, float, float]] = []
    group_max_repeats: dict[tuple[str, str, int, float, float], int] = {}
    group_min_repeats: dict[tuple[str, str, int, float, float], int] = {}
    for activation in activations:
        for model_type in model_types:
            for batch_size in batch_sizes:
                for value in sweep_values:
                    p = value if corruption_mode == "replacement" else 0.0
                    sigma = value if corruption_mode == "additive" else 0.0
                    key = (activation, model_type, batch_size, float(p), float(sigma))
                    if corruption_mode == "replacement" and abs(p) < 1e-12:
                        group_max_repeats[key] = 0
                        group_min_repeats[key] = 0
                        continue
                    group_keys.append(key)
                    group_max_repeats[key] = max_repeats
                    group_min_repeats[key] = min_repeats

    def _make_config(key: tuple[str, str, int, float, float]) -> TrainConfig:
        activation, model_type, batch_size, p, sigma = key
        batch_size = int(batch_size)
        run_epochs = epochs
        if dynamical_epoch:
            # Keep epochs / batch_size approximately fixed across the sweep.
            run_epochs = max(1, int(round(epoch_per_batch_ratio * batch_size)))
        run_seed = random.randint(1, 1_000_000_000)
        return TrainConfig(
            activation=activation,
            model_type=model_type,
            corruption_mode=corruption_mode,
            p=p,
            sigma=sigma,
            mlp_hidden_sizes=[width] * mlp_depth,
            epochs=run_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_type=loss_type,
            seed=run_seed,
            num_workers=data_workers,
            cpu_threads=cpu_threads_per_worker,
            brightness_scale=brightness_scale,
            custom_split=custom_split,
            test_fraction=test_fraction,
            split_seed=split_seed,
            split_source=split_source,
            exact_sample_counts=exact_sample_counts,
            exact_train_samples=exact_train_samples,
            exact_test_samples=exact_test_samples,
            max_train_samples=max_train_samples,
            use_cuda=use_cuda,
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix_tag = f"_{suffix}" if suffix else ""
    suffix_note = f" (suffix={suffix})" if suffix else ""
    print(
        f"[{timestamp}] Launching {len(group_keys)} groups with max_workers={max_workers} "
        f"(stderr_target={stderr_target}, min_repeats={min_repeats}, max_repeats={max_repeats})..."
        f"{suffix_note}"
    )
    if dynamical_epoch:
        print(
            f"[{timestamp}] dynamical_epoch=True with epoch_per_batch_ratio={epoch_per_batch_ratio} "
            f"(epochs = max(1, round(ratio * batch_size)))."
        )

    results: list[dict] = []
    seen_values: set[float] = set()
    run_start_times = {}
    total_runs = 0
    stats_by_group = {k: {"n": 0, "mean": 0.0, "m2": 0.0} for k in group_keys}
    queue = deque()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(cpu_cores,),
    ) as executor:
        in_flight: dict = {}
        desc = f"Runs{suffix_note} (batch_sizes={batch_sizes})"
        pbar = tqdm(total=0, desc=desc, unit="run")

        def _enqueue(key: tuple[str, str, int, float, float]) -> None:
            nonlocal total_runs
            cfg = _make_config(key)
            queue.append((key, cfg))
            total_runs += 1
            pbar.total = total_runs
            pbar.refresh()

        for key in group_keys:
            _enqueue(key)

        while queue or in_flight:
            while queue and len(in_flight) < max_in_flight:
                key, cfg = queue.popleft()
                fut = executor.submit(train_and_evaluate, cfg)
                in_flight[fut] = (key, cfg)
                run_start_times[fut] = datetime.now()
                pbar.set_postfix_str(
                    f"active batch={int(cfg.batch_size)} epochs={int(cfg.epochs)}"
                )
            for future in as_completed(in_flight):
                res = future.result()
                key, cfg = in_flight[future]
                # Ensure downstream grouping has batch size even if train_and_evaluate omits it.
                res["batch_size"] = int(cfg.batch_size)
                res["epochs"] = int(cfg.epochs)
                results.append(res)
                _update_running_stats(stats_by_group[key], float(res["test_accuracy"]))
                pbar.update(1)
                pbar.set_postfix_str(
                    f"done batch={int(cfg.batch_size)} epochs={int(cfg.epochs)}"
                )
                strength_value = res["p"] if corruption_mode == "replacement" else res["sigma"]
                if strength_value not in seen_values:
                    seen_values.add(strength_value)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] new {sweep_label} encountered: {strength_value:.3f}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                batch_size = key[2]
                elapsed = datetime.now() - run_start_times.get(future, datetime.now())
                print(
                    f"[{timestamp}] done: model={res['model_type']} act={res['activation']} "
                    f"{sweep_label}={strength_value:.3f} batch={batch_size} acc={res['test_accuracy']:.4f} "
                    f"elapsed={elapsed}"
                )
                n_done = stats_by_group[key]["n"]
                stderr = _stderr_from_stats(stats_by_group[key])
                if n_done < group_min_repeats[key]:
                    _enqueue(key)
                elif n_done < group_max_repeats[key] and stderr > stderr_target:
                    _enqueue(key)
                del in_flight[future]
                break
        pbar.close()

    cache_key = build_cache_key(
        max_repeats=max_repeats,
        epochs=epochs,
        dynamical_epoch=dynamical_epoch,
        epoch_per_batch_ratio=epoch_per_batch_ratio,
        test_fraction=test_fraction,
        exact_sample_counts=exact_sample_counts,
        exact_train_samples=exact_train_samples,
        exact_test_samples=exact_test_samples,
        loss_type=loss_type,
        corruption_mode=corruption_mode,
        ps=ps,
        sigmas=sigmas,
        batch_sizes=batch_sizes,
        width=width,
        mlp_depth=mlp_depth,
    )
    per_run_path = os.path.join(output_dir, f"results_per_run_{cache_key}{suffix_tag}.csv")
    summary_path = os.path.join(output_dir, f"results_summary_{cache_key}{suffix_tag}.csv")
    write_csv(per_run_path, results)

    summary_rows = summarize_runs(results)
    write_csv(summary_path, summary_rows)

    pdf_path = os.path.join("results", "figures", f"accuracy_vs_{cache_key}{suffix_tag}.pdf")
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "activations": activations,
        "model_types": model_types,
        "corruption_mode": corruption_mode,
        "ps": ps,
        "sigmas": sigmas,
        "batch_sizes": batch_sizes,
        "width": width,
        "mlp_depth": mlp_depth,
        "min_repeats": min_repeats,
        "max_repeats": max_repeats,
        "stderr_target": stderr_target,
        "epochs": epochs,
        "dynamical_epoch": dynamical_epoch,
        "epoch_per_batch_ratio": epoch_per_batch_ratio,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "test_fraction": test_fraction,
        "exact_sample_counts": exact_sample_counts,
        "exact_train_samples": exact_train_samples,
        "exact_test_samples": exact_test_samples,
        "max_workers": max_workers,
        "max_in_flight": max_in_flight,
        "data_workers": data_workers,
        "cpu_threads_per_worker": cpu_threads_per_worker,
        "cpu_cores": cpu_cores,
        "brightness_scale": brightness_scale,
        "custom_split": custom_split,
        "split_seed": split_seed,
        "split_source": split_source,
        "output_dir": output_dir,
        "seed": seed,
        "max_train_samples": max_train_samples,
        "use_cuda": use_cuda,
        "total_runs": len(results),
    }
    write_summary_pdf(pdf_path, summary_rows, metadata)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Saved:")
    print(f"  {per_run_path}")
    print(f"  {summary_path}")
    print(f"  {pdf_path}")
    elapsed = datetime.now() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Elapsed: {elapsed}")
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    runtime_log_path = os.path.join(output_dir, f"runtime_log_{script_name}.csv")
    append_runtime_log(
        runtime_log_path,
        {
            "timestamp_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": int(elapsed.total_seconds()),
            "script": "run_experiments_noisy_training_data_batchsize_sweep.py",
            "cache_key": cache_key,
            "output_dir": output_dir,
            "total_runs": len(results),
        },
    )


if __name__ == "__main__":
    main()
