#!/usr/bin/env python3
"""Compute per-example SNR from existing nested-ensemble test_outputs.csv[.gz].

For scalar output z[k,m,t], first average over initializations m, then use
SNR[t] = mean_k(mean_m(z))**2 / var_k(mean_m(z), ddof=1).
No model loading, training, torch, or dataset download is needed.
See README_snr_scaling.md for usage and interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Edit these defaults, or override them on the command line.
ROOT_DIR = Path("/data2/jt577/2026/papers/noisy_networks/06.2026/fig3/trained_mlp_nested_ensembles")
NUM_TEST_POINTS = 500
RANDOM_SEED = 22334
CHUNK_ROWS = 100_000

# Keep distinct experimental conditions in separate fits and figures.
CONDITION_KEYS = (
    "dataset", "noise_type", "noise_probability", "activation", "depth", "width",
    "weight_std", "bias_std", "epochs", "batch_size", "lr", "optimizer",
    "weight_decay", "loss_name", "num_classes", "seed",
    "num_noise_datasets", "num_models_per_noise_dataset",
)


def load_run(folder: Path) -> dict:
    meta = json.loads((folder / "metadata.json").read_text())
    cfg = meta["config"]
    layout = meta.get("nested_ensemble_layout", cfg)
    n, d, nt = (int(meta[k]) for k in ("num_train", "feature_dim", "num_test"))
    k = int(layout["num_noise_datasets"])
    m = int(layout["num_models_per_noise_dataset"])
    if min(n, d, nt, m) < 1 or k < 2:
        raise ValueError(f"Invalid dimensions or fewer than two noisy datasets: {folder}")
    filename = cfg.get("test_outputs_filename", "test_outputs.csv.gz")
    path = folder / Path(filename).name  # Metadata may contain an old absolute path.
    if not path.exists() and (folder / "test_outputs.csv").exists():
        path = folder / "test_outputs.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    condition = {key: cfg.get(key) for key in CONDITION_KEYS}
    condition.update(dataset=meta.get("dataset", cfg.get("dataset")),
                     num_noise_datasets=k, num_models_per_noise_dataset=m)
    condition["test_is_clean"] = meta.get("test_is_clean")
    condition["normalization"] = meta.get("normalization")
    key = json.dumps(condition, sort_keys=True)
    group_id = hashlib.sha256(key.encode()).hexdigest()[:12]
    hw = cfg.get("img_hw")
    if hw and int(hw[0]) * int(hw[1]) != d:
        raise ValueError(f"img_hw and feature_dim disagree: {folder}")
    return dict(folder=folder, path=path, meta=meta, cfg=cfg, N=n, d=d, nt=nt,
                K=k, M=m, condition=condition, group_id=group_id,
                d_label=f"{hw[0]}x{hw[1]}" if hw else str(d))


def sampled_indices(dataset: str, num_test: int, count: int, seed: int) -> np.ndarray:
    # Same dataset and test length -> same points across every N, d and noise type.
    salt = int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:8], "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, salt]))
    return np.sort(rng.choice(num_test, size=min(count, num_test), replace=False))


def read_selected(run: dict, indices: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    """Stream input once; store only K x M x selected-test scalar outputs.

    The attached trainer guarantees contiguous zero-based indices and
    model_index = noise_dataset_index * M + init_index. Validate this layout,
    duplicates, labels, dimensions, and completeness for every selected point.
    """
    header = pd.read_csv(run["path"], nrows=0).columns
    logit_cols = sorted((c for c in header if c.startswith("logit_")),
                        key=lambda c: int(c.split("_")[1]))
    c = int(run["cfg"]["num_classes"])
    if logit_cols != [f"logit_{i}" for i in range(c)]:
        raise ValueError("Logit columns do not match num_classes")
    if args.output == "class" and not 0 <= args.class_index < c:
        raise ValueError(f"class-index must be between 0 and {c - 1}")
    if args.output == "margin" and c < 2:
        raise ValueError("Margin needs at least two classes")
    id_cols = ["test_index", "noise_dataset_index", "init_index", "model_index",
               "true_label", "train_size", "feature_dim"]
    values = np.empty((run["K"], run["M"], len(indices)), dtype=np.float64)
    seen = np.zeros(values.shape, dtype=bool)
    labels = np.full(len(indices), -1, dtype=np.int64)
    for chunk_number, chunk in enumerate(pd.read_csv(
        run["path"], usecols=id_cols + logit_cols, chunksize=args.chunk_rows,
    ), start=1):
        part = chunk.loc[chunk["test_index"].isin(indices)]
        if part.empty:
            continue
        raw = part[id_cols].to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError("Missing or noninteger identifiers in selected rows")
        t, k, m, model, y, n, d = raw.astype(np.int64).T
        if ((k < 0) | (k >= run["K"]) | (m < 0) | (m >= run["M"])).any():
            raise ValueError("Noise/init indices disagree with metadata")
        if (model != k * run["M"] + m).any():
            raise ValueError("Model IDs disagree with the supplied trainer's nested layout")
        if ((n != run["N"]) | (d != run["d"]) | (y < 0) | (y >= c)).any():
            raise ValueError("Labels or N/d values disagree with metadata")
        ti = np.searchsorted(indices, t)
        flat = np.ravel_multi_index((k, m, ti), values.shape)
        if len(np.unique(flat)) != len(flat) or seen[k, m, ti].any():
            raise ValueError("Duplicate (noise_dataset_index, init_index, test_index) rows")
        # Check labels both within this chunk and across chunks.
        for pos in np.unique(ti):
            ys = np.unique(y[ti == pos])
            if len(ys) != 1 or labels[pos] not in (-1, ys[0]):
                raise ValueError(f"Inconsistent true_label for test_index={indices[pos]}")
            labels[pos] = ys[0]
        logits = part[logit_cols].to_numpy(dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite logits in selected rows")
        true_logit = logits[np.arange(len(y)), y]
        if args.output == "true_class":
            scalar = true_logit
        elif args.output == "class":
            scalar = logits[:, args.class_index]
        else:
            # Linear margin: averaging over models commutes with this transform.
            scalar = true_logit - (logits.sum(axis=1) - true_logit) / (c - 1)
        values[k, m, ti] = scalar
        seen[k, m, ti] = True
        if chunk_number % 10 == 0:
            print(f"    Read {chunk_number * args.chunk_rows:,} input rows", flush=True)
    if not seen.all():
        raise ValueError(f"Incomplete selected outputs: missing {int((~seen).sum())} "
                         "(dataset, initialization, test point) entries")
    return values, labels


def snr_statistics(values: np.ndarray) -> dict[str, np.ndarray]:
    per_noise = values.mean(axis=1)
    signal = per_noise.mean(axis=0)
    variance = per_noise.var(axis=0, ddof=1)
    # Do not divide by K: noise is variation across datasets, not the SEM.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        snr = signal**2 / variance
        log_snr = 2 * np.log(np.abs(signal)) - np.log(variance)
    valid = (variance > 0) & (signal != 0) & np.isfinite(log_snr)
    status = np.full(signal.shape, "ok", dtype=object)
    status[signal == 0] = "zero_signal"
    status[variance == 0] = "zero_variance"
    status[(signal == 0) & (variance == 0)] = "zero_signal_and_variance"
    status[~valid & (status == "ok")] = "nonfinite"
    # Diagnostic only; keep the user's requested uncorrected estimator.
    init_noise = (values.var(axis=1, ddof=1).mean(axis=0) / values.shape[1]
                  if values.shape[1] > 1 else np.full(signal.shape, np.nan))
    return dict(signal_mean=signal, dataset_variance=variance, snr=snr,
                log_snr=np.where(valid, log_snr, np.nan), valid_log=valid,
                status=status, estimated_init_variance_in_dataset_mean=init_noise)


def make_plots(points: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = []
    fits = []
    for group_id, group in points.groupby("group_id", sort=True):
        good = group.loc[group["valid_log"].astype(str).str.lower() == "true"]
        fig, ax = plt.subplots(figsize=(8, 8))
        folder_rows = []
        for folder, sub in group.groupby("folder", sort=True):
            ok = sub.loc[sub["valid_log"].astype(str).str.lower() == "true"]
            row = dict(group_id=group_id, folder=folder, N=int(sub.N.iloc[0]),
                       d=int(sub.d.iloc[0]), log_Nd=float(sub.log_Nd.iloc[0]),
                       num_points=len(sub), num_valid=len(ok),
                       mean_log_snr=float(ok.log_snr.mean()),
                       std_log_snr=float(ok.log_snr.std(ddof=1)),
                       median_log_snr=float(ok.log_snr.median()),
                       q25_log_snr=float(ok.log_snr.quantile(.25)),
                       q75_log_snr=float(ok.log_snr.quantile(.75)))
            summaries.append(row)
            folder_rows.append(row)
        cells = pd.DataFrame(folder_rows).dropna(subset=["mean_log_snr"])
        fit = dict(group_id=group_id, noise_type=str(group.noise_type.iloc[0]),
                   num_folders=len(group.folder.unique()), num_valid_points=len(good),
                   slope=np.nan, intercept=np.nan, r_squared=np.nan,
                   slope_1_intercept=np.nan, separate_N_exponent=np.nan,
                   separate_d_exponent=np.nan, separate_intercept=np.nan)
        if not cells.empty:
            # Standard deviation across ALL valid test-point log SNRs (not SEM).
            # If a folder has just one valid point, its SD is undefined: show
            # its mean marker but no error bar, and retain NaN in the summary.
            ax.errorbar(cells.log_Nd, cells.mean_log_snr,
                        yerr=cells.std_log_snr.to_numpy(), fmt="s", linestyle="none",
                        color="black", markersize=6, markeredgewidth=0.9,
                        ecolor="0.5", elinewidth=0.9, capsize=3, capthick=0.9,
                        label=r"Mean $\pm 1\sigma$", zorder=5)
        # Fit folder means with equal folder weights. Dots within a folder share
        # trained models, so do not report naive pointwise regression p-values.
        if len(cells) >= 2 and cells.log_Nd.nunique() >= 2:
            x = cells.log_Nd.to_numpy()
            y = cells.mean_log_snr.to_numpy()
            slope, intercept = np.linalg.lstsq(
                np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
            residual = y - (slope * x + intercept)
            total = np.sum((y - y.mean())**2)
            r2 = 1 - np.sum(residual**2) / total if total > 0 else np.nan
            b1 = np.mean(y - x)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, slope * xx + intercept, "--", color="black", linewidth=2,
                    label=fr"Fit: $\mathrm{{slope}}={slope:.2f}$")
            fit.update(slope=slope, intercept=intercept, r_squared=r2,
                       slope_1_intercept=b1)
            design = np.column_stack([np.log(cells.N), np.log(cells.d), np.ones(len(cells))])
            if np.linalg.matrix_rank(design) == 3:
                a, b, c = np.linalg.lstsq(design, y, rcond=None)[0]
                fit.update(separate_N_exponent=a, separate_d_exponent=b,
                           separate_intercept=c)
        fits.append(fit)
        first = group.iloc[0]
        ax.set(xlabel=r"$\log(Nd)$", ylabel=r"$\log(\mathrm{SNR})$")
        ax.xaxis.label.set_size(26)
        ax.yaxis.label.set_size(26)
        ax.tick_params(labelsize=20)
        if not good.empty:
            ax.legend(fontsize=17)
        else:
            ax.text(.5, .5, "No finite positive SNR values", transform=ax.transAxes,
                    ha="center")
        # Equal numerical spans AND equal physical units make slope 1 look 45°.
        # Center each range separately: log SNR can have a nonzero intercept.
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        xmid, ymid = np.mean(xlim), np.mean(ylim)
        ax.set_xlim(xmid - span / 2, xmid + span / 2)
        ax.set_ylim(ymid - span / 2, ymid + span / 2)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        stem = outdir / f"snr_scaling__{first['noise_type']}__{group_id}"
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
    pd.DataFrame(summaries).to_csv(outdir / "snr_folder_summary.csv", index=False)
    pd.DataFrame(fits).to_csv(outdir / "snr_scaling_fits.csv", index=False)
    (outdir / "snr_plot_settings.json").write_text(json.dumps(dict(
        marker="square", error_bars="Plus/minus one sample standard deviation of log SNR across test points",
        error_bar_linewidth=0.9, error_bar_capsize=3, error_bar_capthick=0.9,
        error_bar_ddof=1, axis_label_fontsize=26, tick_fontsize=20, legend_fontsize=17,
        fitted_line_style="dashed", fitted_slope_legend_decimals=2,
        equal_axis_scaling=True, means_and_fits="All valid sampled test points",
    ), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--outdir", type=Path, help="Default: ROOT/snr_scaling/OUTPUT_NOISE")
    parser.add_argument("--num-test-points", type=int, default=NUM_TEST_POINTS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--noise-type", choices=["all", "replacement", "additive_gaussian"], default="all")
    parser.add_argument("--output", choices=["true_class", "class", "margin"], default="true_class",
                        help="Raw true-class logit, fixed-class logit, or true minus mean other logits")
    parser.add_argument("--class-index", type=int, default=0, help="Used only with --output class")
    parser.add_argument("--plot-only", type=Path, metavar="SNR_POINTS_CSV",
                        help="Regenerate figures/summaries from saved small CSV without reading raw outputs")
    args = parser.parse_args()
    if args.num_test_points < 1 or args.chunk_rows < 1 or args.seed < 0:
        parser.error("Sample size and chunk rows must be positive; seed must be nonnegative")
    mode = f"class_{args.class_index}" if args.output == "class" else args.output
    outdir = args.outdir or (args.plot_only.parent if args.plot_only else
                            args.root / "snr_scaling" / f"{mode}_{args.noise_type}")
    outdir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        make_plots(pd.read_csv(args.plot_only), outdir)
        print(f"Saved figures and summaries to {outdir}")
        return
    folders = sorted(p.parent for p in args.root.rglob("metadata.json")
                     if p.parent.name.startswith("mlp_ensemble__"))
    if not folders:
        raise FileNotFoundError(f"No mlp_ensemble__* folders with metadata.json under {args.root}")
    runs = []
    for folder in folders:
        # Filter before loading output files, so unused conditions need not be complete.
        cfg = json.loads((folder / "metadata.json").read_text())["config"]
        if args.noise_type == "all" or cfg["noise_type"] == args.noise_type:
            runs.append(load_run(folder))
    if not runs:
        raise ValueError("No runs match the noise-type filter")
    frames, selection_rows = [], []
    dataset_samples = {}
    conditions = {}
    for i, run in enumerate(runs, 1):
        dataset = run["condition"]["dataset"]
        if dataset not in dataset_samples:
            indices = sampled_indices(dataset, run["nt"], args.num_test_points, args.seed)
            dataset_samples[dataset] = dict(nt=run["nt"], indices=indices, labels=None)
        sample = dataset_samples[dataset]
        if sample["nt"] != run["nt"]:
            raise ValueError(f"Different test-set lengths for {dataset}; run separately")
        indices = sample["indices"]
        print(f"[{i}/{len(runs)}] N={run['N']}, d={run['d']}, "
              f"{run['cfg']['noise_type']}, K={run['K']}, M={run['M']}: {run['folder']}", flush=True)
        try:
            values, labels = read_selected(run, indices, args)
        except Exception as exc:
            raise RuntimeError(f"Failed validating {run['path']}: {exc}") from exc
        if sample["labels"] is None:
            sample["labels"] = labels
            selection_rows.extend(dict(dataset=dataset, test_index=int(t), true_label=int(y))
                                  for t, y in zip(indices, labels))
        elif not np.array_equal(sample["labels"], labels):
            raise ValueError(f"Test labels/order disagree across folders for {dataset}")
        frame = pd.DataFrame(snr_statistics(values))
        frame["test_index"], frame["true_label"] = indices, labels
        for key, val in dict(group_id=run["group_id"], folder=str(run["folder"]),
                             dataset=dataset, noise_type=run["cfg"]["noise_type"],
                             p=run["cfg"]["noise_probability"], N=run["N"], d=run["d"],
                             d_label=run["d_label"], K=run["K"], M=run["M"], output=mode,
                             log_Nd=float(np.log(run["N"]) + np.log(run["d"]))).items():
            frame[key] = val
        frames.append(frame)
        conditions[run["group_id"]] = run["condition"]
        print(f"    {len(frame)} points; {int(frame.valid_log.sum())} valid for log plot", flush=True)
    points = pd.concat(frames, ignore_index=True)
    points.to_csv(outdir / "snr_points.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(outdir / "selected_test_points.csv", index=False)
    manifest = dict(root=str(args.root.resolve()), seed=args.seed,
                    requested_num_test_points=args.num_test_points, output=mode,
                    variance_ddof=1, logarithm="natural", conditions=conditions,
                    estimator="mean_k(mean_m(output))^2 / sample_var_k(mean_m(output))",
                    fit="Equal-weight OLS on folder mean log SNR; descriptive, no independence-based p-values")
    (outdir / "snr_analysis_metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
    make_plots(points, outdir)
    print(f"Saved {len(points):,} per-point rows, summaries, fits and PNG/PDF plots to {outdir}")


if __name__ == "__main__":
    main()
