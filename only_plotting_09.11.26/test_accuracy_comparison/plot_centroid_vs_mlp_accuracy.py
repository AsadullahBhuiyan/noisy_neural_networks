"""Plot both accuracy grids for every fit_summary*.json by default.

Optional filters (for example --dataset mnist --depth 5) select a subset.
Titles and filenames retain the fit configuration to distinguish experiments.
"""
import argparse
import csv
import json
import math
import shlex
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
CASE_FIELDS = (
    "dataset", "eval_noise_type", "fit_noise_type", "fit_p", "N",
    "d_label", "loss", "activation", "depth", "width",
)
PLOT_GRIDS = {
    "p0.9_to_1.0": [i / 100 for i in range(90, 101)],
    "p0_to_1_deciles": [i / 10 for i in range(11)],
}


def load_case(summary_path):
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    settings = dict(summary["settings"])
    settings["fit_noise_type"] = settings.get("fit_noise_type") or settings["eval_noise_type"]
    for field in CASE_FIELDS:
        if settings.get(field) is None:
            raise ValueError(f"Missing {field!r} in fit summary settings")
    # Prefer the adjacent CSV so a copied results directory is self-contained.
    recorded_csv = Path(summary["output_csv"])
    local_csv = summary_path.parent / recorded_csv.name
    csv_path = local_csv if local_csv.is_file() else recorded_csv
    if not csv_path.is_absolute():
        csv_path = PROJECT_DIR / csv_path
    return settings, csv_path


def read_rows(settings, csv_path):
    dataset = settings["dataset"]
    noise_type = settings["eval_noise_type"]
    fit_noise_type = settings["fit_noise_type"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["noise_type"] == noise_type
            and row.get("dataset", dataset) == dataset
        ]
    pattern = (
        f"mlp_ensemble__dataset={dataset}__{noise_type}__*__N={settings['N']}__d={settings['d_label']}"
        f"__loss={settings['loss']}__act={settings['activation']}"
        f"__L={settings['depth']}__width={settings['width']}__*"
    )
    ensemble_root = Path(settings.get("ensemble_root", PROJECT_DIR / "trained_mlp_nested_ensembles"))
    if not ensemble_root.is_absolute():
        ensemble_root = PROJECT_DIR / ensemble_root
    summarized_runs = {row.get("ensemble_dir") for row in rows}
    missing_runs = [
        path.name
        for path in ensemble_root.glob(pattern)
        if (path / "test_outputs.csv.gz").is_file()
        and path.name not in summarized_runs
    ]
    if missing_runs:
        command = shlex.join([
            sys.executable, str(PROJECT_DIR / "eval_noisy_centroid_and_network_accuracy.py"),
            "--single-case", "--only-missing", "--dataset", dataset, "--eval-noise-type", noise_type,
            "--fit-noise-type", fit_noise_type, "--fit-p", str(settings["fit_p"]),
            "--N", str(settings["N"]), "--d-label", settings["d_label"], "--loss", settings["loss"],
            "--activation", settings["activation"], "--depth", str(settings["depth"]),
            "--width", str(settings["width"]), "--ensemble-root", str(ensemble_root),
            "--output-dir", str(csv_path.parent),
        ])
        warnings.warn(
            f"{csv_path.name} is missing {len(missing_runs)} matching runs. "
            f"Refresh the summary, then rerun this script:\n{command}", stacklevel=2,
        )
    return sorted(rows, key=lambda row: float(row["noise_probability"]))


def select_rows(rows, grid):
    selected = []
    for p in grid:
        matches = [row for row in rows if math.isclose(
            float(row["noise_probability"]), p, rel_tol=0, abs_tol=1e-9
        )]
        if len(matches) != 1:
            raise ValueError(f"Expected one summary row for p={p:g}; found {len(matches)}. Refresh the summary first.")
        selected.append(matches[0])
    return selected


def plot_case(summary_path, settings, csv_path, output_dir):
    rows = read_rows(settings, csv_path)
    # Validate both grids before writing either plot.
    selections = {name: select_rows(rows, grid) for name, grid in PLOT_GRIDS.items()}
    outputs = []
    for name, selected in selections.items():
        p_vals = [float(row["noise_probability"]) for row in selected]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(
            p_vals,
            [100 * float(row["empirical_mean_test_accuracy"]) for row in selected],
            yerr=[100 * float(row["empirical_std_test_accuracy"]) for row in selected],
            fmt="s-", label="Neural Network", color="red", linewidth=4, zorder=1,
        )
        ax.errorbar(
            p_vals,
            [100 * float(row["mean_test_accuracy"]) for row in selected],
            yerr=[100 * float(row["std_test_accuracy"]) for row in selected],
            fmt="o--", label="Analytical model", color="blue", linewidth=3, zorder=2,
        )
        ax.set_xlabel(r"$p$", fontsize=20)
        ax.set_ylabel("Test accuracy (%)", fontsize=20)
        ax.set_title(
            f"{settings['dataset']} | eval={settings['eval_noise_type']}\n"
            f"L={settings['depth']}, width={settings['width']}, act={settings['activation']}, "
            f"loss={settings['loss']}, N={settings['N']}, d={settings['d_label']}\n"
            f"fit={settings['fit_noise_type']}, fit p={settings['fit_p']:g}",
            fontsize=11,
        )
        ax.set_ylim(0, 100)
        ax.margins(x=0.02)
        if name == "p0_to_1_deciles":
            ax.set_xticks(PLOT_GRIDS[name])
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=16, loc="upper right")
        fig.tight_layout()
        # The summary stem preserves every configuration tag, including future ones.
        case_id = summary_path.stem.removeprefix("fit_summary__")
        output = output_dir / f"accuracy_comparison__{case_id}__{name}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300)
        plt.close(fig)
        outputs.append(output)
        print(f"Wrote {output} ({len(selected)} points)")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-cases", action="store_true", help="Plot all matching summaries (the default).")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_DIR / "trained_centroids")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the input directory.")
    parser.add_argument("--dataset")
    parser.add_argument("--noise-type", "--eval-noise-type", dest="eval_noise_type")
    parser.add_argument("--fit-noise-type")
    parser.add_argument("--fit-p", type=float)
    parser.add_argument("--N", type=int)
    parser.add_argument("--d-label")
    parser.add_argument("--loss")
    parser.add_argument("--activation")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--width", type=int)
    args = parser.parse_args()
    summary_paths = sorted(args.input_dir.resolve().glob("fit_summary*.json"))
    if not summary_paths:
        parser.error(f"No fit_summary*.json files found in {args.input_dir}")
    output_dir = (args.output_dir or args.input_dir).resolve()
    plotted = failed = 0
    for summary_path in summary_paths:
        try:
            settings, csv_path = load_case(summary_path)
            if any(
                getattr(args, field) is not None and getattr(args, field) != settings[field]
                for field in CASE_FIELDS
            ):
                continue
            plot_case(summary_path, settings, csv_path, output_dir)
            plotted += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failed += 1
            print(f"Failed {summary_path.name}: {exc}", file=sys.stderr)
    if not plotted and not failed:
        parser.error("No fit summaries match the requested filters")
    print(f"Plotted {plotted} cases ({plotted * len(PLOT_GRIDS)} plots); {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
