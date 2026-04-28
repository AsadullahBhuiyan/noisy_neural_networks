from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


################################################################################
# USER INPUTS
#
# Edit only this block for normal use. The input is the CSV produced by
# summarize_nested_ensemble_test_outputs.py.
################################################################################

INPUT_CSV = "nested_means_stdvs.csv"
OUTPUT_CSV = "nested_by_d_testpoint_means.csv"
SUMMARY_CSV = "nested_by_d_summary.csv"

# Which std column to average. "nested_std" is the sample std across noisy
# dataset means in the upstream summarizer.
STD_COLUMN = "nested_std"
MEAN_COLUMN = "nested_mean"

################################################################################
# END USER INPUTS
################################################################################


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else _script_dir() / path


def _float_or_nan(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[mid])
    return float(0.5 * (sorted_values[mid - 1] + sorted_values[mid]))


def _load_rows(input_csv: Path) -> tuple[list[dict[str, str]], list[int]]:
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_csv} has no header.")
        required = {"feature_dim", "test_index", "true_label", "class_id", MEAN_COLUMN, STD_COLUMN}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{input_csv} is missing required columns: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{input_csv} has no data rows.")

    class_ids = sorted({int(row["class_id"]) for row in rows})
    return rows, class_ids


def _aggregate(rows: list[dict[str, str]], class_ids: list[int]):
    by_d_test: dict[tuple[int, int], dict[int, dict[str, float]]] = defaultdict(dict)
    true_label_by_d_test: dict[tuple[int, int], int] = {}
    metadata_by_d: dict[int, dict[str, str]] = {}
    std_values_by_d: dict[int, list[float]] = defaultdict(list)

    for row in rows:
        feature_dim = int(row["feature_dim"])
        test_index = int(row["test_index"])
        class_id = int(row["class_id"])
        key = (feature_dim, test_index)

        mean_value = _float_or_nan(row[MEAN_COLUMN])
        std_value = _float_or_nan(row[STD_COLUMN])
        by_d_test[key][class_id] = {
            "mean": mean_value,
            "std": std_value,
        }
        true_label_by_d_test[key] = int(row["true_label"])
        if _is_finite(std_value):
            std_values_by_d[feature_dim].append(std_value)

        if feature_dim not in metadata_by_d:
            metadata_by_d[feature_dim] = {
                "noise_type": row.get("noise_type", ""),
                "noise_probability": row.get("noise_probability", ""),
                "train_size": row.get("train_size", ""),
                "img_h": row.get("img_h", ""),
                "img_w": row.get("img_w", ""),
                "loss_name": row.get("loss_name", ""),
                "activation": row.get("activation", ""),
                "depth": row.get("depth", ""),
                "width": row.get("width", ""),
                "used_num_noise_datasets": row.get("used_num_noise_datasets", ""),
                "used_init_count_min": row.get("used_init_count_min", ""),
                "used_init_count_max": row.get("used_init_count_max", ""),
            }

    summary_by_d = {}
    for feature_dim, std_values in std_values_by_d.items():
        summary_by_d[feature_dim] = {
            "avg_std_all_testpoints_labels": _mean(std_values),
            "median_std_all_testpoints_labels": _median(std_values),
            "min_std_all_testpoints_labels": float(min(std_values)),
            "max_std_all_testpoints_labels": float(max(std_values)),
            "num_std_values": int(len(std_values)),
        }

    return by_d_test, true_label_by_d_test, metadata_by_d, summary_by_d


def _main_fieldnames(class_ids: list[int]) -> list[str]:
    fields = [
        "feature_dim",
        "test_index",
        "true_label",
        "avg_std_all_testpoints_labels_for_d",
        "avg_std_this_testpoint_all_labels",
        "num_labels",
        "noise_type",
        "noise_probability",
        "train_size",
        "img_h",
        "img_w",
        "loss_name",
        "activation",
        "depth",
        "width",
        "used_num_noise_datasets",
        "used_init_count_min",
        "used_init_count_max",
    ]
    fields.extend(f"mean_label_{class_id}" for class_id in class_ids)
    fields.extend(f"std_label_{class_id}" for class_id in class_ids)
    return fields


def _summary_fieldnames() -> list[str]:
    return [
        "feature_dim",
        "avg_std_all_testpoints_labels",
        "median_std_all_testpoints_labels",
        "min_std_all_testpoints_labels",
        "max_std_all_testpoints_labels",
        "num_std_values",
        "num_testpoints",
        "num_labels",
        "noise_type",
        "noise_probability",
        "train_size",
        "img_h",
        "img_w",
        "loss_name",
        "activation",
        "depth",
        "width",
        "used_num_noise_datasets",
        "used_init_count_min",
        "used_init_count_max",
    ]


def write_outputs() -> tuple[Path, Path]:
    input_csv = _resolve_path(INPUT_CSV)
    output_csv = _resolve_path(OUTPUT_CSV)
    summary_csv = _resolve_path(SUMMARY_CSV)

    rows, class_ids = _load_rows(input_csv)
    by_d_test, true_label_by_d_test, metadata_by_d, summary_by_d = _aggregate(rows, class_ids)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_main_fieldnames(class_ids))
        writer.writeheader()
        for feature_dim, test_index in sorted(by_d_test):
            class_payload = by_d_test[(feature_dim, test_index)]
            std_this_test = [
                class_payload[class_id]["std"]
                for class_id in class_ids
                if class_id in class_payload and _is_finite(class_payload[class_id]["std"])
            ]
            out = {
                "feature_dim": int(feature_dim),
                "test_index": int(test_index),
                "true_label": int(true_label_by_d_test[(feature_dim, test_index)]),
                "avg_std_all_testpoints_labels_for_d": summary_by_d[feature_dim][
                    "avg_std_all_testpoints_labels"
                ],
                "avg_std_this_testpoint_all_labels": _mean(std_this_test) if std_this_test else "",
                "num_labels": int(len(class_payload)),
                **metadata_by_d[feature_dim],
            }
            for class_id in class_ids:
                values = class_payload.get(class_id)
                out[f"mean_label_{class_id}"] = "" if values is None else values["mean"]
                out[f"std_label_{class_id}"] = "" if values is None else values["std"]
            writer.writerow(out)

    test_counts_by_d: dict[int, int] = defaultdict(int)
    for feature_dim, _test_index in by_d_test:
        test_counts_by_d[feature_dim] += 1

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_summary_fieldnames())
        writer.writeheader()
        for feature_dim in sorted(summary_by_d):
            writer.writerow(
                {
                    "feature_dim": int(feature_dim),
                    **summary_by_d[feature_dim],
                    "num_testpoints": int(test_counts_by_d[feature_dim]),
                    "num_labels": int(len(class_ids)),
                    **metadata_by_d[feature_dim],
                }
            )

    json_path = summary_csv.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "output_csv": str(output_csv),
                "summary_csv": str(summary_csv),
                "mean_column": MEAN_COLUMN,
                "std_column": STD_COLUMN,
                "class_ids": class_ids,
                "feature_dims": sorted(int(v) for v in summary_by_d),
                "description": (
                    "The main CSV has one row per feature_dim and test_index. "
                    "avg_std_all_testpoints_labels_for_d is the average of STD_COLUMN "
                    "over all rows with that feature_dim, i.e. all selected test points and labels. "
                    "mean_label_k stores the nested ensemble mean for class k at that test point."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_csv, summary_csv


def main() -> None:
    output_csv, summary_csv = write_outputs()
    print(f"Saved per-d/test-point means to {output_csv}", flush=True)
    print(f"Saved per-d std summary to {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
