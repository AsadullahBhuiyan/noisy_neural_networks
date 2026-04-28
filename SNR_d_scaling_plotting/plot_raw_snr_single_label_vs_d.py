import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

INPUT = "nested_by_d_testpoint_means.csv"
LABEL = 9
FONT_SIZE = 24
TICK_SIZE = 20
LINE_WIDTH = 2.0
ALPHA = 0.3
OUTPUT = f"snr_label{LABEL}_true{LABEL}_collapsed_vs_d.png"

curves = defaultdict(list)
d_values = set()
for r in csv.DictReader(open(INPUT, newline="")):
    if int(r["true_label"]) != LABEL:
        continue
    d = int(r["feature_dim"])
    d_values.add(d)
    mean = float(r[f"mean_label_{LABEL}"])
    std = float(r["avg_std_all_testpoints_labels_for_d"])
    snr = mean * mean / (std * std)
    curves[int(r["test_index"])].append((d, snr))

xticks = sorted(d_values)

plt.figure(figsize=(7.5, 5.2))
for test_index, pts in sorted(curves.items()):
    pts = sorted(pts)
    d = np.array([p[0] for p in pts], dtype=float)
    snr = np.array([p[1] for p in pts], dtype=float)
    a, b = np.polyfit(d, snr, 1)
    plt.plot(
        d,
        (snr - b) / a,
        linewidth=LINE_WIDTH,
        alpha=ALPHA,
        color="b",
        marker="o",
        markersize=10,
    )

plt.plot(xticks, xticks, "k--", linewidth=3.0)
plt.xticks(xticks, [str(v) for v in xticks], fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.locator_params(axis="y", nbins=4)
plt.xlabel("d", fontsize=FONT_SIZE)
plt.ylabel("Normalized SNR", fontsize=FONT_SIZE)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=300)
