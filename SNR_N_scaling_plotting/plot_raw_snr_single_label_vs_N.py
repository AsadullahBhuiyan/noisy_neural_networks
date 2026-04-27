import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

INPUT = "nested_by_N_testpoint_means.csv"
OUTPUT = "snr_label6_true6_collapsed.png"
LABEL = 6
XTICKS = [100, 200, 300, 400]
FONT_SIZE = 24
TICK_SIZE = 20
LINE_WIDTH = 2.0
ALPHA = 0.3

curves = defaultdict(list)
for r in csv.DictReader(open(INPUT, newline="")):
    if int(r["true_label"]) != LABEL:
        continue
    N = int(r["train_size"])
    mean = float(r[f"mean_label_{LABEL}"])
    std = float(r["avg_std_all_testpoints_labels_for_N"])
    curves[int(r["test_index"])].append((N, mean * mean / (std * std)))

plt.figure(figsize=(7.5, 5.2))
for test_index, pts in sorted(curves.items()):
    pts = sorted(pts)
    N = np.array([p[0] for p in pts], dtype=float)
    snr = np.array([p[1] for p in pts], dtype=float)
    a, b = np.polyfit(N, snr, 1)
    plt.plot(N, (snr - b) / a, linewidth=LINE_WIDTH, alpha=ALPHA, color="b", marker="o", markersize=10)

plt.plot(XTICKS, XTICKS, "k--", linewidth=3.0)
plt.xticks(XTICKS, fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.locator_params(axis="y", nbins=4)
plt.xlabel("N", fontsize=FONT_SIZE)
plt.ylabel("Normalized SNR", fontsize=FONT_SIZE)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=300)
