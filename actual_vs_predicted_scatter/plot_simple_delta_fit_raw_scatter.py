import csv
import json
from pathlib import Path
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent / "subset__test=1000"
CSV_PATH = BASE / "per_test_class_comparison.csv"
JSON_PATH = BASE / "comparison_database.json"
OUTPUT_PATH = BASE / "simple_delta_fit_raw_scatter.png"


with JSON_PATH.open("r", encoding="utf-8") as handle:
    database = json.load(handle)

fit = database["summary"]["simple_delta_fit_raw"]
a = float(fit["a"])
b = float(fit["b"])
r2 = fit.get("r2")

x = []
y = []
with CSV_PATH.open("r", newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        target = float(row["empirical_mean"])
        delta = float(row["simple_delta_xstar_x_centered"])
        y.append(target)
        x.append(a * delta + b)

lo = min(min(x), min(y))
hi = max(max(x), max(y))
pad = 0.05 * (hi - lo)
lo -= pad
hi += pad

plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=30, alpha=0.05, edgecolors="none", color="b")
plt.plot([lo, hi], [lo, hi], "k--", linewidth=2.0, color="r")
plt.xlim(lo, hi)
plt.ylim(lo, hi)
plt.ylabel(r"Actual outputs", fontsize=24)
plt.xlabel(r"Predicted outputs", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
if r2 is not None:
    plt.text(0.04, 0.96, rf"$R^2 = {float(r2):.3f}$", transform=plt.gca().transAxes, va="top", fontsize=20)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
print(f"saved {OUTPUT_PATH}")