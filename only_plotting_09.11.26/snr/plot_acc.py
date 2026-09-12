import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/data2/jt577/2026/papers/noisy_networks/06.2026/fig3/trained_mlp_nested_ensembles/empirical_mean_model_accuracy_summary.csv"
outdir = "/data2/jt577/2026/papers/noisy_networks/06.2026/fig3/trained_mlp_nested_ensembles"

df = pd.read_csv(csv_path)

# Keep only additive_gaussian rows
df = df[df["folder"].str.contains("replacement", na=False)].copy()

df = df.dropna(subset=["N", "d", "d_label", "mean_model_accuracy", "std_model_accuracy"]).copy()

# Make sure numeric columns are numeric
df["N"] = pd.to_numeric(df["N"])
df["d"] = pd.to_numeric(df["d"])
df["mean_model_accuracy"] = pd.to_numeric(df["mean_model_accuracy"])
df["std_model_accuracy"] = pd.to_numeric(df["std_model_accuracy"])

plt.figure(figsize=(8, 6))

styles = [
    {"color": "#1f77b4", "marker": "o"},
    {"color": "#d62728", "marker": "s"},
    {"color": "#2ca02c", "marker": "^"},
    {"color": "#9467bd", "marker": "D"},
]

for i, d_value in enumerate(sorted(df["d"].unique())):
    subdf = df[df["d"] == d_value].sort_values("N")
    d_label = subdf["d_label"].iloc[0]

    plt.errorbar(
        subdf["N"],
        subdf["mean_model_accuracy"] * 100,
        yerr=subdf["std_model_accuracy"] * 100,
        fmt="o-",
        capsize=3,
        markersize=10,
        linewidth=3,
        label=fr"$d={d_label}$",
        color=styles[i]["color"],
        marker=styles[i]["marker"]
    )

plt.xlabel(r"$N$", fontsize=20)
plt.ylabel("Test accuracy (%)", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig(f"{outdir}/mean_test_accuracy_vs_N_fixed_d_replacement.png", dpi=300)
plt.show()