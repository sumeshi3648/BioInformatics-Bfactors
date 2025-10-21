#!/usr/bin/env python3
"""
QC and visualization for nuclear receptor structure dataset.
Saves plots to PNGs instead of showing interactive windows.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# --- make matplotlib non-interactive & fast ---
import matplotlib
matplotlib.use("Agg")           # render to files, not GUI
import matplotlib.pyplot as plt

# Paths
meta_path = Path("data/meta/structures.csv")
bf_path   = Path("data/processed/ca_bfactors.csv")
plot_dir  = Path("data/meta/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# Load
meta = pd.read_csv(meta_path)
bf   = pd.read_csv(bf_path)

proteins = pd.read_csv("proteins.csv")

# Which proteins have at least one PDB in meta
covered = set(meta["uniprot"])
proteins["has_structure"] = proteins["uniprot"].isin(covered)

missing_proteins = proteins[~proteins["has_structure"]]

print("\nProteins with no structures:")
print(missing_proteins)

# Some people named the column "b_factor" in older versions, normalize:
if "bfactor" not in bf.columns and "b_factor" in bf.columns:
    bf = bf.rename(columns={"b_factor": "bfactor"})

print("Loaded:")
print(f"  Structures meta: {len(meta)} rows")
print(f"  B-factor rows:   {len(bf)}")
print(f"  Unique PDBs in B-factors: {bf['pdb_id'].nunique()}")

# coverage
counts = meta.groupby("symbol")["pdb_id"].nunique().sort_values(ascending=False)
print("\nStructures per protein (top 15):")
print(counts.head(15))

missing = counts[counts == 0]
if not missing.empty:
    print("\nProteins with no PDB structures:")
    print(missing.index.tolist())

#resolution / R-free QC
# bad_res = meta[meta["resolution"] > 3.0]
# if len(bad_res) > 0:
#     print(f"\n{len(bad_res)} structures have resolution worse than 3.0 Å:")
#     print(bad_res[["pdb_id", "symbol", "resolution"]].to_string(index=False))
# else:
#     print("\nAll structures have acceptable resolution (≤3.5 Å)")

# if meta["rfree"].isna().any():
#     print(f"{meta['rfree'].isna().sum()} structures missing R-free.")

# plot - Resolution distribution per subfamily
plt.figure(figsize=(10,6))
meta.boxplot(column="resolution", by="subfamily", rot=45)
plt.ylabel("Resolution (Å)")
plt.title("Distribution of structure resolution per NR subfamily")
plt.suptitle("")
plt.tight_layout()
plt.savefig(plot_dir / "resolution_by_subfamily.png", dpi=300)
plt.close()

#plot: Average B-factor per receptor
avg_b = bf.groupby("symbol")["bfactor"].mean().sort_values()
plt.figure(figsize=(10,12))
avg_b.plot(kind="barh")
plt.xlabel("Average Cα B-factor")
plt.title("Average flexibility per receptor (all structures)")
plt.tight_layout()
plt.savefig(plot_dir / "avg_bfactor_per_receptor.png", dpi=300)
plt.close()

#plot: Std B-factor per receptor
avg_b = bf.groupby("symbol")["bfactor"].std().sort_values()
plt.figure(figsize=(10,12))
avg_b.plot(kind="barh")
plt.xlabel("Std Cα B-factor")
plt.title("Std flexibility per receptor (all structures)")
plt.tight_layout()
plt.savefig(plot_dir / "std_bfactor_per_receptor.png", dpi=300)
plt.close()

#plot resolution vs mean B-factor per structure
merged = (bf.groupby("pdb_id")["bfactor"].mean()
            .reset_index()
            .merge(meta, on="pdb_id", how="left"))

plt.figure(figsize=(7,6))
plt.scatter(merged["resolution"], merged["bfactor"], alpha=0.6)
plt.xlabel("Resolution (Å)")
plt.ylabel("Mean Cα B-factor")

a, b = np.polyfit(merged["resolution"], merged["bfactor"], 1)
x = np.array([merged["resolution"].min(), merged["resolution"].max()])
plt.plot(x, a*x + b, color="red", linestyle="--", label=f"y={a:.1f}x+{b:.1f}")
plt.legend()

df = pd.read_csv("data/meta/structure_summary.csv")
df.dropna(subset=["mean", "resolution"], inplace=True)
corr = df["resolution"].corr(df["mean"])
print(f"\nCorrelation between resolution and mean B-factor: {corr:.3f}")


plt.title("Resolution vs mean B-factor per structure")
plt.tight_layout()
plt.savefig(plot_dir / "resolution_vs_mean_bfactor.png", dpi=300)
plt.close()

#resolution vs std b factor
merged = (bf.groupby("pdb_id")["bfactor"].std()
            .reset_index()
            .merge(meta, on="pdb_id", how="left"))

plt.figure(figsize=(10,6))
plt.scatter(merged["resolution"], merged["bfactor"], alpha=0.6)
plt.xlabel("Resolution (Å)")
plt.ylabel("Std Cα B-factor")
plt.title("Resolution vs std B-factor per structure")
plt.tight_layout()
plt.savefig(plot_dir / "resolution_vs_std_bfactor.png", dpi=300)
plt.close()

#summary table
summary = (bf.groupby("pdb_id")["bfactor"]
             .agg(mean="mean", std="std", count="count")
             .merge(meta, on="pdb_id", how="left"))
summary_out = Path("data/meta/structure_summary.csv")
summary.to_csv(summary_out, index=False)

print("\n📄 Summary saved:", summary_out)
print("Plots saved to:", plot_dir)
print("QC complete (non-interactive).")

# --- Compute global correlation (all structures combined) ---
summary = pd.read_csv("data/meta/structure_summary.csv")
summary = summary.dropna(subset=["mean", "resolution"])

corr_global = summary["resolution"].corr(summary["mean"])
print(f"\nGlobal correlation (resolution vs mean B-factor): {corr_global:.3f}")


# --- Per-receptor correlation ---
corr_per_receptor = (
    summary.dropna(subset=["mean", "resolution"])
           .groupby("symbol")
           .apply(lambda df: df["resolution"].corr(df["mean"]))
           .reset_index(name="correlation")
)
# --- Per-subfamily correlation ---
print("\nCorrelation per receptor:")
print(corr_per_receptor.sort_values("correlation", ascending=False))
corr_per_receptor.to_csv("data/meta/correlation_per_receptor.csv", index=False)


corr_per_subfam = (
    summary.dropna(subset=["mean", "resolution"])
           .groupby("subfamily")
           .apply(lambda df: df["resolution"].corr(df["mean"]))
           .reset_index(name="correlation")
)

print("\nCorrelation per subfamily:")
print(corr_per_subfam.sort_values("correlation", ascending=False))
corr_per_subfam.to_csv("data/meta/correlation_per_subfamily.csv", index=False)

# --- Mean ± SD B-factors and all resolutions per receptor ---
b_summary = (
    summary.groupby("symbol")
           .agg(mean_bfactor=("mean", "mean"),
                std_bfactor=("mean", "std"),
                n_structures=("pdb_id", "count"))
           .reset_index()
)

res_lookup = (summary.groupby("symbol")["resolution"]
                     .apply(lambda x: ", ".join(sorted(map(lambda v: f"{v:.2f}", x.unique()))))
                     .reset_index(name="resolutions"))

b_summary = b_summary.merge(res_lookup, on="symbol", how="left")
b_summary.to_csv("data/meta/bfactor_resolution_summary_per_receptor.csv", index=False)

print("\nSaved: bfactor_resolution_summary_per_receptor.csv")


# Plot: Mean ± SD B-factors per receptor
b_summary = pd.read_csv("data/meta/bfactor_resolution_summary_per_receptor.csv")
b_summary = b_summary.sort_values("mean_bfactor", ascending=True)

plt.figure(figsize=(10, 12))
plt.barh(b_summary["symbol"],
         b_summary["mean_bfactor"],
         xerr=b_summary["std_bfactor"],
         color="skyblue",
         ecolor="gray",
         capsize=3)

plt.xlabel("Mean Cα B-factor (Å²)")
plt.ylabel("Receptor (symbol)")
plt.title("Mean ± SD B-factors per receptor")
plt.tight_layout()
plt.savefig("data/meta/plots/mean_sd_bfactor_per_receptor.png", dpi=300)
plt.close()
