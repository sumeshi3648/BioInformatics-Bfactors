#!/usr/bin/env python3
"""
QC and visualization for nuclear receptor structure dataset.
Saves plots to PNGs instead of showing interactive windows.
"""

from pathlib import Path
import pandas as pd

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

# Some people named the column "b_factor" in older versions, normalize:
if "bfactor" not in bf.columns and "b_factor" in bf.columns:
    bf = bf.rename(columns={"b_factor": "bfactor"})

print("Loaded:")
print(f"  Structures meta: {len(meta)} rows")
print(f"  B-factor rows:   {len(bf)}")
print(f"  Unique PDBs in B-factors: {bf['pdb_id'].nunique()}")

# 1) Coverage
counts = meta.groupby("symbol")["pdb_id"].nunique().sort_values(ascending=False)
print("\nStructures per protein (top 15):")
print(counts.head(15))

missing = counts[counts == 0]
if not missing.empty:
    print("\nProteins with no PDB structures:")
    print(missing.index.tolist())

# 2) Resolution / R-free QC
bad_res = meta[meta["resolution"] > 3.0]
if len(bad_res) > 0:
    print(f"\n{len(bad_res)} structures have resolution worse than 3.5 Å:")
    print(bad_res[["pdb_id", "symbol", "resolution"]].to_string(index=False))
else:
    print("\nAll structures have acceptable resolution (≤3.5 Å)")

if meta["rfree"].isna().any():
    print(f"{meta['rfree'].isna().sum()} structures missing R-free.")

# 3) Plot: Resolution distribution per subfamily
plt.figure(figsize=(10,6))
meta.boxplot(column="resolution", by="subfamily", rot=45)
plt.ylabel("Resolution (Å)")
plt.title("Distribution of structure resolution per NR subfamily")
plt.suptitle("")
plt.tight_layout()
plt.savefig(plot_dir / "resolution_by_subfamily.png", dpi=300)
plt.close()

# 4) Plot: Average B-factor per receptor
avg_b = bf.groupby("symbol")["bfactor"].mean().sort_values()
plt.figure(figsize=(10,12))
avg_b.plot(kind="barh")
plt.xlabel("Average Cα B-factor")
plt.title("Average flexibility per receptor (all structures)")
plt.tight_layout()
plt.savefig(plot_dir / "avg_bfactor_per_receptor.png", dpi=300)
plt.close()

# 5) Plot: Resolution vs mean B-factor per structure
merged = (bf.groupby("pdb_id")["bfactor"].mean()
            .reset_index()
            .merge(meta, on="pdb_id", how="left"))

plt.figure(figsize=(7,6))
plt.scatter(merged["resolution"], merged["bfactor"], alpha=0.6)
plt.xlabel("Resolution (Å)")
plt.ylabel("Mean Cα B-factor")
plt.title("Resolution vs mean B-factor per structure")
plt.tight_layout()
plt.savefig(plot_dir / "resolution_vs_mean_bfactor.png", dpi=300)
plt.close()

# 6) Summary table
summary = (bf.groupby("pdb_id")["bfactor"]
             .agg(mean="mean", std="std", count="count")
             .merge(meta, on="pdb_id", how="left"))
summary_out = Path("data/meta/structure_summary.csv")
summary.to_csv(summary_out, index=False)

print("\n📄 Summary saved:", summary_out)
print("Plots saved to:", plot_dir)
print("QC complete (non-interactive).")
