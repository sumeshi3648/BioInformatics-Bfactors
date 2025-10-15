#!/usr/bin/env python3
"""
Fetch nuclear receptor structures for given UniProt IDs, filter by quality,
and extract Cα (alpha-carbon) B-factors into tidy CSV files.

Outputs:
  data/meta/structures.csv
  data/processed/ca_bfactors.csv
  data/raw/mmcif/<pdb_id>.cif
"""

import argparse
import csv
import json
import re
from pathlib import Path
import time

import pandas as pd
import requests
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from tqdm import tqdm

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_CIF_URL      = "https://files.rcsb.org/download/{pdb}.cif"               # mmCIF download

def read_table(path):
    df = pd.read_csv(path)
    required = {"subfamily","symbol","uniprot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"proteins.csv missing columns: {missing}")
    df = df.dropna(subset=["uniprot"])
    df["uniprot"] = df["uniprot"].str.strip()
    return df

def uniprot_to_pdbs(acc: str) -> set[str]:
    """
    Map a UniProt accession to PDB entry IDs using the RCSB Search API.
    Robust to empty/HTML responses and transient errors.
    """
    if not acc or not isinstance(acc, str):
        return set()

    payload = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": acc
            }
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True}
    }

    for attempt in range(3):
        try:
            r = requests.post(
                RCSB_SEARCH_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if r.status_code != 200 or not r.content or not r.content.strip():
                time.sleep(1.5 * (attempt + 1))
                continue
            data = r.json() 
            return {hit["identifier"].lower() for hit in data.get("result_set", [])}
        except Exception:
            time.sleep(1.5 * (attempt + 1))  

    return set()

def download_mmcif(pdb_id, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{pdb_id}.cif"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = RCSB_CIF_URL.format(pdb=pdb_id.upper())
    resp = requests.get(url, timeout=60)
    if resp.status_code == 200:
        path.write_bytes(resp.content)
        return path
    else:
        raise RuntimeError(f"Failed to download {pdb_id}: HTTP {resp.status_code}")

def _first_float(d, keys):
    for k in keys:
        if k in d:
            try:
                # Some mmCIF values are lists; take first element
                val = d[k][0] if isinstance(d[k], list) else d[k]
                return float(val)
            except Exception:
                continue
    return None

def _first_str(d, keys):
    for k in keys:
        if k in d:
            val = d[k][0] if isinstance(d[k], list) else d[k]
            return str(val)
    return None

def parse_quality_and_method(cif_path):
    """Return dict with method, resolution (Å), Rfree, plus pdb_id."""
    d = MMCIF2Dict(str(cif_path))
    pdb_id = _first_str(d, ["_entry.id"])
    method = _first_str(d, ["_exptl.method"])
    # resolution: commonly _refine.ls_d_res_high or _reflns.d_resolution_high
    resolution = _first_float(d, ["_refine.ls_d_res_high", "_reflns.d_resolution_high"])
    # R-free: commonly _refine.ls_R_factor_R_free
    rfree = _first_float(d, ["_refine.ls_R_factor_R_free", "_refine.ls_R_factor_Rfree"])
    return {
        "pdb_id": pdb_id,
        "method": method,
        "resolution": resolution,
        "rfree": rfree,
    }

def extract_ca_bfactors(cif_path):
    """
    Return list of dicts: pdb_id, chain_id, resnum, icode, resname, atom=Bfactor for Cα atoms only.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem.upper(), str(cif_path))
    rows = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip hetero/water
                hetflag = residue.id[0].strip()
                if hetflag != "":   # e.g., 'W' for water or 'H_' for ligands
                    continue
                # find Cα
                if "CA" in residue:
                    atom = residue["CA"]
                    rows.append({
                        "pdb_id": cif_path.stem.upper(),
                        "model_id": model.id,
                        "chain_id": chain.id,
                        "resnum": residue.id[1],
                        "icode": residue.id[2].strip() if residue.id[2] != " " else "",
                        "resname": residue.get_resname(),
                        "bfactor": float(atom.get_bfactor())
                    })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteins", default="proteins.csv", help="CSV with columns: subfamily,symbol,uniprot")
    ap.add_argument("--outdir", default="data", help="Base data directory")
    ap.add_argument("--min_res", type=float, default=1.8, help="Min (i.e., best) resolution Å to accept")
    ap.add_argument("--max_res", type=float, default=3.5, help="Max (i.e., worst) resolution Å to accept")
    ap.add_argument("--max_rfree", type=float, default=0.30, help="Max R-free to accept")
    ap.add_argument("--method", default="X-RAY DIFFRACTION", help="Experimental method required")
    ap.add_argument("--limit_per_uniprot", type=int, default=0, help="Optional cap for quick tests (0=no cap)")
    args = ap.parse_args()

    base = Path(args.outdir)
    raw_dir = base / "raw" / "mmcif"
    meta_dir = base / "meta"
    proc_dir = base / "processed"
    for p in [raw_dir, meta_dir, proc_dir]:
        p.mkdir(parents=True, exist_ok=True)

    prot_df = read_table(args.proteins)

    structures_meta = []
    all_ca = []

    for _, row in prot_df.iterrows():
        acc = row["uniprot"]
        subfam = row["subfamily"]
        symbol = row["symbol"]
        pdbs = sorted(uniprot_to_pdbs(acc))
        if args.limit_per_uniprot > 0:
            pdbs = pdbs[:args.limit_per_uniprot]

        for pdb_id in tqdm(pdbs, desc=f"{acc}"):
            try:
                cif_path = download_mmcif(pdb_id, raw_dir)
                q = parse_quality_and_method(cif_path)
                q.update({
                    "uniprot": acc,
                    "subfamily": subfam,
                    "symbol": symbol
                })

                # Quality filters (project brief)
                if q["method"] != args.method:
                    continue
                if q["resolution"] is None or q["rfree"] is None:
                    continue
                if not (args.min_res <= q["resolution"] <= args.max_res):
                    continue
                if q["rfree"] > args.max_rfree:
                    continue

                structures_meta.append(q)

                # Extract Cα B-factors
                ca_rows = extract_ca_bfactors(cif_path)
                for r in ca_rows:
                    r.update({
                        "uniprot": acc,
                        "subfamily": subfam,
                        "symbol": symbol
                    })
                all_ca.extend(ca_rows)

            except Exception as e:
                # keep going
                # You may want to log these into a file
                print(f"[WARN] {pdb_id} skipped: {e}")

    # Write outputs
    if structures_meta:
        pd.DataFrame(structures_meta).drop_duplicates(subset=["pdb_id"]).to_csv(meta_dir / "structures.csv", index=False)
    if all_ca:
        pd.DataFrame(all_ca).to_csv(proc_dir / "ca_bfactors.csv", index=False)

    print("Done.")
    print(f"Structures meta: {meta_dir / 'structures.csv'}")
    print(f"Cα B-factors: {proc_dir / 'ca_bfactors.csv'}")

if __name__ == "__main__":
    main()