"""Virtual screen: score every compound in a library against one target protein.

Reads a CSV with columns [name, smiles] (name optional) and a protein sequence
either as a string, a FASTA file, or stdin. Prints a ranked table and writes
results to --out.

Usage:
    python scripts/screen.py --protein BRAF.fasta --library compounds.csv --out hits.csv
    python scripts/screen.py --protein-seq "MKTII..." --library compounds.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests


def read_protein(args) -> str:
    if args.protein_seq:
        return args.protein_seq.strip()
    if args.protein:
        text = Path(args.protein).read_text()
        # Strip FASTA header lines if present
        return "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))
    sys.exit("Provide --protein <fasta> or --protein-seq <sequence>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protein", help="FASTA file with one protein sequence")
    p.add_argument("--protein-seq", help="Protein sequence as a string")
    p.add_argument("--library", required=True, help="CSV with a 'smiles' column (optional 'name' column)")
    p.add_argument("--api", default="http://127.0.0.1:8000", help="GraphDTI server URL")
    p.add_argument("--threshold", type=float, default=0.61, help="Probability cutoff for 'binder'")
    p.add_argument("--out", default="hits.csv", help="Output CSV with ranked predictions")
    p.add_argument("--topn", type=int, default=10, help="How many hits to print to console")
    args = p.parse_args()

    protein_seq = read_protein(args)
    library = pd.read_csv(args.library)
    if "smiles" not in library.columns:
        sys.exit(f"--library CSV must have a 'smiles' column. Got: {list(library.columns)}")
    if "name" not in library.columns:
        library["name"] = [f"compound_{i}" for i in range(len(library))]

    print(f"Target protein: {len(protein_seq)} residues, starts {protein_seq[:30]!r}")
    print(f"Library: {len(library)} compounds")
    print(f"Scoring against {args.api}/predict ...")

    rows = []
    for i, r in enumerate(library.itertuples(index=False), 1):
        try:
            resp = requests.post(
                f"{args.api}/predict",
                json={"smiles": r.smiles, "protein_sequence": protein_seq},
                timeout=60,
            )
            resp.raise_for_status()
            prob = resp.json()["probability"]
            rows.append({"name": r.name, "smiles": r.smiles, "probability": prob})
        except Exception as e:
            rows.append({"name": r.name, "smiles": r.smiles, "probability": None, "error": str(e)[:80]})
        if i % 25 == 0:
            print(f"  ... {i}/{len(library)}")

    out_df = pd.DataFrame(rows).sort_values("probability", ascending=False, na_position="last")
    out_df["binder"] = out_df["probability"].apply(
        lambda x: "" if x is None else ("YES" if x >= args.threshold else "no")
    )
    out_df.to_csv(args.out, index=False)

    print(f"\n=== Top {args.topn} hits (threshold = {args.threshold}) ===")
    cols = ["name", "probability", "binder", "smiles"]
    head = out_df.head(args.topn).copy()
    head["smiles"] = head["smiles"].str.slice(0, 50)
    head["probability"] = head["probability"].round(4)
    print(head[cols].to_string(index=False))
    n_binders = (out_df["probability"] >= args.threshold).sum()
    print(f"\n{n_binders}/{len(out_df)} compounds scored as binders. Full results in {args.out}")


if __name__ == "__main__":
    main()
