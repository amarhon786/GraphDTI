"""Drug-repurposing screen: score every approved drug in the library against EGFR.

Loads the trained checkpoint directly (no FastAPI server needed). The library
CSV has a `class` column; we keep it through to the ranked output so we can
group findings by therapeutic class.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # don't fight any GPU training

import pandas as pd

from graphdti.serving.app import load_predictor


def main():
    target_fasta = Path("experiments/EGFR_kinase.fasta")
    library_csv = Path("experiments/fda_approved_library.csv")
    ckpt = Path("checkpoints/dti_real_medium.pt")
    out_csv = Path("experiments/EGFR_repurposing_hits.csv")

    # Read the protein sequence (skip FASTA header)
    text = target_fasta.read_text()
    seq = "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))

    library = pd.read_csv(library_csv)
    print(f"Target: EGFR kinase domain ({len(seq)} residues)")
    print(f"Library: {len(library)} compounds across {library['class'].nunique()} classes")
    print(f"Model: {ckpt.name}\n")

    predictor = load_predictor(str(ckpt), threshold=0.5, device="cpu")

    rows = []
    for i, r in enumerate(library.to_dict(orient="records"), 1):
        try:
            p = predictor.predict(r["smiles"], seq)
            rows.append({"name": r["name"], "class": r["class"],
                         "probability": p, "smiles": r["smiles"]})
        except Exception as e:
            rows.append({"name": r["name"], "class": r["class"], "probability": None,
                         "smiles": r["smiles"], "error": str(e)[:80]})
        if i % 10 == 0:
            print(f"  ...scored {i}/{len(library)}")

    df = pd.DataFrame(rows).sort_values("probability", ascending=False, na_position="last")
    df.to_csv(out_csv, index=False)
    df_ok = df.dropna(subset=["probability"])

    print("\n" + "=" * 80)
    print(f"TOP 20 PREDICTED EGFR BINDERS")
    print("=" * 80)
    for i, row in enumerate(df_ok.head(20).to_dict(orient="records"), 1):
        cls = row["class"]
        marker = "*" if "EGFR" in cls else ("o" if "kinase" in cls else " ")
        print(f"  {i:2d}. {marker} {row['name']:25s}  prob={row['probability']:.4f}   [{cls}]")

    nonkin = df_ok[~df_ok["class"].str.contains("kinase|EGFR", case=False, na=False)]
    print("\n" + "=" * 80)
    print(f"TOP 10 NON-KINASE DRUGS BY PREDICTED BINDING (potential repurposing candidates)")
    print("=" * 80)
    for i, row in enumerate(nonkin.head(10).to_dict(orient="records"), 1):
        print(f"  {i:2d}.   {row['name']:25s}  prob={row['probability']:.4f}   [{row['class']}]")

    print("\n" + "=" * 80)
    print(f"SANITY CHECK -- where the known EGFR drugs landed")
    print("=" * 80)
    egfr_known = df_ok[df_ok["class"].str.contains("EGFR", case=False, na=False)]
    for row in egfr_known.to_dict(orient="records"):
        rank = (df_ok["probability"] > row["probability"]).sum() + 1
        print(f"  rank #{rank:2d} of {len(df_ok)}  *  {row['name']:25s}  prob={row['probability']:.4f}")

    print(f"\nFull ranking saved to: {out_csv}")


if __name__ == "__main__":
    main()
