"""Screen BCAA metabolites + modulators across the BCAA-catabolism pathway.

Context: BCAA catabolism appears to be a protective phenotype in ATAA (medial
VSMCs). BCKDK is the inhibitory kinase that shuts off BCAA oxidation; BCKDK
inhibition is therefore therapeutic.

This screen tests: which metabolites/compounds does the model predict bind
each of the pathway enzymes — with BCKDK being the most actionable target.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
import pandas as pd
from graphdti.serving.app import load_predictor


def read_fasta(p):
    return "".join(l.strip() for l in Path(p).read_text().splitlines() if not l.startswith(">"))


def fmt(x):
    if x is None or pd.isna(x):
        return "  err  "
    return f"{x:.3f}"


def main():
    targets = {
        "BCKDK":  read_fasta("experiments/BCKDK_canonical.fasta"),
        "BCKDHA": read_fasta("experiments/BCKDHA_canonical.fasta"),
        "BCAT1":  read_fasta("experiments/BCAT1_canonical.fasta"),
        "BCAT2":  read_fasta("experiments/BCAT2_canonical.fasta"),
        "EGFR":   read_fasta("experiments/EGFR_canonical.fasta"),  # off-pathway control
    }
    library = pd.read_csv("experiments/bcaa_metabolites_library.csv")

    print("Targets:")
    for n, s in targets.items():
        print(f"  {n:8s} {len(s):3d} aa")
    print(f"Library: {len(library)} compounds")
    print()

    predictor = load_predictor("checkpoints/dti_real_medium.pt", threshold=0.5, device="cpu")

    rows = []
    for r in library.to_dict(orient="records"):
        result = dict(r)
        for tname, tseq in targets.items():
            try:
                result[tname] = predictor.predict(r["smiles"], tseq)
            except Exception:
                result[tname] = None
        rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv("experiments/bcaa_pathway_results.csv", index=False)

    # Sort by BCKDK binding -- the main therapeutic target
    df_b = df.sort_values("BCKDK", ascending=False, na_position="last")

    print("=" * 130)
    print("RANKED BY PREDICTED BCKDK BINDING  (BCKDK INHIBITION = enhances BCAA catabolism = potentially protective in ATAA)")
    print("=" * 130)
    print(f"  {'name':<25s} {'BCKDK':>7s} {'BCKDHA':>7s} {'BCAT1':>7s} {'BCAT2':>7s} {'EGFR':>7s}   {'class':<24s} note")
    print("  " + "-" * 122)
    for r in df_b.to_dict(orient="records"):
        tag = ""
        if "+CTRL" in str(r["note"]):       tag = "[+CTRL]"
        elif "-CTRL" in str(r["note"]):     tag = "[-CTRL]"
        elif r["class"] == "BCAA":          tag = "[BCAA]"
        elif r["class"] == "BCKA":          tag = "[BCKA]"
        elif "BCAA-derived" in str(r["class"]): tag = "[BCAAd]"
        print(f"  {r['name']:<25s} {fmt(r['BCKDK']):>7s} {fmt(r['BCKDHA']):>7s} {fmt(r['BCAT1']):>7s} {fmt(r['BCAT2']):>7s} {fmt(r['EGFR']):>7s}   {r['class']:<24s} {tag} {r['note'][:50]}")

    print()
    print("=" * 130)
    print("FOCUSED: where do the actual BCAA metabolites sit?")
    print("=" * 130)
    natural = df[df["class"].isin(["BCAA", "BCKA", "BCAA-derived"])].sort_values("BCKDK", ascending=False)
    for r in natural.to_dict(orient="records"):
        print(f"  {r['name']:<25s} BCKDK={fmt(r['BCKDK'])}  BCKDHA={fmt(r['BCKDHA'])}  BCAT1={fmt(r['BCAT1'])}  BCAT2={fmt(r['BCAT2'])}  ({r['class']})")

    # Positive controls
    pos = df[df["note"].str.contains(r"\[\+CTRL\]", na=False, regex=True)]
    print()
    print("  Positive control(s) (known BCKDK inhibitors):")
    for r in pos.to_dict(orient="records"):
        print(f"    {r['name']:<25s} BCKDK={fmt(r['BCKDK'])}  BCKDHA={fmt(r['BCKDHA'])}  BCAT1={fmt(r['BCAT1'])}  BCAT2={fmt(r['BCAT2'])}  EGFR={fmt(r['EGFR'])}")

    neg = df[df["note"].str.contains(r"\[-CTRL\]", na=False, regex=True)]
    print()
    print("  Negative controls:")
    for r in neg.to_dict(orient="records"):
        print(f"    {r['name']:<25s} BCKDK={fmt(r['BCKDK'])}  BCKDHA={fmt(r['BCKDHA'])}  EGFR={fmt(r['EGFR'])}")

    print()
    print(f"Full table saved to experiments/bcaa_pathway_results.csv")


if __name__ == "__main__":
    main()
