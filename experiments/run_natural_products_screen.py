"""Polypharmacology screen: 30 natural products vs BRAF V600E (melanoma kinase).

Most natural products with claimed anticancer activity (curcumin, resveratrol,
green-tea catechins, soy isoflavones, etc.) lack a clear single molecular
target. The literature is full of vague pathway claims but specific kinase
binding is rarely characterized.

This screen asks: which of these naturally-occurring compounds does GraphDTI
predict will bind BRAF V600E (the melanoma driver), and how do they compare
to the FDA-approved BRAF inhibitors?
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
import pandas as pd
from graphdti.serving.app import load_predictor


def main():
    # Canonical BRAF V600E kinase domain (UniProt P15056 residues 457-717, V600E mutation)
    fasta_path = Path("experiments/BRAF_canonical.fasta")
    seq = "".join(line.strip() for line in fasta_path.read_text().splitlines() if not line.startswith(">"))
    library = pd.read_csv("experiments/natural_products_library.csv")
    ckpt = "checkpoints/dti_real_medium.pt"

    print(f"Target: BRAF V600E kinase domain ({len(seq)} residues)")
    print(f"Library: {len(library)} compounds")
    print(f"Model: {Path(ckpt).name}\n")

    p = load_predictor(ckpt, threshold=0.5, device="cpu")

    rows = []
    for i, r in enumerate(library.to_dict(orient="records"), 1):
        try:
            prob = p.predict(r["smiles"], seq)
            rows.append({**r, "probability": prob, "ok": True})
        except Exception as e:
            rows.append({**r, "probability": None, "ok": False, "error": str(e)[:80]})
        if i % 10 == 0:
            print(f"  ...scored {i}/{len(library)}")

    df = pd.DataFrame(rows)
    df_ok = df[df["ok"]].sort_values("probability", ascending=False)

    df_ok.to_csv("experiments/BRAF_natural_products_hits.csv", index=False)

    print("\n" + "=" * 90)
    print(f"FULL RANKING vs BRAF V600E ({len(df_ok)} scored successfully)")
    print("=" * 90)
    for i, row in enumerate(df_ok.to_dict(orient="records"), 1):
        marker = "[CTRL+]" if row["source"] == "Positive control" else (
                  "[CTRL-]" if row["source"] == "Negative control" else "       ")
        print(f"  {i:2d}. {marker}  {row['name']:30s}  prob={row['probability']:.4f}  "
              f"[{row['class']}, {row['source']}]")

    # Discard controls, look at natural products
    nat = df_ok[~df_ok["source"].isin(["Positive control", "Negative control"])]
    pos = df_ok[df_ok["source"] == "Positive control"]
    neg = df_ok[df_ok["source"] == "Negative control"]

    print("\n" + "=" * 90)
    print("KEY METRICS")
    print("=" * 90)
    print(f"  Median positive-control probability:  {pos['probability'].median():.4f}")
    print(f"  Median natural-product probability:   {nat['probability'].median():.4f}")
    print(f"  Median negative-control probability:  {neg['probability'].median():.4f}")

    # Failed
    failed = df[~df["ok"]]
    if len(failed) > 0:
        print(f"\n  ({len(failed)} SMILES failed to parse: {', '.join(failed['name'])})")

    # The interesting question: which natural products score in the top range
    print("\n" + "=" * 90)
    print("NATURAL-PRODUCT HITS ABOVE THE LOWEST POSITIVE CONTROL")
    print("=" * 90)
    pos_min = pos["probability"].min()
    interesting = nat[nat["probability"] >= pos_min]
    if len(interesting):
        for row in interesting.to_dict(orient="records"):
            print(f"  *  {row['name']:30s}  prob={row['probability']:.4f}   ({row['source']}, {row['class']})")
        print(f"\n  → {len(interesting)} natural products score >= the weakest BRAF-approved drug ({pos_min:.4f}).")
        print(f"  → These are testable hypotheses: claim that BRAF V600E is bound by these natural products.")
    else:
        print("  None — all natural products score below the weakest positive control.")
        print("  Reasonable interpretation: model says these natural products are unlikely to be specific BRAF binders.")

    print(f"\nFull ranking saved to: experiments/BRAF_natural_products_hits.csv")


if __name__ == "__main__":
    main()
