"""Test metformin against the AMPK-SIRT1 pathway.

Context (clinical motivation):
  Recent literature suggests metformin reduces ascending thoracic aortic
  aneurysm (TAA) progression by clearing senescent VSMCs in the tunica media
  via the AMPK -> SIRT1 -> PGC-1alpha mitochondrial-homeostasis axis.

What this experiment actually tests:
  Direct *binding* of each compound to:
    - AMPK alpha-2 kinase domain (direct druggable target)
    - SIRT1 catalytic domain (direct druggable target)
    - EGFR kinase domain (unrelated-pathway negative control)

What this experiment does NOT test:
  - Pathway activation (metformin's textbook mechanism is INDIRECT:
    Complex I inhibition -> AMP/ATP rise -> AMPK gamma-subunit AMP binding)
  - Phenotypic effects on VSMCs or aortic tissue
  - PGC-1alpha (it's a transcriptional coactivator, no druggable pocket)

Interpretation guide:
  - Metformin scoring LOW on AMPK = consistent with indirect mechanism (good)
  - Metformin scoring HIGH on AMPK = surprising, would suggest direct binding
  - Direct AMPK activators (A-769662, MK-8722, etc.) should score HIGH on AMPK
  - Resveratrol scoring on SIRT1 = consistent with controversial literature
  - Negative controls (caffeine, ethanol) should score LOW on everything
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
import pandas as pd
from graphdti.serving.app import load_predictor


def read_fasta(path: Path) -> str:
    return "".join(l.strip() for l in path.read_text().splitlines() if not l.startswith(">"))


def main():
    targets = {
        "AMPK_a2": read_fasta(Path("experiments/AMPK_alpha2_canonical.fasta")),
        "SIRT1":   read_fasta(Path("experiments/SIRT1_canonical.fasta")),
        "mTOR":    read_fasta(Path("experiments/mTOR_canonical.fasta")),
        "EGFR":    read_fasta(Path("experiments/EGFR_canonical.fasta")),  # off-pathway control
    }
    library = pd.read_csv("experiments/metformin_AMPK_library.csv")

    print(f"Targets:")
    for n, s in targets.items():
        print(f"  {n:10s}  {len(s)} aa")
    print(f"Library: {len(library)} compounds")
    print()

    predictor = load_predictor("checkpoints/dti_real_medium.pt", threshold=0.5, device="cpu")

    rows = []
    for r in library.to_dict(orient="records"):
        result = {"name": r["name"], "class": r["class"], "note": r["note"], "smiles": r["smiles"]}
        for tname, tseq in targets.items():
            try:
                result[tname] = predictor.predict(r["smiles"], tseq)
            except Exception as e:
                result[tname] = None
                result[f"{tname}_err"] = str(e)[:60]
        rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv("experiments/metformin_AMPK_results.csv", index=False)

    def fmt(x):
        if x is None or pd.isna(x):
            return "  err  "
        return f" {x:.3f} "

    # AMPK-sorted view
    df_ampk = df.sort_values("AMPK_a2", ascending=False, na_position="last")
    print("=" * 115)
    print("RANKED BY PREDICTED AMPK alpha-2 BINDING (full pathway map)")
    print("=" * 115)
    print(f"  {'name':<25s} {'AMPK_a2':>8s} {'SIRT1':>8s} {'mTOR':>8s} {'EGFR':>8s}   {'class':<22s} {'note':s}")
    print("  " + "-" * 110)
    for r in df_ampk.to_dict(orient="records"):
        marker = ""
        if r["name"] == "Metformin":         marker = "[METFORMIN]"
        elif "+mTOR CTRL" in r["note"]:       marker = "[+mTOR]"
        elif "POSITIVE" in r["note"]:         marker = "[+AMPK]"
        elif "Negative" in r["note"]:         marker = "[-CTRL]"
        print(f"  {r['name']:<25s} {fmt(r['AMPK_a2']):>8s} {fmt(r['SIRT1']):>8s} {fmt(r['mTOR']):>8s} {fmt(r['EGFR']):>8s}   {r['class']:<22s} {marker}")

    # mTOR-sorted view
    df_mtor = df.sort_values("mTOR", ascending=False, na_position="last")
    print()
    print("=" * 115)
    print("RANKED BY PREDICTED mTOR BINDING")
    print("=" * 115)
    print(f"  {'name':<25s} {'mTOR':>8s} {'AMPK_a2':>8s} {'SIRT1':>8s} {'EGFR':>8s}   {'class':<22s} {'note':s}")
    print("  " + "-" * 110)
    for r in df_mtor.head(15).to_dict(orient="records"):
        marker = ""
        if r["name"] == "Metformin":         marker = "[METFORMIN]"
        elif "+mTOR CTRL" in r["note"]:       marker = "[+mTOR]"
        elif "POSITIVE" in r["note"]:         marker = "[+AMPK]"
        print(f"  {r['name']:<25s} {fmt(r['mTOR']):>8s} {fmt(r['AMPK_a2']):>8s} {fmt(r['SIRT1']):>8s} {fmt(r['EGFR']):>8s}   {r['class']:<22s} {marker}")

    # Focus: metformin specifically
    met = df[df["name"] == "Metformin"].iloc[0]
    print()
    print("=" * 115)
    print("THE QUESTION: where does metformin bind across the AMPK -> mTOR -> SIRT1 axis?")
    print("=" * 115)
    print(f"  Metformin vs AMPK alpha-2:  prob = {met['AMPK_a2']:.4f}")
    print(f"  Metformin vs mTOR:          prob = {met['mTOR']:.4f}")
    print(f"  Metformin vs SIRT1:         prob = {met['SIRT1']:.4f}")
    print(f"  Metformin vs EGFR (ctrl):   prob = {met['EGFR']:.4f}")

    print()
    print("  Direct AMPK activators (positive controls for AMPK column):")
    for r in df[df["note"].str.contains("AMPK CTRL|POSITIVE", na=False)].to_dict(orient="records"):
        if r["AMPK_a2"] is None: continue
        print(f"    {r['name']:<20s}  AMPK={r['AMPK_a2']:.3f}  mTOR={r['mTOR']:.3f}  SIRT1={r['SIRT1']:.3f}  EGFR={r['EGFR']:.3f}")

    print()
    print("  Direct mTOR inhibitors (positive controls for mTOR column):")
    for r in df[df["note"].str.contains("mTOR CTRL", na=False)].to_dict(orient="records"):
        if r["mTOR"] is None: continue
        print(f"    {r['name']:<20s}  mTOR={r['mTOR']:.3f}  AMPK={r['AMPK_a2']:.3f}  SIRT1={r['SIRT1']:.3f}  EGFR={r['EGFR']:.3f}")

    print()
    print("  Negative controls:")
    for r in df[df["note"].str.contains("Negative control", na=False)].to_dict(orient="records"):
        print(f"    {r['name']:<20s}  AMPK={r['AMPK_a2']:.3f}  mTOR={r['mTOR']:.3f}  SIRT1={r['SIRT1']:.3f}  EGFR={r['EGFR']:.3f}")

    print(f"\nFull table saved to experiments/metformin_AMPK_results.csv")


if __name__ == "__main__":
    main()
