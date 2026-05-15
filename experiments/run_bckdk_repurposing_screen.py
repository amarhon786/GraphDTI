"""Drug-repurposing screen: 85 FDA-approved drugs vs BCKDK.

Rationale: in the BCAA-catabolism / ATAA story, inhibiting BCKDK is the
therapeutic move (relieves the phosphorylation block on BCKDHA, allows
BCAA oxidation). The model was trained on drug-like compounds, so a
DRUG screen against BCKDK should be more productive than a metabolite
screen (which suffered from out-of-distribution issues).

Calibration anchor: BT2 (the textbook BCKDK inhibitor) scored 0.36 in our
earlier screen. We add BT2 here as a positive control and rank everything
relative to it.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
import pandas as pd
from graphdti.serving.app import load_predictor


def main():
    seq = "".join(l.strip() for l in Path("experiments/BCKDK_canonical.fasta").read_text().splitlines() if not l.startswith(">"))
    library = pd.read_csv("experiments/fda_approved_library.csv")

    # Add BT2 as positive control + a few metabolite anchors for context
    extras = pd.DataFrame([
        {"name": "BT2", "smiles": "O=C(O)c1sc2cc(Cl)ccc2c1Cl", "class": "POSITIVE CONTROL (BCKDK inhibitor)"},
        {"name": "Alpha-KIC", "smiles": "CC(C)CC(=O)C(=O)O", "class": "ANCHOR (BCKA, natural allosteric inhibitor)"},
        {"name": "Leucine", "smiles": "CC(C)C[C@H](N)C(=O)O", "class": "ANCHOR (BCAA substrate)"},
    ])
    library = pd.concat([library, extras], ignore_index=True)

    predictor = load_predictor("checkpoints/dti_real_medium.pt", threshold=0.5, device="cpu")
    print(f"Target: BCKDK kinase-fold domain ({len(seq)} aa)")
    print(f"Library: {len(library)} compounds (85 FDA-approved + 3 anchors)")
    print()

    rows = []
    for i, r in enumerate(library.to_dict(orient="records"), 1):
        try:
            p = predictor.predict(r["smiles"], seq)
            rows.append({**r, "probability": p, "ok": True})
        except Exception:
            rows.append({**r, "probability": None, "ok": False})
        if i % 20 == 0:
            print(f"  ...scored {i}/{len(library)}")

    df = pd.DataFrame(rows)
    ok = df[df["ok"]].sort_values("probability", ascending=False).reset_index(drop=True)
    ok.to_csv("experiments/BCKDK_repurposing_hits.csv", index=False)

    # Find BT2 rank
    bt2_row = ok[ok["name"] == "BT2"].iloc[0]
    bt2_rank = (ok["probability"] > bt2_row["probability"]).sum() + 1
    bt2_prob = bt2_row["probability"]

    print("=" * 110)
    print("TOP 20 PREDICTED BCKDK BINDERS (FDA-approved drug library)")
    print("=" * 110)
    for i, r in enumerate(ok.head(20).to_dict(orient="records"), 1):
        marker = ""
        if r["name"] == "BT2":               marker = "[BT2 +CTRL]"
        elif r["class"].startswith("ANCHOR"): marker = "[anchor]"
        bt2_above = ">" if r["probability"] > bt2_prob else " "
        print(f"  {i:2d}. {bt2_above} {r['name']:28s}  prob={r['probability']:.4f}   [{r['class']}] {marker}")

    print()
    print("=" * 110)
    print(f"CALIBRATION CHECK: BT2 (gold-standard BCKDK inhibitor) ranks #{bt2_rank} at {bt2_prob:.4f}")
    print(f"  -> Compounds scoring above BT2 are the 'plausible repurposing hits' the model is flagging")
    print(f"  -> Caveat: model is calibration-noisy on BCKDK; treat the LIST as candidates, not certainties")
    print("=" * 110)

    above_bt2 = ok[ok["probability"] > bt2_prob]
    above_bt2_drugs = above_bt2[~above_bt2["class"].isin(["POSITIVE CONTROL (BCKDK inhibitor)",
                                                            "ANCHOR (BCKA, natural allosteric inhibitor)",
                                                            "ANCHOR (BCAA substrate)"])]
    print()
    print(f"Drugs scoring above the BT2 baseline ({len(above_bt2_drugs)} candidates):")
    for r in above_bt2_drugs.to_dict(orient="records"):
        print(f"  - {r['name']:28s}  prob={r['probability']:.4f}   [{r['class']}]")

    print(f"\nFull table saved to experiments/BCKDK_repurposing_hits.csv")


if __name__ == "__main__":
    main()
