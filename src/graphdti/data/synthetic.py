"""Synthetic DTI dataset generator.

Produces a dataframe of (smiles, protein_sequence, label) triples whose labels
carry real, learnable signal: a ligand binds a protein iff its Morgan
fingerprint Tanimoto to that protein's "reference ligand" exceeds a threshold.
This lets the pipeline be exercised end-to-end without BindingDB access.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from graphdti.data.featurize import morgan_fingerprint, tanimoto

# A small pool of valid drug-like SMILES — enough diversity to train a tiny model.
SEED_SMILES = [
    "CCO", "CCN", "CCC", "CCCC", "CC(C)O", "CC(C)N", "c1ccccc1", "c1ccncc1",
    "c1ccc2ccccc2c1", "CC(=O)O", "CC(=O)N", "CC(=O)Nc1ccccc1", "CN(C)C=O",
    "O=C1CCCCC1", "O=C1CCCN1", "C1CCOCC1", "C1CCNCC1", "C1CCSCC1",
    "Nc1ccc(O)cc1", "Nc1ccc(Cl)cc1", "Nc1ccc(F)cc1", "Nc1ccc(Br)cc1",
    "COc1ccc(N)cc1", "COc1ccc(C=O)cc1", "OC(=O)c1ccccc1", "OC(=O)c1ccncc1",
    "Cc1ccc(N)cc1", "Cc1ccc(O)cc1", "Cc1ccc(C(=O)O)cc1",
    "CCN(CC)CC", "CCOC(=O)C", "CCOCC", "CCSCC", "CCC(=O)N",
    "c1ccc(-c2ccccc2)cc1", "c1ccc(-c2ccncc2)cc1", "c1ccc(Cn2ccnc2)cc1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",   # ibuprofen
    "CN1CCC[C@H]1c1cccnc1",          # nicotine
    "CC(=O)Nc1ccc(O)cc1",            # paracetamol
    "OC(=O)c1ccccc1O",               # salicylic acid
    "CC(=O)OC1=CC=CC=C1C(=O)O",      # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",    # caffeine
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)NCC(O)c1ccc(O)c(O)c1",     # isoprenaline
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose
    "CCN(CC)C(=O)c1ccccc1",
    "Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1",
    "CC[C@H](C)[C@H](NC(=O)OC(C)(C)C)C(=O)O",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "Cc1ccc2nc(N)sc2c1",
    "COc1ccc2nc(N)sc2c1",
    "Nc1nc(N)nc(Cl)n1",
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
    "CC(C)C[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O",
    "CCN1CCN(CCO)CC1", "C1CN(CCO)CCN1", "c1cnc2[nH]ccc2c1",
    "Brc1ccc2ncccc2c1", "Clc1ccc2ncccc2c1", "Fc1ccc2ncccc2c1",
    "CC1=CC(=O)C=CC1=O", "O=C1C=CC(=O)C=C1",
    "OCC1OC(O)C(O)C(O)C1O", "CC(C)(C)OC(=O)N1CCNCC1",
    "CCCCCCCCCCCCC(=O)O",  # tridecanoic acid
    "CCCCCCC(=O)Nc1ccc(O)cc1",
    "CCOC(=O)c1ccc(N)cc1", "COC(=O)c1ccc(N)cc1",
    "Sc1nc2ccccc2[nH]1", "Sc1ncc(Br)c(=O)[nH]1",
    "C[C@@H](O)[C@H](N)C(=O)O",  # threonine
    "N[C@@H](Cc1ccc(O)cc1)C(=O)O",  # tyrosine
    "N[C@@H](Cc1c[nH]cn1)C(=O)O",   # histidine
    "N[C@@H](CCSC)C(=O)O",          # methionine
    "N[C@@H](CCC(=O)O)C(=O)O",      # glutamic acid
    "N[C@@H](CO)C(=O)O",            # serine
    "OC(=O)CCCCC(=O)O",             # adipic acid
    "CC1CCCCC1", "C1CCCC1", "C1CCC1",
    "ClCCOCCCl", "FCCOCCF",
]


def _make_protein(rng: random.Random, length: int = 200) -> str:
    return "".join(rng.choices("ACDEFGHIKLMNPQRSTVWY", k=length))


def generate(
    n_train: int,
    n_val: int,
    n_proteins: int = 12,
    threshold: float = 0.25,
    seed: int = 0,
    out_dir: str | Path = "data/synthetic",
) -> dict[str, int]:
    """Generate train.csv and val.csv with columns: smiles, protein_sequence, label, protein_id.

    For each of `n_proteins` synthetic targets we pick a random "reference ligand"
    from SEED_SMILES; a (ligand, target) pair is labeled positive iff the
    ligand's Tanimoto similarity to the reference exceeds `threshold`.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Filter the seed pool to SMILES RDKit can parse (defensive: all should pass).
    valid = []
    for s in SEED_SMILES:
        fp = morgan_fingerprint(s)
        if fp is not None:
            valid.append((s, fp))
    if len(valid) < 20:
        raise RuntimeError("Seed SMILES pool too small after RDKit validation.")

    proteins = []
    for i in range(n_proteins):
        seq = _make_protein(rng, length=rng.randint(120, 260))
        ref_smi, ref_fp = valid[rng.randrange(len(valid))]
        proteins.append({"id": f"P{i:03d}", "sequence": seq, "ref_smiles": ref_smi, "ref_fp": ref_fp})

    def sample_split(n: int) -> pd.DataFrame:
        rows = []
        # Balanced sampling: alternate positive / negative bias.
        for k in range(n):
            target = proteins[rng.randrange(len(proteins))]
            smi, fp = valid[rng.randrange(len(valid))]
            sim = tanimoto(fp, target["ref_fp"])
            label = int(sim >= threshold)
            rows.append(
                {
                    "smiles": smi,
                    "protein_sequence": target["sequence"],
                    "protein_id": target["id"],
                    "label": label,
                    "tanimoto": float(sim),
                }
            )
        return pd.DataFrame(rows)

    train_df = sample_split(n_train)
    val_df = sample_split(n_val)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.drop(columns=["tanimoto"]).to_csv(out / "train.csv", index=False)
    val_df.drop(columns=["tanimoto"]).to_csv(out / "val.csv", index=False)
    (out / "proteins.json").write_text(
        json.dumps(
            [{"id": p["id"], "sequence": p["sequence"], "ref_smiles": p["ref_smiles"]} for p in proteins],
            indent=2,
        )
    )

    return {
        "train": len(train_df),
        "val": len(val_df),
        "train_pos_rate": float(train_df["label"].mean()),
        "val_pos_rate": float(val_df["label"].mean()),
    }


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=400)
    p.add_argument("--n-proteins", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/synthetic")
    args = p.parse_args()
    stats = generate(args.n_train, args.n_val, args.n_proteins, args.threshold, args.seed, args.out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
