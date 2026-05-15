"""BindingDB loader.

BindingDB ships a tall TSV (~hundreds of columns). We extract:
- Ligand SMILES (from 'Ligand SMILES')
- Target sequence (from 'BindingDB Target Chain Sequence' or 'Target Sequence')
- Affinity (Ki / IC50 / Kd in nM) → binary label using a configurable nM threshold
- UniProt family annotation (when present) → kinase-family holdout split

Run as a module:
    python -m graphdti.data.bindingdb --tsv path/BindingDB_All.tsv \
        --out data/bindingdb --affinity-col Ki --max-pairs 1200000
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LIGAND_COL = "Ligand SMILES"
SEQUENCE_COLS = [
    "BindingDB Target Chain Sequence 1",
    "BindingDB Target Chain Sequence",
    "Target Sequence",
]
UNIPROT_COLS = [
    "UniProt (SwissProt) Primary ID of Target Chain 1",
    "UniProt (SwissProt) Entry Name of Target Chain 1",
    "UniProt (SwissProt) Primary ID of Target Chain",
    "UniProt (SwissProt) Entry Name of Target Chain",
]
AFFINITY_COLS = {"Ki": "Ki (nM)", "IC50": "IC50 (nM)", "Kd": "Kd (nM)"}

# Common kinase UniProt prefixes / suffixes used for the held-out family split.
KINASE_FAMILY_PATTERNS = [
    re.compile(r".*KIN(ASE)?.*", re.IGNORECASE),
    re.compile(r"^(EGFR|ABL1|SRC|MAPK|JAK[123]?|CDK\d+|AURK[ABC]|BRAF|ALK|MET|RET|FLT3|BTK)$"),
]


@dataclass
class LoaderConfig:
    affinity_col: str = "Ki"
    affinity_threshold_nm: float = 1000.0  # <= 1µM = binder
    max_pairs: int | None = None
    min_smiles_len: int = 3
    min_sequence_len: int = 30
    seed: int = 0


def _parse_affinity(s) -> float | None:
    """BindingDB affinity cells can be like '12', '>10000', '<0.5', or empty."""
    if pd.isna(s):
        return None
    try:
        if isinstance(s, (int, float)):
            return float(s)
        text = str(s).strip().replace(",", "")
        text = text.lstrip("<>=~ ")
        return float(text)
    except (ValueError, TypeError):
        return None


def _is_kinase(uniprot_name: str | None) -> bool:
    if not uniprot_name:
        return False
    return any(p.search(uniprot_name) for p in KINASE_FAMILY_PATTERNS)


def load_bindingdb(tsv_path: str | Path, cfg: LoaderConfig) -> pd.DataFrame:
    """Stream the TSV, filter to one affinity column, return a tidy dataframe.

    Output columns: smiles, protein_sequence, protein_id, label, affinity_nm, is_kinase
    """
    aff_col = AFFINITY_COLS[cfg.affinity_col]
    needed = [LIGAND_COL, aff_col] + SEQUENCE_COLS + UNIPROT_COLS

    # We read in chunks because BindingDB is huge (~GB).
    chunks = []
    reader = pd.read_csv(
        tsv_path,
        sep="\t",
        usecols=lambda c: c in needed,
        chunksize=200_000,
        low_memory=False,
        on_bad_lines="skip",
    )
    for chunk in reader:
        seq_col = next((c for c in SEQUENCE_COLS if c in chunk.columns), None)
        if seq_col is None:
            continue
        uniprot_col = next((c for c in UNIPROT_COLS if c in chunk.columns), None)

        chunk = chunk.rename(columns={LIGAND_COL: "smiles", seq_col: "protein_sequence"})
        chunk["affinity_nm"] = chunk[aff_col].map(_parse_affinity)
        chunk["protein_id"] = chunk[uniprot_col].astype(str) if uniprot_col else "UNK"
        chunk["is_kinase"] = chunk["protein_id"].map(_is_kinase)

        chunk = chunk[["smiles", "protein_sequence", "protein_id", "affinity_nm", "is_kinase"]].dropna(
            subset=["smiles", "protein_sequence", "affinity_nm"]
        )
        chunk = chunk[
            (chunk["smiles"].str.len() >= cfg.min_smiles_len)
            & (chunk["protein_sequence"].str.len() >= cfg.min_sequence_len)
        ]
        chunks.append(chunk)
        if cfg.max_pairs and sum(len(c) for c in chunks) >= cfg.max_pairs:
            break

    if not chunks:
        raise RuntimeError(f"No usable rows in {tsv_path}")
    df = pd.concat(chunks, ignore_index=True)
    if cfg.max_pairs:
        df = df.iloc[: cfg.max_pairs].copy()
    df["label"] = (df["affinity_nm"] <= cfg.affinity_threshold_nm).astype(int)
    return df


def split_kinase_holdout(df: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Train/val/test split where the test set is held-out kinase targets.

    - test = all rows whose target is a kinase
    - remaining rows are randomly split 90/10 into train/val
    """
    rng = np.random.default_rng(seed)
    test = df[df["is_kinase"]].copy()
    pool = df[~df["is_kinase"]].copy()
    perm = rng.permutation(len(pool))
    cut = int(0.9 * len(pool))
    train = pool.iloc[perm[:cut]].reset_index(drop=True)
    val = pool.iloc[perm[cut:]].reset_index(drop=True)
    return {"train": train, "val": val, "test": test.reset_index(drop=True)}


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True)
    p.add_argument("--out", default="data/bindingdb")
    p.add_argument("--affinity-col", default="Ki", choices=list(AFFINITY_COLS))
    p.add_argument("--threshold-nm", type=float, default=1000.0)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = LoaderConfig(
        affinity_col=args.affinity_col,
        affinity_threshold_nm=args.threshold_nm,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    df = load_bindingdb(args.tsv, cfg)
    splits = split_kinase_holdout(df, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, part in splits.items():
        path = out / f"{name}.csv"
        part[["smiles", "protein_sequence", "protein_id", "label"]].to_csv(path, index=False)
        stats[name] = {"n": int(len(part)), "pos_rate": float(part["label"].mean()) if len(part) else 0.0}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
