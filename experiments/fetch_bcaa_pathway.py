"""Fetch BCAA-catabolism pathway proteins from UniProt.

Key proteins for the ATAA / BCAA-catabolism protective-phenotype hypothesis:
- BCKDK   (O14874) — the inhibitory kinase. Phosphorylates BCKDHA and SHUTS OFF
                     BCAA oxidation. INHIBITING BCKDK is therapeutic.
- BCKDHA  (P12694) — E1-alpha subunit of BCKDH complex. Rate-limiting enzyme
                     of BCAA catabolism.
- BCAT1   (P54687) — Cytosolic branched-chain aminotransferase. Step 1 of
                     BCAA catabolism (transamination).
- BCAT2   (O15382) — Mitochondrial BCAT.
- DBT     (P11182) — E2 subunit of BCKDH.
- DLD     (P09622) — E3 subunit of BCKDH (shared with PDH, KGDH).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

TARGETS = {
    "BCKDK":  ("O14874", (130, 412)),   # kinase / HATPase catalytic domain
    "BCKDHA": ("P12694", (46, 445)),    # mature protein (after mitochondrial transit peptide)
    "BCAT1":  ("P54687", (1, 386)),     # full cytosolic protein
    "BCAT2":  ("O15382", (28, 392)),    # mature mitochondrial protein
    "DBT":    ("P11182", (61, 482)),    # E2 transacylase, mature
    "DLD":    ("P09622", (36, 509)),    # E3 dihydrolipoamide dehydrogenase, mature
}


def fetch(acc):
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    with urllib.request.urlopen(url, timeout=30) as r:
        return "".join(l for l in r.read().decode().splitlines() if not l.startswith(">"))


def main():
    out_dir = Path("experiments")
    for name, (acc, (a, b)) in TARGETS.items():
        full = fetch(acc)
        seq = full[a - 1:b]
        path = out_dir / f"{name}_canonical.fasta"
        with open(path, "w") as f:
            f.write(f">{name}_domain UniProt {acc} residues {a}-{b}\n{seq}\n")
        print(f"{name:7s}  {len(seq):3d} aa  ({acc} residues {a}-{b})")


if __name__ == "__main__":
    main()
