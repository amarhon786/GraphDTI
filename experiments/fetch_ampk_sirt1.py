"""Fetch AMPK alpha-2 kinase domain and SIRT1 catalytic domain from UniProt."""
from __future__ import annotations

import urllib.request
from pathlib import Path

TARGETS = {
    "AMPK_alpha2": {
        "uniprot": "P54646",                # PRKAA2 (catalytic subunit, alpha-2)
        "kinase_domain": (16, 280),         # canonical Ser/Thr kinase domain
        "mutations": {},
        "description": "AMP-activated protein kinase, alpha-2 catalytic subunit (kinase domain)",
    },
    "SIRT1": {
        "uniprot": "Q96EB6",
        "kinase_domain": (244, 498),        # catalytic sirtuin domain
        "mutations": {},
        "description": "SIRT1 NAD-dependent deacetylase (catalytic domain)",
    },
    "mTOR": {
        "uniprot": "P42345",
        "kinase_domain": (2182, 2491),      # catalytic kinase domain (PI3K-like)
        "mutations": {},
        "description": "mechanistic Target Of Rapamycin (catalytic kinase domain)",
    },
}


def fetch(acc):
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode()
    return "".join(l for l in text.splitlines() if not l.startswith(">"))


def main():
    out_dir = Path("experiments")
    for name, info in TARGETS.items():
        full = fetch(info["uniprot"])
        a, b = info["kinase_domain"]
        seq = full[a - 1:b]
        path = out_dir / f"{name}_canonical.fasta"
        with open(path, "w") as f:
            f.write(f">{name}_domain UniProt {info['uniprot']} residues {a}-{b}\n{seq}\n")
        print(f"{name}: {len(seq)} aa  ({info['uniprot']} residues {a}-{b}) -> {path}")


if __name__ == "__main__":
    main()
