"""Fetch canonical kinase-domain sequences from UniProt and apply mutations.

Compares current app PROTEINS dict against the canonical sequences to identify
any truncations or errors, then writes corrected FASTA files.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

# UniProt entry + kinase domain ranges (1-indexed, inclusive)
KINASES = {
    "BRAF": {
        "uniprot": "P15056",
        "kinase_domain": (457, 717),  # UniProt-annotated catalytic kinase domain
        "mutations": {600: "E"},      # V600E
        "description": "BRAF kinase domain with V600E mutation (melanoma)",
    },
    "EGFR": {
        "uniprot": "P00533",
        "kinase_domain": (712, 979),
        "mutations": {},
        "description": "EGFR kinase domain (lung cancer)",
    },
    "ABL1": {
        "uniprot": "P00519",
        "kinase_domain": (242, 493),
        "mutations": {},
        "description": "ABL1 kinase domain (CML)",
    },
}


def fetch_uniprot_fasta(accession: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode()
    # Strip header, concat all sequence lines
    return "".join(line for line in text.splitlines() if not line.startswith(">"))


def apply_mutations(seq: str, mutations: dict[int, str], domain_start_1based: int) -> str:
    """Apply mutations using FULL-PROTEIN 1-based positions, given the kinase domain start."""
    chars = list(seq)
    for full_pos_1based, new_aa in mutations.items():
        idx_in_domain = full_pos_1based - domain_start_1based
        if 0 <= idx_in_domain < len(chars):
            chars[idx_in_domain] = new_aa
        else:
            raise ValueError(f"Mutation position {full_pos_1based} outside domain {domain_start_1based}-{domain_start_1based+len(chars)-1}")
    return "".join(chars)


def slice_kinase_domain(full_seq: str, start_1based: int, end_1based: int) -> str:
    return full_seq[start_1based - 1:end_1based]


# Current sequences from the app (the suspicious BRAF one is the focus)
CURRENT_APP_SEQUENCES = {
    "BRAF": (
        "DLTVKIGDFGLATEKSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMTGQLPYSNINNRDQII"
        "FMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKKKRDERPLFPQILASIELLARSLPKIHRSASEPSLNRAGFQTEDFSLY"
        "ACASPKTPIQAGGYGAFPVH"
    ),
    "EGFR": (
        "FKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQ"
        "LMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHA"
        "EGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCW"
        "MIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFY"
    ),
    "ABL1": (
        "MGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSR"
        "NAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLH"
        "YPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIK"
        "HPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGEN"
        "HLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEK"
        "DYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGV"
    ),
}


def main():
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    report_lines = ["# Canonical kinase sequence audit\n"]

    for name, info in KINASES.items():
        acc = info["uniprot"]
        print(f"Fetching {name} from UniProt {acc}...")
        full = fetch_uniprot_fasta(acc)
        kd = slice_kinase_domain(full, *info["kinase_domain"])
        if info["mutations"]:
            kd_mut = apply_mutations(kd, info["mutations"], info["kinase_domain"][0])
        else:
            kd_mut = kd

        current = CURRENT_APP_SEQUENCES[name]
        match = (kd_mut == current)
        canonical_len = len(kd_mut)
        current_len = len(current)

        print(f"  Canonical kinase domain length: {canonical_len}")
        print(f"  Currently used in app:          {current_len}")
        print(f"  Exact match: {match}")
        if not match:
            # Find where current sits inside canonical, if at all
            if current in kd_mut:
                offset = kd_mut.index(current)
                print(f"  [!] Current sequence is a SUBSTRING of canonical, starting at canonical position {offset}")
                missing_n = offset
                missing_c = canonical_len - offset - current_len
                print(f"    Missing {missing_n} residues from N-terminus")
                print(f"    Missing {missing_c} residues from C-terminus")
            else:
                print("  [!] Current sequence does NOT match canonical even as substring (possible sequence errors)")
        print()

        # Verify V600E for BRAF
        if name == "BRAF":
            v600_idx = 600 - info["kinase_domain"][0]
            print(f"  V600 site in kinase domain (0-indexed): {v600_idx}")
            print(f"  WT residue at that position:  {kd[v600_idx]}")
            print(f"  Mutant residue at that position: {kd_mut[v600_idx]}")
            print(f"  Context (residues 595-605 of full BRAF): {kd_mut[v600_idx-5:v600_idx+6]}")
            print()

        # Save
        fasta_path = out_dir / f"{name}_canonical.fasta"
        with open(fasta_path, "w") as f:
            mut_tag = "_V600E" if name == "BRAF" else ""
            f.write(f">{name}{mut_tag}_kinase_domain UniProt {acc} residues {info['kinase_domain'][0]}-{info['kinase_domain'][1]}\n")
            f.write(kd_mut + "\n")
        print(f"  Saved: {fasta_path}")
        print()

        report_lines.append(f"## {name}")
        report_lines.append(f"- UniProt: {acc}")
        report_lines.append(f"- Kinase domain: residues {info['kinase_domain'][0]}–{info['kinase_domain'][1]} ({canonical_len} aa)")
        report_lines.append(f"- Currently in app: {current_len} aa")
        report_lines.append(f"- Match: {'[OK] exact' if match else '[FAIL] MISMATCH'}")
        if not match and current in kd_mut:
            offset = kd_mut.index(current)
            report_lines.append(f"- Truncation: missing {offset} residues N-term, missing {canonical_len - offset - current_len} residues C-term")
        report_lines.append("")

    (out_dir / "kinase_audit_report.md").write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
