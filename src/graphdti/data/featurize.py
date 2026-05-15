"""RDKit molecule → PyG graph, plus protein tokenization and Morgan fingerprints."""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from torch_geometric.data import Data

# silence RDKit's noisy info/warning channel
RDLogger.DisableLog("rdApp.*")
_MORGAN_GENERATORS: dict[tuple[int, int], "GetMorganGenerator"] = {}

from graphdti.config import (
    AA_TO_IDX,
    ATOMIC_NUMS,
    MAX_PROTEIN_LEN,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    NUM_DEGREES,
    NUM_FORMAL_CHARGES,
    NUM_HS,
    NUM_HYBRIDIZATIONS,
)

_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]

_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _one_hot(value, choices) -> list[float]:
    """One-hot with an "other" bucket appended at the end."""
    vec = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    vec[idx] = 1.0
    return vec


def _atom_features(atom: Chem.Atom) -> list[float]:
    feats = []
    feats += _one_hot(atom.GetAtomicNum(), ATOMIC_NUMS)
    feats += _one_hot(min(atom.GetDegree(), NUM_DEGREES - 2), list(range(NUM_DEGREES - 1)))
    feats += _one_hot(min(atom.GetTotalNumHs(), NUM_HS - 2), list(range(NUM_HS - 1)))
    feats += _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS[:-1])
    fc = max(-3, min(3, atom.GetFormalCharge()))
    feats += _one_hot(fc, list(range(-3, 3)))
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.IsInRing()))
    assert len(feats) == (
        NUM_ATOM_TYPES + NUM_DEGREES + NUM_HS + NUM_HYBRIDIZATIONS + NUM_FORMAL_CHARGES + 2
    ), len(feats)
    return feats


def _bond_features(bond: Chem.Bond) -> list[float]:
    feats = _one_hot(bond.GetBondType(), _BOND_TYPES[:-1])
    feats.append(float(bond.GetIsConjugated()))
    feats.append(float(bond.IsInRing()))
    assert len(feats) == NUM_BOND_TYPES + 2
    return feats


def smiles_to_graph(smiles: str) -> Data | None:
    """Parse a SMILES into a PyG `Data` object. Returns None on invalid input."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = _bond_features(bond)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(feats)
        edge_attr.append(feats)

    if not edge_index:
        # single-atom molecule: add a self-loop so message passing has something to do
        edge_index = [[0, 0]]
        edge_attr = [[0.0] * (NUM_BOND_TYPES + 2)]

    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        smiles=smiles,
    )


def encode_protein(sequence: str, max_len: int = MAX_PROTEIN_LEN) -> torch.Tensor:
    """Tokenize a protein sequence to a 1-D LongTensor of length `max_len` (zero-padded)."""
    seq = sequence.strip().upper()[:max_len]
    ids = [AA_TO_IDX.get(aa, 0) for aa in seq]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def _morgan_generator(radius: int, n_bits: int):
    key = (radius, n_bits)
    gen = _MORGAN_GENERATORS.get(key)
    if gen is None:
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        _MORGAN_GENERATORS[key] = gen
    return gen


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Return a binary Morgan fingerprint as numpy array. None on invalid SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = _morgan_generator(radius, n_bits).GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    inter = int(np.bitwise_and(fp_a, fp_b).sum())
    union = int(np.bitwise_or(fp_a, fp_b).sum())
    return inter / union if union else 0.0
