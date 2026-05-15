import torch

from graphdti.config import MAX_PROTEIN_LEN, ModelConfig
from graphdti.data.featurize import (
    encode_protein,
    morgan_fingerprint,
    smiles_to_graph,
    tanimoto,
)


def test_smiles_to_graph_basic():
    g = smiles_to_graph("CCO")
    assert g is not None
    assert g.x.size(0) == 3  # 3 heavy atoms
    assert g.edge_index.size(1) >= 2  # at least one bond, doubled
    cfg = ModelConfig()
    assert g.x.size(1) == cfg.atom_in_dim
    assert g.edge_attr.size(1) == cfg.bond_in_dim


def test_smiles_to_graph_invalid():
    assert smiles_to_graph("not_a_real_smiles!!!") is None


def test_encode_protein_padding():
    seq = "ACDEFGHIK"
    enc = encode_protein(seq)
    assert enc.shape == (MAX_PROTEIN_LEN,)
    assert enc.dtype == torch.long
    # first 9 tokens non-zero, rest zero-padded
    assert (enc[:9] != 0).all()
    assert (enc[9:] == 0).all()


def test_encode_protein_unknown_chars():
    enc = encode_protein("ACDXYZ")  # X is unknown
    # tokens 0..2 valid, tokens 3,4,5 -> X is invalid (0), Y valid, Z invalid (0)
    assert enc[0] != 0 and enc[1] != 0 and enc[2] != 0
    assert int(enc[3]) == 0  # X -> unknown


def test_morgan_and_tanimoto():
    fp1 = morgan_fingerprint("CCO")
    fp2 = morgan_fingerprint("CCO")
    fp3 = morgan_fingerprint("c1ccccc1")
    assert fp1 is not None and fp3 is not None
    assert tanimoto(fp1, fp2) == 1.0
    assert 0.0 <= tanimoto(fp1, fp3) < 1.0
