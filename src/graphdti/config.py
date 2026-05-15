"""Shared configuration. Keep this small — module-level constants only."""
from dataclasses import dataclass

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 reserved for <unk>
NUM_AA_TOKENS = len(AMINO_ACIDS) + 1
MAX_PROTEIN_LEN = 1000

ATOMIC_NUMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53]  # 14 common elements; rest -> "other"
NUM_ATOM_TYPES = len(ATOMIC_NUMS) + 1
NUM_DEGREES = 6
NUM_HS = 5
NUM_HYBRIDIZATIONS = 6
NUM_FORMAL_CHARGES = 7  # -3..+3
NUM_BOND_TYPES = 4  # single, double, triple, aromatic


@dataclass
class ModelConfig:
    atom_in_dim: int = (
        NUM_ATOM_TYPES + NUM_DEGREES + NUM_HS + NUM_HYBRIDIZATIONS + NUM_FORMAL_CHARGES + 2
    )  # +2 for aromaticity and ring membership
    bond_in_dim: int = NUM_BOND_TYPES + 2  # +2 for conjugation and ring
    gin_hidden: int = 256
    gin_layers: int = 5
    protein_embed_dim: int = 64
    protein_channels: int = 128
    protein_out_dim: int = 256
    head_hidden: int = 256
    dropout: float = 0.2
    projection_dim: int = 128  # contrastive projection head
