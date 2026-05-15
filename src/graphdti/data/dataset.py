"""Dataset wrapper around a `(smiles, protein_sequence, label)` CSV.

Uses `torch.utils.data.Dataset` rather than `torch_geometric.data.Dataset`
because we batch with a custom `collate` that already calls `Batch.from_data_list`;
the PyG Dataset base would re-wrap and add complexity we don't need.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from graphdti.data.featurize import encode_protein, smiles_to_graph


@dataclass
class DTIPair:
    graph: Data
    protein: torch.Tensor  # LongTensor[max_len]
    label: int
    protein_id: str
    smiles: str


class DTIDataset(Dataset):
    """Reads a CSV, pre-featurizes each row, drops un-parseable SMILES."""

    def __init__(self, csv_path: str | Path):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        if "protein_id" not in self.df.columns:
            self.df["protein_id"] = "UNK"
        self._items: list[DTIPair] = []
        for row in self.df.itertuples(index=False):
            graph = smiles_to_graph(row.smiles)
            if graph is None:
                continue
            self._items.append(
                DTIPair(
                    graph=graph,
                    protein=encode_protein(row.protein_sequence),
                    label=int(row.label),
                    protein_id=str(row.protein_id),
                    smiles=row.smiles,
                )
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> DTIPair:
        return self._items[idx]


def collate(batch: list[DTIPair]):
    """Custom collate: batch graphs via PyG, stack protein tensors, return label tensor."""
    graphs = Batch.from_data_list([b.graph for b in batch])
    proteins = torch.stack([b.protein for b in batch], dim=0)
    labels = torch.tensor([b.label for b in batch], dtype=torch.float32)
    return graphs, proteins, labels
