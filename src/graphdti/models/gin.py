"""GIN encoder over molecular graphs with edge features (GINE)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class GINEncoder(nn.Module):
    """Stack of GINEConv layers + BatchNorm + ReLU, returning per-node embeddings.

    Outputs `x_final` (N, hidden) where N is total nodes across the batch.
    Use a readout module to pool to per-graph embeddings.
    """

    def __init__(
        self,
        atom_in_dim: int,
        bond_in_dim: int,
        hidden: int = 256,
        num_layers: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.atom_proj = nn.Linear(atom_in_dim, hidden)
        self.edge_proj = nn.Linear(bond_in_dim, hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        x = self.atom_proj(x)
        e = self.edge_proj(edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index, e)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # residual
        return x
