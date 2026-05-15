"""Gated attention pooling over per-node embeddings (Li et al. 2016)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax as scatter_softmax


class AttentiveReadout(nn.Module):
    """Gated attention: a = softmax(W_a h) ; g = sigmoid(W_g h) ; pool sum(a * g * h)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
        self.attn = nn.Linear(in_dim, 1)
        self.value = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (graph_emb [B, out_dim], attn_weights [N, 1])."""
        logits = self.attn(h)  # [N, 1]
        weights = scatter_softmax(logits, batch)  # softmax within each graph
        v = self.value(h) * self.gate(h)
        weighted = v * weights
        graph_emb = global_add_pool(weighted, batch)
        return graph_emb, weights
