"""Protein sequence encoder: embedding + stacked dilated 1D convolutions."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphdti.config import NUM_AA_TOKENS


class ProteinCNN(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        channels: int = 128,
        out_dim: int = 256,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed = nn.Embedding(NUM_AA_TOKENS, embed_dim, padding_idx=0)
        layers = []
        in_ch = embed_dim
        for k, d in zip(kernel_sizes, dilations):
            pad = (k - 1) // 2 * d
            layers.append(nn.Conv1d(in_ch, channels, kernel_size=k, padding=pad, dilation=d))
            layers.append(nn.BatchNorm1d(channels))
            layers.append(nn.ReLU(inplace=True))
            in_ch = channels
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Linear(channels, out_dim)
        self.dropout = dropout

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L] long
        mask = (tokens != 0).float().unsqueeze(1)  # [B, 1, L]
        x = self.embed(tokens).transpose(1, 2)  # [B, embed, L]
        h = self.convs(x)  # [B, channels, L]
        h = h * mask  # zero-out padding before pooling
        denom = mask.sum(dim=2).clamp(min=1.0)
        pooled = h.sum(dim=2) / denom  # masked mean pool
        pooled = F.dropout(pooled, p=self.dropout, training=self.training)
        return self.proj(pooled)
