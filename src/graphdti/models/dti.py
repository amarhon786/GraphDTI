"""End-to-end DTI model: ligand graph + protein sequence → binding logit."""
from __future__ import annotations

import torch
import torch.nn as nn

from graphdti.config import ModelConfig
from graphdti.models.gin import GINEncoder
from graphdti.models.protein import ProteinCNN
from graphdti.models.readout import AttentiveReadout


class ProjectionHead(nn.Module):
    """MLP projection head used during contrastive pretraining (SimCLR-style)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)


class GraphDTIModel(nn.Module):
    """GIN(ligand) + CNN(protein) + bilinear interaction + MLP head."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.ligand_encoder = GINEncoder(
            atom_in_dim=cfg.atom_in_dim,
            bond_in_dim=cfg.bond_in_dim,
            hidden=cfg.gin_hidden,
            num_layers=cfg.gin_layers,
            dropout=cfg.dropout,
        )
        self.readout = AttentiveReadout(in_dim=cfg.gin_hidden, out_dim=cfg.gin_hidden)
        self.protein_encoder = ProteinCNN(
            embed_dim=cfg.protein_embed_dim,
            channels=cfg.protein_channels,
            out_dim=cfg.protein_out_dim,
            dropout=cfg.dropout,
        )

        d_l, d_p = cfg.gin_hidden, cfg.protein_out_dim
        self.bilinear = nn.Bilinear(d_l, d_p, cfg.head_hidden)
        self.head = nn.Sequential(
            nn.Linear(cfg.head_hidden + d_l + d_p, cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
        # projection head for contrastive pretraining
        self.projection = ProjectionHead(d_l, cfg.projection_dim)

    def encode_ligand(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (graph_embedding [B, d_l], attention_weights [N, 1])."""
        node_h = self.ligand_encoder(x, edge_index, edge_attr)
        graph_emb, attn = self.readout(node_h, batch)
        return graph_emb, attn

    def forward(self, graphs, proteins: torch.Tensor) -> torch.Tensor:
        """Returns binding logits [B]."""
        return self.forward_with_x(
            graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch, proteins
        )

    def forward_with_x(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_idx: torch.Tensor,
        proteins: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward starting from raw atom features `x`.

        Exposed separately so attribution methods can compute gradients
        with respect to `x` (Integrated Gradients in `interpret/shap_graph.py`).
        """
        ligand_emb, _ = self.encode_ligand(x, edge_index, edge_attr, batch_idx)
        protein_emb = self.protein_encoder(proteins)
        cross = self.bilinear(ligand_emb, protein_emb)
        joint = torch.cat([cross, ligand_emb, protein_emb], dim=-1)
        return self.head(joint).squeeze(-1)

    def predict_proba(self, graphs, proteins: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(graphs, proteins))
