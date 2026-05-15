"""Contrastive pretraining of the ligand GIN encoder.

Loss: SimCLR-style InfoNCE. For each anchor ligand:
- positive view = stochastic graph augmentation of the same molecule
  (random edge dropout + random atom-feature masking)
- negatives = all other in-batch graphs and their augmentations
- HARD negatives = each anchor's Morgan-fingerprint top-K nearest
  neighbors are deliberately included in the same batch. These look
  similar to a fingerprint baseline but are structurally different,
  so the graph model is forced to learn a more discriminative
  representation than what Morgan fingerprints alone can express.

Only the GIN encoder + readout + projection head are updated here.
The protein encoder and DTI head are trained downstream.
"""
from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from graphdti.config import ModelConfig
from graphdti.data.featurize import morgan_fingerprint, smiles_to_graph, tanimoto
from graphdti.models import GraphDTIModel


def augment_graph(data, edge_drop_p: float = 0.15, feat_mask_p: float = 0.15):
    """Return a stochastic view of a PyG graph: random edge drop + atom feature masking."""
    data = deepcopy(data)
    if data.edge_index.size(1) > 2:
        n_edges = data.edge_index.size(1)
        keep = torch.rand(n_edges) > edge_drop_p
        # symmetric: drop edge (i,j) and (j,i) together — bonds are stored both ways
        if keep.sum() < 2:
            keep[0] = True
            keep[1] = True
        data.edge_index = data.edge_index[:, keep]
        data.edge_attr = data.edge_attr[keep]
    if data.x.size(0) > 0:
        mask = (torch.rand(data.x.size(0)) < feat_mask_p).unsqueeze(-1)
        data.x = data.x * (~mask)
    return data


class HardNegativeIndex:
    """For each anchor SMILES, find Morgan-FP top-K nearest neighbor SMILES."""

    def __init__(self, smiles: list[str], k: int = 4, max_index: int = 5000):
        # subsample for speed if the pool is huge
        rng = random.Random(0)
        if len(smiles) > max_index:
            smiles = rng.sample(smiles, max_index)
        self.smiles = []
        self.fps = []
        for s in smiles:
            fp = morgan_fingerprint(s)
            if fp is not None:
                self.smiles.append(s)
                self.fps.append(fp)
        self.fps_mat = np.stack(self.fps).astype(np.uint8) if self.fps else None
        self.k = k

    def neighbors(self, smiles: str) -> list[str]:
        if not self.smiles:
            return []
        fp = morgan_fingerprint(smiles)
        if fp is None:
            return []
        inter = np.bitwise_and(self.fps_mat, fp).sum(axis=1)
        union = np.bitwise_or(self.fps_mat, fp).sum(axis=1).clip(min=1)
        sims = inter / union
        idx = np.argsort(-sims)[:self.k + 1]  # +1 in case self is in the index
        out = []
        for i in idx:
            if self.smiles[i] != smiles:
                out.append(self.smiles[i])
            if len(out) >= self.k:
                break
        return out


def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric InfoNCE over two views (each already L2-normalized).

    z_a, z_b: [B, d]. Positives are paired by index; negatives are all other
    rows in the concatenated 2B set.
    """
    z = torch.cat([z_a, z_b], dim=0)  # [2B, d]
    sim = z @ z.t() / temperature  # [2B, 2B]
    B = z_a.size(0)
    # mask self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=sim.device)
    sim.masked_fill_(mask, float("-inf"))
    targets = torch.arange(2 * B, device=sim.device)
    targets = (targets + B) % (2 * B)  # positive index is the "other view"
    return F.cross_entropy(sim, targets)


def _collect_ligand_smiles(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    return df["smiles"].drop_duplicates().tolist()


def pretrain(
    data_dir: str | Path,
    out_path: str | Path,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.1,
    hard_neg_k: int = 2,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(data_dir)
    smiles = _collect_ligand_smiles(data_dir / "train.csv")
    # featurize once
    graphs: dict[str, Data] = {}
    for s in smiles:
        g = smiles_to_graph(s)
        if g is not None:
            graphs[s] = g
    smiles = list(graphs.keys())
    if len(smiles) < 4:
        raise RuntimeError("Need at least 4 unique parseable SMILES to pretrain.")

    hard_idx = HardNegativeIndex(smiles, k=hard_neg_k)

    cfg = ModelConfig()
    model = GraphDTIModel(cfg).to(device)
    # only ligand encoder + readout + projection get optimized
    params = list(model.ligand_encoder.parameters()) + list(model.readout.parameters()) + list(model.projection.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    history = []
    model.train()
    for epoch in range(epochs):
        random.shuffle(smiles)
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(range(0, len(smiles), batch_size), desc=f"pretrain ep{epoch}")
        for start in pbar:
            anchors = smiles[start : start + batch_size]
            if len(anchors) < 2:
                continue
            # Augment each anchor with its FP hard negatives in the batch.
            batch_smiles = list(anchors)
            for s in anchors:
                for nb in hard_idx.neighbors(s):
                    if nb not in batch_smiles:
                        batch_smiles.append(nb)

            view_a = [augment_graph(graphs[s]) for s in batch_smiles]
            view_b = [augment_graph(graphs[s]) for s in batch_smiles]
            ga = Batch.from_data_list(view_a).to(device)
            gb = Batch.from_data_list(view_b).to(device)

            za, _ = model.encode_ligand(ga.x, ga.edge_index, ga.edge_attr, ga.batch)
            zb, _ = model.encode_ligand(gb.x, gb.edge_index, gb.edge_attr, gb.batch)
            za = model.projection(za)
            zb = model.projection(zb)

            loss = info_nce_loss(za, zb, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        history.append({"epoch": epoch, "loss": epoch_loss / max(n_batches, 1)})

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__, "history": history}, out)

    return {"history": history, "path": str(out_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--hard-neg-k", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    result = pretrain(
        args.data, args.out, args.epochs, args.batch_size, args.lr,
        args.temperature, args.hard_neg_k, args.device, args.seed,
    )
    print(json.dumps(result["history"], indent=2))


if __name__ == "__main__":
    main()
