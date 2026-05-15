"""SHAP-on-graph attribution.

Two complementary signals:

1. `atom_attributions` — Integrated Gradients over atom features. Approximates
   SHAP values for the atom-feature inputs by integrating the model gradient
   along a straight-line path from a zero baseline to the real input. The
   per-atom score sums attributions across feature channels; positive means
   "pushes binding probability up", negative means "pushes it down".

2. `residue_occlusion` — windowed masking of contiguous protein residue spans.
   The drop in predicted probability when a span is masked attributes the
   contribution of that span. Surfaces residue-level binding hypotheses.

Both share the `Attribution` return type so the API can serialize them
uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
from torch_geometric.data import Batch


@dataclass
class Attribution:
    tokens: list[str]
    scores: list[float]
    method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def atom_attributions(
    model,
    graph,
    protein: torch.Tensor,
    steps: int = 32,
    device: str | None = None,
) -> Attribution:
    """Integrated-Gradients atom attributions for a single ligand–protein pair.

    Returns one score per atom (positive = drives binding probability up).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    batch = Batch.from_data_list([graph]).to(device)
    protein = protein.unsqueeze(0).to(device) if protein.dim() == 1 else protein.to(device)
    x_real = batch.x.detach()
    x_base = torch.zeros_like(x_real)

    total_grad = torch.zeros_like(x_real)
    for step in range(1, steps + 1):
        alpha = step / steps
        x_interp = (x_base + alpha * (x_real - x_base)).requires_grad_(True)
        logit = model.forward_with_x(x_interp, batch.edge_index, batch.edge_attr, batch.batch, protein)
        prob = torch.sigmoid(logit).sum()
        grad = torch.autograd.grad(prob, x_interp)[0]
        total_grad = total_grad + grad
    avg_grad = total_grad / steps
    ig = (x_real - x_base) * avg_grad  # [N, F]
    atom_scores = ig.sum(dim=-1).detach().cpu().tolist()

    smiles = getattr(graph, "smiles", None)
    tokens = [f"a{i}" for i in range(len(atom_scores))] if not smiles else _atom_tokens(smiles, len(atom_scores))
    return Attribution(tokens=tokens, scores=atom_scores, method="integrated_gradients")


def _atom_tokens(smiles: str, n_atoms: int) -> list[str]:
    """Pull element symbols from RDKit if available; otherwise fall back to indices."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() != n_atoms:
            return [f"a{i}" for i in range(n_atoms)]
        return [f"{a.GetSymbol()}{a.GetIdx()}" for a in mol.GetAtoms()]
    except Exception:
        return [f"a{i}" for i in range(n_atoms)]


def residue_occlusion(
    model,
    graph,
    protein: torch.Tensor,
    sequence: str,
    window: int = 15,
    stride: int = 5,
    device: str | None = None,
) -> Attribution:
    """Mask sliding windows of the protein sequence; attribute by drop in probability."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    batch = Batch.from_data_list([graph]).to(device)
    base_protein = protein.unsqueeze(0).to(device) if protein.dim() == 1 else protein.to(device)

    with torch.no_grad():
        base_prob = float(torch.sigmoid(model(batch, base_protein)).item())

    L = min(len(sequence), base_protein.size(1))
    if L == 0:
        return Attribution(tokens=[], scores=[], method="residue_occlusion")

    # per-residue score = average drop across all windows that cover that residue
    sums = torch.zeros(L)
    counts = torch.zeros(L)
    for start in range(0, L, stride):
        end = min(start + window, L)
        masked = base_protein.clone()
        masked[:, start:end] = 0  # 0 = padding/unknown token
        with torch.no_grad():
            p = float(torch.sigmoid(model(batch, masked)).item())
        drop = base_prob - p
        sums[start:end] += drop
        counts[start:end] += 1
        if end >= L:
            break

    counts = counts.clamp(min=1.0)
    scores = (sums / counts).tolist()
    tokens = list(sequence[:L])
    return Attribution(tokens=tokens, scores=scores, method="residue_occlusion")
