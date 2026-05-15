"""CLI: print atom and residue attributions for a single ligand–protein pair."""
from __future__ import annotations

import argparse
import json

from graphdti.data.featurize import encode_protein, smiles_to_graph
from graphdti.interpret import atom_attributions, residue_occlusion
from graphdti.serving.app import load_predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--smiles", required=True)
    p.add_argument("--protein", required=True)
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--stride", type=int, default=5)
    args = p.parse_args()

    predictor = load_predictor(args.ckpt)
    graph = smiles_to_graph(args.smiles)
    if graph is None:
        raise SystemExit(f"Could not parse SMILES: {args.smiles!r}")
    protein = encode_protein(args.protein)

    prob = predictor.predict(args.smiles, args.protein)
    atom = atom_attributions(predictor.model, graph, protein, steps=args.ig_steps, device=predictor.device)
    res = residue_occlusion(
        predictor.model, graph, protein, args.protein,
        window=args.window, stride=args.stride, device=predictor.device,
    )

    print(json.dumps({"probability": prob, "atoms": atom.to_dict(), "residues": res.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
