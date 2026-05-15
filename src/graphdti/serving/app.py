"""FastAPI serving app for GraphDTI.

`build_app(ckpt_path)` constructs the app with a loaded predictor.
`run()` is the console-script entry point — reads CKPT, HOST, PORT from env
or CLI flags and starts uvicorn.

Endpoints:
  GET  /health
  POST /predict   → binding probability
  POST /explain   → probability + atom / residue attributions
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException

from graphdti.config import ModelConfig
from graphdti.data.featurize import encode_protein, smiles_to_graph
from graphdti.interpret import atom_attributions, residue_occlusion
from graphdti.models import GraphDTIModel
from graphdti.serving.schemas import (
    AttributionPayload,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from torch_geometric.data import Batch


@dataclass
class Predictor:
    model: GraphDTIModel
    device: str
    threshold: float
    version: str

    def predict(self, smiles: str, protein_sequence: str) -> float:
        graph, protein = self._featurize(smiles, protein_sequence)
        batch = Batch.from_data_list([graph]).to(self.device)
        protein = protein.unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = float(self.model.predict_proba(batch, protein).item())
        return prob

    def explain(self, req: ExplainRequest) -> ExplainResponse:
        graph, protein = self._featurize(req.smiles, req.protein_sequence)
        prob = self.predict(req.smiles, req.protein_sequence)

        atom_payload, res_payload = None, None
        if "atom" in req.methods:
            attr = atom_attributions(self.model, graph, protein, steps=req.ig_steps, device=self.device)
            atom_payload = AttributionPayload(**attr.to_dict())
        if "residue" in req.methods:
            attr = residue_occlusion(
                self.model,
                graph,
                protein,
                req.protein_sequence,
                window=req.occlusion_window,
                stride=req.occlusion_stride,
                device=self.device,
            )
            res_payload = AttributionPayload(**attr.to_dict())

        return ExplainResponse(
            probability=prob,
            predicted_label=int(prob >= self.threshold),
            threshold=self.threshold,
            model_version=self.version,
            atom_attributions=atom_payload,
            residue_attributions=res_payload,
        )

    def _featurize(self, smiles: str, protein: str):
        graph = smiles_to_graph(smiles)
        if graph is None:
            raise HTTPException(status_code=422, detail=f"Could not parse SMILES: {smiles!r}")
        if not protein.strip():
            raise HTTPException(status_code=422, detail="Protein sequence is empty")
        return graph, encode_protein(protein)


def load_predictor(ckpt_path: str | Path, threshold: float = 0.5, device: str | None = None) -> Predictor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = ModelConfig(**state["cfg"]) if "cfg" in state else ModelConfig()
    model = GraphDTIModel(cfg).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    version = state.get("version") or Path(ckpt_path).stem
    return Predictor(model=model, device=device, threshold=threshold, version=version)


def build_app(ckpt_path: str | Path | None = None, threshold: float = 0.5) -> FastAPI:
    app = FastAPI(title="GraphDTI", version="0.1.0")
    state: dict = {"predictor": None}

    if ckpt_path:
        state["predictor"] = load_predictor(ckpt_path, threshold=threshold)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        p: Predictor | None = state["predictor"]
        if p is None:
            return HealthResponse(status="no-model")
        return HealthResponse(status="ok", model_version=p.version, device=p.device)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        p: Predictor | None = state["predictor"]
        if p is None:
            raise HTTPException(status_code=503, detail="No model loaded.")
        prob = p.predict(req.smiles, req.protein_sequence)
        return PredictResponse(
            probability=prob,
            predicted_label=int(prob >= p.threshold),
            threshold=p.threshold,
            model_version=p.version,
        )

    @app.post("/explain", response_model=ExplainResponse)
    def explain(req: ExplainRequest) -> ExplainResponse:
        p: Predictor | None = state["predictor"]
        if p is None:
            raise HTTPException(status_code=503, detail="No model loaded.")
        return p.explain(req)

    return app


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.environ.get("GRAPHDTI_CKPT"))
    parser.add_argument("--host", default=os.environ.get("GRAPHDTI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRAPHDTI_PORT", "8000")))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    import uvicorn

    app = build_app(args.ckpt, threshold=args.threshold)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
