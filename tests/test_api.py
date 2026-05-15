"""FastAPI integration tests using httpx.TestClient (no live network)."""
from __future__ import annotations

from pathlib import Path

import torch
from fastapi.testclient import TestClient

from graphdti.config import ModelConfig
from graphdti.models import GraphDTIModel
from graphdti.serving.app import build_app


def _make_tiny_checkpoint(path: Path) -> ModelConfig:
    cfg = ModelConfig(gin_hidden=32, gin_layers=2, protein_channels=16, protein_out_dim=32,
                      head_hidden=16, projection_dim=16)
    model = GraphDTIModel(cfg)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__}, path)
    return cfg


def test_health_without_model():
    app = build_app(ckpt_path=None)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "no-model"


def test_predict_and_explain(tmp_path):
    ckpt = tmp_path / "tiny.pt"
    _make_tiny_checkpoint(ckpt)
    app = build_app(ckpt_path=ckpt)
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    payload = {"smiles": "CCO", "protein_sequence": "ACDEFGHIK" * 5}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["predicted_label"] in (0, 1)

    r = client.post(
        "/explain",
        json={**payload, "methods": ["atom", "residue"], "ig_steps": 4, "occlusion_window": 5, "occlusion_stride": 3},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["atom_attributions"]["scores"]
    assert body["residue_attributions"]["scores"]
    assert len(body["residue_attributions"]["tokens"]) == len("ACDEFGHIK" * 5)


def test_predict_invalid_smiles(tmp_path):
    ckpt = tmp_path / "tiny.pt"
    _make_tiny_checkpoint(ckpt)
    app = build_app(ckpt_path=ckpt)
    client = TestClient(app)
    r = client.post("/predict", json={"smiles": "NOT_A_SMILES!!!", "protein_sequence": "ACDE"})
    assert r.status_code == 422
