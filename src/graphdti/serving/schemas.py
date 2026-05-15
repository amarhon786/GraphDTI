"""Pydantic request/response schemas for the FastAPI service."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1, description="Ligand SMILES (RDKit-parseable)")
    protein_sequence: str = Field(..., min_length=1, description="One-letter amino acid sequence")


class PredictResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    predicted_label: int
    threshold: float
    model_version: str


class AttributionPayload(BaseModel):
    tokens: list[str]
    scores: list[float]
    method: str


class ExplainRequest(PredictRequest):
    methods: list[Literal["atom", "residue"]] = Field(default_factory=lambda: ["atom", "residue"])
    ig_steps: int = Field(32, ge=4, le=128)
    occlusion_window: int = Field(15, ge=3, le=64)
    occlusion_stride: int = Field(5, ge=1, le=64)


class ExplainResponse(PredictResponse):
    atom_attributions: AttributionPayload | None = None
    residue_attributions: AttributionPayload | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "no-model"]
    model_version: str | None = None
    device: str | None = None
