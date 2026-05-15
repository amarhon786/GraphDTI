"""SageMaker inference handler (alternative to the FastAPI server entrypoint).

If you'd rather use SageMaker's built-in `multi-model-server` /
`sagemaker-inference-toolkit` rather than the bundled FastAPI app, point your
container at this module. SageMaker calls model_fn → input_fn → predict_fn →
output_fn for each invocation.
"""
from __future__ import annotations

import json
from pathlib import Path

from graphdti.serving.app import load_predictor
from graphdti.serving.schemas import ExplainRequest, PredictRequest


def model_fn(model_dir: str):
    """Load the model artefact written by SageMaker (model.tar.gz unpacked here)."""
    ckpt = Path(model_dir) / "dti.pt"
    if not ckpt.exists():
        # search recursively in case the tarball nested it
        matches = list(Path(model_dir).rglob("dti.pt"))
        if not matches:
            raise FileNotFoundError(f"No dti.pt under {model_dir}")
        ckpt = matches[0]
    return load_predictor(ckpt)


def input_fn(request_body: bytes | str, content_type: str = "application/json"):
    if content_type not in ("application/json", "application/x-json"):
        raise ValueError(f"Unsupported content type: {content_type}")
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    return json.loads(request_body)


def predict_fn(payload: dict, predictor) -> dict:
    if payload.get("explain"):
        req = ExplainRequest(**{k: v for k, v in payload.items() if k != "explain"})
        return predictor.explain(req).model_dump()
    req = PredictRequest(**payload)
    prob = predictor.predict(req.smiles, req.protein_sequence)
    return {
        "probability": prob,
        "predicted_label": int(prob >= predictor.threshold),
        "threshold": predictor.threshold,
        "model_version": predictor.version,
    }


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    return json.dumps(prediction)
