"""Evaluation utilities: AUROC, PR-AUC, and a CLI to score a checkpoint on val/test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from graphdti.config import ModelConfig
from graphdti.data.dataset import DTIDataset, collate
from graphdti.models import GraphDTIModel


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """AUROC, PR-AUC, best F1 over the PR curve, and the threshold that achieves it."""
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "pr_auc": float("nan"), "best_f1": float("nan"), "best_threshold": 0.5}
    auroc = float(roc_auc_score(y, p))
    pr_auc = float(average_precision_score(y, p))
    precisions, recalls, thresholds = precision_recall_curve(y, p)
    # F1 across the curve; thresholds is len(precisions)-1
    f1s = 2 * precisions[:-1] * recalls[:-1] / np.clip(precisions[:-1] + recalls[:-1], 1e-9, None)
    if len(f1s) == 0:
        return {"auroc": auroc, "pr_auc": pr_auc, "best_f1": float("nan"), "best_threshold": 0.5}
    best = int(np.argmax(f1s))
    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "best_f1": float(f1s[best]),
        "best_threshold": float(thresholds[best]),
    }


def evaluate(ckpt_path: str | Path, data_csv: str | Path, batch_size: int = 64, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = ModelConfig(**state["cfg"]) if "cfg" in state else ModelConfig()
    model = GraphDTIModel(cfg).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    ds = DTIDataset(data_csv)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    ys, ps = [], []
    with torch.no_grad():
        for graphs, proteins, labels in loader:
            graphs = graphs.to(device)
            proteins = proteins.to(device)
            probs = model.predict_proba(graphs, proteins).cpu().numpy()
            ys.append(labels.numpy())
            ps.append(probs)
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    metrics = compute_metrics(y, p)
    metrics["n"] = int(len(y))
    metrics["positive_rate"] = float(y.mean()) if len(y) else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True, help="dir containing val.csv (and test.csv if present)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data = Path(args.data)
    results = {}
    for split in ["val", "test"]:
        path = data / f"{split}.csv"
        if path.exists():
            results[split] = evaluate(args.ckpt, path, args.batch_size, args.device)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
