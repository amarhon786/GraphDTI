"""Supervised DTI training. Loads a pretrained ligand encoder (optional)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphdti.config import ModelConfig
from graphdti.data.dataset import DTIDataset, collate
from graphdti.models import GraphDTIModel
from graphdti.training.evaluate import compute_metrics


def _load_pretrained(model: GraphDTIModel, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu")
    model_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    # load only ligand-encoder-related keys so we don't clobber freshly-init protein encoder
    own = model.state_dict()
    to_load = {k: v for k, v in model_state.items() if k in own and v.shape == own[k].shape}
    missing = own.keys() - to_load.keys()
    model.load_state_dict({**own, **to_load})
    print(f"[init] loaded {len(to_load)} tensors from {path}; {len(missing)} kept at init.")


def train(
    data_dir: str | Path,
    out_path: str | Path,
    init_ckpt: str | Path | None = None,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(data_dir)
    train_ds = DTIDataset(data_dir / "train.csv")
    val_path = data_dir / "val.csv"
    val_ds = DTIDataset(val_path) if val_path.exists() else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
        if val_ds
        else None
    )

    cfg = ModelConfig()
    model = GraphDTIModel(cfg).to(device)
    if init_ckpt:
        _load_pretrained(model, init_ckpt)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    history = []
    best_auc = -1.0
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"train ep{epoch}")
        for graphs, proteins, labels in pbar:
            graphs = graphs.to(device)
            proteins = proteins.to(device)
            labels = labels.to(device)
            logits = model(graphs, proteins)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * labels.size(0)
            n += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()

        train_loss = running / max(n, 1)
        val_metrics = (
            _evaluate_loader(model, val_loader, device) if val_loader is not None else {}
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(json.dumps(history[-1], indent=2))

        auc = val_metrics.get("auroc", -1.0)
        if auc > best_auc:
            best_auc = auc
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg.__dict__, "history": history},
                out,
            )

    if best_auc < 0:
        # no val: save the last-epoch model
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__, "history": history}, out)

    return {"history": history, "best_auroc": best_auc, "path": str(out_path)}


def _evaluate_loader(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for graphs, proteins, labels in loader:
            graphs = graphs.to(device)
            proteins = proteins.to(device)
            probs = model.predict_proba(graphs, proteins).cpu().numpy()
            ys.append(labels.numpy())
            ps.append(probs)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return compute_metrics(y, p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=None, help="optional pretrained checkpoint")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    res = train(args.data, args.out, args.init, args.epochs, args.batch_size, args.lr,
                device=args.device, seed=args.seed)
    print(json.dumps(res["history"], indent=2))


if __name__ == "__main__":
    main()
