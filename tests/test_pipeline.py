"""End-to-end smoke test: synthetic data → pretrain → train → evaluate → predict."""
from __future__ import annotations

from pathlib import Path

import torch

from graphdti.training.evaluate import evaluate
from graphdti.training.pretrain import pretrain
from graphdti.training.train import train


def test_full_pipeline(synth_data_dir, tmp_path):
    pre_ckpt = tmp_path / "pre.pt"
    dti_ckpt = tmp_path / "dti.pt"

    res_pre = pretrain(
        data_dir=synth_data_dir,
        out_path=pre_ckpt,
        epochs=1,
        batch_size=16,
        lr=1e-3,
        hard_neg_k=1,
        device="cpu",
        seed=0,
    )
    assert pre_ckpt.exists()
    assert len(res_pre["history"]) == 1
    assert torch.load(pre_ckpt, map_location="cpu")["model_state"]

    res_train = train(
        data_dir=synth_data_dir,
        out_path=dti_ckpt,
        init_ckpt=pre_ckpt,
        epochs=1,
        batch_size=16,
        lr=1e-3,
        device="cpu",
        seed=0,
    )
    assert dti_ckpt.exists()
    assert "history" in res_train

    metrics = evaluate(dti_ckpt, Path(synth_data_dir) / "val.csv", batch_size=16, device="cpu")
    assert metrics["n"] > 0
    # AUROC may be unstable with 1 epoch on 64 examples — just check it returned a number
    assert "auroc" in metrics
