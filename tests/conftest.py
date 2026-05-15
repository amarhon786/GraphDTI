"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from graphdti.data.synthetic import generate


@pytest.fixture(scope="session")
def synth_data_dir(tmp_path_factory) -> Path:
    """Tiny synthetic dataset shared across the test session."""
    out = tmp_path_factory.mktemp("synth")
    generate(n_train=64, n_val=32, n_proteins=4, threshold=0.2, seed=42, out_dir=out)
    return out
