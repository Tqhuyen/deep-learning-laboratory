from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Every test starts from the same RNG state, on CPU unless it asks otherwise."""
    torch.manual_seed(0)
    yield
