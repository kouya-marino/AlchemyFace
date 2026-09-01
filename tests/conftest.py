"""Shared pytest fixtures.

Tests that need the real ONNX weights are marked ``models`` and skip
automatically when the weights cannot be found, so the default ``make test``
run needs neither a download nor a camera.
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Directory holding the real ONNX weights, or skip the test."""
    raw = os.environ.get("ALCHEMYFACE_MODEL_DIR")
    if not raw:
        pytest.skip("ALCHEMYFACE_MODEL_DIR is not set")
    path = Path(raw)
    if not path.is_dir():
        pytest.skip(f"model directory does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def pkl_dir() -> Path:
    """Directory of real .pkl face databases, or skip.

    These are production databases holding real people's embeddings, so they
    live in the git-ignored _local/ tree and are never committed. Tests using
    them assert on counts and shapes, never on names.
    """
    path = Path(__file__).resolve().parent.parent / "_local" / "pkl"
    if not path.is_dir():
        pytest.skip(f"pkl fixtures not present: {path}")
    return path
