"""SFace hands back an unnormalised (1, 128) row. The embedder's contract is
a flat, unit-length (128,) vector, which is what makes cosine similarity a
dot product everywhere downstream."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.detection import YuNetDetector
from alchemyface.embedding import SFaceEmbedder
from alchemyface.types import Face


def synthetic_face_image() -> np.ndarray:
    """A deterministic image with something face-shaped in it. It does not
    need to be a real face — alignCrop only uses the landmarks we supply."""
    rng = np.random.default_rng(seed=20260828)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def synthetic_face() -> Face:
    return Face(
        bbox=(80, 60, 120, 120),
        landmarks=np.array(
            [[110, 95], [170, 95], [140, 125], [115, 155], [165, 155]],
            dtype=np.float32,
        ),
        confidence=0.99,
    )


def test_embedder_declares_its_dimension() -> None:
    assert SFaceEmbedder.dim == 128


@pytest.mark.models
def test_embed_returns_a_flat_128_vector(model_dir: Path) -> None:
    embedder = SFaceEmbedder(model_dir=model_dir)
    vector = embedder.embed(synthetic_face_image(), synthetic_face())
    assert vector.shape == (128,)
    assert vector.dtype == np.float32


@pytest.mark.models
def test_embed_returns_a_unit_vector(model_dir: Path) -> None:
    # Measured: raw SFace output has an L2 norm around 10. The embedder must
    # normalise, or every cosine score downstream is wrong.
    embedder = SFaceEmbedder(model_dir=model_dir)
    vector = embedder.embed(synthetic_face_image(), synthetic_face())
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.models
def test_the_same_input_embeds_identically(model_dir: Path) -> None:
    embedder = SFaceEmbedder(model_dir=model_dir)
    image, face = synthetic_face_image(), synthetic_face()
    np.testing.assert_allclose(embedder.embed(image, face), embedder.embed(image, face), atol=1e-6)


@pytest.mark.models
def test_dot_product_of_a_vector_with_itself_is_one(model_dir: Path) -> None:
    embedder = SFaceEmbedder(model_dir=model_dir)
    vector = embedder.embed(synthetic_face_image(), synthetic_face())
    assert float(vector @ vector) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.models
def test_detect_then_embed_works_end_to_end(model_dir: Path) -> None:
    detector = YuNetDetector(model_dir=model_dir, score_threshold=0.3)
    embedder = SFaceEmbedder(model_dir=model_dir)
    image = synthetic_face_image()
    for face in detector.detect(image):
        assert embedder.embed(image, face).shape == (128,)
