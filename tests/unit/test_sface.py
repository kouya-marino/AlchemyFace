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


def synthetic_face_offset() -> Face:
    """A second, differently-placed face so two embeddings are not identical."""
    return Face(
        bbox=(40, 30, 100, 100),
        landmarks=np.array(
            [[70, 60], [120, 60], [95, 88], [74, 112], [116, 112]],
            dtype=np.float32,
        ),
        confidence=0.97,
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


def test_normalize_defaults_to_true() -> None:
    import inspect

    sig = inspect.signature(SFaceEmbedder.__init__)
    assert sig.parameters["normalize"].default is True


@pytest.mark.models
def test_normalize_false_returns_the_raw_vector(model_dir: Path) -> None:
    # Raw SFace output has an L2 norm around 10. The robot's .pkl schema stores
    # it unchanged, so this option has to give it back untouched.
    embedder = SFaceEmbedder(model_dir=model_dir, normalize=False)
    vector = embedder.embed(synthetic_face_image(), synthetic_face())
    assert vector.shape == (128,)
    assert vector.dtype == np.float32
    assert float(np.linalg.norm(vector)) > 2.0


@pytest.mark.models
def test_normalized_and_raw_differ_only_in_magnitude(model_dir: Path) -> None:
    image, face = synthetic_face_image(), synthetic_face()
    unit = SFaceEmbedder(model_dir=model_dir).embed(image, face)
    rawv = SFaceEmbedder(model_dir=model_dir, normalize=False).embed(image, face)
    np.testing.assert_allclose(rawv / np.linalg.norm(rawv), unit, atol=1e-5)


@pytest.mark.models
def test_cosine_is_identical_either_way(model_dir: Path) -> None:
    """The property that makes normalize=False safe to store.

    If this ever fails, every .pkl written with raw vectors would rank
    differently from one written with unit vectors, and the option would be a
    correctness hazard rather than a formatting choice.
    """
    image = synthetic_face_image()
    a, b = synthetic_face(), synthetic_face_offset()
    unit_e = SFaceEmbedder(model_dir=model_dir)
    raw_e = SFaceEmbedder(model_dir=model_dir, normalize=False)

    ua, ub = unit_e.embed(image, a), unit_e.embed(image, b)
    ra, rb = raw_e.embed(image, a), raw_e.embed(image, b)

    cos_unit = float(ua @ ub)
    cos_raw = float(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb)))
    assert cos_unit == pytest.approx(cos_raw, abs=1e-5)
