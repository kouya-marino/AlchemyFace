"""Recognizer owns no algorithm — it sequences three protocols and applies a
threshold. These tests cover exactly that, using fakes."""

from __future__ import annotations

import numpy as np
import pytest

from alchemyface import Recognizer
from alchemyface.errors import NoFaceDetectedError
from alchemyface.store import InMemoryStore
from tests.fakes import FakeDetector, FakeEmbedder, make_face

IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)


def build(faces: list | None = None, threshold: float = 0.363) -> Recognizer:
    embedder = FakeEmbedder()
    return Recognizer(
        detector=FakeDetector(faces),
        embedder=embedder,
        store=InMemoryStore(dim=embedder.dim),
        threshold=threshold,
    )


def test_injected_components_are_used_as_given() -> None:
    recognizer = build()
    recognizer.detect(IMAGE)
    assert recognizer.detector.calls == 1  # type: ignore[attr-defined]


def test_store_defaults_to_the_embedder_dimension() -> None:
    assert build().store.dim == FakeEmbedder.dim  # type: ignore[attr-defined]


def test_enroll_stores_one_entry_and_returns_its_id() -> None:
    recognizer = build()
    entry_id = recognizer.enroll("ada", IMAGE)
    assert isinstance(entry_id, str)
    assert len(recognizer.store) == 1


def test_enroll_keeps_metadata() -> None:
    recognizer = build()
    recognizer.enroll("ada", IMAGE, metadata={"team": "analytical"})
    (match,) = recognizer.store.search(np.eye(8, dtype=np.float32)[10 % 8])
    assert match.metadata == {"team": "analytical"}


def test_enroll_picks_the_largest_face() -> None:
    small, large = make_face(w=3, h=3), make_face(w=5, h=40)
    recognizer = build([small, large])
    recognizer.enroll("ada", IMAGE)
    # FakeEmbedder keys on bbox width, so the stored vector identifies which
    # face was chosen: index 5, not index 3.
    (match,) = recognizer.store.search(np.eye(8, dtype=np.float32)[5])
    assert match.score == pytest.approx(1.0)


def test_enroll_raises_when_there_is_no_face() -> None:
    recognizer = build(faces=[])
    with pytest.raises(NoFaceDetectedError):
        recognizer.enroll("ada", IMAGE)


def test_identify_returns_one_recognition_per_face() -> None:
    recognizer = build([make_face(w=1), make_face(w=2), make_face(w=3)])
    assert len(recognizer.identify(IMAGE)) == 3


def test_identify_returns_nothing_for_an_empty_frame() -> None:
    assert build(faces=[]).identify(IMAGE) == []


def test_identify_matches_an_enrolled_face() -> None:
    recognizer = build()
    recognizer.enroll("ada", IMAGE)
    (recognition,) = recognizer.identify(IMAGE)
    assert recognition.match is not None
    assert recognition.match.label == "ada"
    assert recognition.match.score == pytest.approx(1.0)


def test_identify_reports_unknown_below_the_threshold() -> None:
    recognizer = build([make_face(w=10)])
    recognizer.enroll("ada", IMAGE)
    # A different width maps to an orthogonal vector: score 0.0 < 0.363.
    recognizer.detector.faces = [make_face(w=11)]  # type: ignore[attr-defined]
    (recognition,) = recognizer.identify(IMAGE)
    assert recognition.match is None
    assert recognition.face.bbox[2] == 11


def test_identify_against_an_empty_gallery_is_unknown_not_an_error() -> None:
    # The prototype crashed here: match() returned None and the caller
    # unpacked it regardless.
    (recognition,) = build().identify(IMAGE)
    assert recognition.match is None


def test_threshold_of_zero_accepts_an_orthogonal_match() -> None:
    recognizer = build([make_face(w=10)], threshold=0.0)
    recognizer.enroll("ada", IMAGE)
    recognizer.detector.faces = [make_face(w=11)]  # type: ignore[attr-defined]
    (recognition,) = recognizer.identify(IMAGE)
    assert recognition.match is not None
    assert recognition.match.label == "ada"


def test_default_threshold_is_the_sface_operating_point() -> None:
    assert build().threshold == pytest.approx(0.363)


def test_public_exports_are_importable_from_the_package_root() -> None:
    import alchemyface

    for name in ("Recognizer", "Face", "Match", "Recognition", "AlchemyFaceError"):
        assert hasattr(alchemyface, name), name
