"""The value types carry data and nothing else, so these tests pin down the
two things that are easy to get wrong: area arithmetic, and the fact that a
dataclass holding an ndarray cannot use a generated __eq__."""

from __future__ import annotations

import numpy as np
import pytest

from alchemyface.errors import (
    AlchemyFaceError,
    ModelDownloadError,
    ModelNotFoundError,
    NoFaceDetectedError,
)
from alchemyface.types import Face, Match, Recognition


def _landmarks() -> np.ndarray:
    return np.arange(10, dtype=np.float32).reshape(5, 2)


def test_face_area_is_width_times_height() -> None:
    face = Face(bbox=(10, 20, 30, 40), landmarks=_landmarks(), confidence=0.9)
    assert face.area == 1200


def test_face_is_frozen() -> None:
    face = Face(bbox=(0, 0, 1, 1), landmarks=_landmarks(), confidence=0.5)
    with pytest.raises(AttributeError):
        face.confidence = 0.1  # type: ignore[misc]


def test_two_identical_faces_compare_by_identity_not_value() -> None:
    # landmarks is an ndarray; a generated __eq__ would return an array and
    # blow up with "truth value of an array is ambiguous". eq=False avoids it.
    a = Face(bbox=(0, 0, 1, 1), landmarks=_landmarks(), confidence=0.5)
    b = Face(bbox=(0, 0, 1, 1), landmarks=_landmarks(), confidence=0.5)
    assert a != b
    assert a == a


def test_match_compares_by_value() -> None:
    a = Match(label="ada", score=0.9, entry_id="x", metadata={})
    b = Match(label="ada", score=0.9, entry_id="x", metadata={})
    assert a == b


def test_recognition_holds_no_match_when_unknown() -> None:
    face = Face(bbox=(0, 0, 1, 1), landmarks=_landmarks(), confidence=0.5)
    assert Recognition(face=face, match=None).match is None


@pytest.mark.parametrize(
    "error",
    [ModelNotFoundError, ModelDownloadError, NoFaceDetectedError],
)
def test_every_error_is_catchable_as_the_base(error: type[Exception]) -> None:
    with pytest.raises(AlchemyFaceError):
        raise error("boom")
