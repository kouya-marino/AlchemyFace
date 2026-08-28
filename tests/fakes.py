"""Stand-ins for the two model-backed components.

The pipeline's job is sequencing and thresholding, not inference. Swapping in
fakes lets every branch of it be tested in milliseconds with no 37 MB download,
which is the entire reason Detector and Embedder are protocols.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from alchemyface.types import Face


def make_face(x: int = 0, y: int = 0, w: int = 10, h: int = 10) -> Face:
    return Face(
        bbox=(x, y, w, h),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        confidence=0.99,
    )


class FakeDetector:
    """Returns a canned list of faces and records how often it was called."""

    def __init__(self, faces: list[Face] | None = None) -> None:
        self.faces = faces if faces is not None else [make_face()]
        self.calls = 0

    def detect(self, image: NDArray[np.uint8]) -> list[Face]:
        self.calls += 1
        return list(self.faces)


class FakeEmbedder:
    """Maps each face to a one-hot unit vector, keyed by bbox width.

    One-hot vectors are mutually orthogonal, so two different faces score
    exactly 0.0 against each other and a face scores exactly 1.0 against
    itself. That makes threshold assertions exact rather than approximate.
    """

    dim = 8

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, image: NDArray[np.uint8], face: Face) -> NDArray[np.float32]:
        self.calls += 1
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[face.bbox[2] % self.dim] = 1.0
        return vector
