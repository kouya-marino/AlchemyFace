"""SFace embeddings through OpenCV's DNN runtime.

``alignCrop`` warps the face to a canonical 112x112 using the five landmarks,
and ``feature`` turns that into a ``(1, 128) float32`` row. That row is *not*
normalised — its L2 norm is around 10 — so this class flattens and normalises
it. Once every vector is unit length, cosine similarity is a dot product, which
is exactly what OpenCV's own ``FaceRecognizerSF.match`` computes and why
scikit-learn is not a dependency.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from alchemyface.detection import row_from_face
from alchemyface.models import EMBEDDER, resolve
from alchemyface.types import Face


class SFaceEmbedder:
    """Implements :class:`~alchemyface.embedding.base.Embedder` with SFace."""

    dim: int = 128

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        model_dir: Path | str | None = None,
    ) -> None:
        path = Path(model_path) if model_path else resolve(EMBEDDER, model_dir)
        # The `Xxx.create` class-method form is what cv2's bundled type stubs
        # declare; the module-level `FaceRecognizerSF_create` alias is not.
        self._recognizer = cv2.FaceRecognizerSF.create(str(path), "")

    def embed(self, image: NDArray[np.uint8], face: Face) -> NDArray[np.float32]:
        """A unit-length 128-d embedding of one detected face."""
        aligned = self._recognizer.alignCrop(image, row_from_face(face))
        raw = self._recognizer.feature(aligned)
        flat = np.asarray(raw, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(flat))
        if norm == 0.0:
            raise ValueError("SFace returned a zero embedding for this face")
        return (flat / norm).astype(np.float32)
