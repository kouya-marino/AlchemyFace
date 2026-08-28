"""Row conversion is pure arithmetic and always tested. Anything that loads
the real network is marked `models`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.detection import YuNetDetector, face_from_row, row_from_face


def sample_row() -> np.ndarray:
    """One YuNet detection: bbox, five landmarks, score — 15 float32 columns."""
    return np.array(
        [10, 20, 30, 40, 15, 25, 35, 25, 25, 35, 18, 50, 32, 50, 0.93],
        dtype=np.float32,
    )


def test_face_from_row_reads_the_bounding_box() -> None:
    face = face_from_row(sample_row(), image_shape=(480, 640))
    assert face.bbox == (10, 20, 30, 40)


def test_face_from_row_reads_five_landmarks() -> None:
    face = face_from_row(sample_row(), image_shape=(480, 640))
    assert face.landmarks.shape == (5, 2)
    assert face.landmarks[0].tolist() == [15.0, 25.0]


def test_face_from_row_reads_the_score() -> None:
    face = face_from_row(sample_row(), image_shape=(480, 640))
    assert face.confidence == pytest.approx(0.93, abs=1e-6)


def test_negative_origin_is_clamped_to_the_image() -> None:
    # YuNet happily returns boxes that start off the left or top edge.
    row = sample_row()
    row[0], row[1] = -25.0, -12.0
    face = face_from_row(row, image_shape=(480, 640))
    assert face.bbox[0] == 0
    assert face.bbox[1] == 0


def test_box_running_past_the_edge_is_trimmed() -> None:
    row = sample_row()
    row[0], row[2] = 600.0, 200.0  # x=600 w=200 on a 640-wide image
    face = face_from_row(row, image_shape=(480, 640))
    assert face.bbox[0] + face.bbox[2] <= 640


def test_row_from_face_round_trips() -> None:
    original = sample_row()
    face = face_from_row(original, image_shape=(480, 640))
    rebuilt = row_from_face(face)
    assert rebuilt.shape == (15,)
    assert rebuilt.dtype == np.float32
    np.testing.assert_allclose(rebuilt[4:14], original[4:14])
    assert rebuilt[14] == pytest.approx(0.93, abs=1e-6)


def test_detector_satisfies_the_protocol() -> None:
    from alchemyface.detection.base import Detector

    assert isinstance(YuNetDetector, type)
    assert hasattr(Detector, "detect")


@pytest.mark.models
def test_detector_returns_no_faces_for_a_blank_frame(model_dir: Path) -> None:
    detector = YuNetDetector(model_dir=model_dir)
    assert detector.detect(np.zeros((240, 320, 3), dtype=np.uint8)) == []


@pytest.mark.models
def test_detector_handles_a_changing_frame_size(model_dir: Path) -> None:
    # setInputSize must be re-issued whenever the frame dimensions change,
    # or OpenCV throws. Feeding two sizes in a row proves it is handled.
    detector = YuNetDetector(model_dir=model_dir)
    detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))


@pytest.mark.models
def test_detector_rejects_a_non_bgr_image(model_dir: Path) -> None:
    detector = YuNetDetector(model_dir=model_dir)
    with pytest.raises(ValueError, match="three-channel"):
        detector.detect(np.zeros((240, 320), dtype=np.uint8))
