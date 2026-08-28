"""VideoSource is a thin wrapper over cv2.VideoCapture. Everything except the
"open a real camera" path is tested with a stub capture object."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alchemyface import capture
from alchemyface.capture import VideoSource

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


class StubCapture:
    """Stands in for cv2.VideoCapture."""

    def __init__(self, frames: int = 2, opens: bool = True) -> None:
        self._remaining = frames
        self._opens = opens
        self.released = False
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self._opens

    def set(self, prop: int, value: float) -> bool:
        self.properties[prop] = value
        return True

    def read(self) -> tuple[bool, Any]:
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, FRAME.copy()

    def release(self) -> None:
        self.released = True


@pytest.fixture()
def stub(monkeypatch: pytest.MonkeyPatch) -> StubCapture:
    instance = StubCapture()
    monkeypatch.setattr(capture.cv2, "VideoCapture", lambda source: instance)
    return instance


def test_read_returns_a_frame(stub: StubCapture) -> None:
    with VideoSource() as source:
        assert source.read() is not None


def test_read_returns_none_when_the_stream_ends(stub: StubCapture) -> None:
    with VideoSource() as source:
        source.read()
        source.read()
        assert source.read() is None


def test_frames_iterates_until_the_stream_ends(stub: StubCapture) -> None:
    with VideoSource() as source:
        assert len(list(source.frames())) == 2


def test_context_manager_releases_the_capture(stub: StubCapture) -> None:
    with VideoSource():
        pass
    assert stub.released is True


def test_release_is_idempotent(stub: StubCapture) -> None:
    source = VideoSource()
    source.release()
    source.release()
    assert stub.released is True


def test_a_camera_that_will_not_open_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture.cv2, "VideoCapture", lambda source: StubCapture(opens=False))
    with pytest.raises(RuntimeError, match="could not open"):
        VideoSource(source=7)


def test_requested_resolution_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    instance = StubCapture()
    monkeypatch.setattr(capture.cv2, "VideoCapture", lambda source: instance)
    VideoSource(width=1280, height=720).release()
    assert instance.properties[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert instance.properties[cv2.CAP_PROP_FRAME_HEIGHT] == 720


@pytest.mark.camera
def test_a_real_camera_yields_a_frame() -> None:
    with VideoSource() as source:
        assert source.read() is not None
