"""Every exception AlchemyFace raises descends from AlchemyFaceError, so a
caller can wrap the library in one except clause and never see a bare
cv2.error or urllib exception leak through."""

from __future__ import annotations


class AlchemyFaceError(Exception):
    """Base class for every error AlchemyFace raises."""


class ModelNotFoundError(AlchemyFaceError):
    """Weights were not on disk and downloading them was not permitted."""


class ModelDownloadError(AlchemyFaceError):
    """A download failed, or what arrived did not match its checksum."""


class NoFaceDetectedError(AlchemyFaceError):
    """An operation required a face and the image had none."""


class PickleSchemaError(AlchemyFaceError):
    """A pickle did not hold a recognisable face database."""
