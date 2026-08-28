"""Resolution order and download integrity. Nothing here touches the network:
the download tests point the spec's URL at a file:// URL in a tmp_path, which
urllib handles natively."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from alchemyface import models
from alchemyface.errors import ModelDownloadError, ModelNotFoundError
from alchemyface.models import ModelSpec

PAYLOAD = b"pretend onnx bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture()
def local_spec(tmp_path: Path) -> ModelSpec:
    """A spec whose URL is a real file on disk, served over file://."""
    source = tmp_path / "source.onnx"
    source.write_bytes(PAYLOAD)
    return ModelSpec(
        key="test",
        filename="canonical.onnx",
        url=source.as_uri(),
        sha256=DIGEST,
        aliases=("legacy_name.onnx",),
    )


def test_candidates_puts_canonical_name_first(local_spec: ModelSpec) -> None:
    assert local_spec.candidates == ("canonical.onnx", "legacy_name.onnx")


def test_cache_dir_honours_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert models.cache_dir() == tmp_path / "alchemyface" / "models"


def test_explicit_model_dir_wins_over_env(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit, env = tmp_path / "explicit", tmp_path / "env"
    for d in (explicit, env):
        d.mkdir()
        (d / "canonical.onnx").write_bytes(PAYLOAD)
    monkeypatch.setenv("ALCHEMYFACE_MODEL_DIR", str(env))
    assert (
        models.find_local(local_spec, model_dir=explicit) == explicit / "canonical.onnx"
    )


def test_env_var_is_used_when_no_explicit_dir(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALCHEMYFACE_MODEL_DIR", str(tmp_path))
    (tmp_path / "canonical.onnx").write_bytes(PAYLOAD)
    assert models.find_local(local_spec) == tmp_path / "canonical.onnx"


def test_alias_filename_is_found(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The prototype's weights use different filenames; they must still resolve.
    monkeypatch.setenv("ALCHEMYFACE_MODEL_DIR", str(tmp_path))
    (tmp_path / "legacy_name.onnx").write_bytes(PAYLOAD)
    assert models.find_local(local_spec) == tmp_path / "legacy_name.onnx"


def test_find_local_returns_none_when_absent(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALCHEMYFACE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(models, "cache_dir", lambda: tmp_path / "nope")
    assert models.find_local(local_spec) is None


def test_download_writes_file_and_verifies_checksum(
    tmp_path: Path, local_spec: ModelSpec
) -> None:
    dest = models.download(local_spec, dest_dir=tmp_path / "cache")
    assert dest.read_bytes() == PAYLOAD
    assert dest.name == "canonical.onnx"


def test_download_rejects_a_checksum_mismatch(
    tmp_path: Path, local_spec: ModelSpec
) -> None:
    wrong = ModelSpec(
        key=local_spec.key,
        filename=local_spec.filename,
        url=local_spec.url,
        sha256="0" * 64,
        aliases=local_spec.aliases,
    )
    cache = tmp_path / "cache"
    with pytest.raises(ModelDownloadError, match="checksum"):
        models.download(wrong, dest_dir=cache)
    # A bad download must not leave the file behind, nor a .part turd.
    assert list(cache.iterdir()) == []


def test_download_raises_on_a_missing_source(tmp_path: Path) -> None:
    spec = ModelSpec(
        key="gone",
        filename="gone.onnx",
        url=(tmp_path / "absent.onnx").as_uri(),
        sha256=DIGEST,
    )
    with pytest.raises(ModelDownloadError):
        models.download(spec, dest_dir=tmp_path / "cache")


def test_resolve_prefers_a_local_file_over_downloading(
    tmp_path: Path, local_spec: ModelSpec
) -> None:
    (tmp_path / "canonical.onnx").write_bytes(b"local wins")
    resolved = models.resolve(local_spec, model_dir=tmp_path)
    assert resolved.read_bytes() == b"local wins"


def test_resolve_raises_when_download_is_forbidden(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALCHEMYFACE_MODEL_DIR", raising=False)
    monkeypatch.setattr(models, "cache_dir", lambda: tmp_path / "empty")
    with pytest.raises(ModelNotFoundError, match="canonical.onnx"):
        models.resolve(local_spec, allow_download=False)


def test_registry_pins_both_real_models() -> None:
    assert set(models.MODELS) == {"detector", "embedder"}
    for spec in models.MODELS.values():
        assert len(spec.sha256) == 64
        # raw.githubusercontent serves LFS pointers, not weights.
        assert spec.url.startswith("https://media.githubusercontent.com/media/")
