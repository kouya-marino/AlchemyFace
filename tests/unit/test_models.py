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
    assert models.find_local(local_spec, model_dir=explicit) == explicit / "canonical.onnx"


def test_env_var_is_used_when_no_explicit_dir(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALCHEMYFACE_MODEL_DIR", str(tmp_path))
    (tmp_path / "canonical.onnx").write_bytes(PAYLOAD)
    assert models.find_local(local_spec) == tmp_path / "canonical.onnx"


def test_alias_filename_is_found(tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_download_writes_file_and_verifies_checksum(tmp_path: Path, local_spec: ModelSpec) -> None:
    dest = models.download(local_spec, dest_dir=tmp_path / "cache")
    assert dest.read_bytes() == PAYLOAD
    assert dest.name == "canonical.onnx"


def test_download_rejects_a_checksum_mismatch(tmp_path: Path, local_spec: ModelSpec) -> None:
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


def test_resolve_prefers_a_local_file_over_downloading(tmp_path: Path, local_spec: ModelSpec) -> None:
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


# ------------------------------------------- naming a directory means that one
#
# `find_local` deliberately falls back to the environment variable and the
# cache. A caller that named one directory needs the opposite, and conflating
# the two meant `download-models --model-dir X` reported "already present"
# because a copy sat in the cache, leaving X empty.


def test_find_in_searches_only_the_named_directory(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "canonical.onnx").write_bytes(PAYLOAD)
    monkeypatch.setattr(models, "cache_dir", lambda: cache)
    wanted = tmp_path / "wanted"
    wanted.mkdir()

    # find_local is satisfied by the cache copy; find_in must not be.
    assert models.find_local(local_spec, model_dir=wanted) == cache / "canonical.onnx"
    assert models.find_in(local_spec, wanted) is None

    (wanted / "legacy_name.onnx").write_bytes(PAYLOAD)
    assert models.find_in(local_spec, wanted) == wanted / "legacy_name.onnx"


def test_resolve_downloads_into_the_directory_it_was_given(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It used to drop model_dir on the download path and write to the cache."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(models, "cache_dir", lambda: cache)
    wanted = tmp_path / "wanted"

    resolved = models.resolve(local_spec, model_dir=wanted)
    assert resolved.parent == wanted
    assert resolved.read_bytes() == PAYLOAD
    assert not cache.exists()


def test_resolve_still_uses_the_cache_when_no_directory_is_named(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setattr(models, "cache_dir", lambda: cache)
    assert models.resolve(local_spec).parent == cache


def test_download_reports_an_unwritable_destination_cleanly(tmp_path: Path, local_spec: ModelSpec) -> None:
    """Every other model failure here is an AlchemyFaceError; a caller should
    not have to catch OSError for this one case."""
    import os
    import stat

    readonly = tmp_path / "readonly"
    readonly.mkdir()
    os.chmod(readonly, stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(ModelDownloadError, match="cannot write to"):
            models.download(local_spec, dest_dir=readonly)
    finally:
        os.chmod(readonly, 0o755)
