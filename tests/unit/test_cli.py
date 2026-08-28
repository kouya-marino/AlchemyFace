"""CLI wiring only. The heavy lifting is already covered; these tests check
argument handling, exit codes and output shape, with the Recognizer patched
out so nothing loads a model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from alchemyface import cli
from alchemyface.models import ModelSpec
from alchemyface.store import InMemoryStore
from tests.fakes import FakeDetector, FakeEmbedder, make_face

runner = CliRunner()


@pytest.fixture()
def image_file(tmp_path: Path) -> Path:
    """A real PNG on disk, so cv2.imread succeeds without any face in it."""
    import cv2

    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.zeros((60, 60, 3), dtype=np.uint8))
    return path


@pytest.fixture()
def fake_recognizer(monkeypatch: pytest.MonkeyPatch):
    """Patch the CLI's Recognizer factory to build a fake-backed one."""
    from alchemyface.pipeline import Recognizer

    embedder = FakeEmbedder()
    recognizer = Recognizer(
        detector=FakeDetector([make_face(w=4)]),
        embedder=embedder,
        store=InMemoryStore(dim=embedder.dim),
    )
    monkeypatch.setattr(cli, "_build_recognizer", lambda **kwargs: recognizer)
    return recognizer


def test_version_prints_the_package_version() -> None:
    from alchemyface import __version__

    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help_lists_every_command() -> None:
    result = runner.invoke(cli.app, ["--help"])
    for command in ("version", "download-models", "enroll", "identify"):
        assert command in result.stdout


def test_enroll_writes_a_gallery(
    image_file: Path, tmp_path: Path, fake_recognizer
) -> None:
    gallery = tmp_path / "g.npz"
    result = runner.invoke(
        cli.app,
        [
            "enroll",
            "--name",
            "ada",
            "--image",
            str(image_file),
            "--gallery",
            str(gallery),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert gallery.exists()
    assert "ada" in result.stdout


def test_enroll_appends_to_an_existing_gallery(
    image_file: Path, tmp_path: Path, fake_recognizer
) -> None:
    gallery = tmp_path / "g.npz"
    for name in ("ada", "grace"):
        result = runner.invoke(
            cli.app,
            [
                "enroll",
                "--name",
                name,
                "--image",
                str(image_file),
                "--gallery",
                str(gallery),
            ],
        )
        assert result.exit_code == 0, result.stdout
    store = InMemoryStore(dim=FakeEmbedder.dim)
    store.load(gallery)
    assert len(store) == 2


def test_enroll_exits_nonzero_when_no_face_is_found(
    image_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alchemyface.pipeline import Recognizer

    embedder = FakeEmbedder()
    empty = Recognizer(
        detector=FakeDetector([]),
        embedder=embedder,
        store=InMemoryStore(dim=embedder.dim),
    )
    monkeypatch.setattr(cli, "_build_recognizer", lambda **kwargs: empty)
    result = runner.invoke(
        cli.app,
        [
            "enroll",
            "--name",
            "ada",
            "--image",
            str(image_file),
            "--gallery",
            str(tmp_path / "g.npz"),
        ],
    )
    assert result.exit_code == 1
    assert "no face" in result.output.lower()


def test_enroll_exits_nonzero_for_a_missing_image(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        [
            "enroll",
            "--name",
            "ada",
            "--image",
            str(tmp_path / "absent.png"),
            "--gallery",
            str(tmp_path / "g.npz"),
        ],
    )
    assert result.exit_code != 0


def test_identify_reports_a_known_face(
    image_file: Path, tmp_path: Path, fake_recognizer
) -> None:
    gallery = tmp_path / "g.npz"
    runner.invoke(
        cli.app,
        [
            "enroll",
            "--name",
            "ada",
            "--image",
            str(image_file),
            "--gallery",
            str(gallery),
        ],
    )
    result = runner.invoke(
        cli.app, ["identify", "--image", str(image_file), "--gallery", str(gallery)]
    )
    assert result.exit_code == 0, result.stdout
    assert "ada" in result.stdout


def test_identify_reports_unknown_against_an_empty_gallery(
    image_file: Path, tmp_path: Path, fake_recognizer
) -> None:
    gallery = tmp_path / "empty.npz"
    InMemoryStore(dim=FakeEmbedder.dim).save(gallery)
    result = runner.invoke(
        cli.app, ["identify", "--image", str(image_file), "--gallery", str(gallery)]
    )
    assert result.exit_code == 0
    assert "unknown" in result.stdout.lower()


def test_download_models_reports_each_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_download(spec: ModelSpec, dest_dir: Path | None = None) -> Path:
        seen.append(spec.key)
        return tmp_path / spec.filename

    monkeypatch.setattr(cli, "download", fake_download)
    monkeypatch.setattr(cli, "find_local", lambda spec, model_dir=None: None)
    result = runner.invoke(cli.app, ["download-models"])
    assert result.exit_code == 0
    assert sorted(seen) == ["detector", "embedder"]


def test_download_models_skips_what_is_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli, "find_local", lambda spec, model_dir=None: tmp_path / spec.filename
    )
    monkeypatch.setattr(
        cli, "download", lambda spec, dest_dir=None: pytest.fail("should not download")
    )
    result = runner.invoke(cli.app, ["download-models"])
    assert result.exit_code == 0
    assert "already" in result.stdout.lower()
