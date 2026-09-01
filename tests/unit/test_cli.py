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


def test_enroll_writes_a_gallery(image_file: Path, tmp_path: Path, fake_recognizer) -> None:
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


def test_enroll_appends_to_an_existing_gallery(image_file: Path, tmp_path: Path, fake_recognizer) -> None:
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


def test_identify_reports_a_known_face(image_file: Path, tmp_path: Path, fake_recognizer) -> None:
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
    result = runner.invoke(cli.app, ["identify", "--image", str(image_file), "--gallery", str(gallery)])
    assert result.exit_code == 0, result.stdout
    assert "ada" in result.stdout


def test_identify_reports_unknown_against_an_empty_gallery(image_file: Path, tmp_path: Path, fake_recognizer) -> None:
    gallery = tmp_path / "empty.npz"
    InMemoryStore(dim=FakeEmbedder.dim).save(gallery)
    result = runner.invoke(cli.app, ["identify", "--image", str(image_file), "--gallery", str(gallery)])
    assert result.exit_code == 0
    assert "unknown" in result.stdout.lower()


def test_download_models_reports_each_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_download(spec: ModelSpec, dest_dir: Path | None = None) -> Path:
        seen.append(spec.key)
        return tmp_path / spec.filename

    monkeypatch.setattr(cli, "download", fake_download)
    monkeypatch.setattr(cli, "find_local", lambda spec, model_dir=None: None)
    result = runner.invoke(cli.app, ["download-models"])
    assert result.exit_code == 0
    assert sorted(seen) == ["detector", "embedder"]


def test_download_models_skips_what_is_already_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_local", lambda spec, model_dir=None: tmp_path / spec.filename)
    monkeypatch.setattr(cli, "download", lambda spec, dest_dir=None: pytest.fail("should not download"))
    result = runner.invoke(cli.app, ["download-models"])
    assert result.exit_code == 0
    assert "already" in result.stdout.lower()


# ------------------------------------------------------------------- resize


def make_image(path: Path, width: int = 200, height: int = 100) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (10, 120, 200)).save(path)
    return path


def test_resize_is_registered() -> None:
    assert "resize" in runner.invoke(cli.app, ["--help"]).output


def test_resize_a_folder(tmp_path: Path) -> None:
    from PIL import Image

    src = tmp_path / "in"
    for name in ("a.png", "b.png"):
        make_image(src / name, 200, 100)
    result = runner.invoke(
        cli.app, ["resize", "--folder", str(src), "--output", str(tmp_path / "out"), "--ratio", "0.5"]
    )
    assert result.exit_code == 0, result.output
    assert "2 resized, 0 failed" in result.output
    assert Image.open(tmp_path / "out" / "a.png").size == (100, 50)


def test_resize_a_folder_defaults_its_output(tmp_path: Path) -> None:
    src = tmp_path / "photos"
    make_image(src / "a.png")
    result = runner.invoke(cli.app, ["resize", "--folder", str(src)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "photos_resized" / "a.png").is_file()


def test_resize_a_single_image(tmp_path: Path) -> None:
    src = make_image(tmp_path / "in.png", 200, 100)
    result = runner.invoke(cli.app, ["resize", "--image", str(src), "--output", str(tmp_path / "out.png")])
    assert result.exit_code == 0, result.output
    assert "200x100 -> 100x50" in result.output


def test_resize_needs_exactly_one_source(tmp_path: Path) -> None:
    src = tmp_path / "in"
    make_image(src / "a.png")
    both = runner.invoke(cli.app, ["resize", "--folder", str(src), "--image", str(src / "a.png")])
    assert both.exit_code == 2
    assert "exactly one" in both.output
    assert runner.invoke(cli.app, ["resize"]).exit_code == 2


def test_resize_clamps_an_absurd_ratio(tmp_path: Path) -> None:
    src = make_image(tmp_path / "in.png", 100, 100)
    result = runner.invoke(
        cli.app,
        ["resize", "--image", str(src), "--output", str(tmp_path / "o.png"), "--ratio", "99"],
    )
    assert result.exit_code == 0, result.output
    assert "clamped to 5.00" in result.output


def test_resize_reports_an_empty_folder(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(cli.app, ["resize", "--folder", str(empty)])
    assert result.exit_code == 1
    assert "no images found" in result.output.lower()


def test_resize_refuses_to_overwrite_its_own_source(tmp_path: Path) -> None:
    src = make_image(tmp_path / "in.png")
    result = runner.invoke(cli.app, ["resize", "--image", str(src), "--output", str(src)])
    assert result.exit_code == 1
    assert "same file" in result.output
