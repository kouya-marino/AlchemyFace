"""The Resize tab, driven through its real widgets.

The resizing itself is covered display-free in test_resize_data.py. What is left
here is the wiring: that the two source modes work, that defaults are filled in,
that failures are logged rather than raised, and that the ratio is clamped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.gui


def an_image(path: Path, width: int = 200, height: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (90, 140, 200)).save(path)
    return path


@pytest.fixture()
def resize(app):  # type: ignore[no-untyped-def]
    return app.resize_view


# ------------------------------------------------------------------- the tab


def test_the_resize_tab_exists(app) -> None:  # type: ignore[no-untyped-def]
    assert "Resize" in app.tab_labels()


def test_the_ratio_starts_at_a_half(resize) -> None:  # type: ignore[no-untyped-def]
    assert resize.ratio == pytest.approx(0.5)


@pytest.mark.parametrize("given,expected", [(0.25, 0.25), (99.0, 5.0), (0.0, 0.05)])
def test_the_ratio_is_clamped(resize, given: float, expected: float) -> None:  # type: ignore[no-untyped-def]
    resize._ratio_var.set(given)
    assert resize.ratio == pytest.approx(expected)


# ----------------------------------------------------------------- a folder


def test_resizing_a_folder(resize, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    src = tmp_path / "in"
    for name in ("a.png", "b.png"):
        an_image(src / name, 200, 100)
    assert resize.resize_folder_now(src, tmp_path / "out") == 2
    assert Image.open(tmp_path / "out" / "a.png").size == (100, 50)
    assert "Done. 2 resized, 0 failed." in resize.log_text()


def test_the_output_folder_defaults_beside_the_source(resize, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    src = tmp_path / "photos"
    an_image(src / "a.png")
    assert resize.resize_folder_now(src) == 1
    assert (tmp_path / "photos_resized" / "a.png").is_file()


def test_one_bad_file_is_logged_and_the_rest_proceed(resize, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    src = tmp_path / "in"
    an_image(src / "good.png")
    (src / "broken.png").write_bytes(b"not an image")
    assert resize.resize_folder_now(src, tmp_path / "out") == 1
    log = resize.log_text()
    assert "FAILED" in log
    assert "Done. 1 resized, 1 failed." in log


def test_an_empty_folder_reports(resize, app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert resize.resize_folder_now(empty, tmp_path / "out") == 0
    assert "No images found" in app.reporter.last_error


def test_a_missing_folder_reports(resize, app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    assert resize.resize_folder_now(tmp_path / "absent", tmp_path / "out") == 0
    assert app.reporter.errors


def test_resizing_a_folder_onto_itself_is_refused(resize, app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Filenames are preserved, so this would overwrite every original."""
    src = tmp_path / "in"
    an_image(src / "a.png")
    assert resize.resize_folder_now(src, src) == 0
    assert "same folder" in app.reporter.last_error


# ------------------------------------------------------------ single image


def test_resizing_one_image(resize, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    src = an_image(tmp_path / "in.png", 200, 100)
    assert resize.resize_image_now(src, tmp_path / "out.png") is True
    assert Image.open(tmp_path / "out.png").size == (100, 50)
    assert "200x100 -> 100x50" in resize.log_text()


def test_the_output_image_defaults_beside_the_source(resize, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    src = an_image(tmp_path / "photo.png")
    assert resize.resize_image_now(src) is True
    assert (tmp_path / "photo_resized.png").is_file()


def test_resizing_an_image_onto_itself_is_refused(resize, app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    src = an_image(tmp_path / "in.png")
    assert resize.resize_image_now(src, src) is False
    assert "same file" in app.reporter.last_error


def test_an_unreadable_image_reports(resize, app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    assert resize.resize_image_now(bad, tmp_path / "out.png") is False
    assert app.reporter.errors


def test_a_missing_image_reports(resize, app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    assert resize.resize_image_now(tmp_path / "absent.png", tmp_path / "out.png") is False
    assert app.reporter.errors


# ------------------------------------------------------------------- the log


def test_the_log_can_be_cleared(resize, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    resize.resize_image_now(an_image(tmp_path / "a.png"), tmp_path / "b.png")
    assert resize.log_text().strip()
    resize.clear_log()
    assert not resize.log_text().strip()


def test_the_status_bar_is_updated(resize, app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    resize.resize_image_now(an_image(tmp_path / "a.png"), tmp_path / "b.png")
    assert "Resized" in app.status


# ------------------------------------------------------- refusing bad input
#
# A resize cannot be undone, so the tab asks rather than guesses. Silently
# clamping meant a typo — 50 for 0.5 — rewrote a folder at a size nobody asked
# for, with nothing on screen to say it had happened.


def test_an_empty_source_folder_is_refused_not_crashed(resize) -> None:  # type: ignore[no-untyped-def]
    """Path("") is PosixPath("."), which passed the is_dir() guard and then
    failed deep inside with an uncaught ValueError."""
    resize._folder_in.set("")
    assert resize.resize_folder_now() == 0
    assert any("source folder" in message.lower() for _title, message in resize.reporter.errors)


def test_an_empty_source_image_is_refused(resize) -> None:  # type: ignore[no-untyped-def]
    resize._image_in.set("")
    assert resize.resize_image_now() is False
    assert any("source image" in message.lower() for _title, message in resize.reporter.errors)


@pytest.mark.parametrize("bad", [50.0, 0.0, -1.0, 5.5])
def test_an_out_of_range_ratio_is_refused_rather_than_clamped(resize, tmp_path: Path, bad: float) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "in"
    folder.mkdir()
    an_image(folder / "one.jpg")
    resize._ratio_var.set(bad)
    assert resize.resize_folder_now(folder, tmp_path / "out") == 0
    assert any("between" in message for _title, message in resize.reporter.errors)
    assert not (tmp_path / "out").exists()


def test_a_non_numeric_ratio_is_refused(resize, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "in"
    folder.mkdir()
    an_image(folder / "one.jpg")
    resize._ratio_var.set(float("nan"))
    assert resize.resize_folder_now(folder, tmp_path / "out") == 0
    assert resize.reporter.errors


def test_an_unsupported_extension_is_refused_before_pillow_sees_it(resize, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A .tif would resize fine and then be ignored by the Build tab's scan."""
    from PIL import Image

    odd = tmp_path / "photo.tif"
    Image.new("RGB", (40, 30)).save(odd)
    assert resize.resize_image_now(odd) is False
    assert any("Unsupported file extension" in message for _title, message in resize.reporter.errors)
    assert not (tmp_path / "photo_resized.tif").exists()


# --------------------------------------------------------------- the log


def test_the_log_gets_a_header_and_a_summary(resize, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "in"
    folder.mkdir()
    an_image(folder / "one.jpg")
    an_image(folder / "two.jpg")
    resize.resize_folder_now(folder, tmp_path / "out")
    log = resize.log_text()
    assert "Resizing 2 image(s) at ratio 0.50" in log
    assert f"  src: {folder}" in log
    assert "Done. 2 resized, 0 failed." in log


def test_the_log_is_cleared_between_runs(resize, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Appending under the previous run made it unclear which lines were which."""
    folder = tmp_path / "in"
    folder.mkdir()
    an_image(folder / "one.jpg")
    resize.resize_folder_now(folder, tmp_path / "out1")
    resize.resize_folder_now(folder, tmp_path / "out2")
    assert resize.log_text().count("Resizing 1 image(s)") == 1
    assert "out1" not in resize.log_text()


def test_single_image_mode_logs_a_summary_too(resize, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    source = an_image(tmp_path / "one.jpg")
    assert resize.resize_image_now(source) is True
    log = resize.log_text()
    assert "Resizing 1 image at ratio 0.50" in log
    assert "Done. 1 resized, 0 failed." in log
