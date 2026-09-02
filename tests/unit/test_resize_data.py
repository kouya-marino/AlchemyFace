"""Bulk image resizing — the whole thing, with no Tk anywhere.

This tab exists because YuNet's largest anchors miss a face that fills most of
the frame, which is what a phone selfie looks like. Shrinking the photo brings
the face back into the detector's range.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from alchemyface.gui.resize_data import (
    DEFAULT_RATIO,
    MAX_RATIO,
    MIN_RATIO,
    ResizeOutcome,
    default_output_folder,
    plan_folder,
    resize_folder,
    resize_one,
)


def an_image(path: Path, width: int = 200, height: int = 100, mode: str = "RGB") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (width, height), color=(120, 30, 200) if mode == "RGB" else 128).save(path)
    return path


# ------------------------------------------------------------------- ratio


def test_ratio_defaults_are_the_originals() -> None:
    assert (MIN_RATIO, MAX_RATIO, DEFAULT_RATIO) == (0.05, 5.0, 0.5)


# ------------------------------------------------------------- resize_one


def test_halving_an_image(tmp_path: Path) -> None:
    src = an_image(tmp_path / "in.png", 200, 100)
    result = resize_one(src, tmp_path / "out.png", 0.5)
    assert (result.old_width, result.old_height) == (200, 100)
    assert (result.new_width, result.new_height) == (100, 50)
    assert Image.open(tmp_path / "out.png").size == (100, 50)


def test_doubling_an_image(tmp_path: Path) -> None:
    src = an_image(tmp_path / "in.png", 60, 40)
    result = resize_one(src, tmp_path / "out.png", 2.0)
    assert (result.new_width, result.new_height) == (120, 80)


def test_an_extreme_ratio_still_leaves_one_pixel(tmp_path: Path) -> None:
    src = an_image(tmp_path / "in.png", 10, 10)
    result = resize_one(src, tmp_path / "out.png", 0.001)
    assert (result.new_width, result.new_height) == (1, 1)


def test_the_output_directory_is_created(tmp_path: Path) -> None:
    src = an_image(tmp_path / "in.png")
    resize_one(src, tmp_path / "deep" / "deeper" / "out.png", 0.5)
    assert (tmp_path / "deep" / "deeper" / "out.png").is_file()


def test_writing_over_the_source_is_refused(tmp_path: Path) -> None:
    # Resizing a file onto itself destroys the original irrecoverably.
    src = an_image(tmp_path / "in.png")
    with pytest.raises(ValueError, match="same file"):
        resize_one(src, src, 0.5)


def test_a_jpeg_keeps_its_format_and_stays_rgb(tmp_path: Path) -> None:
    src = an_image(tmp_path / "in.jpg", 80, 60)
    resize_one(src, tmp_path / "out.jpg", 0.5)
    out = Image.open(tmp_path / "out.jpg")
    assert out.format == "JPEG"
    assert out.mode in ("RGB", "L")


def test_an_rgba_source_can_be_written_as_jpeg(tmp_path: Path) -> None:
    """JPEG has no alpha channel, so the mode has to be converted first."""
    src = tmp_path / "in.png"
    Image.new("RGBA", (60, 40), (1, 2, 3, 128)).save(src)
    resize_one(src, tmp_path / "out.jpg", 0.5)
    assert Image.open(tmp_path / "out.jpg").mode == "RGB"


def test_a_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        resize_one(tmp_path / "absent.png", tmp_path / "out.png", 0.5)


def test_a_non_image_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(OSError):
        resize_one(bad, tmp_path / "out.png", 0.5)


# --------------------------------------------------------------- planning


def test_default_output_folder_appends_resized(tmp_path: Path) -> None:
    assert default_output_folder(tmp_path / "photos").name == "photos_resized"


def test_planning_lists_every_image_and_keeps_filenames(tmp_path: Path) -> None:
    src = tmp_path / "in"
    for name in ("b.png", "a.jpg", "c.webp"):
        an_image(src / name)
    (src / "notes.txt").write_text("ignore me")
    jobs = plan_folder(src, tmp_path / "out", 0.5)
    assert [j[0].name for j in jobs] == ["a.jpg", "b.png", "c.webp"]
    assert [j[1].name for j in jobs] == ["a.jpg", "b.png", "c.webp"]
    assert all(j[1].parent == tmp_path / "out" for j in jobs)


def test_planning_an_empty_folder_yields_nothing(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert plan_folder(tmp_path / "empty", tmp_path / "out", 0.5) == []


def test_planning_refuses_the_same_folder(tmp_path: Path) -> None:
    src = tmp_path / "in"
    an_image(src / "a.png")
    with pytest.raises(ValueError, match="same folder"):
        plan_folder(src, src, 0.5)


def test_planning_a_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        plan_folder(tmp_path / "absent", tmp_path / "out", 0.5)


# ------------------------------------------------------------ resize_folder


def test_resizing_a_folder_reports_each_file(tmp_path: Path) -> None:
    src = tmp_path / "in"
    for name in ("a.png", "b.png"):
        an_image(src / name, 100, 50)
    outcomes = resize_folder(src, tmp_path / "out", 0.5)
    assert len(outcomes) == 2
    assert all(o.ok for o in outcomes)
    assert all(o.result is not None and o.result.new_width == 50 for o in outcomes)


def test_one_bad_file_does_not_stop_the_rest(tmp_path: Path) -> None:
    src = tmp_path / "in"
    an_image(src / "good.png", 100, 50)
    (src / "broken.png").write_bytes(b"not an image")
    an_image(src / "also-good.png", 100, 50)
    outcomes = resize_folder(src, tmp_path / "out", 0.5)
    assert len(outcomes) == 3
    assert sum(1 for o in outcomes if o.ok) == 2
    failed = [o for o in outcomes if not o.ok]
    assert len(failed) == 1 and failed[0].source.name == "broken.png"
    assert failed[0].error


def test_outcomes_render_a_log_line(tmp_path: Path) -> None:
    src = tmp_path / "in"
    an_image(src / "a.png", 100, 50)
    (src / "bad.png").write_bytes(b"nope")
    lines = [str(o) for o in resize_folder(src, tmp_path / "out", 0.5)]
    assert any("100x50 -> 50x25" in line for line in lines)
    assert any("FAILED" in line for line in lines)


def test_an_outcome_is_a_frozen_value_type(tmp_path: Path) -> None:
    an_image(tmp_path / "a.png")
    (outcome,) = resize_folder(tmp_path, tmp_path / "out", 0.5)
    assert isinstance(outcome, ResizeOutcome)
    with pytest.raises(AttributeError):
        outcome.ok = False  # type: ignore[misc]


@pytest.mark.models
def test_resizing_rescues_a_face_yunet_cannot_see(tmp_path: Path, model_dir: Path) -> None:
    """The reason this tab exists, demonstrated rather than asserted.

    YuNet's largest anchors miss a face that fills most of the frame — a phone
    selfie. Shrinking the photo brings it back into range.

    An earlier version of this project claimed a particular test photo showed
    this. It did not; detection had merely failed. So the failing case is
    constructed here, and the test would fail if resizing stopped helping.

    Detection is not monotonic in the ratio: it depends on the face matching an
    anchor scale, so a middling ratio can miss where a larger and a smaller one
    both succeed. Hence the search rather than a single value.
    """
    import cv2

    from alchemyface.detection import YuNetDetector

    sources = sorted((Path(__file__).resolve().parents[2] / "_local" / "test_images").glob("*"))
    if not sources:
        pytest.skip("test images not present")

    detector = YuNetDetector(model_dir=model_dir)
    original = cv2.imread(str(sources[1]))
    found = detector.detect(original)
    if not found:
        pytest.skip("no face in the reference image")

    # Crop tight around the face and enlarge: a selfie held at arm's length.
    x, y, w, h = found[0].bbox
    pad = int(w * 0.08)
    crop = original[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]
    selfie = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    selfie_path = tmp_path / "selfie.jpg"
    cv2.imwrite(str(selfie_path), selfie)

    assert detector.detect(selfie) == [], "the constructed selfie was detectable after all"

    rescued = []
    for ratio in (0.5, 0.35, 0.25, 0.15):
        out = tmp_path / f"r{ratio}.jpg"
        resize_one(selfie_path, out, ratio)
        if detector.detect(cv2.imread(str(out))):
            rescued.append(ratio)
    assert rescued, "no ratio recovered the face; resizing no longer helps"
