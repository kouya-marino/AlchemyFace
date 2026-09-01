"""The Build tab's data layer — no Tk, no images, no models.

The original app kept all of this inside an 816-line widget, where the canvas
arithmetic and the sidebar state machine could only be exercised by clicking.
Pulled out here, they are ordinary functions with ordinary tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.gui.annotation_data import (
    BgrCache,
    EntryStatus,
    FaceAnnotation,
    ImageEntry,
    default_face_name,
    face_at,
    fit_image,
    records_for_save,
    sidebar_colour,
    sidebar_text,
    validate_for_save,
)
from alchemyface.types import Face


def face(x: int = 10, y: int = 20, w: int = 30, h: int = 40) -> Face:
    return Face(
        bbox=(x, y, w, h),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        confidence=0.9,
    )


def annotated(name: str = "ada", include: bool = True, **kw) -> FaceAnnotation:  # type: ignore[no-untyped-def]
    return FaceAnnotation(face=face(**kw), include=include, name=name, group="staff")


# ------------------------------------------------------- default naming


def test_single_face_takes_the_filename_stem() -> None:
    assert default_face_name("prashant", index=0, total=1) == "prashant"


def test_multiple_faces_are_suffixed_from_one() -> None:
    assert default_face_name("group", index=0, total=3) == "group_face1"
    assert default_face_name("group", index=2, total=3) == "group_face3"


# ------------------------------------------------------- entry status


def test_undetected_entry_is_pending() -> None:
    assert ImageEntry(path=Path("a.jpg")).status is EntryStatus.PENDING


def test_detected_with_no_faces_is_no_face() -> None:
    entry = ImageEntry(path=Path("a.jpg"), detected=True)
    assert entry.status is EntryStatus.NO_FACE


def test_all_faces_excluded_is_none_included() -> None:
    entry = ImageEntry(path=Path("a.jpg"), detected=True, faces=[annotated(include=False)])
    assert entry.status is EntryStatus.NONE_INCLUDED


def test_at_least_one_included_is_included() -> None:
    entry = ImageEntry(
        path=Path("a.jpg"),
        detected=True,
        faces=[annotated(include=False), annotated(include=True)],
    )
    assert entry.status is EntryStatus.INCLUDED
    assert entry.included_count == 1


# ------------------------------------------------------- sidebar labels


def test_pending_row_shows_a_dot() -> None:
    assert sidebar_text(ImageEntry(path=Path("photo.jpg"))).startswith("·")
    assert "photo.jpg" in sidebar_text(ImageEntry(path=Path("photo.jpg")))


def test_no_face_row_shows_a_warning() -> None:
    entry = ImageEntry(path=Path("photo.jpg"), detected=True)
    assert sidebar_text(entry).startswith("⚠")


def test_partially_included_row_shows_the_ratio() -> None:
    entry = ImageEntry(
        path=Path("photo.jpg"),
        detected=True,
        faces=[annotated(include=True), annotated(include=False)],
    )
    text = sidebar_text(entry)
    assert text.startswith("✓")
    assert "(1/2)" in text


def test_nothing_included_row_shows_a_hollow_circle() -> None:
    entry = ImageEntry(path=Path("photo.jpg"), detected=True, faces=[annotated(include=False)])
    text = sidebar_text(entry)
    assert text.startswith("○")
    assert "(0/1)" in text


def test_every_status_has_its_own_colour() -> None:
    entries = [
        ImageEntry(path=Path("a.jpg")),
        ImageEntry(path=Path("a.jpg"), detected=True),
        ImageEntry(path=Path("a.jpg"), detected=True, faces=[annotated(include=False)]),
        ImageEntry(path=Path("a.jpg"), detected=True, faces=[annotated(include=True)]),
    ]
    colours = [sidebar_colour(e) for e in entries]
    assert len(set(colours)) == 4, colours
    assert all(c.startswith("#") for c in colours)


# --------------------------------------------------- canvas arithmetic


def test_fit_shrinks_a_large_image_to_the_canvas() -> None:
    t = fit_image(image_w=1000, image_h=500, canvas_w=400, canvas_h=400)
    assert t.scale == pytest.approx(0.4)
    assert (t.width, t.height) == (400, 200)


def test_fit_never_enlarges_a_small_image() -> None:
    # Upscaling a small photo to fill the canvas would only blur it.
    t = fit_image(image_w=100, image_h=50, canvas_w=800, canvas_h=800)
    assert t.scale == pytest.approx(1.0)
    assert (t.width, t.height) == (100, 50)


def test_fit_centres_the_image() -> None:
    t = fit_image(image_w=100, image_h=50, canvas_w=300, canvas_h=200)
    assert t.offset_x == 100  # (300 - 100) // 2
    assert t.offset_y == 75  # (200 - 50) // 2


def test_fit_survives_a_zero_sized_canvas() -> None:
    # Tk reports 1x1 before the first layout pass.
    t = fit_image(image_w=100, image_h=100, canvas_w=0, canvas_h=0)
    assert t.scale > 0
    assert t.width >= 1 and t.height >= 1


def test_canvas_and_image_coordinates_round_trip() -> None:
    t = fit_image(image_w=1000, image_h=800, canvas_w=500, canvas_h=500)
    cx, cy = t.to_canvas(200, 400)
    ix, iy = t.to_image(cx, cy)
    assert ix == pytest.approx(200, abs=2)
    assert iy == pytest.approx(400, abs=2)


def test_to_canvas_applies_scale_and_offset() -> None:
    t = fit_image(image_w=200, image_h=200, canvas_w=100, canvas_h=400)
    # scale 0.5, image becomes 100x100, centred vertically -> offset_y 150
    assert t.to_canvas(0, 0) == (0, 150)
    assert t.to_canvas(200, 200) == (100, 250)


# -------------------------------------------------------- hit testing


def test_click_inside_a_box_selects_it() -> None:
    faces = [annotated(x=0, y=0, w=10, h=10), annotated(x=50, y=50, w=10, h=10)]
    assert face_at(faces, 55, 55) == 1


def test_click_outside_every_box_selects_nothing() -> None:
    faces = [annotated(x=0, y=0, w=10, h=10)]
    assert face_at(faces, 500, 500) is None


def test_click_on_a_boundary_counts_as_inside() -> None:
    faces = [annotated(x=0, y=0, w=10, h=10)]
    assert face_at(faces, 10, 10) == 0


def test_overlapping_boxes_pick_the_first() -> None:
    # Deterministic rather than arbitrary: whichever the detector listed first.
    faces = [annotated(x=0, y=0, w=100, h=100), annotated(x=10, y=10, w=20, h=20)]
    assert face_at(faces, 15, 15) == 0


def test_hit_testing_an_empty_list_is_none() -> None:
    assert face_at([], 5, 5) is None


# --------------------------------------------------- save validation


def test_included_named_faces_become_records() -> None:
    entries = [
        ImageEntry(path=Path("a.jpg"), detected=True, faces=[annotated("ada")]),
        ImageEntry(path=Path("b.jpg"), detected=True, faces=[annotated("grace")]),
    ]
    assert [r.name for r in records_for_save(entries)] == ["ada", "grace"]
    assert validate_for_save(entries) == []


def test_excluded_faces_are_left_out() -> None:
    entries = [
        ImageEntry(
            path=Path("a.jpg"),
            detected=True,
            faces=[annotated("ada", include=False), annotated("grace")],
        )
    ]
    assert [r.name for r in records_for_save(entries)] == ["grace"]


def test_undetected_entries_are_skipped_entirely() -> None:
    entries = [ImageEntry(path=Path("a.jpg"), faces=[annotated("ada")])]
    assert records_for_save(entries) == []


def test_a_blank_name_on_an_included_face_is_an_error() -> None:
    entries = [ImageEntry(path=Path("photo.jpg"), detected=True, faces=[annotated("  ")])]
    errors = validate_for_save(entries)
    assert len(errors) == 1
    assert "photo.jpg" in errors[0]
    assert "face #1" in errors[0]


def test_a_blank_name_on_an_excluded_face_is_not_an_error() -> None:
    entries = [
        ImageEntry(
            path=Path("photo.jpg"),
            detected=True,
            faces=[annotated("", include=False)],
        )
    ]
    assert validate_for_save(entries) == []


def test_errors_name_every_offending_face() -> None:
    entries = [
        ImageEntry(
            path=Path("photo.jpg"),
            detected=True,
            faces=[annotated(""), annotated("ok"), annotated("")],
        )
    ]
    errors = validate_for_save(entries)
    assert len(errors) == 2
    assert "face #1" in errors[0] and "face #3" in errors[1]


# -------------------------------------------------------- the BGR cache


def test_cache_returns_what_was_put_in() -> None:
    cache = BgrCache(capacity=2)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    cache.put(Path("a.jpg"), image)
    assert cache.get(Path("a.jpg")) is image


def test_cache_misses_are_none() -> None:
    assert BgrCache(capacity=2).get(Path("nope.jpg")) is None


def test_cache_evicts_the_least_recently_used() -> None:
    cache = BgrCache(capacity=2)
    for name in ("a", "b"):
        cache.put(Path(f"{name}.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))
    cache.get(Path("a.jpg"))  # a is now newest
    cache.put(Path("c.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))
    assert cache.get(Path("b.jpg")) is None  # b was oldest
    assert cache.get(Path("a.jpg")) is not None
    assert cache.get(Path("c.jpg")) is not None


def test_cache_respects_its_capacity() -> None:
    cache = BgrCache(capacity=3)
    for i in range(10):
        cache.put(Path(f"{i}.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))
    assert len(cache) == 3


def test_cache_drop_removes_one_entry() -> None:
    cache = BgrCache(capacity=2)
    cache.put(Path("a.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))
    cache.drop(Path("a.jpg"))
    assert cache.get(Path("a.jpg")) is None


def test_cache_clear_empties_it() -> None:
    cache = BgrCache(capacity=4)
    for i in range(3):
        cache.put(Path(f"{i}.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))
    cache.clear()
    assert len(cache) == 0


# ------------------------------------------- failure is not "no face"


def test_a_failed_entry_has_its_own_status() -> None:
    """A failure must never be reported as a finding about the photograph.

    This project's own changelog once claimed YuNet could not detect a face in
    a particular photo. It could — detection had failed because no model was
    loaded, and the UI showed that as "no face detected".
    """
    entry = ImageEntry(path=Path("a.jpg"), detected=True, error="no model loaded")
    assert entry.status is EntryStatus.FAILED
    assert entry.status is not EntryStatus.NO_FACE


def test_a_failed_row_shows_the_reason() -> None:
    entry = ImageEntry(path=Path("a.jpg"), detected=True, error="could not read the image")
    text = sidebar_text(entry)
    assert text.startswith("✗")
    assert "could not read the image" in text


def test_failed_and_no_face_look_different() -> None:
    failed = ImageEntry(path=Path("a.jpg"), detected=True, error="boom")
    empty = ImageEntry(path=Path("a.jpg"), detected=True)
    assert sidebar_text(failed)[0] != sidebar_text(empty)[0]
    assert sidebar_colour(failed) != sidebar_colour(empty)


def test_a_failure_wins_over_an_empty_face_list() -> None:
    # Both conditions hold at once; the failure is the informative one.
    entry = ImageEntry(path=Path("a.jpg"), detected=True, faces=[], error="boom")
    assert entry.status is EntryStatus.FAILED


def test_every_status_still_has_a_distinct_colour() -> None:
    entries = [
        ImageEntry(path=Path("a.jpg")),
        ImageEntry(path=Path("a.jpg"), detected=True, error="boom"),
        ImageEntry(path=Path("a.jpg"), detected=True),
        ImageEntry(path=Path("a.jpg"), detected=True, faces=[annotated(include=False)]),
        ImageEntry(path=Path("a.jpg"), detected=True, faces=[annotated(include=True)]),
    ]
    colours = [sidebar_colour(e) for e in entries]
    assert len(set(colours)) == 5, colours
