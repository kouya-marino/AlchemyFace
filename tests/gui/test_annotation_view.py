"""The Build tab, driven through its real widgets.

The heavy lifting is already covered display-free: the geometry and status logic
in test_annotation_data.py, the threading in test_detect_worker.py. What is left
here is the wiring — that a folder loads, that edits reach the model, that the
save flow validates, and that navigation and re-detect behave.

A fake recognizer stands in for the real one, so nothing downloads a model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.gui.annotation_data import EntryStatus
from alchemyface.types import Face

pytestmark = pytest.mark.gui


def write_image(path: Path, width: int = 80, height: int = 60) -> Path:
    import cv2

    rng = np.random.default_rng(seed=abs(hash(path.name)) % 2**32)
    cv2.imwrite(str(path), rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    return path


def a_face(x: int = 5, y: int = 5, w: int = 20, h: int = 20) -> Face:
    return Face(
        bbox=(x, y, w, h),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        confidence=0.95,
    )


class FakeRecognizer:
    """Detects a fixed number of faces and embeds them deterministically."""

    def __init__(self, faces_per_image: int = 1) -> None:
        self.faces_per_image = faces_per_image
        self.detect_calls = 0
        self.embed_calls = 0

    def detect(self, image: np.ndarray) -> list[Face]:
        self.detect_calls += 1
        return [a_face(x=5 + 25 * i) for i in range(self.faces_per_image)]

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        self.embed_calls += 1
        vector = np.zeros(128, dtype=np.float32)
        vector[face.bbox[0] % 128] = 10.0
        return vector


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    d = tmp_path / "photos"
    d.mkdir()
    for name in ("ada.jpg", "grace.jpg", "linus.jpg"):
        write_image(d / name)
    return d


@pytest.fixture()
def build(app, folder: Path):  # type: ignore[no-untyped-def]
    """Load a folder into the Build tab with a fake recognizer, and settle."""

    def _build(faces_per_image: int = 1):  # type: ignore[no-untyped-def]
        fake = FakeRecognizer(faces_per_image)
        app.set_recognizer(fake)
        view = app.annotation_view
        view.load_folder(folder)
        view.wait_until_settled(timeout=10.0)
        return view, fake

    return _build


# ------------------------------------------------------------------ loading


def test_the_build_tab_exists(app) -> None:  # type: ignore[no-untyped-def]
    assert "Build DB" in app.tab_labels()


def test_loading_a_folder_lists_every_image(build) -> None:  # type: ignore[no-untyped-def]
    view, _ = build()
    assert view.entry_count == 3
    assert view.sidebar_labels()[0].endswith("(1/1)")


def test_every_image_gets_detected_in_the_background(build) -> None:  # type: ignore[no-untyped-def]
    view, fake = build()
    assert fake.detect_calls == 3
    assert all(e.status is EntryStatus.INCLUDED for e in view.entries)


def test_faces_are_named_from_the_filename(build) -> None:  # type: ignore[no-untyped-def]
    view, _ = build()
    names = sorted(f.name for e in view.entries for f in e.faces)
    assert names == ["ada", "grace", "linus"]


def test_multiple_faces_get_suffixed_names(build) -> None:  # type: ignore[no-untyped-def]
    view, _ = build(faces_per_image=2)
    first = view.entries[0]
    assert [f.name for f in first.faces] == [
        f"{first.path.stem}_face1",
        f"{first.path.stem}_face2",
    ]


def test_an_empty_folder_reports_rather_than_crashing(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    app.set_recognizer(FakeRecognizer())
    empty = tmp_path / "empty"
    empty.mkdir()
    app.annotation_view.load_folder(empty)
    assert app.annotation_view.entry_count == 0
    assert "no images" in app.status.lower()


# --------------------------------------------------------------- navigation


def test_navigation_moves_the_selection(build) -> None:  # type: ignore[no-untyped-def]
    view, _ = build()
    assert view.current_index == 0
    view.next_image()
    assert view.current_index == 1
    view.previous_image()
    assert view.current_index == 0


def test_navigation_stops_at_the_ends(build) -> None:  # type: ignore[no-untyped-def]
    view, _ = build()
    view.previous_image()
    assert view.current_index == 0
    for _ in range(10):
        view.next_image()
    assert view.current_index == view.entry_count - 1


# -------------------------------------------------------------- annotation


def test_unticking_include_updates_the_model_and_sidebar(build) -> None:  # type: ignore[no-untyped-def]
    view, _ = build()
    view.set_include(0, False)
    assert view.entries[0].faces[0].include is False
    assert view.entries[0].status is EntryStatus.NONE_INCLUDED
    assert view.sidebar_labels()[0].startswith("○")


def test_renaming_a_face_reaches_the_model(build) -> None:  # type: ignore[no-untyped-def]
    view, _ = build()
    view.set_name(0, "Ada Lovelace")
    assert view.entries[0].faces[0].name == "Ada Lovelace"
    assert view.entries[0].edited is True


def test_a_new_group_is_offered_as_a_preset(app, build) -> None:  # type: ignore[no-untyped-def]
    view, _ = build()
    view.set_group(0, "visitor")
    assert view.entries[0].faces[0].group == "visitor"
    assert "visitor" in app.group_presets


def test_selecting_a_face_by_clicking_its_box(build) -> None:  # type: ignore[no-untyped-def]
    view, _ = build(faces_per_image=2)
    # The second face sits at x=30 in image coordinates.
    assert view.select_face_at_image_point(35, 10) == 1
    assert view.selected_face == 1
    assert view.select_face_at_image_point(1000, 1000) is None


# ------------------------------------------------------------- re-detection


def test_redetect_runs_the_detector_again(build) -> None:  # type: ignore[no-untyped-def]
    view, fake = build()
    before = fake.detect_calls
    view.redetect(confirm=lambda: True)
    view.wait_until_settled(timeout=10.0)
    assert fake.detect_calls == before + 1


def test_redetect_asks_before_discarding_edits(build) -> None:  # type: ignore[no-untyped-def]
    view, fake = build()
    view.set_name(0, "edited")
    before = fake.detect_calls
    view.redetect(confirm=lambda: False)  # user says no
    assert fake.detect_calls == before
    assert view.entries[0].faces[0].name == "edited"


def test_redetect_does_not_ask_when_nothing_was_edited(build) -> None:  # type: ignore[no-untyped-def]
    view, fake = build()
    asked = []
    view.redetect(confirm=lambda: asked.append(True) or True)
    view.wait_until_settled(timeout=10.0)
    assert asked == []


# ------------------------------------------------------------------ saving


def test_save_writes_a_pkl_the_robot_can_read(app, build, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from alchemyface.store import PickleStore

    view, _ = build()
    out = tmp_path / "face_db.pkl"
    assert app.save_database(out) is True

    store = PickleStore()
    store.load(out)
    assert len(store) == 3
    assert sorted(e.label for e in store.entries()) == ["ada", "grace", "linus"]
    # Raw, unnormalised — what the robot's schema expects.
    assert float(np.linalg.norm(store.entries()[0].vector)) > 2.0


def test_save_ids_are_renumbered_from_zero(app, build, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import pickle

    view, _ = build()
    out = tmp_path / "face_db.pkl"
    app.save_database(out)
    with open(out, "rb") as handle:
        data = pickle.load(handle)
    assert [row[0] for row in data] == ["0", "1", "2"]


def test_save_computes_embeddings_only_when_needed(app, build, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    view, fake = build()
    app.save_database(tmp_path / "a.pkl")
    after_first = fake.embed_calls
    app.save_database(tmp_path / "b.pkl")
    assert fake.embed_calls == after_first, "embeddings were recomputed"


def test_save_refuses_when_a_face_has_no_name(app, build, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    view, _ = build()
    view.set_name(0, "   ")
    assert app.save_database(tmp_path / "out.pkl") is False
    assert "no name" in app.reporter.last_error
    assert not (tmp_path / "out.pkl").exists()


def test_save_refuses_when_everything_is_excluded(app, build, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    view, _ = build()
    for i in range(view.entry_count):
        view.select_index(i)
        view.set_include(0, False)
    assert app.save_database(tmp_path / "out.pkl") is False
    assert "nothing to include" in app.reporter.last_error.lower()


def test_excluded_faces_are_not_written(app, build, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from alchemyface.store import PickleStore

    view, _ = build()
    view.select_index(0)
    view.set_include(0, False)
    out = tmp_path / "out.pkl"
    assert app.save_database(out) is True
    store = PickleStore()
    store.load(out)
    assert len(store) == 2


# ------------------------------------------------------- model invalidation


def test_changing_the_recognizer_drops_cached_embeddings(app, build, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    view, fake = build()
    app.save_database(tmp_path / "a.pkl")  # fills the cache
    assert any(f.embedding is not None for e in view.entries for f in e.faces)
    app.set_recognizer(FakeRecognizer())  # a different model
    assert all(f.embedding is None for e in view.entries for f in e.faces)


# ------------------------------------------------------------------ shutdown


def test_closing_the_window_stops_the_worker(app, build) -> None:  # type: ignore[no-untyped-def]
    view, _ = build()
    assert view.worker_is_running
    app.close()
    assert not view.worker_is_running


def test_load_folder_refuses_without_a_model(app, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Starting with no model would fill the sidebar with fake "no face" rows.

    That is exactly how a false claim got into this project's changelog: every
    image failed, and each failure was displayed as a finding about the photo.
    """
    import cv2

    folder = tmp_path / "photos"
    folder.mkdir()
    cv2.imwrite(str(folder / "a.jpg"), np.zeros((40, 40, 3), dtype=np.uint8))
    app.set_recognizer(None)
    # Patch the view's own reference: it captured the bound method at
    # construction, so replacing the attribute on `app` would not reach it.
    monkeypatch.setattr(app.annotation_view, "_ensure_recognizer", lambda: None)
    assert app.annotation_view.load_folder(folder) is False
    assert app.annotation_view.entry_count == 0
    assert "no model" in app.status.lower()


def test_face_card_variables_do_not_accumulate(app, build) -> None:  # type: ignore[no-untyped-def]
    """Re-rendering the face panel must not leak Tcl variables.

    A trace registers a variable inside Tk, so destroying the widget does not
    free it. `_release_face_vars` existed to detach them — but nothing ever
    registered a variable with it, so it looped zero times and the list was
    always empty. An audit measured 480 leaked variables over 80 re-renders.
    """
    view, _ = build(faces_per_image=2)

    def tcl_vars() -> int:
        return len([n for n in app.tk.call("info", "vars") if str(n).startswith("PY_VAR")])

    before = tcl_vars()
    for _ in range(30):
        view.next_image()
        view.previous_image()
    growth = tcl_vars() - before
    assert growth <= 12, f"leaked {growth} Tcl variables over 60 re-renders"


def test_the_build_tab_reports_its_own_unsaved_work(app, build) -> None:  # type: ignore[no-untyped-def]
    """Closing must not silently discard annotations.

    has_unsaved_changes consulted only the Edit tab, so a folder's worth of
    typed names, groups and include ticks vanished without a prompt.
    """
    view, _ = build()
    assert app.has_unsaved_changes() is False
    view.set_name(0, "somebody-i-typed")
    assert view.entries[view.current_index].edited is True
    assert app.has_unsaved_changes() is True


def test_closing_asks_before_discarding_build_work(app, build) -> None:  # type: ignore[no-untyped-def]
    view, _ = build()
    view.set_include(0, False)
    asked: list[bool] = []
    app.close(confirm=lambda: asked.append(True) or False)  # user declines
    assert asked, "the confirm callback was never consulted"
    assert app.closed is False, "the window closed despite the user declining"


def test_the_detection_score_reaches_the_live_detector(app, build) -> None:  # type: ignore[no-untyped-def]
    """The spinbox was ticked as done in todo.md while no such control existed."""

    class Recording:
        def __init__(self) -> None:
            self.applied: list[float] = []

        def set_score_threshold(self, value: float) -> None:
            self.applied.append(value)

    view, fake = build()
    fake.detector = Recording()  # type: ignore[attr-defined]
    app._score_threshold_var.set(0.42)
    assert app.apply_score_threshold() == pytest.approx(0.42)
    assert fake.detector.applied == [pytest.approx(0.42)]  # type: ignore[attr-defined]


def test_the_detection_score_is_clamped(app) -> None:  # type: ignore[no-untyped-def]
    for raw, expected in ((5.0, 0.99), (-1.0, 0.10), (0.55, 0.55)):
        app._score_threshold_var.set(raw)
        assert app.apply_score_threshold() == pytest.approx(expected)
