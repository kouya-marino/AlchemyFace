"""The Edit DB tab, driven through its real widgets.

The session model is covered display-free in test_edit_data.py. What is left
here is wiring: that loading fills the table, that removal and group edits reach
the model, that detected faces become candidates, and that save and save-as
behave.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.store import PickleStore
from alchemyface.types import Face

pytestmark = pytest.mark.gui


def raw(index: int, dim: int = 128, scale: float = 10.0) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index % dim] = scale
    return vector


def a_face(x: int = 5) -> Face:
    return Face(
        bbox=(x, 5, 20, 20),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        confidence=0.95,
    )


class FakeRecognizer:
    def __init__(self, faces_per_image: int = 1) -> None:
        self.faces_per_image = faces_per_image

    def detect(self, image: np.ndarray) -> list[Face]:
        return [a_face(5 + 25 * i) for i in range(self.faces_per_image)]

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        return raw(face.bbox[0], scale=11.0)


def write_image(path: Path) -> Path:
    import cv2

    rng = np.random.default_rng(seed=abs(hash(path.name)) % 2**32)
    cv2.imwrite(str(path), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
    return path


@pytest.fixture()
def database(tmp_path: Path) -> Path:
    store = PickleStore()
    for index, (name, group) in enumerate([("ada", "ceo"), ("grace", "staff"), ("linus", "staff")]):
        store.add(name, raw(index, scale=10.0 + index), {"group": group})
    path = tmp_path / "db.pkl"
    store.save(path)
    return path


@pytest.fixture()
def edit(app, database: Path):  # type: ignore[no-untyped-def]
    app.set_recognizer(FakeRecognizer())
    view = app.edit_view
    assert view.load(database) is True
    return view


# ------------------------------------------------------------------- the tab


def test_the_edit_tab_exists(app) -> None:  # type: ignore[no-untyped-def]
    assert "Edit DB" in app.tab_labels()


def test_loading_fills_the_table(edit) -> None:  # type: ignore[no-untyped-def]
    assert edit.row_count() == 3
    assert [r.name for r in edit.session.rows()] == ["ada", "grace", "linus"]


def test_loading_a_missing_file_reports(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    assert app.edit_view.load(tmp_path / "absent.pkl") is False
    assert app.reporter.errors
    assert app.edit_view.row_count() == 0


def test_loading_a_malformed_file_reports(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    assert app.edit_view.load(bad) is False
    assert "pickle" in app.reporter.last_error.lower()


def test_the_frame_title_marks_unsaved_changes(edit) -> None:  # type: ignore[no-untyped-def]
    assert "*" not in edit.frame_title
    edit.remove_selected([0])
    assert edit.frame_title.endswith("*")


# ------------------------------------------------------------------ removal


def test_removing_selected_rows(edit) -> None:  # type: ignore[no-untyped-def]
    edit.remove_selected([1])
    assert edit.row_count() == 2
    assert [r.name for r in edit.session.rows()] == ["ada", "linus"]


def test_removing_nothing_selected_is_harmless(edit) -> None:  # type: ignore[no-untyped-def]
    edit.remove_selected([])
    assert edit.row_count() == 3
    assert edit.session.dirty is False


# ------------------------------------------------------------ group editing


def test_editing_a_group(app, edit) -> None:  # type: ignore[no-untyped-def]
    edit.set_group(0, "vip")
    assert edit.session.entries[0].group == "vip"
    assert "vip" in app.group_presets


def test_the_table_redraws_after_a_group_edit(edit) -> None:  # type: ignore[no-untyped-def]
    edit.set_group(0, "vip")
    assert "vip" in edit.table_values()[0]


# ---------------------------------------------------------------- additions


def test_processing_an_image_yields_candidates(edit, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    image = write_image(tmp_path / "newbie.jpg")
    assert edit.process_image(image) == 1
    assert [p.name for p in edit.session.pending] == ["newbie"]
    # Candidates are not entries until merged.
    assert edit.row_count() == 3


def test_processing_a_folder_yields_one_candidate_per_face(app, edit, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    app.set_recognizer(FakeRecognizer(faces_per_image=2))
    folder = tmp_path / "photos"
    folder.mkdir()
    for name in ("a.jpg", "b.jpg"):
        write_image(folder / name)
    assert edit.process_folder(folder) == 4
    assert len(edit.session.pending) == 4


def test_candidates_get_suffixed_names_when_an_image_has_several(app, edit, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    app.set_recognizer(FakeRecognizer(faces_per_image=2))
    image = write_image(tmp_path / "pair.jpg")
    edit.process_image(image)
    assert [p.name for p in edit.session.pending] == ["pair_face1", "pair_face2"]


def test_candidates_are_embedded_on_arrival(edit, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    edit.process_image(write_image(tmp_path / "newbie.jpg"))
    assert edit.session.pending[0].embedding is not None


def test_adding_checked_candidates_merges_them(edit, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    edit.process_image(write_image(tmp_path / "newbie.jpg"))
    assert edit.add_checked() == 1
    assert edit.row_count() == 4
    assert edit.session.rows()[-1].name == "newbie"


def test_unticked_candidates_are_not_merged(edit, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    edit.process_image(write_image(tmp_path / "newbie.jpg"))
    edit.set_pending_include(0, False)
    assert edit.add_checked() == 0
    assert edit.row_count() == 3


def test_a_candidate_with_no_name_is_refused(edit, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    edit.process_image(write_image(tmp_path / "newbie.jpg"))
    edit.set_pending_name(0, "   ")
    assert edit.add_checked() == 0
    assert "name" in app_error(edit).lower()
    assert edit.row_count() == 3


def app_error(view) -> str:  # type: ignore[no-untyped-def]
    return view.reporter.last_error


def test_processing_an_image_when_the_model_cannot_load_reports(app, edit, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The tab loads models itself, so "no recognizer" means loading failed."""
    monkeypatch.setattr(app, "ensure_recognizer", lambda: None)
    monkeypatch.setattr(edit, "_get_recognizer", lambda: None)
    assert edit.process_image(write_image(tmp_path / "x.jpg")) == 0
    assert app.reporter.errors


def test_processing_an_unreadable_image_reports(edit, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    assert edit.process_image(broken) == 0
    assert edit.reporter.errors


# ------------------------------------------------------------------- saving


def test_save_writes_over_the_loaded_file(edit, database: Path) -> None:  # type: ignore[no-untyped-def]
    edit.set_group(0, "vip")
    assert edit.save() is True
    check = PickleStore()
    check.load(database)
    assert check.entries()[0].group == "vip"
    assert edit.session.dirty is False


def test_save_as_writes_elsewhere_and_follows_the_path(edit, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "elsewhere.pkl"
    assert edit.save(out) is True
    assert out.exists()
    assert edit.session.path == out


def test_saving_preserves_raw_magnitudes(edit, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "out.pkl"
    edit.save(out)
    check = PickleStore()
    check.load(out)
    norms = [round(float(np.linalg.norm(e.vector)), 1) for e in check.entries()]
    assert norms == [10.0, 11.0, 12.0]


def test_saving_with_nothing_loaded_reports(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    assert app.edit_view.save() is False
    assert app.reporter.errors


# ------------------------------------------------------- close confirmation


def test_closing_with_unsaved_changes_asks(app, edit) -> None:  # type: ignore[no-untyped-def]
    edit.remove_selected([0])
    assert app.has_unsaved_changes() is True


def test_closing_clean_does_not_ask(app, edit) -> None:  # type: ignore[no-untyped-def]
    assert app.has_unsaved_changes() is False


@pytest.mark.models
def test_the_edit_tab_loads_models_on_its_own(app, database: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Edit must not depend on the Build tab having been used first.

    The two tabs are handed different providers: Build gets a plain getter
    because it calls from a worker thread, Edit gets the loading one because it
    works on the main thread. Wiring Edit to the getter left it unable to detect
    anything at all until Build had run, which no fake-recognizer test could
    show.
    """
    import cv2

    assert app.recognizer is None  # nothing loaded yet
    app.edit_view.load(database)

    real = Path("_local/test_images")
    if not real.is_dir():
        pytest.skip("test images not present")
    found = app.edit_view.process_folder(real)
    assert app.recognizer is not None, "the tab did not load a model"
    assert found > 0, f"no faces detected in {real}"
    assert all(p.embedding is not None for p in app.edit_view.session.pending)
    del cv2
