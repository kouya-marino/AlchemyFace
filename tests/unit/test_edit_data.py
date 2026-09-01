"""The Edit DB tab's session model — no Tk, no images, no models.

The original kept this inside a 731-line widget, with the dirty flag, the
pending-additions list and the merge logic tangled into callbacks. Extracted, it
is a small state machine with ordinary tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.gui.edit_data import EditSession, PendingFace
from alchemyface.store import PickleStore
from alchemyface.types import Face


def raw(index: int, dim: int = 4, scale: float = 10.0) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index % dim] = scale
    return vector


def a_face() -> Face:
    return Face(
        bbox=(0, 0, 10, 10),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        confidence=0.9,
    )


def saved_db(path: Path, people: list[tuple[str, str]]) -> PickleStore:
    store = PickleStore(dim=4)
    for index, (name, group) in enumerate(people):
        store.add(name, raw(index, scale=10.0 + index), {"group": group})
    store.save(path)
    return store


@pytest.fixture()
def session(tmp_path: Path) -> EditSession:
    path = tmp_path / "db.pkl"
    saved_db(path, [("ada", "ceo"), ("grace", "staff"), ("linus", "staff")])
    s = EditSession()
    s.load(path)
    return s


def pending(name: str = "new", include: bool = True) -> PendingFace:
    return PendingFace(
        source=Path("photo.jpg"),
        face=a_face(),
        include=include,
        name=name,
        embedding=raw(3, scale=9.0),
    )


# ------------------------------------------------------------------ loading


def test_a_new_session_is_empty_and_clean() -> None:
    s = EditSession()
    assert len(s.entries) == 0
    assert s.dirty is False
    assert s.path is None


def test_loading_reads_every_entry(session: EditSession) -> None:
    assert [e.name for e in session.entries] == ["ada", "grace", "linus"]
    assert [e.group for e in session.entries] == ["ceo", "staff", "staff"]


def test_loading_records_the_path_and_stays_clean(session: EditSession, tmp_path: Path) -> None:
    assert session.path == tmp_path / "db.pkl"
    assert session.dirty is False


def test_loading_replaces_what_was_there(session: EditSession, tmp_path: Path) -> None:
    other = tmp_path / "other.pkl"
    saved_db(other, [("solo", "vip")])
    session.load(other)
    assert [e.name for e in session.entries] == ["solo"]
    assert session.dirty is False


def test_loading_discards_pending_additions(session: EditSession, tmp_path: Path) -> None:
    session.add_pending([pending()])
    other = tmp_path / "other.pkl"
    saved_db(other, [("solo", "vip")])
    session.load(other)
    assert session.pending == []


def test_loading_preserves_the_raw_vectors(session: EditSession) -> None:
    # The whole point of PickleStore: a round trip must not alter a value.
    norms = [float(np.linalg.norm(e.vector)) for e in session.entries]
    assert [round(n, 1) for n in norms] == [10.0, 11.0, 12.0]


# ------------------------------------------------------------------ removal


def test_removing_one_entry(session: EditSession) -> None:
    assert session.remove([1]) == 1
    assert [e.name for e in session.entries] == ["ada", "linus"]
    assert session.dirty is True


def test_removing_several_at_once(session: EditSession) -> None:
    assert session.remove([0, 2]) == 2
    assert [e.name for e in session.entries] == ["grace"]


def test_removal_order_does_not_matter(session: EditSession) -> None:
    # Deleting by ascending index would shift the later ones out from under it.
    assert session.remove([2, 0]) == 2
    assert [e.name for e in session.entries] == ["grace"]


def test_removing_nothing_leaves_it_clean(session: EditSession) -> None:
    assert session.remove([]) == 0
    assert session.dirty is False


def test_out_of_range_indices_are_ignored(session: EditSession) -> None:
    assert session.remove([99, 1]) == 1
    assert len(session.entries) == 2


# ------------------------------------------------------------ group editing


def test_changing_a_group(session: EditSession) -> None:
    session.set_group(0, "vip")
    assert session.entries[0].group == "vip"
    assert session.dirty is True


def test_setting_the_same_group_is_not_a_change(session: EditSession) -> None:
    session.set_group(0, "ceo")
    assert session.dirty is False


def test_group_is_stripped(session: EditSession) -> None:
    session.set_group(0, "  vip  ")
    assert session.entries[0].group == "vip"


def test_a_group_can_be_cleared(session: EditSession) -> None:
    session.set_group(0, "")
    assert session.entries[0].group == ""
    assert session.dirty is True


def test_setting_a_group_out_of_range_is_ignored(session: EditSession) -> None:
    session.set_group(99, "vip")
    assert session.dirty is False


# --------------------------------------------------------------- additions


def test_pending_faces_are_held_separately(session: EditSession) -> None:
    session.add_pending([pending("newbie")])
    assert [p.name for p in session.pending] == ["newbie"]
    # Not merged yet, so the entries table is untouched.
    assert len(session.entries) == 3


def test_adding_pending_faces_does_not_dirty_the_session(session: EditSession) -> None:
    # Nothing has changed about the database until they are merged.
    session.add_pending([pending()])
    assert session.dirty is False


def test_merging_appends_only_the_checked_ones(session: EditSession) -> None:
    session.add_pending([pending("yes"), pending("no", include=False)])
    assert session.merge_checked() == 1
    assert [e.name for e in session.entries][-1] == "yes"
    assert session.dirty is True


def test_merging_clears_the_pending_list(session: EditSession) -> None:
    session.add_pending([pending("yes"), pending("no", include=False)])
    session.merge_checked()
    assert session.pending == []


def test_merging_nothing_checked_changes_nothing(session: EditSession) -> None:
    session.add_pending([pending("no", include=False)])
    assert session.merge_checked() == 0
    assert len(session.entries) == 3
    assert session.dirty is False


def test_merging_refuses_a_face_with_no_name(session: EditSession) -> None:
    session.add_pending([pending("   ")])
    with pytest.raises(ValueError, match="name"):
        session.merge_checked()
    assert len(session.entries) == 3


def test_merging_refuses_a_face_with_no_embedding(session: EditSession) -> None:
    face = pending("someone")
    face.embedding = None
    session.add_pending([face])
    with pytest.raises(ValueError, match="embedding"):
        session.merge_checked()


def test_duplicate_names_are_allowed(session: EditSession) -> None:
    # The robot's matcher resolves by best cosine similarity, so two entries for
    # one person are useful rather than a mistake.
    session.add_pending([pending("ada")])
    assert session.merge_checked() == 1
    assert [e.name for e in session.entries].count("ada") == 2


def test_pending_can_be_discarded(session: EditSession) -> None:
    session.add_pending([pending()])
    session.clear_pending()
    assert session.pending == []
    assert session.dirty is False


# ------------------------------------------------------------------ saving


def test_saving_writes_the_current_state(session: EditSession, tmp_path: Path) -> None:
    session.remove([1])
    out = tmp_path / "out.pkl"
    session.save(out)
    check = PickleStore()
    check.load(out)
    assert [e.label for e in check.entries()] == ["ada", "linus"]


def test_saving_clears_the_dirty_flag_and_records_the_path(session: EditSession, tmp_path: Path) -> None:
    session.remove([1])
    out = tmp_path / "out.pkl"
    session.save(out)
    assert session.dirty is False
    assert session.path == out


def test_saving_over_the_loaded_path(session: EditSession, tmp_path: Path) -> None:
    session.set_group(0, "vip")
    session.save()  # no argument means the loaded path
    reloaded = EditSession()
    reloaded.load(tmp_path / "db.pkl")
    assert reloaded.entries[0].group == "vip"


def test_saving_with_no_path_at_all_raises() -> None:
    with pytest.raises(ValueError, match="path"):
        EditSession().save()


def test_saving_renumbers_ids_from_zero(session: EditSession, tmp_path: Path) -> None:
    import pickle

    session.remove([0])
    out = tmp_path / "out.pkl"
    session.save(out)
    with open(out, "rb") as handle:
        data = pickle.load(handle)
    assert [row[0] for row in data] == ["0", "1"]


def test_saving_preserves_raw_magnitudes(session: EditSession, tmp_path: Path) -> None:
    out = tmp_path / "out.pkl"
    session.save(out)
    check = PickleStore()
    check.load(out)
    norms = [round(float(np.linalg.norm(e.vector)), 1) for e in check.entries()]
    assert norms == [10.0, 11.0, 12.0]


# ------------------------------------------------------------------- rows


def test_rows_are_numbered_for_display(session: EditSession) -> None:
    assert [r.index for r in session.rows()] == [1, 2, 3]


def test_rows_carry_dimension_and_norm(session: EditSession) -> None:
    rows = session.rows()
    assert all(r.dim == 4 for r in rows)
    assert [round(r.norm, 1) for r in rows] == [10.0, 11.0, 12.0]


def test_rows_show_an_em_dash_for_a_blank_group(session: EditSession) -> None:
    session.set_group(0, "")
    assert session.rows()[0].group == "—"


def test_rows_reflect_removals(session: EditSession) -> None:
    session.remove([0])
    assert [r.name for r in session.rows()] == ["grace", "linus"]
    assert [r.index for r in session.rows()] == [1, 2]


# ------------------------------------------------------------------ title


def test_title_marks_unsaved_changes(session: EditSession) -> None:
    assert not session.title_suffix
    session.remove([0])
    assert session.title_suffix == " *"


def test_groups_in_use_are_offered_as_presets(session: EditSession) -> None:
    assert sorted(session.groups_in_use()) == ["ceo", "staff"]


def test_dim_follows_what_is_loaded(session: EditSession) -> None:
    assert session.dim == 4


def test_dim_defaults_to_sface_when_empty() -> None:
    assert EditSession().dim == 128


def test_a_non_128_database_can_be_saved(session: EditSession, tmp_path: Path) -> None:
    """The store validates dimensions, so save() must not assume 128.

    Every real database is 128, so building the store at the default would have
    worked by luck and failed on anything else.
    """
    out = tmp_path / "four.pkl"
    session.save(out)
    check = PickleStore()
    check.load(out)
    assert check.dim == 4
    assert len(check) == 3
