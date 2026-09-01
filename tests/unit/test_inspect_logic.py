"""The Inspect tab's data layer, tested with no display at all.

Everything the tab shows is derived by pure functions over a PickleStore, so
the only part that needs Tk is the widget wiring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.gui.inspect_data import DatabaseSummary, entry_rows, summarise
from alchemyface.store import PickleStore


def raw(i: int, dim: int = 4, scale: float = 10.0) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = scale
    return v


def populated() -> PickleStore:
    store = PickleStore(dim=4)
    store.add("ada", raw(0, scale=11.0), {"group": "ceo"})
    store.add("grace", raw(1, scale=12.0), {"group": "staff"})
    store.add("ada", raw(2, scale=13.0), {"group": "staff"})
    return store


# ------------------------------------------------------------- entries()


def test_store_exposes_its_entries_in_order() -> None:
    store = populated()
    rows = store.entries()
    assert [e.label for e in rows] == ["ada", "grace", "ada"]
    assert [e.group for e in rows] == ["ceo", "staff", "staff"]


def test_entries_hands_out_vector_copies() -> None:
    store = populated()
    got = store.entries()[0].vector
    got[0] = 999.0
    assert store.entries()[0].vector[0] == pytest.approx(11.0)


# ---------------------------------------------------------- entry_rows()


def test_entry_rows_number_from_one_for_display() -> None:
    rows = entry_rows(populated())
    assert [r.index for r in rows] == [1, 2, 3]


def test_entry_rows_report_the_raw_l2_norm() -> None:
    # The whole reason normalize=False exists: this column stays informative.
    rows = entry_rows(populated())
    assert [round(r.norm, 1) for r in rows] == [11.0, 12.0, 13.0]


def test_entry_rows_carry_dimension_and_ids() -> None:
    rows = entry_rows(populated())
    assert all(r.dim == 4 for r in rows)
    assert len({r.entry_id for r in rows}) == 3


def test_entry_rows_preview_is_truncated_and_signed() -> None:
    rows = entry_rows(populated())
    preview = rows[0].preview
    assert preview.startswith("[")
    assert "+11.0000" in preview
    assert preview.endswith("]")


def test_entry_rows_preview_marks_elision_when_longer_than_five() -> None:
    store = PickleStore(dim=8)
    store.add("ada", np.arange(1, 9, dtype=np.float32))
    (row,) = entry_rows(store)
    assert "…" in row.preview


def test_entry_rows_shows_an_em_dash_for_a_blank_group() -> None:
    store = PickleStore(dim=4)
    store.add("ada", raw(0))
    (row,) = entry_rows(store)
    assert row.group == "—"


def test_entry_rows_on_an_empty_store_is_empty() -> None:
    assert entry_rows(PickleStore(dim=4)) == []


# ----------------------------------------------------------- summarise()


def test_summarise_counts_entries_names_and_groups() -> None:
    s = summarise(populated(), None)
    assert s.entries == 3
    assert s.unique_names == 2  # ada appears twice
    assert s.groups == 2  # ceo, staff
    assert s.dim == 4


def test_summarise_ignores_blank_groups_in_the_count() -> None:
    store = PickleStore(dim=4)
    store.add("ada", raw(0))
    store.add("grace", raw(1), {"group": "staff"})
    assert summarise(store, None).groups == 1


def test_summarise_reads_file_size_when_a_path_is_given(tmp_path: Path) -> None:
    store = populated()
    p = tmp_path / "db.pkl"
    store.save(p)
    s = summarise(store, p)
    assert s.size_bytes == p.stat().st_size
    assert s.path == p


def test_summarise_tolerates_a_missing_file(tmp_path: Path) -> None:
    s = summarise(populated(), tmp_path / "gone.pkl")
    assert s.size_bytes == 0


def test_summarise_of_an_empty_store() -> None:
    s = summarise(PickleStore(dim=4), None)
    assert (s.entries, s.unique_names, s.groups) == (0, 0, 0)
    assert "0 entries" in str(s)


def test_summary_text_is_human_readable(tmp_path: Path) -> None:
    p = tmp_path / "db.pkl"
    populated().save(p)
    text = str(summarise(populated(), p))
    for fragment in ("3 entries", "2 unique names", "2 groups", "dim=4", "KB"):
        assert fragment in text, text


def test_summary_is_a_frozen_value_type() -> None:
    s = summarise(populated(), None)
    assert isinstance(s, DatabaseSummary)
    with pytest.raises(AttributeError):
        s.entries = 99  # type: ignore[misc]
