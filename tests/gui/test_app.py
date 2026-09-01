"""The window and the Inspect tab, driven through their real widgets.

These assert on model state and on what the widgets report — never on pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.store import PickleStore

pytestmark = pytest.mark.gui


def sample_db(path: Path, count: int = 3) -> PickleStore:
    store = PickleStore(dim=4)
    for i in range(count):
        vector = np.zeros(4, dtype=np.float32)
        vector[i % 4] = 10.0 + i
        store.add(f"person{i}", vector, {"group": "staff" if i else "ceo"})
    store.save(path)
    return store


# ------------------------------------------------------------------ shell


def test_window_opens_with_a_title_and_the_version(app) -> None:  # type: ignore[no-untyped-def]
    from alchemyface import __version__

    assert "Face DB Builder" in app.title()
    assert __version__ in app.title()


def test_all_four_tabs_are_present(app) -> None:  # type: ignore[no-untyped-def]
    # The same four the application was ported from, in the same order. Tabs
    # arrived one version at a time and none was ever a dead placeholder.
    assert app.tab_labels() == ["Build DB", "Edit DB", "Resize", "Inspect DB"]


def test_status_starts_ready_and_can_be_set(app) -> None:  # type: ignore[no-untyped-def]
    assert app.status == "Ready."
    app.set_status("something happened")
    assert app.status == "something happened"


def test_close_is_idempotent(app) -> None:  # type: ignore[no-untyped-def]
    # tk.Tk.destroy() raises on an already-destroyed window, so a handler
    # firing twice would crash the app on exit.
    app.close()
    app.close()
    app.close()
    assert app.closed is True


# --------------------------------------------------------------- inspect


def test_loading_a_database_fills_the_table(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "db.pkl"
    sample_db(path, count=3)
    assert app.inspect_view.load(path) is True
    assert app.inspect_view.row_count() == 3
    assert app.inspect_view.store is not None
    assert len(app.inspect_view.store) == 3


def test_summary_line_reflects_the_file(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "db.pkl"
    sample_db(path, count=3)
    app.inspect_view.load(path)
    text = app.inspect_view.summary_text
    assert "3 entries" in text
    assert "dim=4" in text
    assert str(path) in text


def test_status_bar_is_updated_by_the_tab(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "db.pkl"
    sample_db(path, count=2)
    app.inspect_view.load(path)
    assert "Loaded 2 entries" in app.status


def test_loading_an_empty_database_is_not_an_error(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "empty.pkl"
    PickleStore(dim=4).save(path)
    assert app.inspect_view.load(path) is True
    assert app.inspect_view.row_count() == 0
    assert "0 entries" in app.inspect_view.summary_text


def test_a_second_load_replaces_the_first(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    first, second = tmp_path / "a.pkl", tmp_path / "b.pkl"
    sample_db(first, count=3)
    sample_db(second, count=1)
    app.inspect_view.load(first)
    app.inspect_view.load(second)
    assert app.inspect_view.row_count() == 1


def test_missing_file_reports_and_clears(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    good = tmp_path / "db.pkl"
    sample_db(good, count=2)
    app.inspect_view.load(good)
    assert app.inspect_view.load(tmp_path / "absent.pkl") is False
    assert app.inspect_view.row_count() == 0
    assert app.inspect_view.store is None
    assert "does not exist" in app.inspect_view.summary_text


def test_malformed_pickle_reports_the_reason(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle at all")
    assert app.inspect_view.load(bad) is False
    assert app.reporter.errors, "nothing was reported"
    assert "pickle" in app.reporter.last_error.lower()


def test_a_real_production_database_loads(app, pkl_dir: Path) -> None:  # type: ignore[no-untyped-def]
    """End to end on a real robot database. Counts only — never names."""
    path = pkl_dir / "db_thirty.pkl"
    if not path.is_file():
        pytest.skip("fixture missing")
    assert app.inspect_view.load(path) is True
    assert app.inspect_view.row_count() == 30
    assert "dim=128" in app.inspect_view.summary_text
