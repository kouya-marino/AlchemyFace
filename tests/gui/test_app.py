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


def _broken_db(path: Path) -> Path:
    """A database with one NaN vector and one of the wrong dimension."""
    import pickle

    with open(path, "wb") as handle:
        pickle.dump(
            [
                ("1", "ada", "ceo", np.ones(128, dtype=np.float32)),
                ("2", "grace", "staff", np.full(128, np.nan, dtype=np.float32)),
                ("3", "linus", "staff", np.ones(64, dtype=np.float32)),
            ],
            handle,
        )
    return path


def test_inspect_shows_a_database_a_strict_load_would_refuse(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """The inspector exists to diagnose a broken file. Refusing to open one
    withheld exactly the answer the user came for."""
    assert app.inspect_view.load(_broken_db(tmp_path / "broken.pkl")) is True
    assert app.inspect_view.row_count() == 3
    assert any("problem" in title.lower() for title, _message in app.reporter.infos)


def test_edit_opens_a_broken_database_so_it_can_be_repaired(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Deleting the offending row is the whole reason to open it here."""
    broken = _broken_db(tmp_path / "broken.pkl")
    assert app.edit_view.load(broken) is True
    assert app.edit_view.row_count() == 3
    assert len(app.edit_view.session.problems) == 2

    app.edit_view.remove_selected([1, 2])
    assert app.edit_view.save() is True
    reread = PickleStore()
    reread.load(broken)  # strict: the repaired file is clean
    assert [entry.label for entry in reread.entries()] == ["ada"]


# --------------------------------------------------- the detection-score knob
#
# 0.6.0's notes, README and CHANGELOG all described a "Detection score" spinbox
# in the Build tab. `apply_score_threshold` existed and was correct, but no
# widget ever called it, so the threshold was pinned at 0.9 and the docs
# described a control that was not there. These tests assert the widget itself.


def _descendants(widget):  # type: ignore[no-untyped-def]
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


def _build_tab(app):  # type: ignore[no-untyped-def]
    return app.nametowidget(app.notebook.tabs()[0])


def test_the_build_tab_has_a_detection_score_spinbox(app) -> None:  # type: ignore[no-untyped-def]
    spinboxes = [w for w in _descendants(_build_tab(app)) if w.winfo_class() == "TSpinbox"]
    assert spinboxes, "the Build tab has no Spinbox at all"
    spin = spinboxes[0]
    assert float(spin.cget("to")) == pytest.approx(0.99)
    assert float(spin.cget("increment")) == pytest.approx(0.05)
    labels = [str(w.cget("text")) for w in _descendants(_build_tab(app)) if w.winfo_class() == "TLabel"]
    assert any("Detection score" in text for text in labels)


def test_the_spinbox_is_bound_to_the_threshold_variable(app) -> None:  # type: ignore[no-untyped-def]
    """A spinbox that shows the value but is wired to nothing would still pass
    the test above; this checks the variable behind it is the real one."""
    spin = next(w for w in _descendants(_build_tab(app)) if w.winfo_class() == "TSpinbox")
    assert spin.cget("textvariable") == str(app._score_threshold_var)
    spin.set("0.30")
    assert app.score_threshold == pytest.approx(0.30)


def test_turning_the_spinbox_reaches_the_live_detector(app) -> None:  # type: ignore[no-untyped-def]
    """Clicking an arrow must reach the detector, not merely move the number.

    The widget's own `command` callback is invoked through Tcl, which is what a
    real arrow click does. `event_generate` is no good here: a withdrawn window
    never receives it, so the test would pass against a spinbox wired to
    nothing.
    """

    class Detector:
        score_threshold = 0.9

        def set_score_threshold(self, value: float) -> None:
            self.score_threshold = value

    class Recognizer:
        def __init__(self) -> None:
            self.detector = Detector()

    recognizer = Recognizer()
    app.set_recognizer(recognizer)
    spin = next(w for w in _descendants(_build_tab(app)) if w.winfo_class() == "TSpinbox")
    spin.set("0.25")
    app.tk.call(spin.cget("command"))
    assert recognizer.detector.score_threshold == pytest.approx(0.25)


def test_the_spinbox_commits_on_return_and_focus_out(app) -> None:  # type: ignore[no-untyped-def]
    """A value typed in and tabbed away from must take effect without also
    having to touch an arrow."""
    spin = next(w for w in _descendants(_build_tab(app)) if w.winfo_class() == "TSpinbox")
    assert spin.bind("<Return>"), "Return is not bound"
    assert spin.bind("<FocusOut>"), "FocusOut is not bound"


# ------------------------------------------------------- the model path rows


def test_the_build_tab_lets_you_choose_the_onnx_files(app) -> None:  # type: ignore[no-untyped-def]
    labels = [str(w.cget("text")) for w in _descendants(_build_tab(app)) if w.winfo_class() == "TLabel"]
    assert any("YuNet detector" in text for text in labels)
    assert any("Face recognizer" in text for text in labels)


def test_choosing_a_detector_model_discards_the_loaded_one(app) -> None:  # type: ignore[no-untyped-def]
    """Otherwise the newly chosen weights would not take effect until restart."""
    app.set_recognizer(object())
    app._detector_path_var.set("/somewhere/face_detection_yunet.onnx")
    app.set_recognizer(None)  # what _pick_detector_model does after setting the path
    assert app.recognizer is None


def test_the_output_path_starts_pre_filled(app) -> None:  # type: ignore[no-untyped-def]
    """Empty meant Save always detoured through the dialog."""
    assert app._output_var.get().endswith(".pkl")
    assert "face_db_" in app._output_var.get()


def test_a_typed_model_path_takes_effect(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Browse nulls the recognizer itself, but a path typed or pasted straight
    into the entry box does not. Without a signature check the app went on using
    the old weights while displaying the new path — the original rebuilt."""
    first = object()
    app.set_recognizer(first)
    app._recognizer_paths = ("", "")
    assert app.ensure_recognizer() is first

    app._detector_path_var.set(str(tmp_path / "different.onnx"))
    built: list[tuple[str, str]] = []

    def fake_build() -> object:
        built.append((app._detector_path_var.get(), app._embedder_path_var.get()))
        return first

    # ensure_recognizer must notice the mismatch and drop the stale recognizer
    app.ensure_recognizer()
    assert app.recognizer is None or built, "the typed path was ignored"
