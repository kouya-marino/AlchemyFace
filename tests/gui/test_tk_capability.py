"""The capability check against real widgets, and both branches of the pump.

The version logic is covered display-free in tests/unit/test_tkcompat.py. What
is left here is that a real widget takes the branch it should, and that a stale
Tk is announced somewhere the user can see it — which, when the window paints
nothing, means the terminal.

The pump tests replace the *instance's* methods with recorders rather than
patching the class and calling through. Letting the real `update()` run inside
the loop is what deadlocks, so a test that did it would hang rather than fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alchemyface.gui.tkcompat import MIN_DRAWABLE_TK, tk_patchlevel

pytestmark = pytest.mark.gui


def test_the_real_tk_reports_a_usable_version(app) -> None:  # type: ignore[no-untyped-def]
    version = tk_patchlevel(app)
    assert version, "Tk would not report its patchlevel"
    assert len(version) >= 2


def test_the_pump_never_calls_full_update(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """On any Tk. `update()` processes every pending event, not just redraws, so
    inside the resize loop it re-enters whatever else is scheduled: it segfaults
    on 8.5.9 and deadlocks the resize suite on 8.6.18. Idle tasks redraw without
    re-entering, which is the whole point."""
    view = app.resize_view
    calls: list[str] = []
    monkeypatch.setattr(view, "update", lambda: calls.append("update"), raising=False)
    monkeypatch.setattr(view, "update_idletasks", lambda: calls.append("update_idletasks"), raising=False)

    view._pump()

    assert calls == ["update_idletasks"]


def test_the_pump_survives_a_window_that_went_away(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A resize can outlive its window; that must not raise out of the loop."""
    import tkinter as tk

    def gone() -> None:
        raise tk.TclError("application has been destroyed")

    monkeypatch.setattr(app.resize_view, "update_idletasks", gone, raising=False)
    app.resize_view._pump()  # must not raise


def test_a_stale_tk_is_announced_on_stderr(app, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """stderr, not only the status bar and a dialog: the symptom is a window
    that shows nothing, so anything inside the window is unreadable."""
    from alchemyface.gui import app as app_module

    monkeypatch.setattr(app_module, "stale_tk_warning", lambda _w: "PRETEND STALE TK")
    app.reporter.clear()
    capsys.readouterr()

    app._warn_if_tk_cannot_draw()

    assert "PRETEND STALE TK" in capsys.readouterr().err
    assert any("PRETEND STALE TK" in message for _title, message in app.reporter.errors)
    assert "too old" in app.status


def test_a_current_tk_says_nothing_at_startup(app, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from alchemyface.gui import app as app_module

    monkeypatch.setattr(app_module, "stale_tk_warning", lambda _w: None)
    app.reporter.clear()
    capsys.readouterr()

    app._warn_if_tk_cannot_draw()

    assert capsys.readouterr().err == ""
    assert app.reporter.errors == []


def test_the_pumps_reasoning_is_recorded_where_it_is_used() -> None:
    """Both failure modes, so nobody "fixes" this back to update() a third time."""
    source = Path("src/alchemyface/gui/resize_view.py").read_text()
    assert "8.5.9" in source, "_pump does not name the Tk that segfaulted"
    assert "8.6" in source, "_pump does not record that a newer Tk deadlocks too"
    assert str(MIN_DRAWABLE_TK[0]) in source
