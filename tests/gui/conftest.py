"""Fixtures for tests that need a real Tk display.

Everything here is marked `gui`. On a headless machine Tk cannot open a
display, so these skip unless one exists — in CI they run under `xvfb-run`.
The default `pytest -m "not gui"` run never touches this file.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def _require_display() -> None:
    """Skip the whole GUI suite when Tk cannot open a display."""
    tkinter = pytest.importorskip("tkinter", reason="tkinter is not installed")
    try:
        root = tkinter.Tk()
    except Exception as exc:  # tkinter.TclError, and anything else Tk raises
        pytest.skip(f"no display available for Tk: {exc}")
    else:
        root.destroy()


@pytest.fixture()
def app() -> Iterator[object]:
    """A live App window with a recording reporter, always destroyed after.

    The reporter matters: constructing a real modal dialog while the detection
    worker is alive segfaulted, so no test may ever open one. Assertions read
    `app.reporter.errors` / `.infos` instead.
    """
    from alchemyface.gui.app import App
    from alchemyface.gui.reporting import RecordingReporter

    window = App(reporter=RecordingReporter())
    window.withdraw()  # keep it off screen; it still builds and behaves
    try:
        yield window
    finally:
        window.close()
