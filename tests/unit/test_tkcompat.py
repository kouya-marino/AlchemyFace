"""Deciding what the loaded Tk can be trusted with.

Display-free: these drive the version logic through a stub, so the branch for a
Tk that is not installed here is still covered. The behaviour of the real Tk is
exercised in tests/gui/test_tk_capability.py.
"""

from __future__ import annotations

import tkinter as tk

import pytest

from alchemyface.gui.tkcompat import MIN_DRAWABLE_TK, stale_tk_warning, tk_patchlevel


class FakeTk:
    """Answers `info patchlevel` with whatever the test wants."""

    def __init__(self, patchlevel: str | None) -> None:
        self._patchlevel = patchlevel

    def call(self, *args: object) -> str:
        assert args == ("info", "patchlevel")
        if self._patchlevel is None:
            raise tk.TclError("no interpreter")
        return self._patchlevel


class FakeWidget:
    def __init__(self, patchlevel: str | None) -> None:
        self.tk = FakeTk(patchlevel)


def widget(patchlevel: str | None):  # type: ignore[no-untyped-def]
    return FakeWidget(patchlevel)


def test_the_threshold_is_eight_six() -> None:
    """8.5.9 opens a window and paints nothing on a current macOS."""
    assert MIN_DRAWABLE_TK == (8, 6)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("8.6.18", (8, 6, 18)),
        ("8.5.9", (8, 5, 9)),
        ("9.0.4", (9, 0, 4)),
        ("8.6", (8, 6)),
        ("8.6.18b2", (8, 6, 18)),  # a pre-release suffix must not break parsing
    ],
)
def test_the_patchlevel_is_read_exactly(raw: str, expected: tuple[int, ...]) -> None:
    """`tkinter.TkVersion` is a float, so it cannot tell 8.6.0 from 8.6.18 and
    reports 8.5.9 as 8.5. The Tcl string is the only precise source."""
    assert tk_patchlevel(widget(raw)) == expected


@pytest.mark.parametrize("raw", ["", "not-a-version", "x.y.z"])
def test_an_unreadable_patchlevel_is_empty_not_a_guess(raw: str) -> None:
    assert tk_patchlevel(widget(raw)) == ()


def test_a_dead_interpreter_is_empty_rather_than_raising() -> None:
    assert tk_patchlevel(widget(None)) == ()


def test_old_tk_gets_a_warning_naming_the_fix() -> None:
    message = stale_tk_warning(widget("8.5.9"))
    assert message is not None
    assert "8.5.9" in message
    assert "blank" in message
    assert "tcl-tk@8" in message
    # the trap that wastes the most time: brew's default tcl-tk is unusable here
    assert "9.x" in message or "3.13" in message


@pytest.mark.parametrize("raw", ["8.6.18", "9.0.4"])
def test_a_current_tk_gets_no_warning(raw: str) -> None:
    assert stale_tk_warning(widget(raw)) is None


def test_an_unknown_version_gets_no_warning() -> None:
    """Better silent than crying wolf at everyone whose Tk will not answer."""
    assert stale_tk_warning(widget(None)) is None
