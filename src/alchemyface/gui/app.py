"""The Face DB Builder main window.

A shell that owns the window, the notebook and the status bar, and nothing
else. Each tab is a self-contained widget that receives what it needs as
callables, so a tab can be built and driven in a test without the window
knowing anything about it.

Tabs arrive one version at a time. This one has Inspect DB; Build, Edit and
Resize follow.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from alchemyface import __version__
from alchemyface.gui.inspect_view import InspectView

WINDOW_TITLE = "AlchemyFace — Face DB Builder"
WINDOW_GEOMETRY = "1280x820"


class App(tk.Tk):
    """The application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title(f"{WINDOW_TITLE}  ({__version__})")
        self.geometry(WINDOW_GEOMETRY)
        self._status_var = tk.StringVar(value="Ready.")
        self._closed = False
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        status_bar = ttk.Frame(self, padding=8)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Separator(self, orient="horizontal").pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status_bar, textvariable=self._status_var).pack(side=tk.LEFT)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        inspect_tab = ttk.Frame(self.notebook)
        self.notebook.add(inspect_tab, text="Inspect DB")
        self.inspect_view = InspectView(inspect_tab, on_status=self.set_status)
        self.inspect_view.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------- state
    def set_status(self, message: str) -> None:
        """Put a message in the status bar. Passed to every tab."""
        self._status_var.set(message)

    @property
    def status(self) -> str:
        return self._status_var.get()

    def tab_labels(self) -> list[str]:
        """Visible tab names, in order."""
        return [self.notebook.tab(tab_id, "text") for tab_id in self.notebook.tabs()]

    def close(self) -> None:
        """Shut down cleanly. Tabs that own threads get told first.

        Registered as the window-close handler, so a tab with a worker cannot
        be left running after the window disappears.

        Idempotent: ``tk.Tk.destroy`` raises ``TclError`` on a window that is
        already gone, which would turn any double close — a handler firing
        twice, or teardown after an explicit close — into a crash.
        """
        if self._closed:
            return
        self._closed = True
        for view in (getattr(self, "inspect_view", None),):
            shutdown = getattr(view, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:  # noqa: BLE001 - closing must not raise
                    pass
        try:
            self.destroy()
        except tk.TclError:
            pass  # already torn down by Tk itself

    @property
    def closed(self) -> bool:
        return self._closed


def main() -> None:
    """Entry point used by the ``alchemyface db`` command."""
    App().mainloop()
