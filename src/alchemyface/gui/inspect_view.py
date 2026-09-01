"""The Inspect DB tab: open a face database and look at what is inside.

The widget only displays. Everything it shows is computed by
:mod:`alchemyface.gui.inspect_data`, which knows nothing about Tk and is tested
without a display. Read-only — nothing here writes a file.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Literal

from alchemyface.errors import AlchemyFaceError
from alchemyface.gui.inspect_data import entry_rows, summarise
from alchemyface.store import PickleStore

Anchor = Literal["nw", "n", "ne", "w", "center", "e", "sw", "s", "se"]
"""The alignments ``ttk.Treeview.column`` accepts. Typed, because its stub
declares a Literal and a plain ``str`` will not satisfy it."""


@dataclass(frozen=True)
class ColumnSpec:
    """One column of the entries table: its key, header, width and alignment."""

    key: str
    heading: str
    width: int
    anchor: Anchor


class InspectView(ttk.Frame):
    """Read-only viewer for a face database pickle."""

    COLUMN_SPECS: tuple[ColumnSpec, ...] = (
        ColumnSpec("idx", "#", 50, "e"),
        ColumnSpec("id", "ID", 220, "w"),
        ColumnSpec("name", "Name", 180, "w"),
        ColumnSpec("group", "Group", 130, "w"),
        ColumnSpec("dim", "Dim", 70, "center"),
        ColumnSpec("norm", "L2 norm", 90, "e"),
        ColumnSpec("preview", "First values", 420, "w"),
    )
    COLUMNS = tuple(spec.key for spec in COLUMN_SPECS)

    def __init__(self, parent: tk.Misc, *, on_status: Callable[[str], None]) -> None:
        super().__init__(parent)
        self._set_status = on_status
        self._path_var = tk.StringVar(value="")
        self._summary_var = tk.StringVar(value="No file loaded.")
        self._store: PickleStore | None = None
        self._build_ui()

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        top = ttk.LabelFrame(self, text="Face database", padding=8)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(top, text="Path:").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(top, textvariable=self._path_var, width=80).grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(top, text="Browse…", command=self._pick_file).grid(row=0, column=2, padx=6, pady=3)
        ttk.Button(top, text="Load", command=self.load).grid(row=0, column=3, padx=6, pady=3)
        top.columnconfigure(1, weight=1)

        summary = ttk.Frame(self, padding=(8, 4))
        summary.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(summary, textvariable=self._summary_var).pack(side=tk.LEFT)

        table = ttk.LabelFrame(self, text="Entries", padding=4)
        table.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)
        self._tree = ttk.Treeview(table, columns=self.COLUMNS, show="headings", selectmode="browse")
        for spec in self.COLUMN_SPECS:
            self._tree.heading(spec.key, text=spec.heading)
            self._tree.column(spec.key, width=spec.width, anchor=spec.anchor)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ----------------------------------------------------------- actions
    def _pick_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Open face database",
            filetypes=[("Pickle", "*.pkl"), ("All files", "*.*")],
            initialdir=str(Path(self._path_var.get()).parent) if self._path_var.get() else str(Path.cwd()),
        )
        if chosen:
            self._path_var.set(chosen)
            self.load()

    def load(self, path: Path | str | None = None) -> bool:
        """Load a database and render it. Returns whether it worked.

        Takes an optional path so tests and callers can drive it without a
        file dialog.
        """
        raw = str(path) if path is not None else self._path_var.get().strip()
        if not raw:
            return self._failed("No file path set.")
        self._path_var.set(raw)
        target = Path(raw)
        if not target.is_file():
            return self._failed(f"File does not exist: {target}")

        store = PickleStore()
        try:
            store.load(target)
        except AlchemyFaceError as exc:
            return self._failed(str(exc))

        self._store = store
        self._render(target)
        return True

    def _failed(self, reason: str) -> bool:
        self._store = None
        self._summary_var.set(f"Load failed: {reason}")
        self._set_status(f"Inspect failed: {reason}")
        self._clear()
        messagebox.showerror("Load failed", reason)
        return False

    def _clear(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

    def _render(self, path: Path) -> None:
        assert self._store is not None
        self._clear()
        summary = summarise(self._store, path)
        self._summary_var.set(str(summary))
        self._set_status(f"Loaded {summary.entries} entries from {path}")
        for row in entry_rows(self._store):
            self._tree.insert(
                "",
                tk.END,
                iid=str(row.index),
                values=(
                    row.index,
                    row.entry_id[:12],
                    row.name,
                    row.group,
                    row.dim,
                    f"{row.norm:.4f}",
                    row.preview,
                ),
            )

    # ------------------------------------------------------------- state
    @property
    def store(self) -> PickleStore | None:
        """The loaded database, or ``None``."""
        return self._store

    @property
    def summary_text(self) -> str:
        return self._summary_var.get()

    def row_count(self) -> int:
        """How many rows the table is showing."""
        return len(self._tree.get_children())
