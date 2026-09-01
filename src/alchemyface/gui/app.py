"""The Face DB Builder main window.

The window owns what the tabs share — the status bar, the group presets, the
recognizer — and delegates everything else. Each tab receives what it needs as
callables, so a tab can be built and driven in a test without the window knowing
anything about it.

Tabs arrive one version at a time: Inspect DB, then Build DB. Edit and Resize
follow.
"""

from __future__ import annotations

import datetime as dt
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

from alchemyface import __version__
from alchemyface.gui.annotation_data import DEFAULT_GROUP
from alchemyface.gui.annotation_view import AnnotationView, RecognizerLike
from alchemyface.gui.edit_db_view import EditDBView
from alchemyface.gui.inspect_view import InspectView
from alchemyface.gui.reporting import DialogReporter, Reporter
from alchemyface.store import PickleStore

WINDOW_TITLE = "AlchemyFace — Face DB Builder"
WINDOW_GEOMETRY = "1360x860"
DEFAULT_SCORE_THRESHOLD = 0.9
"""YuNet's own default. Lower surfaces more faces; higher trims false ones."""


class App(tk.Tk):
    """The application window."""

    def __init__(self, *, reporter: Reporter | None = None) -> None:
        """``reporter`` defaults to modal dialogs.

        Injected rather than hard-coded so a test never constructs a real modal
        — doing so with a worker thread alive segfaulted.
        """
        super().__init__()
        self.reporter: Reporter = reporter or DialogReporter()
        self.title(f"{WINDOW_TITLE}  ({__version__})")
        self.geometry(WINDOW_GEOMETRY)
        self._closed = False

        self._status_var = tk.StringVar(value="Ready.")
        self._group_presets: list[str] = [DEFAULT_GROUP]
        self._recognizer: RecognizerLike | None = None
        self._score_threshold = DEFAULT_SCORE_THRESHOLD

        self._folder_var = tk.StringVar(value="")
        self._output_var = tk.StringVar(value="")
        self._output_chosen = False
        """Once the user picks an output explicitly, opening a folder stops
        overwriting their choice."""

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

    def _on_window_close(self) -> None:
        from tkinter import messagebox

        self.close(
            confirm=lambda: messagebox.askyesno(
                "Unsaved changes",
                "Edit DB has unsaved changes that will be lost. Close anyway?",
            )
        )

    # ================================================================ shared

    def set_status(self, message: str) -> None:
        self._status_var.set(message)

    @property
    def status(self) -> str:
        return self._status_var.get()

    @property
    def group_presets(self) -> list[str]:
        return list(self._group_presets)

    def add_group_preset(self, group: str) -> None:
        value = group.strip()
        if not value or value in self._group_presets:
            return
        self._group_presets.append(value)
        self._group_listbox.insert(tk.END, value)

    # -------------------------------------------------------------- the model

    @property
    def recognizer(self) -> RecognizerLike | None:
        return self._recognizer

    def set_recognizer(self, recognizer: RecognizerLike | None) -> None:
        """Install a recognizer and discard anything embedded with the old one.

        Keeping stale embeddings would silently mix vectors from two models into
        one database, which no threshold can then separate.
        """
        self._recognizer = recognizer
        view = getattr(self, "annotation_view", None)
        if view is not None:
            view.invalidate_embeddings()

    def _current_recognizer(self) -> RecognizerLike | None:
        """Whatever is loaded, and nothing else.

        This is what the Build tab is given, and the Build tab calls it **from
        its worker thread**. So it must never touch a Tk widget or variable:
        writing a StringVar off the main thread is undefined behaviour, and was
        segfaulting the test suite intermittently. Loading — which reports
        progress and can fail — happens in :meth:`ensure_recognizer`, on the
        main thread.
        """
        return self._recognizer

    def ensure_recognizer(self) -> RecognizerLike | None:
        """Load the real recognizer if nothing is installed. Main thread only.

        Embeddings are deliberately **not** normalised: the robot's `.pkl`
        schema stores raw SFace output. Cosine is scale-invariant so matching is
        unaffected, but the `L2 norm` column stays meaningful and new databases
        remain comparable with existing ones.
        """
        if self._recognizer is not None:
            return self._recognizer
        try:
            from alchemyface.detection import YuNetDetector
            from alchemyface.embedding import SFaceEmbedder
            from alchemyface.pipeline import Recognizer

            self._recognizer = Recognizer(
                detector=YuNetDetector(score_threshold=self._score_threshold),
                embedder=SFaceEmbedder(normalize=False),
                store=PickleStore(),
            )
            self.set_status("Models loaded.")
        except Exception as exc:  # noqa: BLE001 - reported in the UI
            self.reporter.error("Models", f"Could not load the models:\n{exc}")
            self.set_status(f"Model load failed: {exc}")
            self._recognizer = None
        return self._recognizer

    # ==================================================================== UI

    def _build_ui(self) -> None:
        status_bar = ttk.Frame(self, padding=8)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Separator(self, orient="horizontal").pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status_bar, textvariable=self._status_var).pack(side=tk.LEFT)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        build_tab = ttk.Frame(self.notebook)
        edit_tab = ttk.Frame(self.notebook)
        inspect_tab = ttk.Frame(self.notebook)
        self.notebook.add(build_tab, text="Build DB")
        self.notebook.add(edit_tab, text="Edit DB")
        self.notebook.add(inspect_tab, text="Inspect DB")
        self._build_build_tab(build_tab)

        self.edit_view = EditDBView(
            edit_tab,
            # ensure_recognizer, not _current_recognizer: this tab detects on
            # the main thread, so it may load models and report progress. The
            # Build tab gets the plain getter because it calls from a worker
            # thread, where touching Tk is undefined behaviour.
            recognizer_provider=self.ensure_recognizer,
            group_presets_provider=lambda: list(self._group_presets),
            on_status=self.set_status,
            on_preset_added=self.add_group_preset,
            reporter=self.reporter,
        )
        self.edit_view.pack(fill=tk.BOTH, expand=True)

        self.inspect_view = InspectView(inspect_tab, on_status=self.set_status, reporter=self.reporter)
        self.inspect_view.pack(fill=tk.BOTH, expand=True)

    def _build_build_tab(self, parent: ttk.Frame) -> None:
        config = ttk.LabelFrame(parent, text="Configuration", padding=8)
        config.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 2))

        ttk.Label(config, text="Input folder:").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(config, textvariable=self._folder_var, width=70).grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        buttons = ttk.Frame(config)
        buttons.grid(row=0, column=2, sticky="w")
        ttk.Button(buttons, text="Browse…", command=self._pick_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="Open", command=self._open_folder).pack(side=tk.LEFT, padx=2)

        ttk.Label(config, text="Output .pkl:").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(config, textvariable=self._output_var, width=70).grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(config, text="Save as…", command=self._pick_output).grid(row=1, column=2, sticky="w", padx=6, pady=3)
        config.columnconfigure(1, weight=1)

        groups = ttk.LabelFrame(parent, text="Group presets", padding=4)
        groups.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        self._group_listbox = tk.Listbox(groups, height=2, selectmode=tk.SINGLE)
        self._group_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        for preset in self._group_presets:
            self._group_listbox.insert(tk.END, preset)
        controls = ttk.Frame(groups)
        controls.pack(side=tk.RIGHT, padx=4)
        self._new_group_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self._new_group_var, width=18).grid(row=0, column=0, padx=2)
        ttk.Button(controls, text="Add", command=self._add_group_clicked).grid(row=0, column=1, padx=2)
        ttk.Button(controls, text="Remove", command=self._remove_group_preset).grid(row=0, column=2, padx=2)

        actions = ttk.Frame(parent, padding=6)
        actions.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Separator(parent, orient="horizontal").pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(actions, text="💾  Save .pkl", command=self._save_clicked, padding=(12, 6)).pack(side=tk.RIGHT)

        self.annotation_view = AnnotationView(
            parent,
            recognizer_provider=self._current_recognizer,
            ensure_recognizer=self.ensure_recognizer,
            group_presets_provider=lambda: list(self._group_presets),
            on_status=self.set_status,
            on_preset_added=self.add_group_preset,
        )
        self.annotation_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------- callbacks

    def _pick_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Select a folder of face images")
        if chosen:
            self._folder_var.set(chosen)

    def _pick_output(self) -> None:
        chosen = filedialog.asksaveasfilename(
            title="Save face database",
            defaultextension=".pkl",
            filetypes=[("Pickle", "*.pkl"), ("All files", "*.*")],
        )
        if chosen:
            self._output_var.set(chosen)
            self._output_chosen = True

    def _add_group_clicked(self) -> None:
        self.add_group_preset(self._new_group_var.get())
        self._new_group_var.set("")

    def _remove_group_preset(self) -> None:
        selection = self._group_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        self._group_presets.pop(index)
        self._group_listbox.delete(index)

    def _open_folder(self) -> None:
        raw = self._folder_var.get().strip()
        if not raw or not Path(raw).is_dir():
            self.reporter.error("Input folder", "Please choose a valid folder.")
            return
        if self.ensure_recognizer() is None:
            return  # ensure_recognizer has already reported why
        if not self._output_chosen:
            stamp = f"face_db_{dt.date.today():%Y%m%d}.pkl"
            self._output_var.set(str(Path(raw) / stamp))
        self.annotation_view.load_folder(Path(raw))

    def _save_clicked(self) -> None:
        raw = self._output_var.get().strip()
        if not raw:
            self.reporter.error("Save failed", "No output path set. Use 'Save as…'.")
            return
        self.save_database(Path(raw))

    # ------------------------------------------------------------------ save

    def save_database(self, path: Path | str) -> bool:
        """Write every included, named face to a robot-format pickle.

        Returns whether it worked, and reports the reason if not. Embeddings are
        computed here for any face that has not needed one yet.
        """
        from alchemyface.gui.annotation_data import records_for_save, validate_for_save

        entries = self.annotation_view.entries
        problems = validate_for_save(entries)
        if problems:
            return self._save_failed("Cannot save:\n  " + "\n  ".join(problems))

        records = records_for_save(entries)
        if not records:
            return self._save_failed("Nothing to include — every detected face is unticked.")

        problems = self.annotation_view.fill_embeddings()
        if problems:
            return self._save_failed("Cannot save:\n  " + "\n  ".join(problems))

        store = PickleStore()
        for record in records:
            if record.embedding is None:
                return self._save_failed(f"{record.name}: embedding missing after computing them.")
            store.add(record.name.strip(), record.embedding, {"group": record.group})

        destination = Path(path)
        try:
            store.save(destination)
        except OSError as exc:
            return self._save_failed(f"Cannot write {destination}: {exc}")

        size_kb = destination.stat().st_size / 1024
        self.set_status(f"Saved {len(store)} entries → {destination}")
        self.reporter.info(
            "Save successful",
            f"Saved {len(store)} face entries ({size_kb:.1f} KB) to:\n{destination}",
        )
        return True

    def _save_failed(self, reason: str) -> bool:
        self.set_status(f"Save failed: {reason.splitlines()[0]}")
        self.reporter.error("Save failed", reason)
        return False

    # -------------------------------------------------------------- lifecycle

    def tab_labels(self) -> list[str]:
        return [self.notebook.tab(tab_id, "text") for tab_id in self.notebook.tabs()]

    def has_unsaved_changes(self) -> bool:
        """Whether any tab holds work that closing would discard."""
        view = getattr(self, "edit_view", None)
        checker = getattr(view, "has_unsaved_changes", None)
        return bool(checker()) if callable(checker) else False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self, confirm: Callable[[], bool] | None = None) -> None:
        """Shut down cleanly. Tabs owning threads are told first.

        ``confirm`` is consulted only when a tab holds unsaved work, so the
        common case does not nag. Returning False cancels the close.

        Idempotent: ``tk.Tk.destroy`` raises on an already-destroyed window, so
        a handler firing twice would otherwise crash on exit.
        """
        if self._closed:
            return
        if confirm is not None and self.has_unsaved_changes() and not confirm():
            return
        self._closed = True
        for name in ("annotation_view", "edit_view", "inspect_view"):
            view = getattr(self, name, None)
            shutdown = getattr(view, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:  # noqa: BLE001 - closing must not raise
                    pass
        try:
            self.destroy()
        except tk.TclError:
            pass


def main() -> None:
    """Entry point used by the ``alchemyface db`` command."""
    App().mainloop()
