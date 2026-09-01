"""The Resize tab: shrink photos until the detector can see the face.

The work itself is in :mod:`alchemyface.gui.resize_data`; this is the Tk around
it. Source and output each accept either a folder or a single image, so the two
rows are symmetric.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

from alchemyface.gui.annotation_data import IMAGE_EXTENSIONS
from alchemyface.gui.reporting import DialogReporter, Reporter
from alchemyface.gui.resize_data import (
    DEFAULT_RATIO,
    MAX_RATIO,
    MIN_RATIO,
    clamp_ratio,
    default_output_folder,
    resize_folder,
    resize_one,
)


class ResizeView(ttk.Frame):
    """Bulk or single-image resizing, with a per-file log."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_status: Callable[[str], None],
        reporter: Reporter | None = None,
    ) -> None:
        super().__init__(parent)
        self._set_status = on_status
        self.reporter: Reporter = reporter or DialogReporter()

        self._folder_in = tk.StringVar(value="")
        self._folder_out = tk.StringVar(value="")
        self._image_in = tk.StringVar(value="")
        self._image_out = tk.StringVar(value="")
        self._ratio_var = tk.DoubleVar(value=DEFAULT_RATIO)
        self._folder_out_chosen = False
        self._image_out_chosen = False
        self._build_ui()

    # ================================================================ public

    @property
    def ratio(self) -> float:
        return clamp_ratio(self._safe_ratio())

    def log_text(self) -> str:
        return self._log.get("1.0", tk.END)

    def clear_log(self) -> None:
        self._log.configure(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)

    def resize_folder_now(self, source: Path | str | None = None, destination: Path | str | None = None) -> int:
        """Resize a whole folder. Returns how many files were written."""
        src = Path(str(source) if source is not None else self._folder_in.get().strip())
        if not str(src) or not src.is_dir():
            return self._failed(f"Not a folder: {src}") and 0
        raw_out = str(destination) if destination is not None else self._folder_out.get().strip()
        dst = Path(raw_out) if raw_out else default_output_folder(src)
        self._folder_in.set(str(src))
        self._folder_out.set(str(dst))

        try:
            outcomes = resize_folder(src, dst, self.ratio)
        except (ValueError, NotADirectoryError) as exc:
            return self._failed(str(exc)) and 0
        if not outcomes:
            self._failed(f"No images found in {src}")
            return 0

        for outcome in outcomes:
            self._append(str(outcome))
        written = sum(1 for o in outcomes if o.ok)
        failed = len(outcomes) - written
        self._append(f"Done. {written} resized, {failed} failed.")
        self._set_status(f"Resized {written} of {len(outcomes)} image(s) at {self.ratio:.2f} → {dst}")
        return written

    def resize_image_now(self, source: Path | str | None = None, destination: Path | str | None = None) -> bool:
        """Resize a single image. Returns whether it worked."""
        src = Path(str(source) if source is not None else self._image_in.get().strip())
        if not str(src) or not src.is_file():
            return self._failed(f"Not a file: {src}")
        raw_out = str(destination) if destination is not None else self._image_out.get().strip()
        dst = Path(raw_out) if raw_out else src.with_name(f"{src.stem}_resized{src.suffix}")
        self._image_in.set(str(src))
        self._image_out.set(str(dst))

        try:
            result = resize_one(src, dst, self.ratio)
        except ValueError as exc:
            return self._failed(str(exc))
        except OSError as exc:
            return self._failed(f"{src.name}: {exc}")
        self._append(f"{src.name}: {result}")
        self._set_status(f"Resized {src.name} at {self.ratio:.2f} → {dst}")
        return True

    def shutdown(self) -> None:
        """Nothing to release; present so the window can call it uniformly."""

    # ============================================================== internals

    def _safe_ratio(self) -> float:
        try:
            return float(self._ratio_var.get())
        except (tk.TclError, ValueError):
            return DEFAULT_RATIO

    def _failed(self, reason: str) -> bool:
        self._append(f"ERROR: {reason}")
        self._set_status(f"Resize: {reason}")
        self.reporter.error("Resize", reason)
        return False

    def _append(self, line: str) -> None:
        self._log.configure(state=tk.NORMAL)
        self._log.insert(tk.END, line + "\n")
        self._log.see(tk.END)

    # -------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        source = ttk.LabelFrame(self, text="Source", padding=8)
        source.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 2))
        self._path_row(source, 0, "Folder:", self._folder_in, "Browse folder…", self._pick_folder_in)
        self._path_row(source, 1, "Image:", self._image_in, "Browse image…", self._pick_image_in)
        source.columnconfigure(1, weight=1)

        output = ttk.LabelFrame(self, text="Output", padding=8)
        output.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        self._path_row(output, 0, "Folder:", self._folder_out, "Browse folder…", self._pick_folder_out)
        self._path_row(output, 1, "Image:", self._image_out, "Save as…", self._pick_image_out)
        output.columnconfigure(1, weight=1)

        controls = ttk.Frame(self, padding=(8, 4))
        controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(controls, text="Ratio:").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls,
            from_=MIN_RATIO,
            to=MAX_RATIO,
            increment=0.05,
            textvariable=self._ratio_var,
            width=6,
            format="%.2f",
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            controls,
            text="  (0.5 = half · 0.25 = quarter · 2.0 = double)",
            foreground="#666666",
        ).pack(side=tk.LEFT)

        actions = ttk.Frame(self, padding=(8, 4))
        actions.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(actions, text="Resize folder", command=lambda: self.resize_folder_now()).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Resize image", command=lambda: self.resize_image_now()).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Clear log", command=self.clear_log).pack(side=tk.RIGHT, padx=4)

        frame = ttk.LabelFrame(self, text="Log", padding=4)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)
        self._log = tk.Text(frame, height=12, wrap="none", font=("TkFixedFont", 10))
        vertical = ttk.Scrollbar(frame, orient="vertical", command=self._log.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=self._log.xview)
        self._log.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self._log.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    def _path_row(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        button: str,
        command: Callable[[], None],
    ) -> None:
        ttk.Label(parent, text=label, width=8).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(parent, text=button, command=command).grid(row=row, column=2, padx=6, pady=3)

    # ---------------------------------------------------------- file pickers

    def _pick_folder_in(self) -> None:
        chosen = filedialog.askdirectory(title="Select source folder")
        if not chosen:
            return
        self._folder_in.set(chosen)
        if not self._folder_out_chosen:
            self._folder_out.set(str(default_output_folder(chosen)))

    def _pick_folder_out(self) -> None:
        chosen = filedialog.askdirectory(title="Select output folder")
        if chosen:
            self._folder_out.set(chosen)
            self._folder_out_chosen = True

    def _pick_image_in(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS))
        chosen = filedialog.askopenfilename(
            title="Select an image", filetypes=[("Images", patterns), ("All files", "*.*")]
        )
        if not chosen:
            return
        self._image_in.set(chosen)
        if not self._image_out_chosen:
            path = Path(chosen)
            self._image_out.set(str(path.with_name(f"{path.stem}_resized{path.suffix}")))

    def _pick_image_out(self) -> None:
        chosen = filedialog.asksaveasfilename(title="Save resized image as")
        if chosen:
            self._image_out.set(chosen)
            self._image_out_chosen = True
