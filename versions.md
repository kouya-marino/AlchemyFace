# Versions

A curated overview of where AlchemyFace has been and where it is going.
`CHANGELOG.md` is the detailed per-release log; this file is the map.

Design: [`docs/superpowers/specs/2026-08-28-alchemyface-design.md`](docs/superpowers/specs/2026-08-28-alchemyface-design.md)
(library) and [`docs/superpowers/specs/2026-09-01-face-db-builder-design.md`](docs/superpowers/specs/2026-09-01-face-db-builder-design.md)
(the GUI app).

Every version below is a branch → pull request → review → merge → tag.
Each one leaves the project working and independently publishable.

---

## Shipped

### 0.1.0 — the library · 2026-08-28

Face detection and recognition as a typed, dependency-light library.

- `Recognizer` facade over three protocols: `Detector`, `Embedder`, `FaceStore`.
- `YuNetDetector`, `SFaceEmbedder` (128-d, L2-normalised), `InMemoryStore`.
- Model weights resolved at runtime and SHA256-verified; 21 KB wheel.
- `VideoSource`; CLI with `download-models`, `enroll`, `identify`, `version`.
- 88 tests, 91% coverage. Published to PyPI.

### 0.2.0 — library groundwork for the app · 2026-09-01

No GUI. Two additions the app needs, both useful on their own.

- `SFaceEmbedder(normalize=False)` — raw SFace output, L2 ≈ 10.
- `PickleStore` — the robot's `list[(id, name, group, vector)]` format as a
  real `FaceStore`, storing vectors verbatim and normalising inside `search`.
- `PickleSchemaError` so malformed input reports a reason.

**Verified:** all three production databases load byte-identically to the
original app's reader, and a database written by `PickleStore` loads back in
that app with its vectors intact. 123 tests, 91% coverage.

Reading is deliberately forgiving — the production files disagree with their own
documented schema (`id` is `int` in one and `str` in two; vectors are
`(1, 128)` in one and `(128,)` in two).

---

## Planned

### 0.3.0 — GUI shell and Inspect DB

The smallest complete slice of GUI, deliberately. Proves the entry point, the
packaging, and headless CI before the large views arrive.

- `alchemyface.gui` package. `Pillow` becomes a base dependency.
- `alchemyface db` launches the app; `import alchemyface` still works with no
  `tkinter` present.
- Main window: notebook, status bar, shared group presets, lazy `Recognizer`.
- **Inspect DB tab** — open any `.pkl`, list `ID · Name · Group · Dim · L2 norm
  · first values`, plus a summary line.
- `gui` pytest marker; GUI tests run under `xvfb-run` in CI.

**Done when:** `alchemyface db` opens, loads all three sample `.pkl` files, and
`pytest -m "not gui"` still runs display-free.

### 0.4.0 — Build DB

The core of the app, and the largest single piece.

- Three-pane annotation view: image sidebar, canvas with numbered face boxes,
  per-face panel (thumbnail · Include · Name · Group).
- Background detection worker: priority queue, eager folder prefetch,
  generation counters discarding stale jobs, BGR LRU cache, clean shutdown.
- Sidebar status glyphs: `·` pending, `⚠` no face, `○ (0/N)`, `✓ (k/N)`.
- Canvas: debounced resize, cached `PhotoImage`, click-to-select a face,
  dashed outline when excluded.
- Detection-score spinbox (0.10–0.99, default 0.9), Re-detect with a
  confirm-if-edited prompt, embeddings computed on demand and cached.
- Save flow: names validated, missing embeddings filled in, IDs renumbered
  from `"0"`, distinct success and failure dialogs.

**Done when:** a folder of photos becomes a `.pkl` the robot loads, and the
UI stays responsive while a large folder detects in the background.

### 0.5.0 — Edit DB

- Load an existing `.pkl`; table of `# · Name · Group · Dim · L2 norm`.
- Remove rows (multi-select, Delete key). Inline group edit via combobox,
  typed values added to presets.
- Add faces from a folder or a single image as pending cards, then merge the
  ticked ones.
- Save over the loaded path or Save as…; duplicates allowed, since the robot
  resolves by best cosine similarity.
- Unsaved-changes marker in the frame title and a confirm on close.

**Done when:** an existing database can be opened, trimmed, extended and saved
without the robot noticing anything but the intended change.

### 0.6.0 — Resize

Bulk pre-resize before enrolment. Tight phone selfies where a face fills more
than half the frame defeat YuNet's largest anchors; shrinking brings them into
range.

- Source folder or single image; matching output; ratio 0.05–5.0, default 0.5.
- EXIF transpose, LANCZOS down / BICUBIC up, format-aware save options.
- Per-file log; same-path overwrite refused.
- `alchemyface resize` CLI subcommand sharing the same helper.

**Done when:** photos that YuNet missed at full size are detected after a
0.5 resize, through both the tab and the CLI.

### 1.0.0 — parity and polish

- Behaviour audit against the original app, tab by tab.
- README rewritten to cover library *and* app; screenshots.
- Docstring and typing pass over the GUI; coverage back above 80% overall.
- `todo.md` emptied of anything blocking, or its remainder deferred explicitly.

**Done when:** nothing the original app does is missing, and the reason for
every deliberate difference is written down.

---

## Deliberate differences from the original app

| | Original | Here |
|---|---|---|
| Face pipeline | own `face_pipeline.py`, 59 lines | the `alchemyface` library |
| Face representation | raw `(N, 15)` numpy rows | `Face` dataclass |
| `.pkl` handling | `_normalize_db` duplicated in two views | one `PickleStore` |
| Inspect tab | inline in `app.py` | its own `inspect_view.py` |
| Bad schema | returns `None` | raises `PickleSchemaError` with a reason |
| Layout | flat, `gui/` beside `main.py` | `src/alchemyface/gui/` |
| Tooling | requirements.txt, pytest | setuptools, ruff, mypy, CI, PyPI |
