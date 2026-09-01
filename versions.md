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

### 0.3.0 — GUI shell and Inspect DB · 2026-09-01

The smallest useful slice of GUI, on purpose. Packaging, the entry point and
headless CI are proven here so the large views that follow have nothing to prove
but themselves.

- `alchemyface db` launches the Face DB Builder. `Pillow` joins the base install.
- **Inspect DB tab** — open any `.pkl` and see every entry, plus a summary line.
- `alchemyface.gui.inspect_data` holds the pure presentation functions, so what
  the tab shows is tested with no display. `StoreEntry` and
  `PickleStore.entries()` give it something to enumerate.
- `gui` pytest marker; CI runs those under `xvfb` with `python3-tk`, and the
  coverage gate moved to that job because it is the only one that sees widgets.

**Verified:** `import alchemyface` works with `tkinter` blocked — three tests
enforce it, since on Debian and Ubuntu `tkinter` is a separate OS package and
the mistake is invisible on a developer machine. A real 30-entry production
database loads in the tab. 162 tests, 95% coverage.

### 0.4.0 — Build DB · 2026-09-01

A folder of photos becomes a database the robot can load.

- Three panes: image sidebar with status glyphs, canvas with numbered face
  boxes, one panel per face (thumbnail · Include · Name · Group).
- Background detection with priorities, whole-folder prefetch, and generation
  counters so opening a different folder abandons the old one.
- Click-to-select on the canvas; excluded faces draw dashed. Re-detect asks
  before discarding edits.
- Save writes included, named faces as a robot-format `.pkl` with raw vectors
  and ids renumbered from `"0"`.
- `annotation_data` and `detect_worker` hold the logic and the threading, both
  free of Tk, so 195 of the 241 tests need no display.

**Verified** on the real models and real photos: 6 images, 7 faces, raw L2 norms
between 9.8 and 13.9. One tight selfie found no face at all — which is what the
Resize tab in 0.6.0 exists to fix.

---

## Planned

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
