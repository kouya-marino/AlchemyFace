# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

_Nothing yet._

## [0.3.0] — 2026-09-01

The desktop application arrives, deliberately as its smallest useful slice: the
window, and one read-only tab. Packaging, the entry point and headless CI are
proven here so the large views that follow have nothing to prove but themselves.

### Added

- `alchemyface db` launches the **Face DB Builder** desktop application.
- **Inspect DB tab** — open any face database `.pkl` and see every entry as
  `# · ID · Name · Group · Dim · L2 norm · First values`, with a summary line
  giving counts, dimension and file size.
- `alchemyface.gui.inspect_data`: `EntryRow`, `DatabaseSummary`, `entry_rows()`
  and `summarise()`. Pure functions over a store, so what the tab shows is
  tested without a display.
- `StoreEntry`, and `PickleStore.entries()`. `search` answers "who is this?";
  this answers "what is in here?", which is what listing a database needs.
- A `gui` pytest marker. `pytest -m "not gui"` is the fast path and needs no
  display — and, as a test proves, no `tkinter` either.

### Changed

- `Pillow>=10` joins the base dependencies. The GUI ships in the base install;
  `tkinter` is standard library, so Pillow is the only addition.
- CI gained a `gui` job running under `xvfb` with `python3-tk`. The coverage
  gate moved there, because it is the only job that can exercise the widgets —
  gating the display-free matrix job would have measured the wrong thing.

### Notes

- **Nothing in the library imports `tkinter`.** On Debian and Ubuntu it is a
  separate OS package, so importing it anywhere in the library would break
  `import alchemyface` for people who only wanted the library. Three tests
  enforce this: two import the public surface in a subprocess with `tkinter`
  blocked, and one statically locates any module-scope import. `alchemyface db`
  fails with an `apt-get install python3-tk` hint rather than a traceback.
- `App.close()` is idempotent. `tk.Tk.destroy()` raises on an already-destroyed
  window, so a close handler firing twice would have crashed on exit.
- 162 tests, 95% coverage locally with weights present; 91% without.

## [0.2.0] — 2026-09-01

Groundwork for the Face DB Builder GUI. Library only — no GUI yet. Both
additions are useful on their own.

### Added

- `SFaceEmbedder(normalize=False)` returns SFace's raw output, L2 norm ~10,
  instead of a unit vector. The default stays `True`, because the `Embedder`
  protocol promises unit length and the rest of the library relies on it.
  Cosine similarity is scale-invariant, so matching is identical either way;
  what changes is the magnitude stored on disk.
- `PickleStore`, a `FaceStore` over the Unitree G1 robot's
  `list[(id, name, group, vector)]` pickle. Stores vectors verbatim and
  normalises inside `search`, so a round trip alters nothing. Reads the
  back-compatible `{name: vector}` dict form too.
- `PickleSchemaError`, so a malformed database reports a reason rather than
  returning `None`.

### Notes

- `PickleStore` is deliberately forgiving on read. The production databases
  disagree with their own documented schema: `id` is an `int` in one file and a
  `str` in two, and the vector is `(1, 128)` in one and `(128,)` in two. Both
  are coerced. A stricter reader would refuse a database the robot loads today.
- Verified against all three production databases: values load byte-identically
  to the original app's reader, and a database written here loads back in that
  app with its vectors intact.
- 123 tests, coverage held at 91%.

## [0.1.0] — 2026-08-28

First release. Extracted from an earlier prototype into a typed, installable library.

### Added

- `Recognizer` facade sequencing three protocols — `Detector`, `Embedder` and
  `FaceStore` — so a different gallery or embedding model is a drop-in.
- `YuNetDetector` for detection and `SFaceEmbedder` for 128-d embeddings, via
  OpenCV's DNN runtime. Embeddings are L2-normalised, making cosine similarity
  a dot product.
- `InMemoryStore`: a numpy gallery with `.npz` save/load. No database required.
- Runtime model resolution — explicit path, `$ALCHEMYFACE_MODEL_DIR`,
  `~/.cache/alchemyface/models/`, then a SHA256-verified download from the
  OpenCV Zoo. Weights are never vendored, keeping the wheel at ~20 KB.
- `VideoSource`, a context-managed wrapper over `cv2.VideoCapture`.
- CLI: `alchemyface download-models | enroll | identify | version`.
- Inline type information (`py.typed`), checked with mypy under
  `disallow_untyped_defs`.

### Notes

- Depends on `opencv-python-headless` rather than `opencv-python`: the package
  makes no GUI calls, and the headless build installs cleanly on CI, in Docker
  and on servers without `libGL`.
- 88 tests at 91% coverage. The default suite needs no models, camera or
  network; model-backed tests are marked and skip when weights are absent.

[Unreleased]: https://github.com/kouya-marino/AlchemyFace/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v0.3.0
[0.2.0]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v0.2.0
[0.1.0]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v0.1.0
