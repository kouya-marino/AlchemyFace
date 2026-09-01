# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

_Nothing yet._

## [1.0.0] — 2026-09-02

Parity. Every tab of the application this was ported from now exists here, on
top of the published `alchemyface` library rather than a private wrapper.

### Added

- **Resize tab** and **`alchemyface resize`** — shrink photos until the detector
  can see the face again. Folder or single image on both the source and output
  side, ratio 0.05–5.0, per-file log, and a refusal to write over the source.
- `alchemyface.gui.resize_data`: `resize_one`, `resize_folder`, `plan_folder`,
  `clamp_ratio`, `default_output_folder`. No Tk, so the whole feature is tested
  without a display.

### Why the Resize tab exists — measured, not inherited

YuNet's largest anchors miss a face filling most of the frame. Constructed from a
real photo:

```
selfie 426x546, face ~93% of frame  ->  0 detected
resized to 0.5  213x273             ->  1 detected
```

Detection is **not monotonic** in the ratio: 0.25 finds nothing while 0.15 works
again, because it depends on the face matching an anchor scale. A test builds
this case and fails if resizing ever stops helping.

This matters because 0.4.0 originally justified the tab with a claim that a
particular test photo could not be detected. That was false — detection had
merely failed — and the claim was corrected in 0.5.0. The justification is now
reproducible rather than asserted.

### Notes

- Four tabs, matching the original: Build DB, Edit DB, Resize, Inspect DB.
- 377 tests; 279 run without a display.

## [0.6.0] — 2026-09-02

An audit of every documented claim against the code. It found 25 confirmed
discrepancies, several of them real defects rather than prose. Resize moves to
0.7.0.

### Added

- **Detection-score control** in the Build tab — a spinbox, 0.10–0.99, default
  0.9, applied to the live detector rather than by rebuilding it, so cached
  embeddings survive. `YuNetDetector.set_score_threshold()` and
  `App.apply_score_threshold()` support it.

  This was **ticked as done in `todo.md` since 0.4.0 while no such control
  existed**, and the bullet describing it was deleted from `versions.md` in the
  same commit that ticked the box. Implementing it was the honest remedy.
- `tests/unit/test_declared_dependencies.py` — asserts `requirements*.txt`
  matches `pyproject.toml` and the README badge matches the declared version.
  Comments claiming files are "kept in step" are not mechanisms.

### Fixed

- **Face-card Tk variables leaked.** `_release_face_vars` existed to detach
  traces, but nothing ever registered a variable with it, so it looped zero
  times. Measured 480 leaked Tcl variables over 80 re-renders; now 0. The
  docstring had confidently described protection that did not exist — and an
  earlier changelog credited this mechanism with removing a warning that the
  `update_idletasks` change in the same commit actually fixed.
- **Closing discarded Build-tab work silently.** `has_unsaved_changes` consulted
  only the Edit tab, so typed names, groups and include ticks vanished with no
  prompt.
- **`requirements.txt` omitted `Pillow`**, so an environment pinned from it got a
  working library and a GUI that crashed on import — while the file's own comment
  claimed it was kept in step with `pyproject.toml`.
- **`PickleStore.load` enumerated the exceptions it expected**, so a pickle whose
  `GLOBAL` opcode names an uninstalled module raised `ModuleNotFoundError` out of
  the library instead of `PickleSchemaError`.
- **`publish.yml` claimed "the same checks as CI"** while installing neither
  `python3-tk` nor `xvfb`, so all GUI tests skipped silently on a release, and it
  had no coverage gate. It now runs them under `xvfb` with the gate.
- **`python -m build` was documented but not installable** — `build` and `twine`
  were missing from the `dev` extra the README tells you to install.

### Corrected documentation

- `errors.py` claimed every exception descends from `AlchemyFaceError`. It does
  not: `cv2.error` escapes when a file on disk is not loadable ONNX, and
  `ValueError` / `RuntimeError` escape from argument and camera failures.
- The README's "a working install is a few megabytes" was wrong by two orders of
  magnitude — the runtime dependencies are ~160 MB installed, mostly OpenCV.
- The spec understated the personal-data inventory: it said the production `.pkl`
  databases were left in the original directory. They are in `_local/`. Being
  wrong there matters more than anywhere else.
- The spec's "L2 between 11.5 and 13.6" was each file's *first* entry, not the
  range. Across all 90 entries it is 9.75–14.59.
- The design notes described `id` type variation as per-file; in one production
  database it is per-entry — 46 `int` and 7 `str` in the same list, which is why
  every id is coerced individually.
- Test counts in four places conflated "collected", "passing" and "run without a
  display". Corrected and disambiguated.
- `detect_worker.submit` claimed re-submission "does nothing"; only *background*
  re-submissions are dropped, by design.
- `cli.py` claimed a module-scope tkinter import would break `import
  alchemyface`; it would not, since the package never imports `cli`.
- The Inspect DB table has seven columns, not the six the README listed, and the
  `alchemyface db` transcript omitted two of its five printed lines.
- `app.py`'s module docstring still said the Edit tab was forthcoming.

### Notes

- Every fix carries a test, and each new test was verified to **fail** with its
  fix reverted — the discipline that would have caught the no-op above.
- 326 tests collected; 260 run without a display.

## [0.5.0] — 2026-09-01

The Edit DB tab, and a correction.

### Added

- **Edit DB tab** — open an existing database, remove entries, edit groups
  inline, add faces from a folder or a single image, save over the loaded path or
  elsewhere. The frame title carries ` *` while unsaved, and closing the window
  asks before discarding.
- `alchemyface.gui.edit_data.EditSession` — entries, pending additions and dirty
  state as one small state machine, tested without a display. Candidates are held
  apart from entries deliberately: a detected face is a *candidate* until merged,
  which is why adding one does not mark the session dirty and merging does.
- `EntryStatus.FAILED`, distinct from `NO_FACE`.

### Fixed

- **A detection failure was displayed as "no face detected".** A broken model, an
  unreadable file or a missing recognizer all produced the same `⚠ no face` row
  as a photograph genuinely containing none. Failures now have their own status,
  glyph, colour and message.

  This mattered: the 0.4.0 entry in `versions.md` claimed one test photo found no
  face and offered it as evidence for the Resize tab. That was wrong. Detection
  had failed because no model was loaded; the photo detects reliably at
  confidence 0.953. `versions.md` carries a correction.
- **`AnnotationView.load_folder` refuses without a model** instead of starting
  and filling the sidebar with failures dressed as findings.
- **The Edit tab could not detect anything until the Build tab had been used.**
  It was wired to the worker-safe getter, which never loads. It now gets the
  loading provider, since it works on the main thread. No fake-recognizer test
  could have shown this — only running it for real.
- **`EditSession.save` assumed 128 dimensions.** Every real database is 128, so
  it would have worked by luck and failed on anything else.

### Notes

- 313 tests collected; 252 of them run without a display. (An earlier draft of
  this line said "312 tests, 240 of them needing no display" — 240 is the count
  under CI's fast-path filter, which also excludes the `models` and `camera`
  markers, not the display-free count.)

## [0.4.1] — 2026-09-01

Documentation only. No code changes.

### Fixed

- **`README.md` described a library-only package.** It had not been touched
  since before 0.2.0, so three releases went to PyPI — where the README *is* the
  project page — with no mention of the desktop application, `PickleStore`,
  `normalize=False`, or the `.pkl` schema. It also claimed the dependencies were
  "OpenCV, NumPy, Typer. Nothing else", which stopped being true when Pillow was
  added in 0.3.0.
- The rewrite covers the library and the app, states the real dependencies,
  explains that the library never imports `tkinter` while the GUI needs it, and
  documents raw versus unit embeddings. Every code example in it was executed
  against the real models before publishing.
- `versions.md` and `todo.md` now require a README update in each version's own
  pull request. Scheduling it for 1.0.0 was the mistake that caused this.

## [0.4.0] — 2026-09-01

The Build DB tab: a folder of photos becomes a database the robot can load.

### Added

- **Build DB tab** — three panes. Sidebar of images with status glyphs, the
  current image with numbered face boxes, and one panel per face carrying a
  thumbnail, Include, Name and Group.
- Background detection with priorities: the image on screen is detected first,
  the rest of the folder follows, and the sidebar fills in without navigating.
- Click a box on the canvas to select that face. Excluded faces draw dashed.
- Re-detect, which asks before discarding edits and does not nag otherwise.
- Group presets shared across tabs; typing a new group adds it.
- Save writes every included, named face as a robot-format `.pkl`, computing
  any embedding not already cached and renumbering ids from `"0"`.
- `alchemyface.gui.annotation_data` — models, sidebar state, canvas geometry and
  save validation as pure functions.
- `alchemyface.gui.detect_worker` — the threading, knowing nothing of Tk or
  OpenCV, so it is tested with a fake detector.

### Notes

- Embeddings are **not** normalised. The robot's schema stores raw SFace output,
  and `SFaceEmbedder(normalize=False)` from 0.2.0 supplies it. Verified on real
  photos: L2 norms between 9.8 and 13.9.
- Installing a different recognizer discards cached embeddings. Keeping them
  would silently mix vectors from two models into one database, which no
  threshold can then separate.
- The detection worker records what it has **completed**, not only what is in
  flight. A caller cannot distinguish "still queued" from "finished but not yet
  collected", and a submission landing in that window used to detect the same
  image twice. `forget()` lets Re-detect ask again deliberately.
- Tk variables bound to the face panels are held by the view. Left to garbage
  collection they were destroyed after the interpreter had gone, raising
  "main thread is not in main loop"; holding them also cut the GUI suite from
  29s to 8s.
- `Image.LANCZOS` replaced with `Image.Resampling.LANCZOS`, the Pillow 10 name.
- Reporting is injected rather than hard-coded. Views took `tkinter.messagebox`
  directly, so testing a failure path meant constructing a real modal dialog.
  `Reporter` / `DialogReporter` / `RecordingReporter` replace it, and no test
  opens a window.
- The recognizer *provider* handed to the Build tab is called from its worker
  thread, so it no longer touches Tk. It previously set a status variable there,
  which is undefined behaviour off the main thread. Loading moved to
  `ensure_recognizer`, called on the main thread when a folder is opened.
- `AnnotationView.shutdown()` waits for its worker. Signalling and walking away
  left a thread holding references to Tk objects that were about to be
  destroyed.
- 241 tests, 195 of them needing no display. The GUI suite was segfaulting
  intermittently on macOS system Tk 8.5.9 — roughly one run in ten — because the
  test settle-helper both drained results and called a full `update`, rendering
  every result twice and churning `ImageTk.PhotoImage`. It now drains once and
  flushes idle tasks: 20 consecutive clean runs. CI uses Tk 8.6, where the
  problem does not arise.

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
- 164 tests collected at this tag, 163 passing; 95% coverage locally with the
  weights present, 91% without.

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
- 130 tests collected at this tag, 129 passing; coverage 91%.

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

[Unreleased]: https://github.com/kouya-marino/AlchemyFace/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v1.0.0
[0.6.0]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v0.6.0
[0.5.0]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v0.5.0
[0.4.1]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v0.4.1
[0.4.0]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v0.4.0
[0.3.0]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v0.3.0
[0.2.0]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v0.2.0
[0.1.0]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v0.1.0
