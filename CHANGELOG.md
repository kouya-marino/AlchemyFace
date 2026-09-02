# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

_Nothing yet._

## [1.2.0] — 2026-09-03

A blank window on macOS, which turned out to be Apple's 2010 Tcl/Tk rather than
anything in this code, and the launcher that had been missing since the port.

### Added

- **A startup warning when Tk is too old to draw.** Apple ships Tcl/Tk 8.5.9,
  and any Python built without pointing at something newer links against it. On
  a current macOS it opens a window and paints nothing — not even a plain
  `tk.Label` with a background colour. The app now detects it and names the fix.

  Written to **stderr** as well as to the status bar and a dialog, because the
  entire symptom is that nothing inside the window can be read. A blank window
  with no explanation costs an hour; this was that hour.

- `alchemyface.gui.tkcompat` — `tk_patchlevel()` and `stale_tk_warning()`. The
  version comes from Tcl's `info patchlevel`, not `tkinter.TkVersion`, which is
  a float and so reports 8.5.9 as `8.5` and cannot tell 8.6.0 from 8.6.18. An
  unreadable version counts as fine rather than as old: crying wolf at everyone
  whose Tk will not answer is worse than the thing being guarded against.

### Changed

- `ResizeView._pump` keeps `update_idletasks()`, and now records **why** for
  good. The original called the full `update()`, and the difference had been
  justified only by a segfault on Tk 8.5.9 — implying a newer Tk would make it
  safe. It does not: restoring `update()` on Tk 8.6.18 **deadlocks** the resize
  suite instead of crashing it.

  The hazard was never the Tk version. `update()` processes *every* pending
  event, not just redraws, so calling it inside this loop re-enters whatever
  else is scheduled — the detection worker's polling among it. Old Tk turned
  that into a segfault; new Tk turns it into a hang. Recorded in both the code
  and the differences table so this is not "fixed" back a third time.

- **`main.py` at the repository root**, so `python main.py` launches the
  application from a bare `git clone` with nothing installed — the way the
  application this was ported from is started, and the first thing anyone who
  knows it will try. It puts `src/` on the path itself, and then hands off to
  the `db` command rather than reimplementing it, so it cannot drift from
  `alchemyface db`.

  An already-importable `alchemyface` is left alone: prepending `src/`
  unconditionally would shadow an installed copy with the working tree, which
  is a surprising thing for a launcher to do when the two differ.

- `main.py` is included in the `ruff` and `mypy` targets in CI and in the README
  — an unchecked file at the root is a file that rots — and the build now fails
  if a top-level `main` module reaches the wheel.

### Notes

- 450 tests; 315 run without a display.
- The first version of the wheel check was wrong in a way worth recording: it
  ran `python -c "import main"` from the workspace, where the checkout's own
  `main.py` sits in the working directory and is importable via `sys.path[0]`.
  It reported a leak that did not exist, and would have failed every CI run. It
  now runs from `/tmp`, and the wheel-listing grep was checked against both a
  planted `main.py` and the real `alchemyface/__main__.py`.

## [1.1.1] — 2026-09-02

A regression 1.0.0 introduced with the Build tab's `.onnx` path choosers, found
by a fan-out audit comparing the port tab-by-tab against the application it was
ported from. 1.1.0 was never released — this was found before it was tagged, so
the two ship together.

### Fixed

- **Save refused for good after changing a model path.** Choosing a different
  detector or recognizer drops the loaded recognizer, and `fill_embeddings`
  asked the *non-loading* provider for one — the provider that exists so the
  detection worker never touches Tk. It got `None` and reported
  `no model loaded` for every face, one line after the status bar promised
  `Model changed — N cached embedding(s) will be recomputed.` Retrying never
  helped; the only escape was Open, which discards every name typed so far.
  It now asks the loading provider, which is safe because its one caller is
  the Save button, on the main thread. The original application self-heals here,
  which is how the audit spotted it.
- **A model path typed into the box was displayed but ignored.** Browse drops
  the recognizer itself; typing or pasting a path did not, so the app went on
  using the old weights. `ensure_recognizer` now compares the paths against what
  the live recognizer was built from and rebuilds on a mismatch — the check the
  original had.
- **Re-detect with no model destroyed the annotation it could not rebuild.** It
  cleared the faces and their typed names, then submitted to a worker with no
  model. It now checks first and leaves the work alone.

### Notes

- The provider split itself was right: the original's single loading provider
  reads Tk variables from the worker thread, which is undefined behaviour. The
  defect was one call site wired to the wrong half.
- 417 tests; 290 run without a display. Each of the four new tests was confirmed
  to fail with its own fix reverted — including one first written so weakly it
  passed against the bug, which is worse than no test.

## [1.1.0] — 2026-09-02 (unreleased)

Two bugs found while writing down how to launch the application, and the
`-m` entry point whose absence prompted the question.

### Added

- **`python -m alchemyface`** — the package is now runnable with `-m`, so
  `python -m alchemyface db` works without the console script being on PATH.
  It is deliberately identical to running `alchemyface`: same commands, same
  help. `python -m alchemyface.gui.app` goes straight to the window, which is
  the closest equivalent to the `python main.py` of the application this was
  ported from.
- `models.find_in(spec, directory)` — searches exactly one directory, where
  `find_local` deliberately falls back to `ALCHEMYFACE_MODEL_DIR` and the cache.

### Fixed

- **`alchemyface download-models --model-dir X` wrote nothing to X** if a copy
  of the weights happened to be in the cache. It printed `already present at
  ~/.cache/…` and left X empty, while the flag's help said "Where to write the
  weights". It now treats only X as present, and the help says what it does.
- **`models.resolve(spec, model_dir=X)` ignored X when it had to download**,
  writing to the cache instead. A caller who named a directory got the weights
  somewhere else with nothing said.
- **`models.download()` let a raw `PermissionError` escape** for an unwritable
  destination — the one model failure in that module a caller had to catch
  `OSError` for. It is now a `ModelDownloadError` naming the directory. This
  mattered more once `resolve` started honouring `model_dir`, since an
  unwritable directory became reachable rather than silently bypassed.

### Notes

- 413 tests; 290 run without a display.

## [1.0.0] — 2026-09-02

Parity. Every tab of the application this was ported from now exists here, on
top of the published `alchemyface` library rather than a private wrapper.

### Added

- **Resize tab** and **`alchemyface resize`** — shrink photos until the detector
  can see the face again. Folder or single image on both the source and output
  side, ratio 0.05–5.0, per-file log, and a refusal to write over the source.
- `alchemyface.gui.resize_data`: `resize_one`, `resize_one_outcome`,
  `resize_folder`, `plan_folder`, `default_output_folder`. No Tk, so the whole
  feature is tested without a display.

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

### Fixed — from a tab-by-tab parity audit

A second fan-out audit compared every tab against the original and
adversarially verified each gap through three lenses. 33 candidates, 30
confirmed. All of them are fixed here.

**The Detection score spinbox never existed.** 0.6.0's notes, this changelog,
`versions.md` and the README all described a spinbox in the Build tab.
`App.apply_score_threshold()` was written and correct, but no widget ever called
it, so the threshold was pinned at 0.9 and every document described a control
that was not there. See the corrected 0.6.0 entry below. The widget now exists,
and four tests assert the widget itself rather than the method behind it —
including one that invokes its Tcl `command` callback, because `event_generate`
is not delivered to a withdrawn window and would have passed against a spinbox
wired to nothing.

Data loss and crashes:

- **Save with an empty table overwrote the loaded database with an empty one.**
  A select-all-delete followed by Save destroyed every entry, with no undo.
  Refused now, at the session layer so the CLI is covered too.
- **Resize folder with an empty Source field raised an uncaught `ValueError`**
  out of the button callback: `Path("")` is `PosixPath(".")`, which passes
  `is_dir()`. Refused with a message.
- **An out-of-range resize ratio was silently clamped** and the run proceeded,
  so a mistyped 50 for 0.5 rewrote a folder at a size nobody asked for. Both the
  tab and `alchemyface resize` now refuse and say why.

Work being thrown away:

- Candidates left unticked were discarded by **Add checked** instead of staying
  pending for a second pass.
- One unnamed ticked candidate made **Add checked** add *nothing*; it now adds
  the usable ones and reports what it skipped.
- Opening a folder with no images **wiped the annotations already loaded**; it
  now keeps them and says so.

Refusing to open repairable databases:

- Inspect DB and Edit DB aborted the whole load on any NaN, infinity, or
  odd-sized vector. That is backwards: a database is opened in the inspector
  *because* something is wrong with it, and in the editor to delete the
  offending rows. `PickleStore.load_leniently()` now salvages what it can and
  reports each problem; `load()` stays strict, so nothing enrols against a
  vector that cannot be matched.

Missing controls and feedback:

- The Build tab could not choose the YuNet / SFace `.onnx` files.
- Pending cards were never redrawn after **Add checked** or a load, so consumed
  candidates stayed on screen and pressing the button again did nothing
  silently.
- Save with no database loaded reported an error instead of opening **Save as…**.
- Detecting an image or folder that yielded no faces said nothing at all.
- The Resize log had no run header, was never cleared between runs, and
  single-image mode printed no summary. It now logs each file as it is written.
- The Edit tab lost its summary line, and its Folder/Image path boxes, so a path
  could not be typed or pasted and Process could not be re-run.
- Escape did not cancel an inline Group edit, and opening the preset dropdown
  committed the half-typed value.
- The face-box number lost its filled colour chip and was unreadable on a light
  photo.
- The status bar never reported the current image's detection result.
- A `{name: vector}` database showed a fresh random hex ID on every load instead
  of the entry's index.
- Save as… / Browse… dialogs did not open where the relevant box already
  pointed, and the Resize image filter missed uppercase extensions.

### Notes

- Four tabs, matching the original: Build DB, Edit DB, Resize, Inspect DB.
- `YuNetDetector` now clamps its own score to 0.05–0.99. It previously accepted
  1.5 and -3.0 unchanged, which cv2 takes without complaint and then returns
  garbage for.
- 404 tests; 281 run without a display.

### Deliberate differences from the original

- The Resize log redraws with `update_idletasks()`, where the original called
  the full `update()`. `update()` re-enters the Tk event loop and reliably
  segfaults the GUI suite on the macOS Tk this is developed against — the same
  crash, for the same reason, that a full `update()` caused in the detection
  view. Progress is still visible; a long batch cannot be closed mid-run.
- Names are written trimmed of surrounding whitespace; the original wrote them
  verbatim. A trailing space is invisible in the table and produces a second,
  silently distinct identity.

## [0.6.0] — 2026-09-02

An audit of every documented claim against the code. It found 25 confirmed
discrepancies, several of them real defects rather than prose. Resize moves to
0.7.0.

### Added

- `YuNetDetector.set_score_threshold()` and `App.apply_score_threshold()`, to
  change the detection score without rebuilding the recognizer so cached
  embeddings survive.

  > **Correction (1.0.0).** This entry originally claimed a **Detection-score
  > spinbox** shipped in the Build tab. It did not. The two methods above were
  > written and are correct, but no widget was ever added to call them, so the
  > threshold stayed pinned at 0.9 and this changelog, `versions.md`, `todo.md`
  > and the README all documented a control that did not exist.
  >
  > The same box had already been ticked in `todo.md` in 0.4.0 with nothing
  > behind it; 0.6.0 was supposed to be the remedy and instead repeated the
  > mistake one layer down — implementing the method and documenting the
  > feature. The 1.0.0 parity audit caught it. The widget now exists and is
  > tested as a widget.
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
