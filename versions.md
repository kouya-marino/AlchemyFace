# Versions

A curated overview of where AlchemyFace has been and where it is going.
`CHANGELOG.md` is the detailed per-release log; this file is the map.

Every version below is a branch → pull request → review → merge → tag.
Each one leaves the project working and independently publishable.

**Every version that changes what a user can do updates `README.md` in the same
pull request.** The README is the PyPI project page, so deferring it means
publishing a description that is wrong. 0.2.0 through 0.4.0 shipped with a
README still describing a library-only package — it had been scheduled for
1.0.0, which was a mistake.

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
that app with its vectors intact. 130 tests at this tag, 91% coverage.

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
database loads in the tab. 164 tests at this tag, 95% coverage with the weights
present.

### 0.4.0 — Build DB · 2026-09-01

A folder of photos becomes a database the robot can load.

- Three panes: image sidebar with status glyphs, canvas with numbered face
  boxes, one panel per face (thumbnail · Include · Name · Group).
- Background detection with priorities, whole-folder prefetch, and generation
  counters so opening a different folder abandons the old one.
- Click-to-select on the canvas; excluded faces draw dashed. Re-detect asks
  before discarding edits.
- Detection-score spinbox (0.10–0.99, default 0.9), applied to the live
  detector. *Shipped late, in 0.5.1: it was dropped from 0.4.0's scope while
  its checkbox was ticked, which an audit caught.*
- Save writes included, named faces as a robot-format `.pkl` with raw vectors
  and ids renumbered from `"0"`.
- `annotation_data` and `detect_worker` hold the logic and the threading, both
  free of Tk, so 207 of the 242 tests at this tag need no display.

**Verified** on the real models and real photos: 6 images, 7 faces, raw L2 norms
between 9.8 and 13.9.

> **Correction.** This entry originally claimed one photo found no face at all,
> and offered it as evidence for the Resize tab. That was wrong. Detection had
> failed because no model was loaded, and the interface displayed the failure as
> "no face detected". The photo detects reliably at confidence 0.953. Fixed in
> 0.5.0, where a failure has its own status and can never be mistaken for a
> finding about the image.

### 0.5.0 — Edit DB · 2026-09-01

- Open an existing database; remove entries, edit groups inline, add faces from a
  folder or a single image, save over the loaded path or elsewhere.
- Unsaved changes marked in the frame title and confirmed on close.
- `EditSession` holds entries, candidates and dirty state, tested display-free.
  Candidates stay apart from entries until merged.
- `EntryStatus.FAILED`: a detection failure is no longer shown as "no face
  detected". That conflation had already produced a false claim in this file.

**Verified** against a real 30-entry production database: removed two entries,
changed a group, detected 7 faces from real photos, merged them, and saved 35
entries with raw magnitudes 9.81–14.59 intact. 313 tests.

### 0.6.0 — documentation audit, and the detection score · 2026-09-02

A fan-out audit checked every documented claim against the code, adversarially
verifying each discrepancy. 25 confirmed; several were defects, not prose.

- `App.apply_score_threshold()` and `YuNetDetector.set_score_threshold()`,
  applied to the live detector so cached embeddings survive.

  > **Correction (1.0.0).** This bullet originally said a **Detection-score
  > spinbox** shipped here. It did not — the methods were written, but no widget
  > ever called them, so the threshold stayed pinned at 0.9. The box had already
  > been ticked in 0.4.0 with nothing behind it; 0.6.0 was meant to be the fix
  > and repeated the mistake one layer down. The 1.0.0 parity audit caught it.
- Face-card Tk variables no longer leak — 480 over 80 re-renders, now 0. The
  release mechanism existed; the line registering variables with it never did.
- Closing now asks before discarding Build-tab annotations.
- `requirements.txt` had omitted Pillow; a test now enforces both requirements
  files against `pyproject.toml`.
- `publish.yml` claimed CI's checks while skipping every GUI test; it now runs
  them under `xvfb` with the coverage gate.
- Corrected: the README's install size (~160 MB, not "a few megabytes"), the
  spec's personal-data inventory, its L2 range, its per-entry id variation, four
  test counts, and three docstrings that described behaviour the code lacked.

**Verified:** every fix has a test, and each was checked to fail with its fix
reverted. 326 tests collected; 260 run without a display.

### 1.0.0 — parity · 2026-09-02

Every tab of the original application, on top of the published library.

- **Resize tab** and **`alchemyface resize`**: folder or single image, ratio
  0.05–5.0, per-file log, source-overwrite refused.
- Four tabs, matching the original: Build DB, Edit DB, Resize, Inspect DB.

**Verified:** the tab's own justification, measured rather than inherited — a
constructed selfie at 426x546 with the face filling ~93% of the frame detects 0
faces, and 1 after resizing to 0.5. Detection is not monotonic in the ratio, so
the test searches several.

**Parity audit.** A fan-out audit compared the port tab-by-tab against the
original, each gap adversarially verified through three lenses. 33 candidates,
30 confirmed, all fixed here. The headline finding was that the **Detection
score spinbox never existed** — the method behind it was written and correct,
but no widget called it, and four documents described the control anyway. The
0.6.0 entry above is corrected. Also fixed: Save with an empty table silently
destroyed the loaded database; Resize with an empty source folder crashed;
an out-of-range ratio was silently clamped and the run proceeded; unticked
candidates were discarded; one unnamed candidate blocked a whole batch; opening
an empty folder wiped loaded annotations; and both Inspect and Edit refused to
open the malformed databases they exist to diagnose and repair. The full list is
in `CHANGELOG.md`.

**The lesson, taken twice now.** 0.4.0 ticked a box for a control that did not
exist. 0.6.0 was meant to remedy that and instead implemented the *method* and
documented the *feature*, leaving the same hole one layer down. What catches
this is not more careful prose but tests that assert the artefact a user touches
— so the new tests query the widget tree and invoke the widget's own Tcl
callback, and every fix in this release was confirmed to fail with itself
reverted. 404 tests; 281 run without a display.

### 1.2.0 — a blank window, and `python main.py` · 2026-09-03

A GUI that opened and showed nothing, which turned out to be Apple's 2010
Tcl/Tk rather than anything in this code.

**The window was blank because the Python was linked against Tcl/Tk 8.5.9.**
Every Python on the machine was: pyenv builds it without tcltk flags unless
told otherwise, and `/usr/bin/python3` uses the system Tk too. On macOS 26 it
opens a window and paints nothing at all — not even a plain `tk.Label` with a
background colour. Proven by installing `python-tk@3.10`, whose Tk 8.6.18 drew
correctly, then rebuilding pyenv 3.10.21 against `tcl-tk@8`. The app now warns
at startup and names the fix, on **stderr** as well as in the window, since a
blank window is exactly the case where nothing in the window can be read.

**Three things worth recording about how this went.** My evidence that the
window worked was a process that did not exit — which is entirely compatible
with a blank window, and the 123 GUI tests cannot catch it either because they
call `withdraw()` and so never test painting. Then I proposed restoring the
original's full `update()` on the theory that Tk 8.6 made it safe; it does not —
it deadlocks the resize suite rather than segfaulting it, because `update()`
re-enters every pending callback and not just redraws. The differences table now
says so, so it is not "fixed" back a third time. And 3.10.21 rather than a
rebuilt 3.10.6, because pyenv-virtualenv stores environments inside the version
directory: reinstalling 3.10.6 would have deleted `acloud_env`, which belongs to
another project.

The three documented ways to launch all required the package to be installed.
A bare checkout had none, which is exactly the case the original covers with a
nine-line `main.py`. Now so does this. It bootstraps `src/` onto the path and
delegates to the `db` command, so it cannot drift from `alchemyface db`; an
already-importable copy is left alone rather than shadowed by the working tree.

`main.py` joined the ruff and mypy targets, and the build now fails if a
top-level `main` reaches the wheel. That check was wrong on the first attempt —
it ran `import main` from the workspace, where the checkout's own `main.py` is
importable via `sys.path[0]`, so it reported a leak that did not exist and would
have failed every run. Caught by testing the check against a planted `main.py`
and the real `alchemyface/__main__.py` before trusting it. 430 tests; 301 run
without a display.

---


### 1.1.1 — a 1.0.0 regression in Save · 2026-09-02

Found by a fan-out audit comparing the port tab-by-tab against the original,
with each claimed regression verified by execution on both trees.

- **Save refused for good after a model-path change** — the feature 1.0.0 added.
  Choosing a different `.onnx` drops the recognizer, and `fill_embeddings` asked
  the non-loading provider (the one that exists so the worker thread never
  touches Tk), got `None`, and reported "no model loaded" for every face — one
  line after promising a recompute. Retrying never helped. The original
  self-heals, which is how it was caught.
- A model path **typed** rather than browsed was displayed but ignored.
- **Re-detect with no model wiped the names** it then could not rebuild.

The provider split was sound — the original reads Tk variables from the worker
thread, which is undefined behaviour. One call site was wired to the wrong half.

One of the four new tests initially passed against the bug, because it left a
recognizer installed; a test that cannot fail is worse than none, so it was
rewritten until reverting the fix broke it. 417 tests; 290 run without a display.

---


### 1.1.0 — the `-m` entry point, and two model-path bugs · 2026-09-02

Prompted by a plain question — how do you launch this? The original was started
with `python main.py`; writing down the equivalent here surfaced that the
package could not be run with `-m` at all, and two bugs in the model paths.

- **`python -m alchemyface`** now works, identical to the console script.
  `python -m alchemyface.gui.app` goes straight to the window.
- **`download-models --model-dir X` wrote nothing to X** when the cache already
  held a copy — it said `already present at ~/.cache/…` and left X empty, while
  the flag promised "Where to write the weights".
- **`resolve(spec, model_dir=X)` ignored X when downloading**, writing to the
  cache instead.
- **`download()` leaked a raw `PermissionError`** for an unwritable destination,
  the one model failure in that module needing an `OSError` catch. Found by
  asking what the `resolve` fix newly made reachable, rather than after the
  fact. 413 tests; 290 run without a display.

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
| Resize progress | full `update()` per file | `update_idletasks()` — `update()` processes *every* pending event, not just redraws, so inside the loop it re-enters the detection worker's polling: it segfaults on Tk 8.5.9 and deadlocks on 8.6.18. Progress still draws; a batch cannot be cancelled mid-run |
| Names | written verbatim | trimmed of surrounding whitespace, which is invisible in the table but makes a second identity |
| Detection score floor | clamped in the GUI only | clamped in `YuNetDetector`, so the library and CLI get it too |
| Malformed `.pkl` | displayed | displayed, with each problem reported; still refused for enrolment |
