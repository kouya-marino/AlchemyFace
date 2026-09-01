# Todo

Actionable work, grouped by the version it belongs to. `versions.md` explains
*why* each version exists; this is the checklist.

Convention: one branch and one pull request per version. Tick items as they
merge, not as they are written.

---

## Now — 0.2.0 · library groundwork

### Setup
- [ ] Copy `onnx/` (37 MB) and `test_images/` (660 KB) from the source app into
      git-ignored `_local/`. Leave the 233 MB zip, the bulk photo folders, the
      69 audio files and the production `.pkl` files where they are.
- [ ] Copy the three sample `.pkl` files into `_local/pkl/` as read-only
      fixtures for schema tests.
- [ ] Branch `feat/v0.2.0-pickle-store`.

### `normalize=False`
- [ ] Add `normalize: bool = True` to `SFaceEmbedder.__init__`.
- [ ] Test: with `normalize=False` the L2 norm is ~10, not 1.
- [ ] Test: cosine similarity between two faces is identical either way —
      this is the property that makes the option safe.
- [ ] Decide and document whether `Recognizer` exposes it. Leaning no: the
      library's own store normalises anyway, so only `.pkl` writers need raw.

### `PickleStore`
- [ ] `PickleSchemaError` in `errors.py`, under `AlchemyFaceError`.
- [ ] `store/pickle.py` implementing `FaceStore`: `add` / `search` / `remove` /
      `__len__`, plus `load(path)` and `save(path)`.
- [ ] IDs renumbered from `"0"` on save, matching the original.
- [ ] Read the back-compatible `{name: vector}` dict form.
- [ ] Raise `PickleSchemaError` with a reason on anything else.
- [ ] `search` must normalise internally so raw stored vectors still rank
      correctly — the store's own cosine, not the caller's problem.
- [ ] Tests: round-trip; dict form; malformed input; empty database;
      mixed dimensions rejected.

### Cross-check against the original
- [ ] Load all three sample `.pkl` files; assert entry counts 53 / 30 / 7.
- [ ] Absorb the real schema variance, measured across all three files:
      `id` is `int` in one and `str` in two; the vector is `(1, 128)` in one and
      `(128,)` in two. Coerce `id` to `str` and flatten the vector before
      validating — a stricter reader would refuse a database the robot loads
      today. Test each variant explicitly.
- [ ] Write a `.pkl` with `PickleStore`, open it in the original app's Inspect
      tab, confirm identical vectors.

### Ship
- [ ] `CHANGELOG.md` entry. Bump version to 0.2.0.
- [ ] PR → review → merge → tag `v0.2.0` → verify on PyPI.

---

## Next — 0.3.0 · GUI shell and Inspect DB

- [ ] Add `Pillow>=10` to base dependencies; note in CHANGELOG that the base
      install now includes the GUI.
- [ ] `src/alchemyface/gui/__init__.py` exporting `App`, imported lazily.
- [ ] **Guard test: `import alchemyface` succeeds with `tkinter` masked out.**
      Without this the D3 constraint breaks silently for library users on
      systems where `python3-tk` is not installed.
- [ ] `alchemyface db` subcommand; import `gui` inside the command body only.
- [ ] `app.py`: notebook, status bar, group presets, lazy `Recognizer` with
      signature caching and cache invalidation when models change.
- [ ] `inspect_view.py`: path entry, Browse/Load, entries table, summary line.
- [ ] `gui` pytest marker; `xvfb-run -a` for GUI tests in `ci.yml`.
- [ ] Confirm `pytest -m "not gui"` still needs no display.

---

## Later

### 0.4.0 · Build DB
- [ ] Port `annotation_view.py`, translating `(N, 15)` rows to `Face`.
- [ ] Keep the concurrency design intact: priority queue, generation counters,
      `queue.Queue` hand-off to the Tk thread, BGR LRU cache, clean shutdown.
- [ ] Keep the responsiveness work: 80 ms debounced canvas resize, cached
      `PhotoImage` keyed on (path, canvas size), 50 ms completion polling.
- [ ] Detection-score spinbox wired to the live detector.
- [ ] Save flow with name validation and on-demand embedding fill-in.

### 0.5.0 · Edit DB
- [ ] Port `edit_db_view.py` onto `PickleStore` — drop the duplicated
      `_normalize_db`.
- [ ] Unsaved-changes tracking and close confirmation.

### 0.6.0 · Resize
- [ ] Port `resize_util.py` and `resize_view.py`.
- [ ] `alchemyface resize` CLI subcommand replacing the hardcoded script.

### 1.0.0 · Parity and polish
- [ ] Tab-by-tab audit against the original.
- [ ] README covering library and app; screenshots.
- [ ] Coverage back above 80% with the GUI included.

---

## Open questions

- [ ] Entry point: `alchemyface db` subcommand, or a separate `alchemyface-db`
      console script? Leaning subcommand — one binary.
- [ ] macOS ships Tk 8.5.9. Every widget used works, but 8.6 fixes real
      rendering bugs. Document, or require 8.6?
- [ ] The robot matches at cosine `0.32`; the library's default is `0.363`.
      Should the app surface the robot's threshold anywhere, or is that purely
      the consumer's business?

## Carried over from the library

- [ ] Backport `contents: read` to the publish workflow of AlchemyCV,
      AlchemyAnnotate, AlchemyDetect and AlchemyCloud. Their publish jobs work
      only because those repos are public.
- [ ] 344 MB of packages left in Homebrew's Python 3.14 by the first Poetry
      mishap. Inert, but worth removing.
