# AlchemyFace — Face DB Builder

**Date:** 2026-09-01
**Status:** design approved in chat; awaiting spec review
**Extends:** `2026-08-28-alchemyface-design.md` (the library this builds on)
**Ports from:** `Arithmer_Data/Face_Reco_App` — a working Tkinter app, ~2,900 lines

## 1. Purpose

AlchemyFace 0.1.0 is a library. This grows it into a library **and** a desktop
application: a Tkinter GUI that turns folders of photos into a `.pkl` face
database consumed by a Unitree G1 robot's `face_recognition_controller`.

The app already exists and works. This is a faithful port into AlchemyFace,
not a redesign — with one substitution: the app's own `face_pipeline.py`
(59 lines wrapping YuNet + SFace) is replaced by the library that already
does exactly that.

## 2. Goals

- Every behaviour of the existing app, preserved.
- The app consumes `alchemyface`'s own `Recognizer` / `Detector` / `Embedder`.
- `.pkl` output stays byte-compatible with databases the robot already loads.
- `import alchemyface` keeps working on systems without `tkinter` installed.
- No personal data reaches the repository, the wheel, or the sdist.

## 3. Non-goals

- Porting to PyQt5. The other four Alchemy apps use Qt; this one stays Tkinter.
  2,400 lines of working, tested GUI is not worth rewriting for consistency.
- Changing the `.pkl` schema, or the robot's matching behaviour.
- Live camera recognition. The app enrols from photos; the library already has
  `VideoSource` for anyone who wants frames.
- Reproducing the app's bulk photo corpus (233 MB zip, 115 photos, 69 mp3s).

## 4. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | AlchemyFace evolves; no new repository | One project, one history, one PyPI name. The library and its first real consumer stay together. |
| D2 | GUI ships in the **base install**, not an extra | Explicit instruction. Adds one dependency, Pillow. `tkinter` is stdlib, so nothing else. |
| D3 | **`alchemyface/__init__.py` never imports the GUI** | On Debian/Ubuntu `tkinter` is a separate apt package. Importing it at package level would break `import alchemyface` for library users — a regression on 0.1.0's API. Only the console-script entry point imports `gui`. |
| D4 | Tkinter retained | See non-goals. Verified: every `ttk` widget the app uses works on macOS's Tk 8.5.9. |
| D5 | `SFaceEmbedder(normalize=False)` added in 0.2.0 | The robot's schema stores raw SFace output (L2 ≈ 10). Cosine is scale-invariant so matching is identical either way — but raw keeps new databases byte-comparable with existing ones and keeps the `L2 norm` column diagnostic. |
| D6 | `PickleStore` implements `FaceStore` | The robot's `list[(id, name, group, vec)]` format becomes a first-class backend, not GUI-private code. Useful to library users independently. |
| D7 | Six versions, one PR each | Each version is independently shippable and leaves the app working. Reviewed and merged before the next begins. |
| D8 | Personal data quarantined in `_local/` | Same control as the library: git-ignored, pruned from the sdist, and a CI step that fails the build if any of it reaches a distribution. |

## 5. Architecture

The library is untouched. Two things are added beside it.

```
src/alchemyface/
├── __init__.py           unchanged — imports NO gui module (D3)
├── types.py              unchanged
├── errors.py             + PickleSchemaError
├── models.py             unchanged
├── pipeline.py           unchanged
├── capture.py            unchanged
├── cli.py                + `db` subcommand launching the GUI
├── detection/            unchanged
├── embedding/
│   └── sface.py          + normalize=False option            (0.2.0)
├── store/
│   ├── memory.py         unchanged
│   └── pickle.py         NEW  robot .pkl schema as FaceStore (0.2.0)
└── gui/                  NEW
    ├── __init__.py       exports App only; imported lazily
    ├── app.py            main window, notebook, save flow    (0.3.0)
    ├── inspect_view.py   Inspect DB tab                      (0.3.0)
    ├── annotation_view.py Build DB tab                       (0.4.0)
    ├── edit_db_view.py   Edit DB tab                         (0.5.0)
    ├── resize_view.py    Resize tab                          (0.6.0)
    └── resize_util.py    shared resize helper                (0.6.0)
```

The original keeps the Inspect tab inline in `app.py`; it moves to its own
`inspect_view.py` so `app.py` stays a shell that wires tabs together rather
than a shell that also happens to contain a feature.

### 5.1 How the app talks to the library

The app keeps the original's dependency-injection shape, which is already
sound: each view receives callables rather than reaching for globals.

```python
AnnotationView(
    parent,
    recognizer_provider=self._get_or_load_recognizer,   # was pipeline_provider
    group_presets_provider=lambda: list(self._group_presets),
    on_status=self._set_status,
    on_preset_added=self._add_preset,
)
```

`FacePipeline.detect()` → `Recognizer.detect()` returning `list[Face]`
instead of an `(N, 15)` array. `FacePipeline.embed(bgr, face_row)` →
`Recognizer.embed(image, face)`. The `Face` dataclass replaces raw rows, so
the GUI stops indexing `bbox[0..3]` into a numpy row and reads
`face.bbox` instead.

### 5.2 The `.pkl` schema

```python
[
    ("0", "Alice", "staff",   np.ndarray(shape=(128,), dtype=float32)),
    ("1", "Bob",   "visitor", np.ndarray(shape=(128,), dtype=float32)),
]
```

`id` is a string counting from `"0"`. `group` is free-form and may be empty.
The vector is **raw** `cv2.FaceRecognizerSF.feature()` output, not
normalised — the robot normalises inside its own cosine helper, at threshold
`0.32`.

`PickleStore` reads both this and the back-compatible `{name: vector}` dict
form. It raises `PickleSchemaError` on anything else rather than returning
`None`, so callers get a reason instead of a bare failure.

**The schema is looser in practice than documented.** Measured across the three
production databases:

| File | Entries | `id` type | Vector shape | L2 norm |
|---|---|---|---|---|
| `face_data_26_08_2025_paloma.pkl` | 53 | **`int`** | **`(1, 128)`** | 13.58 |
| `face_db_20260511_check.pkl` | 30 | `str` | `(128,)` | 11.58 |
| `face_db_test.pkl` | 7 | `str` | `(128,)` | 13.23 |

The original app absorbs both variations silently, via `str(_id)` and
`.flatten()`. `PickleStore` must do the same deliberately: coerce `id` to
`str`, flatten any vector to one dimension, and only then reject. A stricter
reader would refuse a database the robot loads today.

All three store raw vectors (L2 between 11.5 and 13.6), confirming D5.

## 6. Version roadmap

Detail lives in `versions.md`; the summary:

| Version | Ships | Size |
|---|---|---|
| 0.2.0 | `normalize=False`, `PickleStore`, `PickleSchemaError` | library only, no GUI |
| 0.3.0 | GUI shell, Inspect DB tab, `alchemyface db` entry point, Pillow dep | small vertical slice |
| 0.4.0 | Build DB tab — 3-pane annotation view, threaded detect worker, save | largest |
| 0.5.0 | Edit DB tab | large |
| 0.6.0 | Resize tab + `alchemyface resize` CLI | self-contained |
| 1.0.0 | Parity audit, docs, README rewrite | polish |

0.3.0 deliberately carries the smallest possible GUI so that packaging, the
entry point, and headless CI are proven before the 816-line annotation view
lands.

## 7. Workflow

Every version is a branch, a pull request, a review, then a merge.

```
git switch -c feat/v0.2.0-pickle-store
… implement, TDD …
gh pr create --fill
# CI: identity guard · lint · mypy · py3.10-3.12 · leak check
# review, address findings
gh pr merge --squash
git tag v0.2.0 && git push origin v0.2.0     # publish.yml → PyPI
```

`ci.yml` already runs on `pull_request` to `main`, so nothing new is needed
for the PR gate.

## 8. Testing

| Layer | How | Marker |
|---|---|---|
| Library additions | Real vectors, temp files. No display, no models. | none |
| `PickleStore` | Round-trip both schema forms; malformed input raises. | none |
| GUI construction | Instantiate widgets against a real Tk root. | `gui` |
| GUI behaviour | Drive callbacks directly; assert on model state, not pixels. | `gui` |
| Real weights | As today. | `models` |

**`tkinter` cannot open a display on a bare CI runner.** GUI tests run under
`xvfb-run -a`, as `AlchemyCloud/ci.yml` already does for Qt. The existing 88
library tests stay display-free and fast; `pytest -m "not gui"` must remain
the quick path.

A test asserting that `import alchemyface` succeeds with `tkinter` masked out
enforces D3, because that regression is silent otherwise.

## 9. Personal data

The source app holds far more than the library did:

| What | Amount |
|---|---|
| `.pkl` databases | 90 entries — names plus groups including apparent client names |
| `audioperson_*_jp/` | 69 mp3s named after real people |
| photo folders | 115 images |
| `face_images.zip` | 233 MB |

Copied into git-ignored `_local/`: the ONNX models (37 MB, public OpenCV Zoo
weights) and `test_images/` (660 KB, 6 images) so the suite is meaningful.
Left in the original directory: the zip, the bulk photo folders, the audio,
and the production `.pkl` files.

Controls are the library's, unchanged: `_local/` git-ignored, `MANIFEST.in`
prunes it, and a CI step fails the build if a distribution ever contains it.

## 10. Open questions

- **Entry point name.** `alchemyface db` as a CLI subcommand, or a separate
  `alchemyface-db` console script? A subcommand keeps one binary; a separate
  script is more discoverable in a launcher. Leaning subcommand.
- **Tk 8.5 vs 8.6.** macOS ships 8.5.9. Every widget used works, but 8.6 fixes
  real rendering bugs. Worth a documented note rather than a code change.
