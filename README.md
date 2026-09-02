# AlchemyFace

[![PyPI](https://img.shields.io/badge/PyPI-v1.1.1-blue.svg)](https://pypi.org/project/alchemyface/)
[![CI](https://github.com/kouya-marino/AlchemyFace/actions/workflows/ci.yml/badge.svg)](https://github.com/kouya-marino/AlchemyFace/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Face recognition built on [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) —
**a typed Python library and a desktop application for building face databases.**

```bash
pip install alchemyface
alchemyface db          # the Face DB Builder  (or: python -m alchemyface db)
```

## Why

Most Python face-recognition packages pull in dlib, PyTorch or TensorFlow.
AlchemyFace uses two small ONNX models through OpenCV's own DNN runtime, so the
recognition stack itself is small: the wheel is about 30 KB, and the weights are
37 MB fetched once on first use.

Being straight about the total: the dependencies are not small. OpenCV, NumPy and
Pillow come to roughly **160 MB installed** — most of it OpenCV. That is well
under a PyTorch or TensorFlow install, but it is not "a few megabytes", as this
paragraph previously claimed.

---

# The library

```python
import cv2
from alchemyface import Recognizer

r = Recognizer()                       # weights download once, then cached

r.enroll("prashant", cv2.imread("me.jpg"))
r.enroll("alice",    cv2.imread("alice.jpg"))

for recognition in r.identify(cv2.imread("group.jpg")):
    face, match = recognition.face, recognition.match
    if match:
        print(f"{match.label} at {face.bbox} ({match.score:.2f})")
    else:
        print(f"unknown face at {face.bbox}")
```

`identify` returns one `Recognition` per detected face. `match` is `None` when
nothing clears the threshold — the library never invents a label.

### Galleries

`Recognizer` is a facade over three protocols — `Detector`, `Embedder` and
`FaceStore` — so any conforming object can be substituted.

| Store | |
|---|---|
| `InMemoryStore` | numpy matrix, unit vectors, `.npz` save/load. The default. |
| `PickleStore` | the Unitree G1 robot's `list[(id, name, group, vector)]` pickle. Stores vectors **verbatim**. |

```python
from alchemyface.store import PickleStore

store = PickleStore()
store.load("face_db.pkl")
print(len(store), store.dim)
for entry in store.entries():
    print(entry.label, entry.group, entry.vector.shape)
```

### Raw versus unit embeddings

`SFaceEmbedder` returns unit-length vectors by default, because the `Embedder`
protocol promises it and the rest of the library relies on it. Pass
`normalize=False` for SFace's raw output, whose L2 norm is around 10:

```python
from alchemyface.embedding import SFaceEmbedder

SFaceEmbedder().embed(image, face)                    # L2 == 1
SFaceEmbedder(normalize=False).embed(image, face)     # L2 ≈ 10, raw
```

Cosine similarity is scale-invariant, so **matching is identical either way**.
What differs is what lands on disk: the robot's schema stores raw values, and
keeping them means a database written here stays comparable with one written by
anything else, and the `L2 norm` column remains a useful diagnostic rather than
reading `1.0000` for every entry.

### Live video

```python
from alchemyface import Recognizer
from alchemyface.capture import VideoSource

r = Recognizer()
with VideoSource(0, width=1280, height=720) as camera:
    for frame in camera.frames():
        for recognition in r.identify(frame):
            print(recognition.match.label if recognition.match else "unknown")
```

---

# The application

From a checkout, with nothing installed:

```bash
python main.py
```

Or, once the package is installed:

```bash
alchemyface db                      # the console script
python -m alchemyface db            # same thing, no reliance on PATH
python -m alchemyface.gui.app       # straight to the window
```

`main.py` puts `src/` on the path itself and then hands off to the `db` command,
so it cannot drift from `alchemyface db` — same behaviour, same message when
tkinter is missing, same exit code. It is deliberately **not** packaged: a
top-level `main` module in the wheel would claim a name that is not ours, and CI
fails the build if one appears.

A Tkinter desktop app that turns folders of photos into a `.pkl` face database.

### Build DB

Three panes: image sidebar, the current image with numbered face boxes, and one
panel per face.

1. Choose an input folder and click **Open**. Every image is detected in the
   background, the one on screen first, so the sidebar fills in as you work.
   Opening a folder with no images in it keeps what you already have loaded
   rather than clearing it.
2. Each detected face becomes a numbered box on the canvas and a row on the
   right — thumbnail, **Include**, **Name**, **Group**. Names default to the
   filename, or `<stem>_faceN` when an image holds several.
3. Untick **Include** to drop a face; its box turns dashed.
4. Click a box to select that face. **Re-detect** runs YuNet again, asking first
   if you have unsaved edits. The **Detection score** spinbox (arrows step
   0.10–0.99, default 0.9 — YuNet's own; typed values are honoured down to 0.05)
   applies to the live detector immediately, on Return or on leaving the field;
   lower finds more faces, higher is stricter.
5. **YuNet detector** and **Face recognizer** take a path to a different `.onnx`
   file. Leave them empty to use the resolved weights — cache, then
   `ALCHEMYFACE_MODEL_DIR`, then download.
6. **Save .pkl** writes every included, named face, computing any embedding not
   already cached and renumbering ids from `"0"`.

Sidebar glyphs: `·` pending · `✗` detection failed · `⚠` no face ·
`○ (0/N)` nothing included · `✓ (k/N)` k of N included.

A failure and an empty result are shown differently on purpose. Reporting a
broken model or an unreadable file as "no face detected" makes it look like a
finding about the photograph — which once put a false claim in this project's
own roadmap.

### Inspect DB

Read-only viewer for any database: `# · ID · Name · Group · Dim · L2 norm ·
first values`, with a summary line of counts, dimension and file size. Reads the
four-tuple list form and the back-compatible `{name: vector}` dict, numbering
the latter's entries by position.

It opens **malformed** databases too — a NaN, an infinity, an odd-sized vector —
listing every entry and reporting what is wrong with each. Refusing them would
withhold exactly the diagnosis the tab exists to give. Edit DB opens them for
the same reason, so the bad rows can be deleted; saving still validates every
vector, so a repaired file is clean and nothing enrols against a vector that
cannot be matched.

### Edit DB

Open an existing database and change it.

- **Load** any `.pkl`; the table shows `# · Name · Group · Dim · L2 norm · first
  values`. The frame title gains ` *` while there are unsaved changes, and
  closing the window asks before discarding them.
- **Remove selected** — or the Delete key — drops the chosen rows.
- **Double-click a Group cell** to edit it inline from a combobox of presets.
  Anything you type is added to the presets; Escape abandons the edit.
- **Add faces** from a folder or a single image — type or paste a path, or pick
  one with **…**, then press **Process**. Each detected face becomes a candidate
  card — thumbnail, Include, Name, Group — and nothing changes until you press
  **Add checked**. Duplicate names are allowed: the robot resolves by best
  cosine similarity, so a second photo of someone is an improvement.
- **Add checked** takes every usable ticked candidate. One with no name is
  skipped and reported rather than blocking the batch, and anything skipped or
  left unticked stays on the list for a second pass.
- **Save** writes over the loaded path; **Save as…** writes elsewhere. With
  faces added but no database loaded, Save opens Save as… — that is a first
  save, not a mistake. Saving an **empty** table is refused: it would replace a
  real database with an empty one and cannot be undone.
- A line under the path says what is loaded and whether it is saved.

### Resize

Shrink photos until the detector can see the face again.

YuNet's largest anchors miss a face that fills most of the frame — which is what
a phone selfie held at arm's length looks like. Resizing recovers it:

```
selfie 426x546, face ~93% of frame   ->  0 faces detected
resized to 0.5   213x273             ->  1 face detected
```

Detection is **not monotonic** in the ratio, because it depends on the face
matching an anchor scale. In that same example 0.25 finds nothing while 0.15
works again — so if one ratio fails, try another.

- Source and output each take **a folder or a single image**.
- Output defaults beside the source: `photos` → `photos_resized`,
  `face.jpg` → `face_resized.jpg`.
- **Ratio** 0.05–5.0, default 0.5. LANCZOS when shrinking, BICUBIC when growing,
  EXIF orientation applied, and near-lossless saves (JPEG q95 subsampling 0,
  WebP q95 method 6) because these images are about to be enrolled and
  compression artefacts move the embedding.
- Writing over the source is refused — a resize cannot be undone. For the same
  reason a ratio outside 0.05–5.0 is **refused rather than clamped**, in the tab
  and on the command line: silently turning a mistyped 50 into 5.0 would rewrite
  the folder at a size nobody asked for.
- Single-image mode refuses an extension the Build tab would not scan, rather
  than writing a `.tif` that is then ignored.
- A per-file log, cleared and headed at the start of each run and written as
  each file lands; one unreadable file does not abandon the batch.

The same work from the command line:

```bash
alchemyface resize --folder photos/ --ratio 0.5
alchemyface resize --image selfie.jpg --output smaller.jpg --ratio 0.25
```

### The `.pkl` schema

```python
[
    ("0", "Alice", "staff",   np.ndarray(shape=(128,), dtype=float32)),
    ("1", "Bob",   "visitor", np.ndarray(shape=(128,), dtype=float32)),
]
```

Reading is deliberately forgiving. Real databases disagree with this
documentation — `id` is sometimes an `int`, and the vector sometimes `(1, 128)` —
so both are coerced. A stricter reader would refuse a database that works today.

---

## Requirements

`opencv-python-headless`, `numpy`, `typer`, `Pillow`. Python 3.10 or newer.

**The GUI needs `tkinter`, but the library does not.** Nothing in the library
imports it, so `import alchemyface` works on a server, in Docker, or in CI with
no Tk installed — enforced by tests, not hoped for. `alchemyface db` reports what
to install rather than raising:

```
$ alchemyface db
the desktop application needs tkinter, which is not available: No module named '_tkinter'
tkinter ships with Python but is packaged separately on some systems.
  Debian/Ubuntu:  sudo apt-get install python3-tk
  Fedora:         sudo dnf install python3-tkinter
  macOS/Windows:  reinstall Python from python.org
$ echo $?
3
```

The presentation helpers are Tk-free too, so you can render a database in a web
app or a notebook:

```python
from alchemyface.gui.inspect_data import entry_rows, summarise
```

OpenCV is the **headless** build, so there is no `libGL` requirement either.

## Model weights

Resolved in this order, first hit wins:

1. `model_dir=` passed to `Recognizer`
2. `$ALCHEMYFACE_MODEL_DIR`
3. `~/.cache/alchemyface/models/`
4. downloaded from the OpenCV Zoo and SHA256-verified

`alchemyface download-models` pre-fetches. Set `ALCHEMYFACE_MODEL_DIR` to work
offline.

## The recognition threshold

The library defaults to cosine `0.363`, SFace's published operating point. The
G1 robot matches at `0.32`. It is a tunable, not a constant — validate it against
your own data.

## Development

```bash
pyenv install 3.10.6
pyenv virtualenv 3.10.6 alchemyface     # .python-version activates it here
pip install -e ".[dev]"
```

| Command | |
|---|---|
| `pytest tests/ -m "not models and not camera and not gui"` | the fast suite — no display, no models, no network |
| `pytest tests/ -m "gui"` | needs a display; `xvfb-run -a` on a headless box |
| `pytest tests/ -m "not camera"` | everything except the camera |
| `ruff check src tests main.py` · `ruff format src tests main.py` | lint and format |
| `mypy src/alchemyface main.py` | type check |
| `python -m build` | wheel and sdist |

There is deliberately no coverage badge. A hand-written percentage goes stale
silently, and generating a real one needs either a third-party service or a bot
committing to `main` — which the repository's commit-identity check refuses. The
CI badge already means the gate passed, and that gate includes a coverage floor.

The PyPI badge is **static and version-pinned**, not the usual dynamic
`shields.io/pypi/v` one. That badge proved unreliable: PyPI's JSON API reports a
stale `info.version` for some time after an upload, shields.io caches whatever it
read for three hours (`max-age=10800`), and GitHub's image proxy then caches
shields. The three compound, and the badge showed a version four releases behind.
A pinned badge cannot be stale — and CI fails the build if it ever disagrees with
`pyproject.toml`, so it cannot be forgotten either.

Model-backed tests skip unless the weights are present:

```bash
export ALCHEMYFACE_MODEL_DIR="$PWD/_local/onnx"
```

## A note on data

`_local/` is **git-ignored and must stay that way**. It holds face embeddings,
recordings and photographs of real, identifiable people, carried over from the
prototype this grew out of. Under Japan's APPI and GDPR Article 9 those are
sensitive personal data. They are development fixtures: excluded from the wheel,
the sdist and version control, and a CI step fails the build if any of them ever
reach a distribution.

## Links

- [CHANGELOG.md](CHANGELOG.md) — what shipped, per release
- [versions.md](versions.md) — the roadmap
- [todo.md](todo.md) — what is next

## Licence

MIT — see [LICENSE](LICENSE).

Model weights are distributed by the [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
under their own terms — YuNet MIT, SFace Apache-2.0 — and are downloaded at
runtime rather than redistributed here.
