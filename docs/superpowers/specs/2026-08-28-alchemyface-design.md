# AlchemyFace — design

**Date:** 2026-08-28
**Status:** implemented on 2026-08-28. See
`docs/superpowers/plans/2026-08-28-alchemyface-v0.1.md` for the plan that built it,
including three deviations agreed during implementation (`Match.entry_id`,
`identify()` without `k`, and normalisation inside `InMemoryStore`).
**Supersedes:** an earlier internal prototype

## 1. Purpose

Extract the working face-recognition pipeline from an internal prototype into a
small, typed, installable Python library published on PyPI as `alchemyface`.

The prototype does the hard part correctly — YuNet detection, SFace embeddings,
cosine matching against a pgvector gallery — but all of it lives in a single
266-line `main()` alongside camera capture, OpenCV drawing, Postgres writes and
threaded mp3 playback. Nothing in it can be imported, reused or tested.

## 2. Goals

- `pip install alchemyface` gives a usable detect / embed / enroll / identify API.
- Dependency-light: OpenCV, NumPy, Typer. Nothing else in the base install.
- Testable without a camera, a database, or a 37 MB model download.
- Adding a new storage backend or embedding model requires no change to the pipeline.
- No personal data ever reaches version control or a published artefact.

## 3. Non-goals for v0.1

- Liveness or anti-spoofing detection.
- Person detection (the prototype's YOLO block is 100% commented out; dropped).
- Age / gender / emotion attributes (the DeepFace spike is dropped).
- Training or fine-tuning. AlchemyFace consumes pretrained ONNX weights only.
- Persistent database backends. The `FaceStore` protocol ships in v0.1 so
  pgvector and SQLite can be added later without breaking changes; neither
  implementation is in scope now.

## 4. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Distribution name and import name are both `alchemyface` | `face_recognition` is taken on PyPI by a widely-used dlib wrapper. `alchemyface` and `alchemy-face` were both confirmed unregistered on 2026-08-28. |
| D2 | Library plus a thin CLI; the greeter app is not packaged | Keeps app policy (greetings, kiosk loop, event schema) out of a general-purpose library. |
| D3 | `src/` layout | Prevents the working directory from shadowing the installed package — the failure mode that surfaces just before a release. |
| D4 | Components are `typing.Protocol`s behind a `Recognizer` facade | Makes backends swappable and lets the pipeline be tested with fakes. |
| D5 | Model weights downloaded on first use, cached, SHA256-verified | A 37 MB wheel is slow to install and re-uploads on every release. Overridable for offline use. |
| D6 | In-memory store only in v0.1, with `.npz` save/load | Zero dependencies and enough to be genuinely useful. Save/load is serialisation of the one store, not a second backend. |
| D7 | Poetry 2.x with PEP 621 metadata, on pyenv 3.10.6 | Continuity with the prototype's toolchain and its Makefile. |
| D8 | All personal data quarantined in a git-ignored `_local/` | One clearly-named directory is far harder to leak than four scattered at the repo root. |

## 5. Architecture

```
                        ┌──────────────┐
        image ─────────▶│  Recognizer  │◀──── threshold, model_dir
                        └──────┬───────┘
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌───────────┐    ┌───────────┐
        │ Detector │    │ Embedder  │    │ FaceStore │     ← protocols
        └────┬─────┘    └─────┬─────┘    └─────┬─────┘
             ▼                ▼                ▼
      YuNetDetector    SFaceEmbedder     InMemoryStore      ← v0.1 implementations
             │                │
             └────────┬───────┘
                      ▼
                  models.py        resolve / download / verify weights
```

`Recognizer` owns no algorithm. It sequences the three protocols and applies the
threshold. Every implementation is replaceable without touching it.

### 5.1 Layout

```
src/alchemyface/
├── __init__.py           Recognizer, Face, Match, Recognition, __version__
├── types.py              frozen dataclasses; numpy is the only import
├── detection/
│   ├── base.py           Detector protocol
│   └── yunet.py          YuNetDetector  (cv2.FaceDetectorYN)
├── embedding/
│   ├── base.py           Embedder protocol
│   └── sface.py          SFaceEmbedder  (cv2.FaceRecognizerSF)
├── store/
│   ├── base.py           FaceStore protocol
│   └── memory.py         InMemoryStore  (+ save/load .npz)
├── pipeline.py           Recognizer facade
├── models.py             weight resolution, download, SHA256
├── errors.py             AlchemyFaceError hierarchy
├── capture.py            VideoSource
├── cli.py                typer app
└── py.typed
```

### 5.2 Types

```python
@dataclass(frozen=True)
class Face:
    bbox: tuple[int, int, int, int]   # x, y, w, h
    landmarks: np.ndarray             # (5, 2) float32
    confidence: float

@dataclass(frozen=True)
class Match:
    label: str
    score: float                      # cosine similarity, [-1, 1]
    metadata: Mapping[str, Any]

@dataclass(frozen=True)
class Recognition:
    face: Face
    match: Match | None               # None when below threshold
```

### 5.3 Protocols

```python
class Detector(Protocol):
    def detect(self, image: NDArray[np.uint8]) -> list[Face]: ...

class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, image: NDArray[np.uint8], face: Face) -> NDArray[np.float32]: ...

class FaceStore(Protocol):
    def add(self, label: str, vector: NDArray[np.float32],
            metadata: Mapping[str, Any] | None = None) -> str: ...
    def search(self, vector: NDArray[np.float32], k: int = 1) -> list[Match]: ...
    def remove(self, entry_id: str) -> None: ...
    def __len__(self) -> int: ...
```

### 5.4 The facade

```python
class Recognizer:
    def __init__(self, *, detector=None, embedder=None, store=None,
                 model_dir: Path | str | None = None,
                 threshold: float = 0.363) -> None: ...

    def detect(self, image) -> list[Face]: ...
    def embed(self, image, face) -> NDArray[np.float32]: ...
    def enroll(self, label, image, metadata=None) -> str: ...   # largest face
    def identify(self, image, k: int = 1) -> list[Recognition]: ...
```

`enroll` raises `NoFaceDetectedError` when the image has no face. `identify`
returns one `Recognition` per detected face, with `match=None` below threshold —
the caller decides what "unknown" means, rather than the library guessing.

Embeddings are L2-normalised on the way out of `Embedder`, so cosine similarity
is a dot product and the store needs no normalisation logic of its own.

### 5.5 Model resolution

First hit wins:

1. `model_dir=` argument
2. `$ALCHEMYFACE_MODEL_DIR`
3. `~/.cache/alchemyface/models/` (honours `$XDG_CACHE_HOME`)
4. download from the OpenCV Zoo, verify SHA256, write to (3)

Downloads use `urllib.request` from the standard library — no `requests`
dependency. A failed checksum deletes the partial file and raises. Downloads
write to a temporary file and rename on success, so an interrupted download
cannot leave a corrupt cache entry.

### 5.6 Errors

A single `AlchemyFaceError` base, with `ModelNotFoundError`,
`ModelDownloadError` and `NoFaceDetectedError` beneath it. Callers can catch the
base class and never see a bare `cv2.error`.

## 6. Testing

The point of D4 is that the pipeline is testable without heavy fixtures.

| Layer | How | Marker |
|---|---|---|
| `Recognizer`, thresholding, store | `FakeDetector` / `FakeEmbedder` returning canned values | none — always runs |
| `models.py` | temp dirs, monkeypatched env, a local HTTP fixture for download and checksum-failure paths | none |
| `YuNetDetector`, `SFaceEmbedder` | real weights from `$ALCHEMYFACE_MODEL_DIR` | `models` |
| `capture.py` | real camera | `camera` |

`make test` runs `-m "not models and not camera"` and must stay fast and
offline. `make test_all` adds the model tests. Coverage gate is 80%.

The 43 real `(1, 128) float32` embeddings in `_local/face_data/` are useful
store and matching fixtures that need no images — but they are personal data, so
tests using them must be marked and must never assert on real names.

## 7. Migration from the prototype

| Prototype | Destination |
|---|---|
| `scripts/face_recognizer.py` — detect, embed, match | split across `detection/`, `embedding/`, `store/`, `pipeline.py` |
| `scripts/face_recognizer.py` — loop, drawing, audio, event log | `examples/greeter/` (verbatim, unmigrated) |
| `face_recognition/data_capsule.py` | `examples/greeter/`; the seed of a future `store/pgvector.py` |
| `scripts/videocapture.py` | `capture.py` — promoted and actually used |
| `scripts/entry_db.py`, `scripts/get_eventlog.py` | `examples/greeter/` |
| `scripts/deepfacetest.py`, `scripts/capture_test2.py`, `record_video.py` | dropped — spikes; the prototype folder keeps them |
| YOLO block, `yolov8n.pt`, `ultralytics` | dropped — the code is entirely commented out |
| `main.py`, `tests/test_sample_add.py`, template README/Makefile | replaced |

Base dependencies fall from 11 to 3. Removed: `sqlalchemy`, `psycopg2`,
`pgvector`, `pandas`, `scikit-learn`, `ultralytics`, `pygame`. The one thing
`scikit-learn` was used for — `cosine_similarity` — is a dot product on
normalised vectors.

### Prototype bugs not to carry over

- `face_recognizer.py:217` writes to `./recorded_faces/`, which is never created.
- `face_recognizer.py:170` is `if True:`, disabling the unknown-face branch, so
  an unrecognised face is labelled with the nearest gallery entry's identity.
- `face_recognizer.py:54` `match()` returns `None` implicitly when the gallery is
  empty, and the caller unpacks it unconditionally.
- `entry_db.py:22` hardcodes `id=43` inside the loop; every row collides.

## 8. Personal data

`_local/` holds 43 face embeddings, 46 name recordings, and an ID-to-name map
for real, identifiable people. Under Japan's APPI and GDPR Article 9
these are sensitive personal data.

Controls in place:

- `_local/` is git-ignored, as are the original directory names, `*.npy`,
  `*.npz`, `*.mp3` and the csv files, in case anything is reintroduced at the root.
- `tool.poetry.exclude` keeps `_local`, `examples`, `tests` and `docs` out of the sdist.
- Verified after `make build`: wheel and sdist contain only `src/alchemyface`.
- `old_data/` (60 MB of captured face crops) was deliberately not copied.

Before the repository is pushed to any remote, confirm `_local/` is untracked.

## 9. Environment

pyenv supplies the interpreter and the virtualenv; Poetry manages dependencies.

```bash
pyenv virtualenv 3.10.6 alchemyface     # .python-version activates it here
make install
```

`pyenv local` activates through shims and never exports `VIRTUAL_ENV`. Poetry,
installed via Homebrew and running on its own Python 3.14, cannot see a
shim-activated environment: with `virtualenvs.create = false` it will silently
install into its own interpreter instead. The Makefile therefore exports
`VIRTUAL_ENV` from `pyenv prefix alchemyface` for every target, and errors early
if that virtualenv is missing.

`coverage-badge` was replaced by `genbadge`: it imports `pkg_resources`, which
setuptools removed in v81.

## 10. Open questions

The repository is `github.com/kouya-marino/AlchemyFace`; `pyproject.toml` reflects it.

- **Licence.** Undecided, and required before a PyPI release. MIT or Apache-2.0
  are the conventional choices for a library of this kind.
- **Python upper bound.** Currently `>=3.10` with no ceiling. Verified on 3.10.6
  with OpenCV 4.14 and NumPy 2.2.6.

## 11. Next step

An implementation plan, built module by module under TDD, in this order:
`types` → `models` → `store/memory` → `detection/yunet` → `embedding/sface` →
`pipeline` → `cli` → `capture`.
