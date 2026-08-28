# AlchemyFace v0.1 Implementation Plan

> **Toolchain superseded, 2026-08-28.** This document was written against
> Poetry, a Makefile, and black/isort/pylint. The project was subsequently
> aligned with its sibling repositories (AlchemyCV, AlchemyAnnotate,
> AlchemyDetect, AlchemyCloud): **setuptools** with `requirements.txt`,
> **ruff** at line-length 120, no Makefile, and PyPI **Trusted Publishing**.
> Every design decision below still holds — only the tooling changed. Read
> `pyproject.toml` and `.github/workflows/` for what is actually in use.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the empty `alchemyface` package skeleton into a working, typed, tested face-recognition library that detects faces, embeds them, stores them, and identifies them — with no camera or database required to run its test suite.

**Architecture:** Three `typing.Protocol` seams — `Detector`, `Embedder`, `FaceStore` — with one concrete implementation each (`YuNetDetector`, `SFaceEmbedder`, `InMemoryStore`), sequenced by a `Recognizer` facade that owns no algorithm of its own. Weights are resolved at runtime (argument → env var → cache → download), never vendored. Every embedding is L2-normalised on the way out of the embedder, which makes cosine similarity a plain dot product.

**Tech Stack:** Python 3.10.6 (pyenv virtualenv `alchemyface`), OpenCV 4.14 (`cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`), NumPy 2.2, Typer 0.27, pytest 8.4, mypy, black, isort, pylint, Poetry 2.4.

**Spec:** `docs/superpowers/specs/2026-08-28-alchemyface-design.md`

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Base runtime dependencies are frozen at three:** `opencv-python`, `numpy`, `typer`. Do not add a fourth. Downloads use `urllib.request` from the standard library, not `requests`.
- **Python floor is 3.10.** `from __future__ import annotations` at the top of every module so `X | None` syntax is safe.
- **mypy runs with `disallow_untyped_defs = true`.** Every function, including every test, needs annotations. `make check_type` must stay green.
- **Line length 88** (black + isort profile black + pylint). Run `make format` before every commit.
- **Coverage gate is 80%** on `make test`, which runs `-m "not models and not camera"`. That subset must never need the network, a camera, or the real weights.
- **Tests touching real weights are marked `@pytest.mark.models`** and use the `model_dir` fixture in `tests/conftest.py`, which skips when `ALCHEMYFACE_MODEL_DIR` is unset. The Makefile exports it as `$(PROJECT_DIR)/_local/onnx`.
- **Never commit anything from `_local/`,** and never write a test that asserts on a real person's name from `_local/data/id2map.csv`.
- **Run every command through `make`,** or export `VIRTUAL_ENV="$(pyenv prefix alchemyface)"` first. Bare `poetry` without it installs into Homebrew's Python 3.14.
- **Cosine threshold default is `0.363`,** SFace's published operating point.

### Prerequisite

This repository is not yet a git repository. Before executing, run:

```bash
cd /Users/prashantrawat/Workspace/__code__/AlchemyFace
git init
git add .
git commit -m "chore: scaffold AlchemyFace package, tooling and spec"
```

Confirm `git status --porcelain` does not list anything under `_local/` before that first commit. If git is deliberately still deferred, skip every "Commit" step and run `make format check_type test` in its place.

---

## Verified facts

These were measured against the real models on 2026-08-28. Do not re-derive them; do not assume otherwise.

| Fact | Value |
|---|---|
| `FaceDetectorYN.detect()` returns | `(None, ndarray)` where the array is `(N, 15) float32`, or `(None, None)` when nothing is found |
| Row layout | `[x, y, w, h, lm0x, lm0y, … lm4x, lm4y, score]` — cols 0-3 bbox, 4-13 five landmarks, 14 score |
| `FaceRecognizerSF.alignCrop()` returns | `(112, 112, 3) uint8` |
| `FaceRecognizerSF.feature()` returns | `(1, 128) float32`, **not normalised** (observed L2 ≈ 10.0) |
| Normalised dot product vs `cv2.match(..., FR_COSINE)` | identical to 1e-6 — this is why scikit-learn is dropped |
| OpenCV Zoo `raw.githubusercontent.com` URLs | return **131-byte Git-LFS pointer files**, not models. Must use `media.githubusercontent.com/media/...` |

Model digests, verified by download:

| Model | Filename | Bytes | SHA256 |
|---|---|---|---|
| YuNet | `face_detection_yunet_2023mar.onnx` | 232,589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| SFace | `face_recognition_sface_2021dec.onnx` | 38,696,353 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

The prototype's local copies are the same architectures under different filenames — `yunet_n_640_640.onnx` and `face_recognizer_fast.onnx`. They are registered as aliases so `_local/onnx/` resolves offline, and their digests are deliberately **not** checked: local files are trusted, downloads are not.

---

## Deviations from the spec

Three refinements found while validating. Implement the plan, not the spec, where they differ.

1. **`Match` gains an `entry_id: str` field.** The spec's `FaceStore.remove(entry_id)` is unusable otherwise — a caller who matches an entry has no way to name it for removal.
2. **`Recognizer.identify()` drops the `k` parameter.** It returns one `Recognition` per face holding the single best match; a `k` that the return type discards is dead weight. Callers wanting candidates use `store.search(vector, k=…)` directly.
3. **`InMemoryStore` normalises on `add` as well as the embedder.** Idempotent and cheap, and it makes the store correct in isolation when fed raw vectors.

---

## File structure

| File | Responsibility | Task |
|---|---|---|
| `src/alchemyface/types.py` | `Face`, `Match`, `Recognition` value types. NumPy only. | 1 |
| `src/alchemyface/errors.py` | `AlchemyFaceError` and its three subclasses. | 1 |
| `src/alchemyface/models.py` | `ModelSpec`, the registry, cache location, local search, download, checksum. | 2 |
| `src/alchemyface/store/base.py` | `FaceStore` protocol. | 3 |
| `src/alchemyface/store/memory.py` | `InMemoryStore` with `.npz` save/load. | 3 |
| `src/alchemyface/detection/base.py` | `Detector` protocol. | 4 |
| `src/alchemyface/detection/yunet.py` | `YuNetDetector`, and the `(N,15)` row ↔ `Face` conversion. | 4 |
| `src/alchemyface/embedding/base.py` | `Embedder` protocol. | 5 |
| `src/alchemyface/embedding/sface.py` | `SFaceEmbedder`, alignment and L2 normalisation. | 5 |
| `src/alchemyface/pipeline.py` | `Recognizer` facade. | 6 |
| `src/alchemyface/__init__.py` | Public exports. | 6 |
| `src/alchemyface/cli.py` | `download-models`, `enroll`, `identify` commands. | 7 |
| `src/alchemyface/capture.py` | `VideoSource` context manager. | 8 |
| `tests/fakes.py` | `FakeDetector`, `FakeEmbedder` — how the pipeline is tested without models. | 6 |

---

### Task 1: Value types and error hierarchy

**Files:**
- Create: `src/alchemyface/types.py`
- Create: `src/alchemyface/errors.py`
- Test: `tests/unit/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Face(bbox: tuple[int,int,int,int], landmarks: NDArray[np.float32], confidence: float)` with a read-only `.area` property; `Match(label: str, score: float, entry_id: str, metadata: Mapping[str, Any])`; `Recognition(face: Face, match: Match | None)`; `AlchemyFaceError`, `ModelNotFoundError`, `ModelDownloadError`, `NoFaceDetectedError`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_types.py`:

```python
"""The value types carry data and nothing else, so these tests pin down the
two things that are easy to get wrong: area arithmetic, and the fact that a
dataclass holding an ndarray cannot use a generated __eq__."""

from __future__ import annotations

import numpy as np
import pytest

from alchemyface.errors import (
    AlchemyFaceError,
    ModelDownloadError,
    ModelNotFoundError,
    NoFaceDetectedError,
)
from alchemyface.types import Face, Match, Recognition


def _landmarks() -> np.ndarray:
    return np.arange(10, dtype=np.float32).reshape(5, 2)


def test_face_area_is_width_times_height() -> None:
    face = Face(bbox=(10, 20, 30, 40), landmarks=_landmarks(), confidence=0.9)
    assert face.area == 1200


def test_face_is_frozen() -> None:
    face = Face(bbox=(0, 0, 1, 1), landmarks=_landmarks(), confidence=0.5)
    with pytest.raises(AttributeError):
        face.confidence = 0.1  # type: ignore[misc]


def test_two_identical_faces_compare_by_identity_not_value() -> None:
    # landmarks is an ndarray; a generated __eq__ would return an array and
    # blow up with "truth value of an array is ambiguous". eq=False avoids it.
    a = Face(bbox=(0, 0, 1, 1), landmarks=_landmarks(), confidence=0.5)
    b = Face(bbox=(0, 0, 1, 1), landmarks=_landmarks(), confidence=0.5)
    assert a != b
    assert a == a


def test_match_compares_by_value() -> None:
    a = Match(label="ada", score=0.9, entry_id="x", metadata={})
    b = Match(label="ada", score=0.9, entry_id="x", metadata={})
    assert a == b


def test_recognition_holds_no_match_when_unknown() -> None:
    face = Face(bbox=(0, 0, 1, 1), landmarks=_landmarks(), confidence=0.5)
    assert Recognition(face=face, match=None).match is None


@pytest.mark.parametrize(
    "error",
    [ModelNotFoundError, ModelDownloadError, NoFaceDetectedError],
)
def test_every_error_is_catchable_as_the_base(error: type[Exception]) -> None:
    with pytest.raises(AlchemyFaceError):
        raise error("boom")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `ModuleNotFoundError: No module named 'alchemyface.types'`

- [ ] **Step 3: Write the implementation**

Create `src/alchemyface/errors.py`:

```python
"""Every exception AlchemyFace raises descends from AlchemyFaceError, so a
caller can wrap the library in one except clause and never see a bare
cv2.error or urllib exception leak through."""

from __future__ import annotations


class AlchemyFaceError(Exception):
    """Base class for every error AlchemyFace raises."""


class ModelNotFoundError(AlchemyFaceError):
    """Weights were not on disk and downloading them was not permitted."""


class ModelDownloadError(AlchemyFaceError):
    """A download failed, or what arrived did not match its checksum."""


class NoFaceDetectedError(AlchemyFaceError):
    """An operation required a face and the image had none."""
```

Create `src/alchemyface/types.py`:

```python
"""Value types shared across AlchemyFace. NumPy is the only import."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class Face:
    """A detected face: where it is, how to align it, how sure the detector was.

    ``eq=False`` is deliberate. ``landmarks`` is an ndarray, and a generated
    ``__eq__`` would compare element-wise and return an array, so any ``==``
    between two Faces would raise "truth value of an array is ambiguous".
    Identity comparison is the useful default here.
    """

    bbox: tuple[int, int, int, int]
    """Pixel bounding box as ``(x, y, width, height)``, clamped to the image."""

    landmarks: NDArray[np.float32]
    """``(5, 2)`` array: right eye, left eye, nose tip, right and left mouth corner."""

    confidence: float
    """Detector score. Not a recognition score — see :class:`Match`."""

    @property
    def area(self) -> int:
        """Pixel area, used to pick the most prominent face in a frame."""
        return self.bbox[2] * self.bbox[3]


@dataclass(frozen=True)
class Match:
    """A gallery entry that resembles a query embedding."""

    label: str
    score: float
    """Cosine similarity in ``[-1, 1]``. Higher is more alike."""

    entry_id: str
    """Opaque id of the stored entry, as returned by ``FaceStore.add``."""

    metadata: Mapping[str, Any]


@dataclass(frozen=True, eq=False)
class Recognition:
    """One detected face and the best gallery entry for it, if any cleared
    the threshold. ``match is None`` means unknown — the library does not
    invent a label for it."""

    face: Face
    match: Match | None
```

- [ ] **Step 4: Run tests and the type checker**

Run: `make format && make check_type && make test`
Expected: all tests PASS, mypy reports success.

- [ ] **Step 5: Commit**

```bash
git add src/alchemyface/types.py src/alchemyface/errors.py tests/unit/test_types.py
git commit -m "feat: add core value types and error hierarchy"
```

---

### Task 2: Model resolution and download

**Files:**
- Create: `src/alchemyface/models.py`
- Test: `tests/unit/test_models.py`

**Interfaces:**
- Consumes: `ModelDownloadError`, `ModelNotFoundError` from Task 1.
- Produces: `ModelSpec` (frozen dataclass with `key`, `filename`, `url`, `sha256`, `aliases`, and a `.candidates` property); module constants `DETECTOR`, `EMBEDDER`, `MODELS: dict[str, ModelSpec]`; functions `cache_dir() -> Path`, `find_local(spec, model_dir=None) -> Path | None`, `sha256_of(path) -> str`, `download(spec, dest_dir=None) -> Path`, `resolve(spec, model_dir=None, allow_download=True) -> Path`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_models.py`:

```python
"""Resolution order and download integrity. Nothing here touches the network:
the download tests point the spec's URL at a file:// URL in a tmp_path, which
urllib handles natively."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from alchemyface import models
from alchemyface.errors import ModelDownloadError, ModelNotFoundError
from alchemyface.models import ModelSpec

PAYLOAD = b"pretend onnx bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture()
def local_spec(tmp_path: Path) -> ModelSpec:
    """A spec whose URL is a real file on disk, served over file://."""
    source = tmp_path / "source.onnx"
    source.write_bytes(PAYLOAD)
    return ModelSpec(
        key="test",
        filename="canonical.onnx",
        url=source.as_uri(),
        sha256=DIGEST,
        aliases=("legacy_name.onnx",),
    )


def test_candidates_puts_canonical_name_first(local_spec: ModelSpec) -> None:
    assert local_spec.candidates == ("canonical.onnx", "legacy_name.onnx")


def test_cache_dir_honours_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert models.cache_dir() == tmp_path / "alchemyface" / "models"


def test_explicit_model_dir_wins_over_env(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit, env = tmp_path / "explicit", tmp_path / "env"
    for d in (explicit, env):
        d.mkdir()
        (d / "canonical.onnx").write_bytes(PAYLOAD)
    monkeypatch.setenv("ALCHEMYFACE_MODEL_DIR", str(env))
    assert models.find_local(local_spec, model_dir=explicit) == explicit / "canonical.onnx"


def test_env_var_is_used_when_no_explicit_dir(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALCHEMYFACE_MODEL_DIR", str(tmp_path))
    (tmp_path / "canonical.onnx").write_bytes(PAYLOAD)
    assert models.find_local(local_spec) == tmp_path / "canonical.onnx"


def test_alias_filename_is_found(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The prototype's weights use different filenames; they must still resolve.
    monkeypatch.setenv("ALCHEMYFACE_MODEL_DIR", str(tmp_path))
    (tmp_path / "legacy_name.onnx").write_bytes(PAYLOAD)
    assert models.find_local(local_spec) == tmp_path / "legacy_name.onnx"


def test_find_local_returns_none_when_absent(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALCHEMYFACE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(models, "cache_dir", lambda: tmp_path / "nope")
    assert models.find_local(local_spec) is None


def test_download_writes_file_and_verifies_checksum(
    tmp_path: Path, local_spec: ModelSpec
) -> None:
    dest = models.download(local_spec, dest_dir=tmp_path / "cache")
    assert dest.read_bytes() == PAYLOAD
    assert dest.name == "canonical.onnx"


def test_download_rejects_a_checksum_mismatch(
    tmp_path: Path, local_spec: ModelSpec
) -> None:
    wrong = ModelSpec(
        key=local_spec.key,
        filename=local_spec.filename,
        url=local_spec.url,
        sha256="0" * 64,
        aliases=local_spec.aliases,
    )
    cache = tmp_path / "cache"
    with pytest.raises(ModelDownloadError, match="checksum"):
        models.download(wrong, dest_dir=cache)
    # A bad download must not leave the file behind, nor a .part turd.
    assert list(cache.iterdir()) == []


def test_download_raises_on_a_missing_source(tmp_path: Path) -> None:
    spec = ModelSpec(
        key="gone",
        filename="gone.onnx",
        url=(tmp_path / "absent.onnx").as_uri(),
        sha256=DIGEST,
    )
    with pytest.raises(ModelDownloadError):
        models.download(spec, dest_dir=tmp_path / "cache")


def test_resolve_prefers_a_local_file_over_downloading(
    tmp_path: Path, local_spec: ModelSpec
) -> None:
    (tmp_path / "canonical.onnx").write_bytes(b"local wins")
    resolved = models.resolve(local_spec, model_dir=tmp_path)
    assert resolved.read_bytes() == b"local wins"


def test_resolve_raises_when_download_is_forbidden(
    tmp_path: Path, local_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALCHEMYFACE_MODEL_DIR", raising=False)
    monkeypatch.setattr(models, "cache_dir", lambda: tmp_path / "empty")
    with pytest.raises(ModelNotFoundError, match="canonical.onnx"):
        models.resolve(local_spec, allow_download=False)


def test_registry_pins_both_real_models() -> None:
    assert set(models.MODELS) == {"detector", "embedder"}
    for spec in models.MODELS.values():
        assert len(spec.sha256) == 64
        # raw.githubusercontent serves LFS pointers, not weights.
        assert spec.url.startswith("https://media.githubusercontent.com/media/")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `ModuleNotFoundError: No module named 'alchemyface.models'`

- [ ] **Step 3: Write the implementation**

Create `src/alchemyface/models.py`:

```python
"""Locating the ONNX weights AlchemyFace runs on.

Weights are not vendored: a 37 MB wheel is slow to install and re-uploads on
every release. They are resolved at runtime, first hit wins:

1. an explicit ``model_dir`` argument
2. ``$ALCHEMYFACE_MODEL_DIR``
3. ``~/.cache/alchemyface/models/`` (honours ``$XDG_CACHE_HOME``)
4. downloaded from the OpenCV Zoo and verified against a pinned SHA256

Files already on disk are trusted and not checksummed — they may legitimately
be a different build. Downloads always are.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from alchemyface.errors import ModelDownloadError, ModelNotFoundError

# The zoo keeps its weights in Git LFS. raw.githubusercontent.com serves the
# ~130-byte pointer file instead of the model; the media host serves the model.
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

_CHUNK = 1 << 20


@dataclass(frozen=True)
class ModelSpec:
    """Where a model lives, what it is called, and how to know it arrived intact."""

    key: str
    filename: str
    url: str
    sha256: str
    aliases: tuple[str, ...] = ()
    """Other filenames the same architecture ships under. The prototype's
    weights use different names, and should resolve without being renamed."""

    @property
    def candidates(self) -> tuple[str, ...]:
        """Filenames to look for on disk, canonical name first."""
        return (self.filename, *self.aliases)


DETECTOR = ModelSpec(
    key="detector",
    filename="face_detection_yunet_2023mar.onnx",
    url=f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    aliases=("yunet_n_640_640.onnx",),
)

EMBEDDER = ModelSpec(
    key="embedder",
    filename="face_recognition_sface_2021dec.onnx",
    url=f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    aliases=("face_recognizer_fast.onnx",),
)

MODELS: dict[str, ModelSpec] = {spec.key: spec for spec in (DETECTOR, EMBEDDER)}


def cache_dir() -> Path:
    """Where downloaded weights are kept between runs."""
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "alchemyface" / "models"


def _search_paths(model_dir: Path | str | None = None) -> Iterator[Path]:
    if model_dir is not None:
        yield Path(model_dir)
    env = os.environ.get("ALCHEMYFACE_MODEL_DIR")
    if env:
        yield Path(env)
    yield cache_dir()


def find_local(spec: ModelSpec, model_dir: Path | str | None = None) -> Path | None:
    """First existing file matching any of the spec's names, or None."""
    for directory in _search_paths(model_dir):
        for name in spec.candidates:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def sha256_of(path: Path | str) -> str:
    """Hex digest of a file, read in chunks so a 37 MB model is not slurped."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(spec: ModelSpec, dest_dir: Path | str | None = None) -> Path:
    """Fetch a model, verify it, and place it atomically.

    The download goes to a ``.part`` file in the destination directory and is
    renamed only after the checksum passes, so an interrupted or corrupted
    download can never be mistaken for a usable cache entry.
    """
    directory = Path(dest_dir) if dest_dir is not None else cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    handle, raw_tmp = tempfile.mkstemp(dir=directory, suffix=".part")
    os.close(handle)
    tmp = Path(raw_tmp)

    try:
        with urllib.request.urlopen(spec.url, timeout=60) as response:
            with open(tmp, "wb") as out:
                shutil.copyfileobj(response, out, _CHUNK)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"could not download {spec.filename} from {spec.url}: {exc}"
        ) from exc

    actual = sha256_of(tmp)
    if actual != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"checksum mismatch for {spec.filename}: "
            f"expected {spec.sha256}, got {actual}"
        )

    destination = directory / spec.filename
    tmp.replace(destination)
    return destination


def resolve(
    spec: ModelSpec,
    model_dir: Path | str | None = None,
    allow_download: bool = True,
) -> Path:
    """Path to usable weights, downloading them if that is permitted."""
    local = find_local(spec, model_dir)
    if local is not None:
        return local
    if not allow_download:
        raise ModelNotFoundError(
            f"{spec.filename} was not found in any of "
            f"{[str(p) for p in _search_paths(model_dir)]}, "
            "and downloading was disabled. Set ALCHEMYFACE_MODEL_DIR, pass "
            "model_dir=, or run `alchemyface download-models`."
        )
    return download(spec)
```

- [ ] **Step 4: Run tests and the type checker**

Run: `make format && make check_type && make test`
Expected: all tests PASS, mypy success.

- [ ] **Step 5: Verify against the real weights**

Run:

```bash
export VIRTUAL_ENV="$(pyenv prefix alchemyface)"; export PATH="$VIRTUAL_ENV/bin:$PATH"
ALCHEMYFACE_MODEL_DIR=_local/onnx python -c "
from alchemyface.models import DETECTOR, EMBEDDER, find_local
print(find_local(DETECTOR)); print(find_local(EMBEDDER))"
```

Expected: two paths under `_local/onnx`, resolved via the aliases.

- [ ] **Step 6: Commit**

```bash
git add src/alchemyface/models.py tests/unit/test_models.py
git commit -m "feat: resolve, download and verify ONNX weights"
```

---

### Task 3: FaceStore protocol and InMemoryStore

**Files:**
- Create: `src/alchemyface/store/base.py`
- Create: `src/alchemyface/store/memory.py`
- Modify: `src/alchemyface/store/__init__.py`
- Test: `tests/unit/test_memory_store.py`

**Interfaces:**
- Consumes: `Match` from Task 1.
- Produces: `FaceStore` protocol with `add(label, vector, metadata=None) -> str`, `search(vector, k=1) -> list[Match]`, `remove(entry_id) -> None`, `__len__() -> int`; `InMemoryStore(dim: int = 128)` implementing it plus `save(path) -> None` and `load(path) -> None`; re-exported from `alchemyface.store`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_memory_store.py`:

```python
"""The store is pure numpy, so it is tested with hand-made unit vectors and
no models at all."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.store import InMemoryStore


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def basis(index: int, dim: int = 4) -> np.ndarray:
    """A one-hot unit vector — orthogonal to every other basis vector, so
    cosine similarity between two different ones is exactly 0."""
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_new_store_is_empty() -> None:
    assert len(InMemoryStore(dim=4)) == 0


def test_add_returns_a_unique_id() -> None:
    store = InMemoryStore(dim=4)
    first = store.add("ada", basis(0))
    second = store.add("grace", basis(1))
    assert first != second
    assert len(store) == 2


def test_search_on_an_empty_store_returns_nothing() -> None:
    assert InMemoryStore(dim=4).search(basis(0)) == []


def test_search_finds_the_exact_vector_with_score_one() -> None:
    store = InMemoryStore(dim=4)
    entry_id = store.add("ada", basis(0), {"team": "analytical"})
    (match,) = store.search(basis(0))
    assert match.label == "ada"
    assert match.entry_id == entry_id
    assert match.metadata == {"team": "analytical"}
    assert match.score == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero() -> None:
    store = InMemoryStore(dim=4)
    store.add("ada", basis(0))
    (match,) = store.search(basis(1))
    assert match.score == pytest.approx(0.0)


def test_search_ranks_by_descending_score() -> None:
    store = InMemoryStore(dim=4)
    store.add("far", basis(1))
    store.add("near", unit(1.0, 0.1, 0.0, 0.0))
    store.add("middle", unit(1.0, 1.0, 0.0, 0.0))
    labels = [m.label for m in store.search(basis(0), k=3)]
    assert labels == ["near", "middle", "far"]


def test_k_larger_than_the_gallery_is_not_an_error() -> None:
    store = InMemoryStore(dim=4)
    store.add("ada", basis(0))
    assert len(store.search(basis(0), k=50)) == 1


def test_unnormalised_input_is_normalised_on_the_way_in() -> None:
    store = InMemoryStore(dim=4)
    store.add("ada", np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32))
    (match,) = store.search(np.array([9.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert match.score == pytest.approx(1.0)


def test_wrong_dimension_is_rejected() -> None:
    store = InMemoryStore(dim=4)
    with pytest.raises(ValueError, match="dimension"):
        store.add("ada", np.zeros(8, dtype=np.float32))


def test_zero_vector_is_rejected() -> None:
    # A zero vector has no direction, so cosine similarity is undefined.
    store = InMemoryStore(dim=4)
    with pytest.raises(ValueError, match="zero"):
        store.add("ada", np.zeros(4, dtype=np.float32))


def test_a_2d_row_vector_is_accepted() -> None:
    # cv2's feature() hands back (1, 128); callers should not have to ravel it.
    store = InMemoryStore(dim=4)
    store.add("ada", basis(0).reshape(1, 4))
    assert store.search(basis(0).reshape(1, 4))[0].label == "ada"


def test_remove_drops_the_entry_and_its_vector() -> None:
    store = InMemoryStore(dim=4)
    entry_id = store.add("ada", basis(0))
    store.add("grace", basis(1))
    store.remove(entry_id)
    assert len(store) == 1
    assert [m.label for m in store.search(basis(1), k=5)] == ["grace"]


def test_remove_of_an_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        InMemoryStore(dim=4).remove("nope")


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = InMemoryStore(dim=4)
    entry_id = store.add("ada", basis(0), {"team": "analytical", "n": 1})
    store.add("grace", basis(1))
    path = tmp_path / "gallery.npz"
    store.save(path)

    restored = InMemoryStore(dim=4)
    restored.load(path)
    assert len(restored) == 2
    (match,) = restored.search(basis(0), k=1)
    assert match.label == "ada"
    assert match.entry_id == entry_id
    assert match.metadata == {"team": "analytical", "n": 1}


def test_load_replaces_rather_than_appends(tmp_path: Path) -> None:
    source = InMemoryStore(dim=4)
    source.add("ada", basis(0))
    path = tmp_path / "gallery.npz"
    source.save(path)

    target = InMemoryStore(dim=4)
    target.add("grace", basis(1))
    target.load(path)
    assert len(target) == 1
    assert target.search(basis(0))[0].label == "ada"


def test_load_rejects_a_dimension_mismatch(tmp_path: Path) -> None:
    source = InMemoryStore(dim=8)
    source.add("ada", basis(0, dim=8))
    path = tmp_path / "gallery.npz"
    source.save(path)
    with pytest.raises(ValueError, match="dimension"):
        InMemoryStore(dim=4).load(path)


def test_save_of_an_empty_store_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    InMemoryStore(dim=4).save(path)
    restored = InMemoryStore(dim=4)
    restored.load(path)
    assert len(restored) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `ImportError: cannot import name 'InMemoryStore'`

- [ ] **Step 3: Write the protocol**

Create `src/alchemyface/store/base.py`:

```python
"""The gallery seam.

A store owns enrolled embeddings and answers nearest-neighbour queries. It is
a Protocol rather than a base class so a pgvector or SQLite implementation can
be added later without inheriting from anything in this package.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from alchemyface.types import Match


@runtime_checkable
class FaceStore(Protocol):
    """Holds labelled embeddings and finds the closest ones."""

    def add(
        self,
        label: str,
        vector: NDArray[np.float32],
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Store an embedding under a label. Returns its opaque entry id."""

    def search(self, vector: NDArray[np.float32], k: int = 1) -> list[Match]:
        """The ``k`` most similar entries, best first. Empty if the store is."""

    def remove(self, entry_id: str) -> None:
        """Delete one entry. Raises ``KeyError`` if it is not there."""

    def __len__(self) -> int:
        """How many entries are stored."""
```

- [ ] **Step 4: Write the implementation**

Create `src/alchemyface/store/memory.py`:

```python
"""A gallery held in a single numpy matrix.

Every vector is stored L2-normalised, which makes cosine similarity a matrix
product: ``vectors @ query``. That is why scikit-learn is not a dependency.
Brute force over a few thousand faces is well under a millisecond, and it
keeps the default install free of any database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from alchemyface.types import Match


@dataclass(frozen=True)
class _Entry:
    entry_id: str
    label: str
    metadata: dict[str, Any]


def _as_unit_row(vector: NDArray[np.float32], dim: int) -> NDArray[np.float32]:
    """Validate, flatten and normalise. Accepts ``(dim,)`` or ``(1, dim)``."""
    flat = np.asarray(vector, dtype=np.float32).ravel()
    if flat.shape != (dim,):
        raise ValueError(
            f"expected a vector of dimension {dim}, got shape {np.shape(vector)}"
        )
    norm = float(np.linalg.norm(flat))
    if norm == 0.0:
        raise ValueError("cannot store a zero vector: it has no direction")
    return (flat / norm).astype(np.float32)


class InMemoryStore:
    """Implements :class:`~alchemyface.store.base.FaceStore` with numpy."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim
        self._vectors: NDArray[np.float32] = np.empty((0, dim), dtype=np.float32)
        self._entries: list[_Entry] = []

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._entries)

    def add(
        self,
        label: str,
        vector: NDArray[np.float32],
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        row = _as_unit_row(vector, self._dim)
        entry = _Entry(uuid4().hex, label, dict(metadata or {}))
        self._vectors = np.vstack([self._vectors, row])
        self._entries.append(entry)
        return entry.entry_id

    def search(self, vector: NDArray[np.float32], k: int = 1) -> list[Match]:
        if not self._entries:
            return []
        query = _as_unit_row(vector, self._dim)
        scores = self._vectors @ query
        take = min(max(k, 1), len(self._entries))
        # argpartition finds the top-k in O(n); only those k are then sorted.
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        return [
            Match(
                label=self._entries[i].label,
                score=float(scores[i]),
                entry_id=self._entries[i].entry_id,
                metadata=dict(self._entries[i].metadata),
            )
            for i in top
        ]

    def remove(self, entry_id: str) -> None:
        for index, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                del self._entries[index]
                self._vectors = np.delete(self._vectors, index, axis=0)
                return
        raise KeyError(entry_id)

    def save(self, path: Path | str) -> None:
        """Write the gallery to a ``.npz`` file.

        Metadata goes in as one JSON blob rather than an object array, so the
        file loads without ``allow_pickle`` and cannot execute anything.
        """
        np.savez(
            path,
            vectors=self._vectors,
            labels=np.array([e.label for e in self._entries], dtype="U"),
            entry_ids=np.array([e.entry_id for e in self._entries], dtype="U"),
            metadata=np.array(json.dumps([e.metadata for e in self._entries])),
            dim=np.array(self._dim),
        )

    def load(self, path: Path | str) -> None:
        """Replace the gallery with the contents of a ``.npz`` file."""
        with np.load(path, allow_pickle=False) as data:
            dim = int(data["dim"])
            if dim != self._dim:
                raise ValueError(
                    f"gallery dimension {dim} does not match store dimension {self._dim}"
                )
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            labels: list[str] = np.asarray(data["labels"]).tolist()
            entry_ids: list[str] = np.asarray(data["entry_ids"]).tolist()
            metadata = json.loads(str(data["metadata"]))

        self._vectors = vectors.reshape(-1, self._dim)
        self._entries = [
            _Entry(entry_id, label, dict(meta))
            for entry_id, label, meta in zip(entry_ids, labels, metadata)
        ]
```

Modify `src/alchemyface/store/__init__.py` to:

```python
"""Gallery backends. See ``base`` for the protocol they implement."""

from alchemyface.store.base import FaceStore
from alchemyface.store.memory import InMemoryStore

__all__ = ["FaceStore", "InMemoryStore"]
```

- [ ] **Step 5: Run tests and the type checker**

Run: `make format && make check_type && make test`
Expected: all tests PASS, mypy success.

- [ ] **Step 6: Commit**

```bash
git add src/alchemyface/store tests/unit/test_memory_store.py
git commit -m "feat: add FaceStore protocol and in-memory gallery"
```

---

### Task 4: Detector protocol and YuNetDetector

**Files:**
- Create: `src/alchemyface/detection/base.py`
- Create: `src/alchemyface/detection/yunet.py`
- Modify: `src/alchemyface/detection/__init__.py`
- Test: `tests/unit/test_yunet.py`

**Interfaces:**
- Consumes: `Face` (Task 1), `DETECTOR` and `resolve` (Task 2).
- Produces: `Detector` protocol with `detect(image) -> list[Face]`; `YuNetDetector(model_path=None, model_dir=None, score_threshold=0.9, nms_threshold=0.3, top_k=5000)`; module functions `face_from_row(row, image_shape) -> Face` and `row_from_face(face) -> NDArray[np.float32]` — Task 5 uses `row_from_face`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_yunet.py`:

```python
"""Row conversion is pure arithmetic and always tested. Anything that loads
the real network is marked `models`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.detection import YuNetDetector, face_from_row, row_from_face
from alchemyface.types import Face


def sample_row() -> np.ndarray:
    """One YuNet detection: bbox, five landmarks, score — 15 float32 columns."""
    return np.array(
        [10, 20, 30, 40, 15, 25, 35, 25, 25, 35, 18, 50, 32, 50, 0.93],
        dtype=np.float32,
    )


def test_face_from_row_reads_the_bounding_box() -> None:
    face = face_from_row(sample_row(), image_shape=(480, 640))
    assert face.bbox == (10, 20, 30, 40)


def test_face_from_row_reads_five_landmarks() -> None:
    face = face_from_row(sample_row(), image_shape=(480, 640))
    assert face.landmarks.shape == (5, 2)
    assert face.landmarks[0].tolist() == [15.0, 25.0]


def test_face_from_row_reads_the_score() -> None:
    face = face_from_row(sample_row(), image_shape=(480, 640))
    assert face.confidence == pytest.approx(0.93, abs=1e-6)


def test_negative_origin_is_clamped_to_the_image() -> None:
    # YuNet happily returns boxes that start off the left or top edge.
    row = sample_row()
    row[0], row[1] = -25.0, -12.0
    face = face_from_row(row, image_shape=(480, 640))
    assert face.bbox[0] == 0
    assert face.bbox[1] == 0


def test_box_running_past_the_edge_is_trimmed() -> None:
    row = sample_row()
    row[0], row[2] = 600.0, 200.0  # x=600 w=200 on a 640-wide image
    face = face_from_row(row, image_shape=(480, 640))
    assert face.bbox[0] + face.bbox[2] <= 640


def test_row_from_face_round_trips() -> None:
    original = sample_row()
    face = face_from_row(original, image_shape=(480, 640))
    rebuilt = row_from_face(face)
    assert rebuilt.shape == (15,)
    assert rebuilt.dtype == np.float32
    np.testing.assert_allclose(rebuilt[4:14], original[4:14])
    assert rebuilt[14] == pytest.approx(0.93, abs=1e-6)


def test_detector_satisfies_the_protocol() -> None:
    from alchemyface.detection.base import Detector

    assert isinstance(YuNetDetector, type)
    assert hasattr(Detector, "detect")


@pytest.mark.models
def test_detector_returns_no_faces_for_a_blank_frame(model_dir: Path) -> None:
    detector = YuNetDetector(model_dir=model_dir)
    assert detector.detect(np.zeros((240, 320, 3), dtype=np.uint8)) == []


@pytest.mark.models
def test_detector_handles_a_changing_frame_size(model_dir: Path) -> None:
    # setInputSize must be re-issued whenever the frame dimensions change,
    # or OpenCV throws. Feeding two sizes in a row proves it is handled.
    detector = YuNetDetector(model_dir=model_dir)
    detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))


@pytest.mark.models
def test_detector_rejects_a_non_bgr_image(model_dir: Path) -> None:
    detector = YuNetDetector(model_dir=model_dir)
    with pytest.raises(ValueError, match="three-channel"):
        detector.detect(np.zeros((240, 320), dtype=np.uint8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `ImportError: cannot import name 'YuNetDetector'`

- [ ] **Step 3: Write the protocol**

Create `src/alchemyface/detection/base.py`:

```python
"""The detection seam."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from alchemyface.types import Face


@runtime_checkable
class Detector(Protocol):
    """Finds faces in a BGR image."""

    def detect(self, image: NDArray[np.uint8]) -> list[Face]:
        """Every face found, in the detector's own order. Empty list if none."""
```

- [ ] **Step 4: Write the implementation**

Create `src/alchemyface/detection/yunet.py`:

```python
"""YuNet face detection through OpenCV's DNN runtime.

OpenCV hands back an ``(N, 15) float32`` array — bounding box in columns 0-3,
five landmarks in 4-13, score in 14 — or ``None`` when nothing is found. The
box is not clamped to the image, so it can start at a negative coordinate or
run off the right edge; ``face_from_row`` fixes that before anyone slices an
array with it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from alchemyface.models import DETECTOR, resolve
from alchemyface.types import Face

_ROW_WIDTH = 15


def face_from_row(row: NDArray[np.float32], image_shape: tuple[int, ...]) -> Face:
    """Convert one OpenCV detection row into a :class:`Face`, clamped."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x = max(0, int(row[0]))
    y = max(0, int(row[1]))
    w = max(0, min(int(row[2]), width - x))
    h = max(0, min(int(row[3]), height - y))
    return Face(
        bbox=(x, y, w, h),
        landmarks=np.asarray(row[4:14], dtype=np.float32).reshape(5, 2),
        confidence=float(row[14]),
    )


def row_from_face(face: Face) -> NDArray[np.float32]:
    """Rebuild the 15-column row OpenCV needs for ``alignCrop``."""
    row = np.zeros(_ROW_WIDTH, dtype=np.float32)
    row[:4] = face.bbox
    row[4:14] = face.landmarks.reshape(-1)
    row[14] = face.confidence
    return row


class YuNetDetector:
    """Implements :class:`~alchemyface.detection.base.Detector` with YuNet."""

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        model_dir: Path | str | None = None,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        path = Path(model_path) if model_path else resolve(DETECTOR, model_dir)
        # The `Xxx.create` class-method form is what cv2's bundled type stubs
        # declare; the module-level `FaceDetectorYN_create` alias is not, and
        # trips mypy with "Module has no attribute".
        self._detector = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), score_threshold, nms_threshold, top_k
        )
        self._input_size = (320, 320)

    def detect(self, image: NDArray[np.uint8]) -> list[Face]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"expected a three-channel BGR image, got shape {image.shape}"
            )
        height, width = image.shape[:2]
        # The network is built for a fixed input size; it must be told
        # whenever the frame dimensions change or OpenCV raises.
        if (width, height) != self._input_size:
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)

        _, rows = self._detector.detect(image)
        if rows is None:
            return []
        # Iterating a 2-D ndarray yields rows, but the numpy stubs type the
        # element as a scalar, so each row is re-asserted as a float32 array.
        return [
            face_from_row(np.asarray(row, dtype=np.float32), image.shape)
            for row in rows
        ]
```

Modify `src/alchemyface/detection/__init__.py` to:

```python
"""Face detection. See ``base`` for the protocol these implement."""

from alchemyface.detection.base import Detector
from alchemyface.detection.yunet import YuNetDetector, face_from_row, row_from_face

__all__ = ["Detector", "YuNetDetector", "face_from_row", "row_from_face"]
```

- [ ] **Step 5: Run the fast tests, then the model tests**

Run: `make format && make check_type && make test`
Expected: PASS, with the three `models` tests reported as skipped.

Run: `make test_all`
Expected: PASS, with the `models` tests now executing.

- [ ] **Step 6: Commit**

```bash
git add src/alchemyface/detection tests/unit/test_yunet.py
git commit -m "feat: add Detector protocol and YuNet implementation"
```

---

### Task 5: Embedder protocol and SFaceEmbedder

**Files:**
- Create: `src/alchemyface/embedding/base.py`
- Create: `src/alchemyface/embedding/sface.py`
- Modify: `src/alchemyface/embedding/__init__.py`
- Test: `tests/unit/test_sface.py`

**Interfaces:**
- Consumes: `Face` (Task 1), `EMBEDDER` and `resolve` (Task 2), `row_from_face` (Task 4).
- Produces: `Embedder` protocol with a `dim` property and `embed(image, face) -> NDArray[np.float32]`; `SFaceEmbedder(model_path=None, model_dir=None)` whose `embed` returns a **unit-length** `(128,)` vector.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_sface.py`:

```python
"""SFace hands back an unnormalised (1, 128) row. The embedder's contract is
a flat, unit-length (128,) vector, which is what makes cosine similarity a
dot product everywhere downstream."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.detection import YuNetDetector
from alchemyface.embedding import SFaceEmbedder
from alchemyface.types import Face


def synthetic_face_image() -> np.ndarray:
    """A deterministic image with something face-shaped in it. It does not
    need to be a real face — alignCrop only uses the landmarks we supply."""
    rng = np.random.default_rng(seed=20260828)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def synthetic_face() -> Face:
    return Face(
        bbox=(80, 60, 120, 120),
        landmarks=np.array(
            [[110, 95], [170, 95], [140, 125], [115, 155], [165, 155]],
            dtype=np.float32,
        ),
        confidence=0.99,
    )


def test_embedder_declares_its_dimension() -> None:
    assert SFaceEmbedder.dim == 128


@pytest.mark.models
def test_embed_returns_a_flat_128_vector(model_dir: Path) -> None:
    embedder = SFaceEmbedder(model_dir=model_dir)
    vector = embedder.embed(synthetic_face_image(), synthetic_face())
    assert vector.shape == (128,)
    assert vector.dtype == np.float32


@pytest.mark.models
def test_embed_returns_a_unit_vector(model_dir: Path) -> None:
    # Measured: raw SFace output has an L2 norm around 10. The embedder must
    # normalise, or every cosine score downstream is wrong.
    embedder = SFaceEmbedder(model_dir=model_dir)
    vector = embedder.embed(synthetic_face_image(), synthetic_face())
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.models
def test_the_same_input_embeds_identically(model_dir: Path) -> None:
    embedder = SFaceEmbedder(model_dir=model_dir)
    image, face = synthetic_face_image(), synthetic_face()
    np.testing.assert_allclose(
        embedder.embed(image, face), embedder.embed(image, face), atol=1e-6
    )


@pytest.mark.models
def test_dot_product_of_a_vector_with_itself_is_one(model_dir: Path) -> None:
    embedder = SFaceEmbedder(model_dir=model_dir)
    vector = embedder.embed(synthetic_face_image(), synthetic_face())
    assert float(vector @ vector) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.models
def test_detect_then_embed_works_end_to_end(model_dir: Path) -> None:
    detector = YuNetDetector(model_dir=model_dir, score_threshold=0.3)
    embedder = SFaceEmbedder(model_dir=model_dir)
    image = synthetic_face_image()
    for face in detector.detect(image):
        assert embedder.embed(image, face).shape == (128,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `ImportError: cannot import name 'SFaceEmbedder'`

- [ ] **Step 3: Write the protocol**

Create `src/alchemyface/embedding/base.py`:

```python
"""The embedding seam."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from alchemyface.types import Face


@runtime_checkable
class Embedder(Protocol):
    """Turns a detected face into a comparable vector."""

    dim: int
    """Length of the vectors this embedder produces."""

    def embed(self, image: NDArray[np.uint8], face: Face) -> NDArray[np.float32]:
        """A unit-length embedding of ``face`` as it appears in ``image``.

        Implementations MUST return an L2-normalised, one-dimensional array of
        length ``dim``, so that a dot product between two of them is their
        cosine similarity.
        """
```

- [ ] **Step 4: Write the implementation**

Create `src/alchemyface/embedding/sface.py`:

```python
"""SFace embeddings through OpenCV's DNN runtime.

``alignCrop`` warps the face to a canonical 112x112 using the five landmarks,
and ``feature`` turns that into a ``(1, 128) float32`` row. That row is *not*
normalised — its L2 norm is around 10 — so this class flattens and normalises
it. Once every vector is unit length, cosine similarity is a dot product, which
is exactly what OpenCV's own ``FaceRecognizerSF.match`` computes and why
scikit-learn is not a dependency.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from alchemyface.detection import row_from_face
from alchemyface.models import EMBEDDER, resolve
from alchemyface.types import Face


class SFaceEmbedder:
    """Implements :class:`~alchemyface.embedding.base.Embedder` with SFace."""

    dim: int = 128

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        model_dir: Path | str | None = None,
    ) -> None:
        path = Path(model_path) if model_path else resolve(EMBEDDER, model_dir)
        self._recognizer = cv2.FaceRecognizerSF.create(str(path), "")

    def embed(self, image: NDArray[np.uint8], face: Face) -> NDArray[np.float32]:
        aligned = self._recognizer.alignCrop(image, row_from_face(face))
        raw = self._recognizer.feature(aligned)
        flat = np.asarray(raw, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(flat))
        if norm == 0.0:
            raise ValueError("SFace returned a zero embedding for this face")
        return (flat / norm).astype(np.float32)
```

Modify `src/alchemyface/embedding/__init__.py` to:

```python
"""Face embedding. See ``base`` for the protocol these implement."""

from alchemyface.embedding.base import Embedder
from alchemyface.embedding.sface import SFaceEmbedder

__all__ = ["Embedder", "SFaceEmbedder"]
```

- [ ] **Step 5: Run both test suites**

Run: `make format && make check_type && make test && make test_all`
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add src/alchemyface/embedding tests/unit/test_sface.py
git commit -m "feat: add Embedder protocol and SFace implementation"
```

---

### Task 6: Recognizer facade

**Files:**
- Create: `src/alchemyface/pipeline.py`
- Create: `tests/fakes.py`
- Modify: `src/alchemyface/__init__.py`
- Test: `tests/unit/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: `Recognizer(*, detector=None, embedder=None, store=None, model_dir=None, threshold=0.363)` with `detect`, `embed`, `enroll`, `identify`; `FakeDetector`, `FakeEmbedder` in `tests/fakes.py`; `alchemyface.__all__` exporting the public surface.

This is the task the protocol layout exists for: the whole pipeline is exercised with fakes, so these tests need no weights, no camera and no network.

- [ ] **Step 1: Write the fakes**

Create `tests/fakes.py`:

```python
"""Stand-ins for the two model-backed components.

The pipeline's job is sequencing and thresholding, not inference. Swapping in
fakes lets every branch of it be tested in milliseconds with no 37 MB download,
which is the entire reason Detector and Embedder are protocols.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from alchemyface.types import Face


def make_face(x: int = 0, y: int = 0, w: int = 10, h: int = 10) -> Face:
    return Face(
        bbox=(x, y, w, h),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        confidence=0.99,
    )


class FakeDetector:
    """Returns a canned list of faces and records how often it was called."""

    def __init__(self, faces: list[Face] | None = None) -> None:
        self.faces = faces if faces is not None else [make_face()]
        self.calls = 0

    def detect(self, image: NDArray[np.uint8]) -> list[Face]:
        self.calls += 1
        return list(self.faces)


class FakeEmbedder:
    """Maps each face to a one-hot unit vector, keyed by bbox width.

    One-hot vectors are mutually orthogonal, so two different faces score
    exactly 0.0 against each other and a face scores exactly 1.0 against
    itself. That makes threshold assertions exact rather than approximate.
    """

    dim = 8

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, image: NDArray[np.uint8], face: Face) -> NDArray[np.float32]:
        self.calls += 1
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[face.bbox[2] % self.dim] = 1.0
        return vector
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_pipeline.py`:

```python
"""Recognizer owns no algorithm — it sequences three protocols and applies a
threshold. These tests cover exactly that, using fakes."""

from __future__ import annotations

import numpy as np
import pytest

from alchemyface import Recognizer
from alchemyface.errors import NoFaceDetectedError
from alchemyface.store import InMemoryStore
from tests.fakes import FakeDetector, FakeEmbedder, make_face

IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)


def build(faces: list | None = None, threshold: float = 0.363) -> Recognizer:
    embedder = FakeEmbedder()
    return Recognizer(
        detector=FakeDetector(faces),
        embedder=embedder,
        store=InMemoryStore(dim=embedder.dim),
        threshold=threshold,
    )


def test_injected_components_are_used_as_given() -> None:
    recognizer = build()
    recognizer.detect(IMAGE)
    assert recognizer.detector.calls == 1  # type: ignore[attr-defined]


def test_store_defaults_to_the_embedder_dimension() -> None:
    assert build().store.dim == FakeEmbedder.dim  # type: ignore[attr-defined]


def test_enroll_stores_one_entry_and_returns_its_id() -> None:
    recognizer = build()
    entry_id = recognizer.enroll("ada", IMAGE)
    assert isinstance(entry_id, str)
    assert len(recognizer.store) == 1


def test_enroll_keeps_metadata() -> None:
    recognizer = build()
    recognizer.enroll("ada", IMAGE, metadata={"team": "analytical"})
    (match,) = recognizer.store.search(np.eye(8, dtype=np.float32)[10 % 8])
    assert match.metadata == {"team": "analytical"}


def test_enroll_picks_the_largest_face() -> None:
    small, large = make_face(w=3, h=3), make_face(w=5, h=40)
    recognizer = build([small, large])
    recognizer.enroll("ada", IMAGE)
    # FakeEmbedder keys on bbox width, so the stored vector identifies which
    # face was chosen: index 5, not index 3.
    (match,) = recognizer.store.search(np.eye(8, dtype=np.float32)[5])
    assert match.score == pytest.approx(1.0)


def test_enroll_raises_when_there_is_no_face() -> None:
    recognizer = build(faces=[])
    with pytest.raises(NoFaceDetectedError):
        recognizer.enroll("ada", IMAGE)


def test_identify_returns_one_recognition_per_face() -> None:
    recognizer = build([make_face(w=1), make_face(w=2), make_face(w=3)])
    assert len(recognizer.identify(IMAGE)) == 3


def test_identify_returns_nothing_for_an_empty_frame() -> None:
    assert build(faces=[]).identify(IMAGE) == []


def test_identify_matches_an_enrolled_face() -> None:
    recognizer = build()
    recognizer.enroll("ada", IMAGE)
    (recognition,) = recognizer.identify(IMAGE)
    assert recognition.match is not None
    assert recognition.match.label == "ada"
    assert recognition.match.score == pytest.approx(1.0)


def test_identify_reports_unknown_below_the_threshold() -> None:
    recognizer = build([make_face(w=10)])
    recognizer.enroll("ada", IMAGE)
    # A different width maps to an orthogonal vector: score 0.0 < 0.363.
    recognizer.detector.faces = [make_face(w=11)]  # type: ignore[attr-defined]
    (recognition,) = recognizer.identify(IMAGE)
    assert recognition.match is None
    assert recognition.face.bbox[2] == 11


def test_identify_against_an_empty_gallery_is_unknown_not_an_error() -> None:
    # The prototype crashed here: match() returned None and the caller
    # unpacked it regardless.
    (recognition,) = build().identify(IMAGE)
    assert recognition.match is None


def test_threshold_of_zero_accepts_an_orthogonal_match() -> None:
    recognizer = build([make_face(w=10)], threshold=0.0)
    recognizer.enroll("ada", IMAGE)
    recognizer.detector.faces = [make_face(w=11)]  # type: ignore[attr-defined]
    (recognition,) = recognizer.identify(IMAGE)
    assert recognition.match is not None
    assert recognition.match.label == "ada"


def test_default_threshold_is_the_sface_operating_point() -> None:
    assert build().threshold == pytest.approx(0.363)


def test_public_exports_are_importable_from_the_package_root() -> None:
    import alchemyface

    for name in ("Recognizer", "Face", "Match", "Recognition", "AlchemyFaceError"):
        assert hasattr(alchemyface, name), name
```

- [ ] **Step 3: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `ImportError: cannot import name 'Recognizer' from 'alchemyface'`

- [ ] **Step 4: Write the implementation**

Create `src/alchemyface/pipeline.py`:

```python
"""The facade that sequences detection, embedding and storage.

Recognizer deliberately contains no algorithm. It decides *order* and applies
the *threshold*; everything else is delegated to whichever Detector, Embedder
and FaceStore it was handed. That is what makes a pgvector gallery or a
different embedding model a drop-in rather than a rewrite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from alchemyface.detection.base import Detector
from alchemyface.embedding.base import Embedder
from alchemyface.errors import NoFaceDetectedError
from alchemyface.store.base import FaceStore
from alchemyface.types import Face, Recognition

DEFAULT_THRESHOLD = 0.363
"""SFace's published cosine operating point. A tunable, not a constant:
validate it against your own data before relying on it."""


class Recognizer:
    """Detect, embed, enroll and identify faces."""

    def __init__(
        self,
        *,
        detector: Detector | None = None,
        embedder: Embedder | None = None,
        store: FaceStore | None = None,
        model_dir: Path | str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        """Any component left as ``None`` gets the default implementation.

        Constructing the defaults loads the ONNX weights, downloading them on
        first use. Pass all three components to build a Recognizer that touches
        neither disk nor network.
        """
        # Imported here rather than at module scope so that injecting fakes
        # never pays the cost of importing cv2-backed modules.
        from alchemyface.detection.yunet import YuNetDetector
        from alchemyface.embedding.sface import SFaceEmbedder
        from alchemyface.store.memory import InMemoryStore

        self.detector: Detector = (
            detector if detector is not None else YuNetDetector(model_dir=model_dir)
        )
        self.embedder: Embedder = (
            embedder if embedder is not None else SFaceEmbedder(model_dir=model_dir)
        )
        self.store: FaceStore = (
            store if store is not None else InMemoryStore(dim=self.embedder.dim)
        )
        self.threshold = threshold

    def detect(self, image: NDArray[np.uint8]) -> list[Face]:
        """Every face in the image."""
        return self.detector.detect(image)

    def embed(self, image: NDArray[np.uint8], face: Face) -> NDArray[np.float32]:
        """The unit-length embedding of one already-detected face."""
        return self.embedder.embed(image, face)

    def enroll(
        self,
        label: str,
        image: NDArray[np.uint8],
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Add the most prominent face in the image to the gallery.

        Raises :class:`NoFaceDetectedError` if there is no face. Enrolling the
        largest face is a deliberate choice: an enrolment photo with a
        bystander in the background should not silently enrol the bystander.
        """
        faces = self.detect(image)
        if not faces:
            raise NoFaceDetectedError(
                f"no face found in the image supplied for {label!r}"
            )
        face = max(faces, key=lambda candidate: candidate.area)
        return self.store.add(label, self.embed(image, face), metadata)

    def identify(self, image: NDArray[np.uint8]) -> list[Recognition]:
        """One :class:`Recognition` per detected face.

        ``Recognition.match`` is ``None`` when the best candidate falls below
        the threshold, or when the gallery is empty — the caller decides what
        "unknown" should mean rather than the library guessing a label.
        """
        recognitions: list[Recognition] = []
        for face in self.detect(image):
            candidates = self.store.search(self.embed(image, face), k=1)
            best = candidates[0] if candidates else None
            if best is not None and best.score < self.threshold:
                best = None
            recognitions.append(Recognition(face=face, match=best))
        return recognitions
```

Modify `src/alchemyface/__init__.py` to:

```python
"""AlchemyFace — face detection and recognition on YuNet and SFace.

The public surface is deliberately small::

    from alchemyface import Recognizer

    r = Recognizer()
    r.enroll("prashant", image)
    r.identify(frame)

Everything else is a seam. ``Detector``, ``Embedder`` and ``FaceStore`` are
protocols, so any conforming object can be substituted without touching the
pipeline.
"""

from alchemyface.errors import (
    AlchemyFaceError,
    ModelDownloadError,
    ModelNotFoundError,
    NoFaceDetectedError,
)
from alchemyface.pipeline import DEFAULT_THRESHOLD, Recognizer
from alchemyface.types import Face, Match, Recognition

__version__ = "0.1.0"

__all__ = [
    "AlchemyFaceError",
    "DEFAULT_THRESHOLD",
    "Face",
    "Match",
    "ModelDownloadError",
    "ModelNotFoundError",
    "NoFaceDetectedError",
    "Recognition",
    "Recognizer",
    "__version__",
]
```

- [ ] **Step 5: Run tests and the type checker**

Run: `make format && make check_type && make test`
Expected: all tests PASS, mypy success.

- [ ] **Step 6: Prove the real pipeline works end to end**

Run:

```bash
export VIRTUAL_ENV="$(pyenv prefix alchemyface)"; export PATH="$VIRTUAL_ENV/bin:$PATH"
ALCHEMYFACE_MODEL_DIR=_local/onnx python -c "
import numpy as np
from alchemyface import Recognizer
r = Recognizer()
print('detector:', type(r.detector).__name__)
print('embedder:', type(r.embedder).__name__, r.embedder.dim)
print('faces in a blank frame:', r.identify(np.zeros((240,320,3), np.uint8)))
"
```

Expected: `YuNetDetector`, `SFaceEmbedder 128`, and an empty list. No download occurs, because the aliases resolve `_local/onnx`.

- [ ] **Step 7: Commit**

```bash
git add src/alchemyface/pipeline.py src/alchemyface/__init__.py tests/fakes.py tests/unit/test_pipeline.py
git commit -m "feat: add Recognizer facade wiring the three protocols"
```

---

### Task 7: CLI commands

**Files:**
- Modify: `src/alchemyface/cli.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `Recognizer` (Task 6), `InMemoryStore` (Task 3), `MODELS`/`download`/`cache_dir` (Task 2).
- Produces: commands `version`, `download-models`, `enroll`, `identify`. Keep the existing `@app.callback()` — without it Typer promotes a lone command to the top level.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_cli.py`:

```python
"""CLI wiring only. The heavy lifting is already covered; these tests check
argument handling, exit codes and output shape, with the Recognizer patched
out so nothing loads a model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from alchemyface import cli
from alchemyface.models import ModelSpec
from alchemyface.store import InMemoryStore
from tests.fakes import FakeDetector, FakeEmbedder, make_face

runner = CliRunner()


@pytest.fixture()
def image_file(tmp_path: Path) -> Path:
    """A real PNG on disk, so cv2.imread succeeds without any face in it."""
    import cv2

    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.zeros((60, 60, 3), dtype=np.uint8))
    return path


@pytest.fixture()
def fake_recognizer(monkeypatch: pytest.MonkeyPatch):
    """Patch the CLI's Recognizer factory to build a fake-backed one."""
    from alchemyface.pipeline import Recognizer

    embedder = FakeEmbedder()
    recognizer = Recognizer(
        detector=FakeDetector([make_face(w=4)]),
        embedder=embedder,
        store=InMemoryStore(dim=embedder.dim),
    )
    monkeypatch.setattr(cli, "_build_recognizer", lambda **kwargs: recognizer)
    return recognizer


def test_version_prints_the_package_version() -> None:
    from alchemyface import __version__

    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help_lists_every_command() -> None:
    result = runner.invoke(cli.app, ["--help"])
    for command in ("version", "download-models", "enroll", "identify"):
        assert command in result.stdout


def test_enroll_writes_a_gallery(
    image_file: Path, tmp_path: Path, fake_recognizer
) -> None:
    gallery = tmp_path / "g.npz"
    result = runner.invoke(
        cli.app,
        ["enroll", "--name", "ada", "--image", str(image_file), "--gallery", str(gallery)],
    )
    assert result.exit_code == 0, result.stdout
    assert gallery.exists()
    assert "ada" in result.stdout


def test_enroll_appends_to_an_existing_gallery(
    image_file: Path, tmp_path: Path, fake_recognizer
) -> None:
    gallery = tmp_path / "g.npz"
    for name in ("ada", "grace"):
        result = runner.invoke(
            cli.app,
            ["enroll", "--name", name, "--image", str(image_file), "--gallery", str(gallery)],
        )
        assert result.exit_code == 0, result.stdout
    store = InMemoryStore(dim=FakeEmbedder.dim)
    store.load(gallery)
    assert len(store) == 2


def test_enroll_exits_nonzero_when_no_face_is_found(
    image_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alchemyface.pipeline import Recognizer

    embedder = FakeEmbedder()
    empty = Recognizer(
        detector=FakeDetector([]),
        embedder=embedder,
        store=InMemoryStore(dim=embedder.dim),
    )
    monkeypatch.setattr(cli, "_build_recognizer", lambda **kwargs: empty)
    result = runner.invoke(
        cli.app,
        ["enroll", "--name", "ada", "--image", str(image_file),
         "--gallery", str(tmp_path / "g.npz")],
    )
    assert result.exit_code == 1
    assert "no face" in result.stdout.lower()


def test_enroll_exits_nonzero_for_a_missing_image(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["enroll", "--name", "ada", "--image", str(tmp_path / "absent.png"),
         "--gallery", str(tmp_path / "g.npz")],
    )
    assert result.exit_code != 0


def test_identify_reports_a_known_face(
    image_file: Path, tmp_path: Path, fake_recognizer
) -> None:
    gallery = tmp_path / "g.npz"
    runner.invoke(
        cli.app,
        ["enroll", "--name", "ada", "--image", str(image_file), "--gallery", str(gallery)],
    )
    result = runner.invoke(
        cli.app, ["identify", "--image", str(image_file), "--gallery", str(gallery)]
    )
    assert result.exit_code == 0, result.stdout
    assert "ada" in result.stdout


def test_identify_reports_unknown_against_an_empty_gallery(
    image_file: Path, tmp_path: Path, fake_recognizer
) -> None:
    gallery = tmp_path / "empty.npz"
    InMemoryStore(dim=FakeEmbedder.dim).save(gallery)
    result = runner.invoke(
        cli.app, ["identify", "--image", str(image_file), "--gallery", str(gallery)]
    )
    assert result.exit_code == 0
    assert "unknown" in result.stdout.lower()


def test_download_models_reports_each_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_download(spec: ModelSpec, dest_dir: Path | None = None) -> Path:
        seen.append(spec.key)
        return tmp_path / spec.filename

    monkeypatch.setattr(cli, "download", fake_download)
    monkeypatch.setattr(cli, "find_local", lambda spec, model_dir=None: None)
    result = runner.invoke(cli.app, ["download-models"])
    assert result.exit_code == 0
    assert sorted(seen) == ["detector", "embedder"]


def test_download_models_skips_what_is_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "find_local", lambda spec, model_dir=None: tmp_path / spec.filename)
    monkeypatch.setattr(
        cli, "download", lambda spec, dest_dir=None: pytest.fail("should not download")
    )
    result = runner.invoke(cli.app, ["download-models"])
    assert result.exit_code == 0
    assert "already" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `AttributeError: module 'alchemyface.cli' has no attribute '_build_recognizer'`

- [ ] **Step 3: Write the implementation**

Replace `src/alchemyface/cli.py` with:

```python
"""Command line entry point for AlchemyFace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from numpy.typing import NDArray

from alchemyface import __version__
from alchemyface.errors import AlchemyFaceError
from alchemyface.models import MODELS, download, find_local
from alchemyface.pipeline import DEFAULT_THRESHOLD, Recognizer
from alchemyface.store.memory import InMemoryStore

app = typer.Typer(
    name="alchemyface",
    help="Face detection and recognition on YuNet and SFace.",
    no_args_is_help=True,
    add_completion=False,
)


# Typer promotes a lone command to the top level, which would make `alchemyface
# version` an unexpected argument. An explicit callback keeps subcommand mode on
# regardless of how many commands are registered.
@app.callback()
def main() -> None:
    """Face detection and recognition on YuNet and SFace."""


def _build_recognizer(**kwargs: object) -> Recognizer:
    """Indirection so tests can substitute a fake-backed Recognizer."""
    return Recognizer(**kwargs)  # type: ignore[arg-type]


def _read_image(path: Path) -> NDArray[np.uint8]:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        typer.echo(f"could not read an image from {path}", err=True)
        raise typer.Exit(code=2)
    # imread's default IMREAD_COLOR always yields 8-bit 3-channel BGR, so this
    # is a no-op at runtime; it is here to state the dtype for the type checker.
    return np.asarray(image, dtype=np.uint8)


def _load_gallery(recognizer: Recognizer, gallery: Path) -> None:
    store = recognizer.store
    if gallery.exists() and isinstance(store, InMemoryStore):
        store.load(gallery)


def _save_gallery(recognizer: Recognizer, gallery: Path) -> None:
    store = recognizer.store
    if isinstance(store, InMemoryStore):
        gallery.parent.mkdir(parents=True, exist_ok=True)
        store.save(gallery)


@app.command()
def version() -> None:
    """Print the installed AlchemyFace version."""
    typer.echo(__version__)


@app.command("download-models")
def download_models(
    model_dir: Path | None = typer.Option(
        None, "--model-dir", help="Where to write the weights (defaults to the cache)."
    ),
) -> None:
    """Fetch the ONNX weights ahead of first use."""
    for spec in MODELS.values():
        existing = find_local(spec, model_dir)
        if existing is not None:
            typer.echo(f"{spec.key}: already present at {existing}")
            continue
        typer.echo(f"{spec.key}: downloading {spec.filename} …")
        try:
            path = download(spec, dest_dir=model_dir)
        except AlchemyFaceError as exc:
            typer.echo(f"{spec.key}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"{spec.key}: saved to {path}")


@app.command()
def enroll(
    name: str = typer.Option(..., "--name", help="Label to store the face under."),
    image: Path = typer.Option(
        ..., "--image", exists=True, dir_okay=False, help="Photo containing one face."
    ),
    gallery: Path = typer.Option(..., "--gallery", help="Gallery .npz to create or extend."),
    model_dir: Path | None = typer.Option(None, "--model-dir", help="Directory holding the weights."),
) -> None:
    """Add the most prominent face in an image to a gallery."""
    recognizer = _build_recognizer(model_dir=model_dir)
    _load_gallery(recognizer, gallery)
    try:
        recognizer.enroll(name, _read_image(image))
    except AlchemyFaceError as exc:
        typer.echo(f"could not enroll {name}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _save_gallery(recognizer, gallery)
    typer.echo(f"enrolled {name} — gallery now holds {len(recognizer.store)} face(s)")


@app.command()
def identify(
    image: Path = typer.Option(
        ..., "--image", exists=True, dir_okay=False, help="Image to search."
    ),
    gallery: Path = typer.Option(
        ..., "--gallery", exists=True, dir_okay=False, help="Gallery .npz to search against."
    ),
    threshold: float = typer.Option(
        DEFAULT_THRESHOLD, "--threshold", help="Cosine score required to accept a match."
    ),
    model_dir: Path | None = typer.Option(None, "--model-dir", help="Directory holding the weights."),
) -> None:
    """Report who each face in an image looks like."""
    recognizer = _build_recognizer(model_dir=model_dir, threshold=threshold)
    recognizer.threshold = threshold
    _load_gallery(recognizer, gallery)
    recognitions = recognizer.identify(_read_image(image))
    if not recognitions:
        typer.echo("no faces detected")
        return
    for recognition in recognitions:
        x, y, w, h = recognition.face.bbox
        if recognition.match is None:
            typer.echo(f"unknown          at ({x},{y},{w},{h})")
        else:
            typer.echo(
                f"{recognition.match.label:<16} at ({x},{y},{w},{h}) "
                f"score={recognition.match.score:.3f}"
            )


if __name__ == "__main__":  # pragma: no cover
    app()
```

Note: `_build_recognizer` ignores `threshold` when a test substitutes it, which is why `identify` also assigns `recognizer.threshold` directly.

- [ ] **Step 4: Run tests and the type checker**

Run: `make format && make check_type && make test`
Expected: all tests PASS, mypy success.

- [ ] **Step 5: Exercise the CLI by hand**

Run:

```bash
export VIRTUAL_ENV="$(pyenv prefix alchemyface)"; export PATH="$VIRTUAL_ENV/bin:$PATH"
export ALCHEMYFACE_MODEL_DIR=_local/onnx
alchemyface --help
alchemyface download-models          # should report both as already present
```

Expected: four commands listed; both models reported present, nothing downloaded.

- [ ] **Step 6: Commit**

```bash
git add src/alchemyface/cli.py tests/unit/test_cli.py
git commit -m "feat: add download-models, enroll and identify commands"
```

---

### Task 8: VideoSource and documentation

**Files:**
- Create: `src/alchemyface/capture.py`
- Test: `tests/unit/test_capture.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `VideoSource(source=0, width=None, height=None)` — a context manager with `read() -> NDArray[np.uint8] | None`, `frames() -> Iterator[NDArray[np.uint8]]`, `release() -> None`.

This replaces `scripts/videocapture.py` from the prototype, which defined a perfectly reasonable `AbstractVideoCapture` interface and then never used it.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_capture.py`:

```python
"""VideoSource is a thin wrapper over cv2.VideoCapture. Everything except the
"open a real camera" path is tested with a stub capture object."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alchemyface import capture
from alchemyface.capture import VideoSource

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


class StubCapture:
    """Stands in for cv2.VideoCapture."""

    def __init__(self, frames: int = 2, opens: bool = True) -> None:
        self._remaining = frames
        self._opens = opens
        self.released = False
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self._opens

    def set(self, prop: int, value: float) -> bool:
        self.properties[prop] = value
        return True

    def read(self) -> tuple[bool, Any]:
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, FRAME.copy()

    def release(self) -> None:
        self.released = True


@pytest.fixture()
def stub(monkeypatch: pytest.MonkeyPatch) -> StubCapture:
    instance = StubCapture()
    monkeypatch.setattr(capture.cv2, "VideoCapture", lambda source: instance)
    return instance


def test_read_returns_a_frame(stub: StubCapture) -> None:
    with VideoSource() as source:
        assert source.read() is not None


def test_read_returns_none_when_the_stream_ends(stub: StubCapture) -> None:
    with VideoSource() as source:
        source.read()
        source.read()
        assert source.read() is None


def test_frames_iterates_until_the_stream_ends(stub: StubCapture) -> None:
    with VideoSource() as source:
        assert len(list(source.frames())) == 2


def test_context_manager_releases_the_capture(stub: StubCapture) -> None:
    with VideoSource():
        pass
    assert stub.released is True


def test_release_is_idempotent(stub: StubCapture) -> None:
    source = VideoSource()
    source.release()
    source.release()
    assert stub.released is True


def test_a_camera_that_will_not_open_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        capture.cv2, "VideoCapture", lambda source: StubCapture(opens=False)
    )
    with pytest.raises(RuntimeError, match="could not open"):
        VideoSource(source=7)


def test_requested_resolution_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    instance = StubCapture()
    monkeypatch.setattr(capture.cv2, "VideoCapture", lambda source: instance)
    VideoSource(width=1280, height=720).release()
    assert instance.properties[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert instance.properties[cv2.CAP_PROP_FRAME_HEIGHT] == 720


@pytest.mark.camera
def test_a_real_camera_yields_a_frame() -> None:
    with VideoSource() as source:
        assert source.read() is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `ModuleNotFoundError: No module named 'alchemyface.capture'`

- [ ] **Step 3: Write the implementation**

Create `src/alchemyface/capture.py`:

```python
"""Reading frames from a camera or a video file.

A thin wrapper over ``cv2.VideoCapture`` that fails loudly when the device
will not open, releases itself on the way out of a ``with`` block, and offers
an iterator so callers do not have to write the read-check-read loop by hand.
"""

from __future__ import annotations

from types import TracebackType
from typing import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray


class VideoSource:
    """A camera index or a path to a video file, as a context manager."""

    def __init__(
        self,
        source: int | str = 0,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"could not open video source {source!r}")
        if width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self._released = False

    def read(self) -> NDArray[np.uint8] | None:
        """The next frame, or ``None`` once the stream is exhausted."""
        ok, frame = self._capture.read()
        return frame if ok else None

    def frames(self) -> Iterator[NDArray[np.uint8]]:
        """Yield frames until the stream ends."""
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def release(self) -> None:
        """Release the device. Safe to call more than once."""
        if not self._released:
            self._capture.release()
            self._released = True

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
```

- [ ] **Step 4: Run tests and the type checker**

Run: `make format && make check_type && make test`
Expected: all tests PASS (the `camera` test skipped), mypy success.

- [ ] **Step 5: Update the README**

In `README.md`, delete the "Status: pre-release" blockquote — the API it warned about now exists. Then add this section immediately after the "Bring your own components" section:

```markdown
### Live video

```python
from alchemyface import Recognizer
from alchemyface.capture import VideoSource

r = Recognizer()
r.store.load("gallery.npz")

with VideoSource(0, width=1280, height=720) as camera:
    for frame in camera.frames():
        for face, match in ((x.face, x.match) for x in r.identify(frame)):
            print(match.label if match else "unknown", face.bbox)
```
```

- [ ] **Step 6: Full verification**

Run:

```bash
export VIRTUAL_ENV="$(pyenv prefix alchemyface)"; export PATH="$VIRTUAL_ENV/bin:$PATH"
make format check_formatted check_type lint test test_all build
unzip -l dist/*.whl | grep -icE '_local|face_data|sounds|\.npy|\.onnx|\.mp3'
```

Expected: every target passes, and the final count is `0`.

- [ ] **Step 7: Commit**

```bash
git add src/alchemyface/capture.py tests/unit/test_capture.py README.md
git commit -m "feat: add VideoSource and document live-video usage"
```

---

## Done when

- [ ] `make check_formatted check_type lint test` all pass.
- [ ] `make test_all` passes with `ALCHEMYFACE_MODEL_DIR=_local/onnx`.
- [ ] Coverage is at or above 80%.
- [ ] `make build` produces a wheel containing only `alchemyface/**` — no `_local`, no `.onnx`, no `.npy`, no `.mp3`.
- [ ] `git status --porcelain` never lists anything under `_local/`.
- [ ] The three spec open questions are resolved before any `poetry publish`: licence, the real repository URL in `pyproject.toml`, and whether to declare a Python upper bound.
