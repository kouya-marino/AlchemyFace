# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

_Nothing yet._

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

[Unreleased]: https://github.com/kouya-marino/AlchemyFace/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kouya-marino/AlchemyFace/releases/tag/v0.1.0
