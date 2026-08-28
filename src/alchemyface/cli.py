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
    # Imported lazily so `alchemyface version` does not pay for loading cv2.
    import cv2  # pylint: disable=import-outside-toplevel

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
    gallery: Path = typer.Option(
        ..., "--gallery", help="Gallery .npz to create or extend."
    ),
    model_dir: Path | None = typer.Option(
        None, "--model-dir", help="Directory holding the weights."
    ),
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
        ...,
        "--gallery",
        exists=True,
        dir_okay=False,
        help="Gallery .npz to search against.",
    ),
    threshold: float = typer.Option(
        DEFAULT_THRESHOLD, "--threshold", help="Cosine score required to accept."
    ),
    model_dir: Path | None = typer.Option(
        None, "--model-dir", help="Directory holding the weights."
    ),
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
