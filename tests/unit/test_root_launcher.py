"""The repository's ``main.py``.

The installed entry points are covered in test_cli.py. This file covers the
launcher at the repository root, which exists for a bare ``git clone`` with
nothing installed — the way the application this was ported from is started.

Nothing here opens a window; tests/gui/test_root_launcher_gui.py does that.
"""

from __future__ import annotations

import builtins
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "main.py"


def load_launcher() -> ModuleType:
    """Import ``main.py`` by path, since it is not part of the package."""
    spec = importlib.util.spec_from_file_location("_root_main", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_launcher_exists() -> None:
    """`python main.py` is the first thing anyone who knows the original will
    try, and for a long time this repository had no answer to it."""
    assert LAUNCHER.is_file()


def test_it_points_src_at_the_checkout_it_lives_in() -> None:
    module = load_launcher()
    assert module.SRC == REPO_ROOT / "src"
    # Resolved, not relative, so it works when run from another directory.
    assert module.SRC.is_absolute()


def test_an_installed_package_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepending src/ unconditionally would shadow an installed copy with the
    working tree — a surprising thing for a launcher to do when they differ."""
    module = load_launcher()
    before = list(sys.path)
    assert module.ensure_importable() is False
    assert sys.path == before


def test_a_bare_checkout_gets_src_on_the_path(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_launcher()
    real_import = builtins.__import__

    def hide_alchemyface(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "alchemyface" and str(module.SRC) not in sys.path:
            raise ImportError("hidden for the test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", hide_alchemyface)
    monkeypatch.delitem(sys.modules, "alchemyface", raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if str(module.SRC) != p])

    assert module.ensure_importable() is True
    assert str(module.SRC) in sys.path


def test_it_raises_rather_than_hiding_a_genuinely_missing_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no src/ to fall back on, the ImportError is the honest answer."""
    module = load_launcher()
    real_import = builtins.__import__

    def hide_alchemyface(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "alchemyface":
            raise ImportError("hidden for the test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", hide_alchemyface)
    monkeypatch.delitem(sys.modules, "alchemyface", raising=False)
    monkeypatch.setattr(module, "SRC", tmp_path / "no-such-src")

    with pytest.raises(ImportError):
        module.ensure_importable()


def test_it_delegates_to_the_db_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delegating rather than reimplementing is what keeps `python main.py`
    from drifting away from `alchemyface db` — same behaviour, same message
    when tkinter is missing, same exit code."""
    from alchemyface import cli

    called: list[list[str]] = []
    monkeypatch.setattr(cli, "app", lambda argv: called.append(list(argv)))
    load_launcher().main()
    assert called == [["db"]]


def test_the_launcher_is_not_shipped_in_the_wheel() -> None:
    """It is a checkout convenience. A root module in the wheel would install a
    top-level `main` into site-packages, which is not ours to claim.

    Read from pyproject as text rather than parsed: `tomllib` arrived in 3.11
    and this suite supports 3.10. CI asserts the built wheel itself, which is
    the artefact this is standing in for.
    """
    config = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'where = ["src"]' in config
    assert 'include = ["alchemyface*"]' in config
    assert "py-modules" not in config and "py_modules" not in config


@pytest.mark.parametrize("target", ["ruff check", "ruff format --check", "mypy"])
def test_the_launcher_is_covered_by_the_linters(target: str) -> None:
    """An unchecked file at the root is a file that rots."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    lines = [line for line in workflow.splitlines() if target in line]
    assert lines, f"{target} does not appear in ci.yml"
    assert all("main.py" in line for line in lines), f"{target} does not include main.py: {lines}"


def test_it_passes_its_own_linters() -> None:
    """Not merely listed in CI — actually clean, checked here so a local run
    catches it too."""
    for command in (
        [sys.executable, "-m", "ruff", "check", str(LAUNCHER)],
        [sys.executable, "-m", "ruff", "format", "--check", str(LAUNCHER)],
    ):
        done = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT, timeout=120)
        assert done.returncode == 0, f"{command[2:]} failed:\n{done.stdout}{done.stderr}"
