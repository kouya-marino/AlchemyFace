"""`import alchemyface` must not require tkinter.

On Debian and Ubuntu, tkinter is a separate OS package (`python3-tk`) that a
`pip install` cannot pull in. If any library module imported the GUI, then
`import alchemyface` would fail on those systems — a silent regression of the
0.1.0 library API for everyone who never wanted the GUI.

This is the only test that catches it, because on a developer machine tkinter
is present and the mistake is invisible.
"""

from __future__ import annotations

import subprocess
import sys

TK_MODULES = ("tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox")


def test_library_imports_with_tkinter_unavailable() -> None:
    """Import the whole public surface in a subprocess where tkinter is blocked."""
    script = """
import sys

class Blocker:
    def find_module(self, name, path=None):
        if name == "tkinter" or name.startswith("tkinter."):
            return self
        return None
    def load_module(self, name):
        raise ImportError(f"No module named {name!r} (blocked by test)")

sys.meta_path.insert(0, Blocker())
for name in list(sys.modules):
    if name == "tkinter" or name.startswith("tkinter."):
        del sys.modules[name]

import alchemyface
from alchemyface import Recognizer, Face, Match, Recognition
from alchemyface.store import PickleStore, InMemoryStore
from alchemyface.detection import YuNetDetector
from alchemyface.embedding import SFaceEmbedder
from alchemyface import models, capture, pipeline, types, errors

try:
    import tkinter
except ImportError:
    pass
else:
    raise AssertionError("tkinter was importable — the blocker did not work")

print("OK", alchemyface.__version__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"importing alchemyface without tkinter failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.stdout.startswith("OK")


def test_cli_module_imports_without_tkinter() -> None:
    """The CLI is a console-script entry point, so it must import too.

    It may only reach for the GUI inside the body of the command that needs it.
    """
    script = """
import sys

class Blocker:
    def find_module(self, name, path=None):
        if name == "tkinter" or name.startswith("tkinter."):
            return self
        return None
    def load_module(self, name):
        raise ImportError(f"No module named {name!r} (blocked by test)")

sys.meta_path.insert(0, Blocker())
from alchemyface.cli import app
print("OK")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert result.stdout.strip() == "OK"


def test_no_library_module_imports_tkinter_at_module_scope() -> None:
    """A static check, so the failure names the offending file.

    The subprocess tests above prove the behaviour; this one localises it.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent / "src" / "alchemyface"
    offenders = []
    for path in root.rglob("*.py"):
        if "gui" in path.relative_to(root).parts:
            continue  # the GUI is allowed to import tkinter, obviously
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import tkinter", "from tkinter")):
                # An import inside a function body is fine; only module scope
                # is a problem, and module scope means no leading whitespace.
                if not line.startswith((" ", "\t")):
                    offenders.append(f"{path.relative_to(root)}:{lineno}")
    assert not offenders, f"tkinter imported at module scope in: {offenders}"


def test_db_command_fails_helpfully_when_tkinter_is_missing() -> None:
    """Without tkinter the command must explain the fix, not traceback."""
    script = """
import sys

class Blocker:
    def find_module(self, name, path=None):
        if name == "tkinter" or name.startswith("tkinter."):
            return self
        return None
    def load_module(self, name):
        raise ImportError(f"No module named {name!r} (blocked by test)")

sys.meta_path.insert(0, Blocker())
for name in list(sys.modules):
    if name == "tkinter" or name.startswith("tkinter."):
        del sys.modules[name]

from typer.testing import CliRunner
from alchemyface.cli import app

result = CliRunner().invoke(app, ["db"])
print("EXIT", result.exit_code)
print(result.output)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "EXIT 3" in result.stdout, result.stdout
    assert "python3-tk" in result.stdout, result.stdout


def test_db_command_is_registered() -> None:
    from typer.testing import CliRunner

    from alchemyface.cli import app

    out = CliRunner().invoke(app, ["--help"]).output
    assert "db" in out
