"""`import alchemyface` must not require tkinter.

On Debian and Ubuntu, tkinter is a separate OS package (`python3-tk`) that a
`pip install` cannot pull in. If any library module imported the GUI, then
`import alchemyface` would fail on those systems — a silent regression of the
0.1.0 library API for everyone who never wanted the GUI.

These are the only tests that catch it, because on a developer machine tkinter
is present and the mistake is invisible.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Installed as a meta-path finder in a subprocess so tkinter genuinely cannot be
# imported. `find_spec` is the modern API and the only one that exists from
# Python 3.12, which removed `find_module` — an earlier version of this file
# used the legacy API and so blocked nothing on 3.12. The self-check at the end
# of each script is what exposed that; without it these tests would have passed
# vacuously.
BLOCKER = """
import sys


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "tkinter" or name.startswith("tkinter."):
            raise ImportError(f"No module named {name!r} (blocked by test)")
        return None


sys.meta_path.insert(0, Blocker())
for _name in list(sys.modules):
    if _name == "tkinter" or _name.startswith("tkinter."):
        del sys.modules[_name]


def assert_tkinter_is_blocked():
    try:
        import tkinter  # noqa: F401
    except ImportError:
        return
    raise AssertionError("tkinter was importable - the blocker did not work")
"""


def run_blocked(body: str) -> subprocess.CompletedProcess[str]:
    """Run `body` in a subprocess where tkinter cannot be imported."""
    return subprocess.run(
        [sys.executable, "-c", BLOCKER + body],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_library_imports_with_tkinter_unavailable() -> None:
    result = run_blocked("""
import alchemyface
from alchemyface import Recognizer, Face, Match, Recognition, StoreEntry
from alchemyface.store import PickleStore, InMemoryStore
from alchemyface.detection import YuNetDetector
from alchemyface.embedding import SFaceEmbedder
from alchemyface import models, capture, pipeline, types, errors

assert_tkinter_is_blocked()
print("OK", alchemyface.__version__)
""")
    assert result.returncode == 0, (
        f"importing alchemyface without tkinter failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.stdout.startswith("OK"), result.stdout


def test_pure_gui_helpers_import_without_tkinter() -> None:
    """The presentation layer is deliberately Tk-free, so it must import too."""
    result = run_blocked("""
import numpy as np
from alchemyface.gui.inspect_data import entry_rows, summarise
from alchemyface.store import PickleStore

assert_tkinter_is_blocked()
store = PickleStore(dim=4)
vector = np.zeros(4, dtype=np.float32)
vector[0] = 11.0
store.add("ada", vector, {"group": "ceo"})
assert entry_rows(store)[0].norm == 11.0
assert "1 entries" in str(summarise(store, None))
print("OK")
""")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert result.stdout.strip() == "OK"


def test_cli_module_imports_without_tkinter() -> None:
    """The CLI is a console-script entry point, so it must import too.

    It may only reach for the GUI inside the body of the command that needs it.
    """
    result = run_blocked("""
from alchemyface.cli import app

assert_tkinter_is_blocked()
print("OK")
""")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert result.stdout.strip() == "OK"


def test_db_command_fails_helpfully_when_tkinter_is_missing() -> None:
    """Without tkinter the command must explain the fix, not traceback."""
    result = run_blocked("""
from typer.testing import CliRunner
from alchemyface.cli import app

assert_tkinter_is_blocked()
outcome = CliRunner().invoke(app, ["db"])
print("EXIT", outcome.exit_code)
print(outcome.output)
""")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "EXIT 3" in result.stdout, result.stdout
    assert "python3-tk" in result.stdout, result.stdout


def test_db_command_is_registered() -> None:
    from typer.testing import CliRunner

    from alchemyface.cli import app

    assert "db" in CliRunner().invoke(app, ["--help"]).output


def test_no_library_module_imports_tkinter_at_module_scope() -> None:
    """A static check, so a failure names the offending file.

    The subprocess tests prove the behaviour; this one localises it.
    """
    root = Path(__file__).resolve().parent.parent.parent / "src" / "alchemyface"
    offenders = []
    for path in root.rglob("*.py"):
        if "gui" in path.relative_to(root).parts:
            continue  # the GUI is allowed to import tkinter, obviously
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.startswith((" ", "\t")) and line.strip().startswith(("import tkinter", "from tkinter")):
                offenders.append(f"{path.relative_to(root)}:{lineno}")
    assert not offenders, f"tkinter imported at module scope in: {offenders}"
