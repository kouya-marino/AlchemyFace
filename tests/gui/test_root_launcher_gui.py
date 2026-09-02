"""`python main.py` actually opens a window.

The display-free half of this is in tests/unit/test_root_launcher.py. This is
the part that needs Tk, and it runs the launcher as a real subprocess rather
than importing it, because that is how a user invokes it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui

LAUNCHER = Path(__file__).resolve().parents[2] / "main.py"


def test_python_main_py_opens_a_window() -> None:
    """A window blocks on `mainloop()`, so a timeout is the success signal —
    anything else means it exited, which for this script means it failed."""
    try:
        done = subprocess.run(
            [sys.executable, str(LAUNCHER)],
            capture_output=True,
            text=True,
            timeout=25,
        )
    except subprocess.TimeoutExpired:
        return  # still running: the window is up
    pytest.fail(f"main.py exited with {done.returncode}\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}")


def test_it_can_be_run_from_another_directory(tmp_path: Path) -> None:
    """Paths inside the launcher are resolved against the file, not the caller."""
    try:
        done = subprocess.run(
            [sys.executable, str(LAUNCHER)],
            capture_output=True,
            text=True,
            timeout=25,
            cwd=tmp_path,
        )
    except subprocess.TimeoutExpired:
        return
    pytest.fail(f"main.py exited with {done.returncode} when run from {tmp_path}\n{done.stderr}")
