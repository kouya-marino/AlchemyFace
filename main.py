#!/usr/bin/env python3
"""Launch the Face DB Builder from a checkout: ``python main.py``.

For an installed copy, ``alchemyface db`` and ``python -m alchemyface db`` are
the entry points, and either is preferable — they work from any directory. This
file covers the case neither does: a bare ``git clone`` with nothing installed.

It exists because the application this was ported from is started with a plain
``python main.py``, so that is the first thing anyone who knows it will try
here, and because ``src/`` layouts are otherwise unimportable until something is
installed.

The work is delegated to the ``db`` command rather than reimplemented, so this
cannot drift from ``alchemyface db`` — same behaviour, same message when tkinter
is missing, same exit code.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"


def ensure_importable() -> bool:
    """Put ``src/`` on the path if ``alchemyface`` is not already importable.

    Returns whether it had to. An installed copy is deliberately left alone:
    prepending ``src/`` unconditionally would shadow it with the working tree,
    which is a surprising thing for a launcher to do when the two differ.
    """
    try:
        import alchemyface  # noqa: F401
    except ImportError:
        if not SRC.is_dir():
            raise
        sys.path.insert(0, str(SRC))
        return True
    return False


def main() -> None:
    """Run the ``db`` command, as ``alchemyface db`` does."""
    ensure_importable()
    # Imported after the path is arranged, or this would fail in the very case
    # the bootstrap above exists to handle.
    from alchemyface.cli import app

    app(["db"])


if __name__ == "__main__":
    main()
