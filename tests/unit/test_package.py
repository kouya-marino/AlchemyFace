"""Sanity checks on the package itself: it imports, and it is the src/ copy."""

from pathlib import Path

from typer.testing import CliRunner

import alchemyface
from alchemyface.cli import app

runner = CliRunner()


def test_version_is_pep440():
    parts = alchemyface.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_package_is_the_src_layout_copy():
    # A src/ layout exists to stop the CWD shadowing the installed package.
    # If this ever resolves outside src/, the editable install has broken.
    assert Path(alchemyface.__file__).parent.parent.name == "src"


def test_cli_version_matches_package():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == alchemyface.__version__


def test_cli_shows_help_with_no_arguments():
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout
