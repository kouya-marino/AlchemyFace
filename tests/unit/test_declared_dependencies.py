"""requirements.txt and pyproject.toml must not drift apart.

requirements.txt claimed to be "kept in step with pyproject.toml's [project]
dependencies" while missing Pillow — so an environment pinned from it got a
working library and a GUI that crashed on import. A comment is not a mechanism;
this is.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def toml_load(path: Path) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(path.read_text())
    # 3.10 has no tomllib; parse the two arrays we care about.
    text = path.read_text()

    def array(name: str) -> list[str]:
        block = re.search(rf"^{name} = \[(.*?)^\]", text, re.S | re.M)
        return re.findall(r'"([^"]+)"', block.group(1)) if block else []

    return {
        "project": {
            "dependencies": array("dependencies"),
            "optional-dependencies": {"dev": array("dev")},
        }
    }


def names(specs: list[str]) -> set[str]:
    return {re.split(r"[<>=!~ \[]", s, 1)[0].strip().lower() for s in specs if s.strip()}


def requirements(path: Path) -> set[str]:
    lines = [
        line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    return names(lines)


def test_runtime_requirements_match_pyproject() -> None:
    declared = names(toml_load(ROOT / "pyproject.toml")["project"]["dependencies"])
    pinned = requirements(ROOT / "requirements.txt")
    assert pinned == declared, (
        f"requirements.txt and pyproject.toml disagree:\n"
        f"  only in pyproject:        {sorted(declared - pinned)}\n"
        f"  only in requirements.txt: {sorted(pinned - declared)}"
    )


def test_dev_requirements_match_the_dev_extra() -> None:
    extras = toml_load(ROOT / "pyproject.toml")["project"]["optional-dependencies"]
    declared = names(extras["dev"])
    pinned = requirements(ROOT / "requirements-dev.txt")
    assert pinned == declared, (
        f"requirements-dev.txt and the dev extra disagree:\n"
        f"  only in pyproject:            {sorted(declared - pinned)}\n"
        f"  only in requirements-dev.txt: {sorted(pinned - declared)}"
    )


def test_the_readme_badge_matches_the_declared_version() -> None:
    """The same check CI runs, so it fails locally before a push too."""
    text = (ROOT / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', text, re.M).group(1)
    badge = re.search(r"PyPI-v(\d+\.\d+\.\d+)", (ROOT / "README.md").read_text())
    assert badge, "no version-pinned PyPI badge in README.md"
    assert badge.group(1) == declared, f"README badge says v{badge.group(1)}, pyproject declares {declared}"
