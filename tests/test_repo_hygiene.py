"""AST-walk hygiene checks shared across every later slice.

This is the mechanism Phases 2-5 extend: each later phase adds a narrowly scoped
entry to ALLOWED_LARGE_LITERALS (for example, permitting the three dataset-shape
literals only inside `validation.py`) instead of loosening the scan itself.
"""

import ast
from pathlib import Path

from nids.paths import repo_root

LARGE_LITERAL_THRESHOLD = 10_000
"""Any bare integer literal at or above this value is considered a suspicious
hardcoded dataset count unless explicitly allow-listed for its exact module."""

SCANNED_RELATIVE_ROOTS: tuple[str, ...] = ("src/nids", "tests")

# Maps a module's path, relative to the repository root and POSIX-style, to the set
# of large integer literals explicitly permitted in that exact module. Empty in this
# slice: no module has earned an exception yet.
ALLOWED_LARGE_LITERALS: dict[str, frozenset[int]] = {}


def _self_path() -> Path:
    """This module's own path, excluded from the scan it performs.

    It defines LARGE_LITERAL_THRESHOLD itself, so scanning it would be a
    self-referential false positive rather than a finding about project code.
    """
    return Path(__file__).resolve()


def _scanned_files(root: Path) -> list[Path]:
    """Collect every `.py` file under the scanned roots, excluding this module.

    Args:
        root: The repository root to scan from.

    Returns:
        A sorted list of module paths eligible for the literal scan.
    """
    excluded = _self_path()
    collected: list[Path] = []
    for relative_root in SCANNED_RELATIVE_ROOTS:
        base = root / relative_root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            resolved = path.resolve()
            if resolved == excluded:
                continue
            collected.append(resolved)
    return collected


def _large_int_literals(path: Path) -> list[int]:
    """Return every bare integer literal at or above LARGE_LITERAL_THRESHOLD in `path`.

    Args:
        path: A Python module to parse.

    Returns:
        The offending literal values, in source order. Booleans are excluded even
        though `bool` is a subclass of `int` in Python's data model.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
            and node.value >= LARGE_LITERAL_THRESHOLD
        ):
            found.append(node.value)
    return found


def test_no_unexplained_large_integer_literals() -> None:
    root = repo_root()
    violations: list[str] = []
    for path in _scanned_files(root):
        relative = path.relative_to(root.resolve()).as_posix()
        allowed = ALLOWED_LARGE_LITERALS.get(relative, frozenset())
        for value in _large_int_literals(path):
            if value not in allowed:
                violations.append(f"{relative}: {value}")

    assert not violations, (
        "Unexplained large integer literal(s) found (module: value): "
        + ", ".join(violations)
    )
