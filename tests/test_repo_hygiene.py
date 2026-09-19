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
# of large integer literals explicitly permitted in that exact module.
ALLOWED_LARGE_LITERALS: dict[str, frozenset[int]] = {
    # The raw dataset's documented row/column counts. Properties of the unmodified
    # input, not of any cleaning step, and permitted nowhere else (see design
    # Decision "175_341, 82_332 and 45 are the only dataset row/column literals
    # permitted anywhere in src/nids").
    "src/nids/validation.py": frozenset({175_341, 82_332, 45}),
    # AGENTS.md's fixed subsample row cap ("Gaussian Process, spectral
    # clustering and t-SNE run on a stratified subsample of at most 10000
    # rows"). Permitted only as `stratified_subsample`'s `max_rows` default;
    # nowhere else in src/nids or tests.
    "src/nids/sampling.py": frozenset({10_000}),
}


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


_WRITE_MODE_CHARS = frozenset("wax")
"""Mode characters that open a file for writing, appending, or exclusive creation."""


def _calls_raw_data_dir(node: ast.AST) -> bool:
    """True if `node`'s subtree contains a call naming `raw_data_dir`."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name) and func.id == "raw_data_dir":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "raw_data_dir":
                return True
    return False


def _open_mode_argument(call: ast.Call) -> str | None:
    """Return the literal string mode argument of an `open(...)`-shaped call, if any."""
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        if isinstance(call.args[1].value, str):
            return call.args[1].value
    for keyword in call.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                return keyword.value.value
    return None


def _forbidden_write_calls(node: ast.AST) -> list[str]:
    """Return a description of every write-intent call found in `node`'s subtree.

    Matches `open(...)`/`Path.open(...)` with an explicit write, append, or
    exclusive-creation mode; `.to_csv(...)`; and `.mkdir(...)`.
    """
    found: list[str] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        is_open_call = (isinstance(func, ast.Name) and func.id == "open") or (
            isinstance(func, ast.Attribute) and func.attr == "open"
        )
        if is_open_call:
            mode = _open_mode_argument(sub)
            if mode is not None and any(char in mode for char in _WRITE_MODE_CHARS):
                found.append(f"open(..., mode={mode!r})")
        elif isinstance(func, ast.Attribute) and func.attr == "to_csv":
            found.append(".to_csv(...)")
        elif isinstance(func, ast.Attribute) and func.attr == "mkdir":
            found.append(".mkdir(...)")
    return found


def test_no_write_mode_open_under_raw() -> None:
    """No function in `src/nids/**` may write to a path derived from `raw_data_dir()`.

    Threat-matrix boundary: `data/raw/` is read-only (see design's Structural Leak
    Prevention). Checked per function definition (and per module-level statement
    group) rather than per-module, so a module that legitimately reads from
    `data/raw/` in one function and writes elsewhere (for example under
    `results/`) in another function is not flagged.
    """
    root = repo_root()
    base = root / "src" / "nids"
    violations: list[str] = []

    for path in sorted(base.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(root.resolve()).as_posix()

        scopes: list[ast.AST] = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        top_level_statements = [
            stmt
            for stmt in tree.body
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if top_level_statements:
            scopes.append(ast.Module(body=top_level_statements, type_ignores=[]))

        for scope in scopes:
            if not _calls_raw_data_dir(scope):
                continue
            for finding in _forbidden_write_calls(scope):
                violations.append(f"{relative}: {finding}")

    assert not violations, (
        "Write-intent call(s) found in a scope that also calls raw_data_dir() "
        "(module: finding): " + ", ".join(violations)
    )


def test_no_hardcoded_cleaned_partition_counts_in_slice_3_modules() -> None:
    """Slice 3 modules and tests carry no literal cleaned-partition row count.

    The generic `test_no_unexplained_large_integer_literals` scan above
    already covers every module under `src/nids/**` and `tests/**`; this test
    names `cleaning.py`, `data.py`, and `test_cleaning.py` explicitly (data-
    cleaning spec — "No hardcoded cleaned-partition row counts") so a
    regression here is reported without cross-referencing the generic scan,
    and no slice-3 module contributes an entry to `ALLOWED_LARGE_LITERALS`
    (that dict also carries `sampling.py`'s row-cap literal, added in slice 4).
    """
    root = repo_root()
    targets = ("src/nids/cleaning.py", "src/nids/data.py", "tests/test_cleaning.py")
    violations: list[str] = []
    for relative in targets:
        path = root / relative
        assert path.is_file(), f"expected slice-3 module missing: {relative}"
        allowed = ALLOWED_LARGE_LITERALS.get(relative, frozenset())
        assert not allowed, f"{relative} must not appear in ALLOWED_LARGE_LITERALS"
        for value in _large_int_literals(path):
            if value not in allowed:
                violations.append(f"{relative}: {value}")

    assert not violations, (
        "Hardcoded cleaned-partition literal(s) found (module: value): "
        + ", ".join(violations)
    )
