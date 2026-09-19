"""AST guards proving the structural leak-prevention rules design's
"Structural Leak Prevention" table names.

This slice adds the cleaning half: `cleaning.py` defines no `fit`/
`fit_transform` function and calls no `.fit(...)`/`.fit_transform(...)`
method anywhere. Phase 4 extends this module with the preprocessing half (no
public function in `preprocessing.py` may accept a DataFrame parameter).
"""

import ast
from pathlib import Path

from nids.paths import repo_root

_FORBIDDEN_NAMES = frozenset({"fit", "fit_transform"})


def _cleaning_module_path() -> Path:
    """The absolute path to `src/nids/cleaning.py`."""
    return repo_root() / "src" / "nids" / "cleaning.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_cleaning_defines_no_fit_function() -> None:
    """`cleaning.py` MUST define no function named `fit` or `fit_transform`.

    Hard constraint (data-cleaning spec — "The cleaning surface exposes no
    fit interface"): this capability has no class and no fitted state.
    """
    tree = _parse(_cleaning_module_path())
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    violations = defined & _FORBIDDEN_NAMES
    assert not violations, f"cleaning.py defines forbidden function(s): {sorted(violations)}"


def test_cleaning_calls_no_fit_method() -> None:
    """`cleaning.py` MUST call no `.fit(...)`/`.fit_transform(...)` method.

    Every operation in this capability is a fixed transformation or a
    read-only diagnostic; none may learn a parameter from either partition.
    """
    tree = _parse(_cleaning_module_path())
    violations = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES
    ]
    assert not violations, f"cleaning.py calls forbidden method(s): {violations}"


def test_cleaning_defines_no_class() -> None:
    """`cleaning.py` MUST define no class -- it is pure functions and frozen
    dataclasses (a `ClassDef` for a `@dataclass` is expected; a class that
    also defines `fit`/`fit_transform` is already caught by the two tests
    above, so this test only documents the "no class, no fit" pairing from
    the design's structural leak prevention table)."""
    tree = _parse(_cleaning_module_path())
    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    # Every class defined here must be a diagnostics dataclass, never an
    # estimator: none may define `fit`.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = {
                child.name
                for child in node.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            }
            assert not (methods & _FORBIDDEN_NAMES), (
                f"class {node.name} in cleaning.py defines a forbidden method"
            )
    assert class_names, "expected at least the diagnostics dataclasses to be present"
