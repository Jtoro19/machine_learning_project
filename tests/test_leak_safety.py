"""AST guards proving the structural leak-prevention rules design's
"Structural Leak Prevention" table names.

Phase 3 added the cleaning half: `cleaning.py` defines no `fit`/
`fit_transform` function and calls no `.fit(...)`/`.fit_transform(...)`
method anywhere. This slice adds the preprocessing half: no public function
in `preprocessing.py` may accept a DataFrame parameter -- the entire public
surface is one `bool` and one no-argument function (design Decision 5).
"""

import ast
from pathlib import Path

from nids.paths import repo_root

_FORBIDDEN_NAMES = frozenset({"fit", "fit_transform"})

_FORBIDDEN_PARAMETER_NAMES = frozenset({"X", "test", "X_test", "df", "test_df"})
"""Parameter names that would signal a hidden data-fitting path."""

_DATAFRAME_ANNOTATION_NAMES = frozenset({"DataFrame", "pd.DataFrame", "pandas.DataFrame"})
"""Type-annotation spellings that name a DataFrame, in source order they might appear."""


def _cleaning_module_path() -> Path:
    """The absolute path to `src/nids/cleaning.py`."""
    return repo_root() / "src" / "nids" / "cleaning.py"


def _preprocessing_module_path() -> Path:
    """The absolute path to `src/nids/preprocessing.py`."""
    return repo_root() / "src" / "nids" / "preprocessing.py"


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


def test_cleaning_classes_define_no_fit_method() -> None:
    """`cleaning.py` MUST define no class carrying a fit method -- it is pure functions and frozen
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


def _annotation_name(annotation: ast.expr | None) -> str | None:
    """Best-effort textual name of a parameter's type annotation.

    Handles the two shapes this codebase's public signatures actually use:
    a bare name (`bool`) and a dotted attribute (`pd.DataFrame`). Returns
    `None` when there is no annotation at all.
    """
    if annotation is None:
        return None
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        value = annotation.value
        prefix = value.id if isinstance(value, ast.Name) else ast.dump(value)
        return f"{prefix}.{annotation.attr}"
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    return ast.dump(annotation)


def _public_module_level_functions(
    tree: ast.Module,
) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, bool]]:
    """Every public function or method definition `tree`'s public surface exposes.

    Includes top-level, non-underscore-prefixed `FunctionDef`/
    `AsyncFunctionDef` nodes in `tree.body`, AND non-underscore-prefixed
    `FunctionDef`/`AsyncFunctionDef` methods defined directly in the body of
    any top-level `ClassDef`.

    A previous version of this helper walked only `tree.body` for
    `ast.FunctionDef`, so a class method -- or an `async def` function --
    accepting a `pd.DataFrame` parameter would silently evade the two guards
    below (code-review finding). Widened to also inspect
    `ast.AsyncFunctionDef` and methods inside `ast.ClassDef`, so the
    structural leak defense actually covers classes.

    Returns:
        `(function_node, is_method)` pairs, where `is_method` is True for a
        class method (whose leading `self`/`cls` parameter is excluded by
        `_all_parameters` below).
    """
    found: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, bool]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith(
            "_"
        ):
            found.append((node, False))
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and not child.name.startswith("_"):
                    found.append((child, True))
    return found


def _all_parameters(
    func: ast.FunctionDef | ast.AsyncFunctionDef, *, is_method: bool = False
) -> list[ast.arg]:
    """Every positional, positional-only, and keyword-only parameter of `func`.

    Args:
        func: The function or method definition.
        is_method: When True, the leading `self`/`cls` parameter (which
            carries no meaningful annotation) is excluded.
    """
    args = func.args
    parameters = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if is_method and parameters:
        parameters = parameters[1:]
    return parameters


def test_preprocessing_public_functions_take_no_dataframe_parameter() -> None:
    """No public function in `preprocessing.py` may accept a DataFrame or any
    other data parameter (design Decision 5): every parameter of every
    public, module-level function MUST be annotated `bool`, or the function
    MUST take no parameters at all.

    Hard constraint (preprocessing-pipeline spec -- "The preprocessing
    surface exposes no test-fitting path").
    """
    tree = _parse(_preprocessing_module_path())
    functions = _public_module_level_functions(tree)
    assert functions, "expected at least build_preprocessor and build_preprocessor_pair"

    violations: list[str] = []
    for func, is_method in functions:
        parameters = _all_parameters(func, is_method=is_method)
        for parameter in parameters:
            if parameter.arg in _FORBIDDEN_PARAMETER_NAMES:
                violations.append(f"{func.name}({parameter.arg}: forbidden parameter name)")
                continue
            annotation_name = _annotation_name(parameter.annotation)
            if annotation_name != "bool":
                violations.append(
                    f"{func.name}({parameter.arg}: {annotation_name!r}, expected 'bool')"
                )

    assert not violations, (
        "preprocessing.py has a public function parameter that is not a bool "
        f"(or the function is not parameter-free): {violations}"
    )


def test_preprocessing_public_functions_have_no_dataframe_annotation_anywhere() -> None:
    """Belt-and-braces: no parameter anywhere in `preprocessing.py`'s public
    surface is annotated with a DataFrame-shaped type, checked independently
    of the bool-only rule above so a renamed parameter cannot slip a
    DataFrame annotation past it."""
    tree = _parse(_preprocessing_module_path())
    violations: list[str] = []
    for func, is_method in _public_module_level_functions(tree):
        for parameter in _all_parameters(func, is_method=is_method):
            annotation_name = _annotation_name(parameter.annotation)
            if annotation_name in _DATAFRAME_ANNOTATION_NAMES:
                violations.append(f"{func.name}({parameter.arg}: {annotation_name})")

    assert not violations, f"DataFrame-annotated parameter(s) found: {violations}"


def test_widened_guard_catches_dataframe_parameter_in_class_method() -> None:
    """The parameter-shape scan must inspect class methods and `async def`
    functions, not only bare module-level `def` functions -- otherwise a
    class method (or an async function) accepting a `pd.DataFrame` parameter
    would silently evade the structural leak guard above (code-review
    finding). Proven against a synthetic module, since `preprocessing.py`
    itself defines no class."""
    source = (
        "import pandas as pd\n\n"
        "class Sneaky:\n"
        "    def fit(self, df: pd.DataFrame) -> None:\n"
        "        ...\n\n"
        "    async def afit(self, df: pd.DataFrame) -> None:\n"
        "        ...\n\n"
        "    def _private(self, df: pd.DataFrame) -> None:\n"
        "        ...\n"
    )
    tree = ast.parse(source, filename="<synthetic>")

    functions = _public_module_level_functions(tree)
    names = {func.name for func, _ in functions}
    assert names == {"fit", "afit"}, (
        "widened guard must find the public sync and async class methods, and "
        "must still skip the underscore-prefixed private one"
    )
    assert all(is_method for _, is_method in functions)

    violations: list[str] = []
    for func, is_method in functions:
        for parameter in _all_parameters(func, is_method=is_method):
            annotation_name = _annotation_name(parameter.annotation)
            if annotation_name in _DATAFRAME_ANNOTATION_NAMES:
                violations.append(f"{func.name}({parameter.arg}: {annotation_name})")

    assert sorted(violations) == ["afit(df: pd.DataFrame)", "fit(df: pd.DataFrame)"], violations


def test_cleaning_report_never_fits_on_the_test_partition() -> None:
    """Every `.fit(...)` in `cleaning_report.py` MUST take a train-derived argument.

    `cleaning_report.py` is the only module under `src/nids/` that fits an
    estimator outside a Pipeline (a `RareCategoryGrouper`, purely to report the
    categories it learns). The no-recompute guard in `test_repo_hygiene.py` does
    not cover fit targets, so without this test a future `.fit(result.test[...])`
    would pass every existing check while violating the AGENTS.md blocking rule
    against fitting on test data.
    """
    root = repo_root()
    path = root / "src" / "nids" / "cleaning_report.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in {"fit", "fit_transform"}):
            continue
        if not node.args:
            violations.append(f"line {node.lineno}: {func.attr}() with no argument to inspect")
            continue
        argument = node.args[0]
        source = ast.unparse(argument)

        # Reject anything mentioning the test partition anywhere in the
        # subtree. A substring check for ".train" alone is bypassable:
        # `fit(pd.concat([result.train, result.test]))` contains ".train"
        # while fitting on test data, which is the blocking rule this guard
        # exists to prevent.
        mentions_test = any(
            isinstance(sub, ast.Attribute) and sub.attr == "test" for sub in ast.walk(argument)
        )
        if mentions_test:
            violations.append(f"line {node.lineno}: {func.attr}({source}) references .test")
            continue

        # Require the argument's own root to be a `.train` attribute access,
        # so only a train partition (optionally subscripted or column-selected)
        # can be fitted.
        root: ast.AST = argument
        while isinstance(root, (ast.Subscript, ast.Call)):
            root = root.func if isinstance(root, ast.Call) else root.value
        is_train_rooted = isinstance(root, ast.Attribute) and root.attr == "train"
        if not is_train_rooted:
            violations.append(f"line {node.lineno}: {func.attr}({source}) is not train-rooted")

    assert not violations, (
        "cleaning_report.py fits on something that is not the training partition: "
        + "; ".join(violations)
    )
