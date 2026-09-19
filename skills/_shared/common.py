"""Argparse plumbing, file I/O, and results-writer wiring shared by the three
`skills/` scripts.

Dataset column/dtype-shaping logic (schema detection, cleaning, the generic
preprocessing fallback, feature-set computation, and the leakage assertion) lives in
`nids.tabular` (AGENTS.md: "Shared preprocessing lives in `src/`" -- round-3 review
standards finding 3; round-4 review standards finding 3 moved `assert_no_leakage`
there too, since it enforces the same UNSW-specific data rules as everything else in
that module). This module re-exports the small subset of those functions each script
calls directly (`is_unsw_raw_schema`, `compute_features`, `clean_dataset_if_unsw`,
`build_preprocessor_for`, `split_and_dedupe`, `resolve_stratify`,
`assert_no_leakage`), and keeps only CLI/argparse parsing, CSV loading,
`nids.results.ResultsWriter` wiring, and the `choose_n_clusters` unsupervised
cluster-count selection helper (shared only by the two clustering skills, not
a preprocessing concern) as its own code, per design Decision 2 and Decision 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    import numpy as np
    import pandas as pd

    from nids.results import ResultsWriter

_REPO_MARKERS = ("pyproject.toml", "src/nids")


def ensure_repo_root_on_syspath() -> Path:
    """Walk upward from this file to find the repo root and prepend it to `sys.path`.

    Mirrors `nids.paths.repo_root()`'s marker search without importing `nids` first,
    so a script invoked as `python skills/<name>/<script>.py` from any working
    directory can `import nids` and `from skills._shared import common`.

    Round-5 review finding 2: this function used to run only from inside each
    script's `main()`, AFTER `common.py`'s own module-level `from nids.tabular
    import (...)` below had already executed and already required `nids` to be
    importable -- so the docstring's "from any working directory" promise could not
    actually hold unless the package was separately installed (`uv sync`). Fixed:
    this function is now called once, here, at MODULE scope, immediately after its
    own definition and BEFORE the `from nids.tabular import (...)` line -- so `src/`
    (and the repo root) are genuinely on `sys.path` before the first `nids` import
    this module performs, regardless of whether `nids` is installed.

    Returns:
        The resolved repository root.

    Raises:
        RuntimeError: no ancestor directory carries every marker in `_REPO_MARKERS`.
    """
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if all((candidate / marker).exists() for marker in _REPO_MARKERS):
            root_str = str(candidate)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            src_str = str(candidate / "src")
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return candidate
    raise RuntimeError(
        f"Could not locate the repository root walking up from {start}: "
        f"no ancestor carries every marker in {_REPO_MARKERS!r}."
    )


ensure_repo_root_on_syspath()

from nids.tabular import (  # noqa: E402
    assert_no_leakage,
    build_preprocessor_for,
    clean_dataset_if_unsw,
    compute_features,
    is_unsw_raw_schema,
    numeric_and_categorical_columns,
    resolve_stratify,
    split_and_dedupe,
)

__all__ = [
    "SEED",
    "assert_no_leakage",
    "build_arg_parser",
    "build_preprocessor_for",
    "choose_n_clusters",
    "clean_dataset_if_unsw",
    "compute_features",
    "ensure_repo_root_on_syspath",
    "is_unsw_raw_schema",
    "load_dataset",
    "numeric_and_categorical_columns",
    "resolve_exclude",
    "resolve_results_writer",
    "resolve_stratify",
    "split_and_dedupe",
]

SEED: int = 42
"""Default seed for every random operation across the three skills."""


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Return the argparse contract shared by all three skill scripts.

    Args:
        description: Script-specific one-line description shown in `--help`.

    Returns:
        A configured, unparsed `ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset", required=True, help="Path to the input CSV dataset."
    )
    parser.add_argument(
        "--target", required=True, help="Name of the target column."
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Column to exclude from the feature set. Repeatable, and/or a single "
            "comma-separated value (e.g. --exclude sttl,ct_state_ttl)."
        ),
    )
    parser.add_argument(
        "--output", required=True, help="Output folder for the results manifest."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for every random operation (default: 42).",
    )
    return parser


def resolve_exclude(raw_values: list[str]) -> list[str]:
    """Flatten `--exclude`'s repeated-and/or-comma-separated values.

    Args:
        raw_values: The raw `args.exclude` list from argparse (one entry per
            `--exclude` occurrence, each possibly comma-separated).

    Returns:
        A flat, de-duplicated, order-preserving list of column names.
    """
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for piece in raw.split(","):
            name = piece.strip()
            if name and name not in seen:
                seen.add(name)
                resolved.append(name)
    return resolved


def load_dataset(path: str) -> "pd.DataFrame":
    """Read the dataset CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        The parsed `DataFrame`, with pandas' default dtype inference (this
        module is dataset-agnostic, so no column-specific dtype is forced).
    """
    import pandas as pd

    return pd.read_csv(path)


def resolve_results_writer(
    output: str,
    generated_at: "datetime | None" = None,
    *,
    allow_external: bool = False,
) -> "ResultsWriter":
    """Build a `nids.results.ResultsWriter` targeting `output` exactly.

    `ResultsWriter` writes to `<root>/<notebook_id>`; this splits `output` into
    `root = output.parent` and `notebook_id = output.name` so the writer's folder is
    exactly `output` (design Decision 6).

    Round-3 review worth-fixing finding 8: `notebook_id` is validated against
    `nids.paths.NOTEBOOK_ID_PATTERN` explicitly, here, BEFORE `root.mkdir(...)`
    runs -- rather than relying solely on `ResultsWriter.__init__`'s own
    validation (which raises only after `root` has already been created on
    disk) -- so an invalid `--output` name is rejected with a clear,
    pattern-naming error before any filesystem side effect.

    Round-4 review blocking finding 1: `notebook_id` was validated against
    `NOTEBOOK_ID_PATTERN`, but its LOCATION never was -- `root = output_path.parent`
    was passed straight to `ResultsWriter`'s `root=` escape hatch, so
    `--output /tmp/anything` (or any path outside `results/`) wrote a full manifest
    outside the project results tree, bypassing exactly the containment
    `nids.paths.results_dir()` exists to enforce. Fixed: by default, `root.resolve()`
    is now required to be `results_root()` itself or a descendant of it, mirroring
    `results_dir()`'s own `is_relative_to` containment check, and any violation
    raises `ValueError` naming the offending path before `root.mkdir(...)` runs.
    `allow_external=True` is an explicit, documented escape hatch for tests that
    need to target `tmp_path`; the production CLI path (`resolve_results_writer`
    called from a skill script's `main()`) never passes it, so `--output` from a
    real invocation stays contained by default.

    Args:
        output: The output folder path from `--output`.
        generated_at: Forwarded to `ResultsWriter` (used by tests for determinism).
        allow_external: When True, skip the `results_root()` containment check.
            Production CLI call sites MUST NOT pass this; it exists only for tests
            that intentionally target a directory outside `results/` (e.g. `tmp_path`).

    Returns:
        A `ResultsWriter`, unopened context (caller should use it as a `with` block).

    Raises:
        ValueError: `Path(output).name` does not match
            `nids.paths.NOTEBOOK_ID_PATTERN` (lowercase snake_case), or (unless
            `allow_external=True`) `output`'s parent directory resolves outside
            `nids.paths.results_root()`.
    """
    from nids.paths import NOTEBOOK_ID_PATTERN, results_root
    from nids.results import ResultsWriter

    output_path = Path(output)
    notebook_id = output_path.name
    if not NOTEBOOK_ID_PATTERN.match(notebook_id):
        raise ValueError(
            f"Invalid --output folder name {notebook_id!r}: the final path "
            f"component must match {NOTEBOOK_ID_PATTERN.pattern!r} (lowercase "
            "snake_case), e.g. 'results/my_run', not 'results/My-Run'."
        )
    root = output_path.parent

    if not allow_external:
        resolved_root = root.resolve()
        resolved_results_root = results_root().resolve()
        if resolved_root != resolved_results_root and not resolved_root.is_relative_to(
            resolved_results_root
        ):
            raise ValueError(
                f"--output {output!r} resolves outside the project results tree: "
                f"{resolved_root} is not {resolved_results_root} or a descendant of "
                "it. Pass a path under 'results/', e.g. 'results/my_run', not an "
                "absolute or external path."
            )

    root.mkdir(parents=True, exist_ok=True)
    return ResultsWriter(notebook_id, root=root, generated_at=generated_at)


def choose_n_clusters(
    X: "np.ndarray | pd.DataFrame",
    seed: int,
    k_min: int = 2,
    k_max: int = 10,
) -> tuple[int, str]:
    """Choose KMeans' cluster count via an UNSUPERVISED silhouette sweep.

    Round-3 review worth-fixing finding 7: both clustering skills previously
    derived `n_clusters` from `y_train.nunique()` -- a hyperparameter read
    straight off the target -- then scored that same clustering against that
    same target (silhouette AND adjusted Rand index). This function never
    looks at the target: it fits `KMeans(n_clusters=k, ...)` for every `k` in
    `[k_min, k_max]` (clipped so no candidate exceeds `len(X) - 1`, since a
    clustering needs at least one row outside its own cluster to be scored)
    and keeps the `k` with the highest `silhouette_score(X, labels)`,
    computed purely from `X`'s own point geometry.

    Round-4 review blocking finding 2: `silhouette_score` is O(n^2) in the number of
    rows (pairwise distances), evaluated once per candidate `k` (up to 9 times over
    `[k_min, k_max]`). `clustering-reduction` already subsamples to at most 10,000
    rows before calling this function, but `eda-reduction-clustering` does not,
    so on the full UNSW-NB15 train split (~122k rows) this scored `silhouette_score`
    over the ENTIRE split nine times -- exactly the cost class AGENTS.md caps at a
    10,000-row stratified subsample for GP, spectral clustering, and t-SNE. Fixed:
    every `silhouette_score` call here passes `sample_size=10_000, random_state=seed`,
    so it always scores at most a 10,000-row random subsample of `X` (sklearn's own
    `sample_size` parameter), regardless of how many rows the caller passes in or
    whether the caller already subsampled upstream.

    Args:
        X: The (already preprocessed/PCA-transformed) feature matrix to cluster.
        seed: Random seed forwarded to every candidate `KMeans` fit and to
            `silhouette_score`'s own `sample_size`-driven row subsampling.
        k_min: Smallest k to try (default 2 -- silhouette is undefined for k=1).
        k_max: Largest k to try (default 10).

    Returns:
        `(chosen_k, method)`. `method` is `"silhouette_sweep"` when at least one
        candidate k produced a valid (more-than-one-cluster) labeling, else
        `"fixed_minimum"` (the dataset was too small, or degenerate, for the
        sweep to distinguish any candidate; falls back to the smallest
        feasible k).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_rows = len(X)
    upper = min(k_max, n_rows - 1)
    if upper < k_min:
        return max(2, min(k_min, n_rows)), "fixed_minimum"

    best_k: int | None = None
    best_score = float("-inf")
    for k in range(k_min, upper + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, sample_size=10_000, random_state=seed)
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        return k_min, "fixed_minimum"
    return best_k, "silhouette_sweep"
