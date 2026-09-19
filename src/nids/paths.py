"""Repository-root resolution and derived filesystem locations.

Every path used elsewhere in this package is derived at runtime from :func:`repo_root`,
never hardcoded, so the package behaves the same whether it is imported from a notebook,
a test, or an installed non-editable wheel. See design Decision 1 for the full rationale
behind the marker-search strategy used here.
"""

import os
from pathlib import Path
import re

REPO_MARKERS: tuple[str, ...] = ("pyproject.toml", "src/nids")
"""Directory markers that together identify this repository and nothing else."""

REPO_ROOT_ENV_VAR: str = "NIDS_REPO_ROOT"
"""Environment variable that can override marker search, once validated."""

NOTEBOOK_ID_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
"""Allowed shape for a notebook/producer identifier used under ``results/``."""


class RepoRootNotFoundError(RuntimeError):
    """Raised when no ancestor directory carries every entry in :data:`REPO_MARKERS`."""


def _has_all_markers(directory: Path) -> bool:
    """Return True when every entry in REPO_MARKERS exists directly under `directory`.

    Args:
        directory: Candidate directory to check.

    Returns:
        True if `directory` contains every marker in REPO_MARKERS.
    """
    return all((directory / marker).exists() for marker in REPO_MARKERS)


def find_repo_root(start: Path) -> Path | None:
    """Walk upward from `start` returning the first directory carrying every REPO_MARKER.

    Args:
        start: A file or directory path to begin the upward search from. When `start`
            is a file, the search begins at its parent directory.

    Returns:
        The first ancestor directory (including `start` itself, or its parent when
        `start` is a file) that contains every marker in REPO_MARKERS, or None if no
        such directory exists.
    """
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if _has_all_markers(candidate):
            return candidate
    return None


def repo_root() -> Path:
    """Resolve the repository root.

    Resolution order: an upward walk from this module's own file location, then an
    upward walk from the current working directory, then the `NIDS_REPO_ROOT`
    environment variable (only trusted once validated against REPO_MARKERS). Nothing
    is cached, so every call re-resolves from scratch.

    Returns:
        The resolved repository root directory.

    Raises:
        RepoRootNotFoundError: No ancestor of the package file location, the current
            working directory, or a valid `NIDS_REPO_ROOT` carries every marker.
    """
    file_start = Path(__file__).resolve()
    found = find_repo_root(file_start)
    if found is not None:
        return found

    cwd_start = Path.cwd().resolve()
    found = find_repo_root(cwd_start)
    if found is not None:
        return found

    env_value = os.environ.get(REPO_ROOT_ENV_VAR)
    if env_value:
        candidate = Path(env_value).resolve()
        if _has_all_markers(candidate):
            return candidate

    markers = " and ".join(repr(marker) for marker in REPO_MARKERS)
    raise RepoRootNotFoundError(
        f"Could not locate the repository root: no ancestor of {file_start} or "
        f"{cwd_start} contains both {markers}. Set the {REPO_ROOT_ENV_VAR} "
        "environment variable to the repository root to override this search."
    )


def raw_data_dir() -> Path:
    """Return `<repo_root>/data/raw`. Never creates it; this tree is read-only.

    Returns:
        The path to the raw dataset directory. The directory may or may not exist on
        disk; this function never checks for or creates it.
    """
    return repo_root() / "data" / "raw"


def results_root() -> Path:
    """Return `<repo_root>/results`. Never creates it.

    Returns:
        The path to the results directory. This function never checks for or creates
        it.
    """
    return repo_root() / "results"


def results_dir(notebook_id: str, *, create: bool = True) -> Path:
    """Return `<repo_root>/results/<notebook_id>`, creating it and `tables/`, `figures/`.

    Args:
        notebook_id: Identifier for the producer writing results, validated against
            NOTEBOOK_ID_PATTERN before any filesystem call is made.
        create: When True (the default), create the notebook's directory along with
            its `tables/` and `figures/` subdirectories.

    Returns:
        The resolved results directory for `notebook_id`.

    Raises:
        ValueError: `notebook_id` does not match NOTEBOOK_ID_PATTERN.
        RuntimeError: The resolved directory escapes `results_root()`.
    """
    if not NOTEBOOK_ID_PATTERN.match(notebook_id):
        raise ValueError(
            f"Invalid notebook_id {notebook_id!r}: must match "
            f"{NOTEBOOK_ID_PATTERN.pattern!r}."
        )

    root = results_root()
    candidate = root / notebook_id
    resolved = candidate.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise RuntimeError(
            f"Resolved results directory {resolved} escapes results_root() "
            f"{resolved_root}."
        )

    if create:
        (resolved / "tables").mkdir(parents=True, exist_ok=True)
        (resolved / "figures").mkdir(parents=True, exist_ok=True)

    return resolved
