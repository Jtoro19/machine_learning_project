"""The `results/<notebook_id>/` output-contract writer.

`ResultsWriter` is the single implementation of the results-output-contract
capability: every producer under `results/` (starting with
`nids.cleaning_report`) writes its tables, figures, metrics, and notes
through this class instead of hand-rolling folder creation, JSON
serialisation, or manifest emission. See design Decision 9 for the
determinism rationale (one timestamp, in one file, with a
`SOURCE_DATE_EPOCH` override) and the "results.py" module section for the
full design-points table this implementation follows point-for-point.
"""

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from nids import paths

if TYPE_CHECKING:  # keep matplotlib out of import time
    from matplotlib.figure import Figure

SCHEMA_VERSION: Final[str] = "1.0"
"""`manifest.json`'s `schema_version` field. Bumped only on a breaking
manifest-shape change."""

ARTIFACT_NAME_PATTERN = paths.NOTEBOOK_ID_PATTERN
"""Shape every `notebook_id` and every artifact/metric/note name MUST match,
validated before any filesystem call (path-traversal and silent-overwrite
prevention). Identical to `paths.NOTEBOOK_ID_PATTERN`; reused rather than
redefined so the two can never drift apart."""

_REGISTERED_TABLE_EXTENSIONS: Final[frozenset[str]] = frozenset({".csv", ".json"})
_REGISTERED_FIGURE_EXTENSIONS: Final[frozenset[str]] = frozenset({".png"})

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """One `tables[]`/`figures[]` entry in `manifest.json`."""

    path: str
    """POSIX path, relative to the manifest's own folder."""
    title: str
    type: str
    """`"csv"`, `"json"`, or `"png"`."""
    description: str


@dataclass(frozen=True, slots=True)
class MetricEntry:
    """One `metrics[]` entry in `manifest.json`."""

    name: str
    value: int | float | str
    description: str


@dataclass(frozen=True, slots=True)
class NoteEntry:
    """One `notes[]` entry in `manifest.json` (**D2** — the additive `notes` array)."""

    id: str
    text: str


def _validate_name(name: str, *, kind: str) -> None:
    """Reject a `notebook_id`/artifact/metric/note name before any filesystem call.

    Args:
        name: The candidate name.
        kind: Human-readable label used in the raised message (for example
            `"notebook_id"` or `"artifact name"`).

    Raises:
        ValueError: `name` does not match `ARTIFACT_NAME_PATTERN` -- this
            rejects path traversal (`..`), separators (`/`), a leading `/`,
            an embedded extension, and any non-lowercase-snake-case name
            before a single byte is written or a single directory is
            created.
    """
    if not ARTIFACT_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid {kind} {name!r}: must match {ARTIFACT_NAME_PATTERN.pattern!r}."
        )


def _relative_posix(target: Path, *, directory: Path) -> str:
    """Build a manifest-safe relative path, always POSIX-separated.

    Args:
        target: The absolute path of a written artifact.
        directory: The notebook's own output directory.

    Returns:
        `target`'s path relative to `directory`, joined with `/` regardless
        of platform (design's "Windows/Linux path drift" guard).
    """
    return str(PurePosixPath(target.relative_to(directory)))


def _atomic_write_bytes_from_csv(frame: pd.DataFrame, target: Path) -> None:
    """Write `frame` to `target` deterministically, atomically.

    Args:
        frame: The table to write. `index=False` is unconditional -- callers
            that care about row order must `sort_values`/`reset_index`
            before calling `add_table`.
        target: The final `.csv` path. A `<target>.tmp` sibling is written
            first and `os.replace`d into place.
    """
    tmp = target.with_name(target.name + ".tmp")
    frame.to_csv(
        tmp,
        index=False,
        lineterminator="\n",
        float_format="%.6f",
        encoding="utf-8",
    )
    os.replace(tmp, target)


def _atomic_write_json(payload: Any, target: Path) -> None:
    """Write `payload` as deterministic, newline-terminated JSON, atomically.

    Args:
        payload: A JSON-serialisable object.
        target: The final `.json` path. A `<target>.tmp` sibling is written
            first and `os.replace`d into place.
    """
    tmp = target.with_name(target.name + ".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, target)


class ResultsWriter:
    """Write one producer's self-describing `results/<notebook_id>/` folder.

    Every artifact is written to `<path>.tmp` then `os.replace`d into place
    (atomic on POSIX and Windows). `manifest.json` is written last, by
    `write_manifest()` (called automatically by `close()`/`__exit__` on a
    clean run), so "manifest exists" implies "every declared artifact
    exists". See design's "results.py" design-points table for the full
    rationale behind every choice below.
    """

    def __init__(
        self,
        notebook_id: str,
        *,
        root: Path | None = None,
        generated_at: datetime | None = None,
        prune_undeclared: bool = True,
    ) -> None:
        """Validate `notebook_id`, then create the notebook's output folder.

        Args:
            notebook_id: This producer's identifier. Validated against
                `ARTIFACT_NAME_PATTERN` before any filesystem call.
            root: The `results/` root to write under. Defaults to
                `paths.results_root()`, resolved lazily (via the `nids.paths`
                module, not a frozen import) so tests can monkeypatch
                `paths.results_root`.
            generated_at: Explicit timestamp for `manifest.json`'s
                `generated_at` field. Resolution order when omitted:
                `SOURCE_DATE_EPOCH` environment variable (UTC epoch seconds)
                -> `datetime.now(UTC)` (design Decision 9).
            prune_undeclared: When True (the default), `close()` deletes any
                registered-extension file directly under this notebook's own
                `tables/`/`figures/` that the manifest does not declare.

        Raises:
            ValueError: `notebook_id` does not match `ARTIFACT_NAME_PATTERN`.
        """
        _validate_name(notebook_id, kind="notebook_id")
        self.notebook_id = notebook_id
        self.prune_undeclared = prune_undeclared

        base_root = root if root is not None else paths.results_root()
        self._directory = base_root / notebook_id
        self._tables_dir = self._directory / "tables"
        self._figures_dir = self._directory / "figures"
        self._directory.mkdir(parents=True, exist_ok=True)
        self._tables_dir.mkdir(parents=True, exist_ok=True)
        self._figures_dir.mkdir(parents=True, exist_ok=True)

        self._generated_at = self._resolve_generated_at(generated_at)

        self._registered_artifact_names: set[str] = set()
        self._registered_metric_names: set[str] = set()
        self._registered_note_ids: set[str] = set()
        self._tables: list[ArtifactEntry] = []
        self._figures: list[ArtifactEntry] = []
        self._metrics: list[MetricEntry] = []
        self._notes: list[NoteEntry] = []
        self._manifest_written = False

    @staticmethod
    def _resolve_generated_at(generated_at: datetime | None) -> datetime:
        """Resolve `generated_at`: constructor arg -> `SOURCE_DATE_EPOCH` -> now.

        Args:
            generated_at: The constructor's explicit override, if any.

        Returns:
            The resolved, timezone-aware `datetime` to serialise at
            `%Y-%m-%dT%H:%M:%SZ` precision.
        """
        if generated_at is not None:
            return generated_at
        epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if epoch:
            return datetime.fromtimestamp(int(epoch), tz=UTC)
        return datetime.now(UTC)

    @property
    def directory(self) -> Path:
        """This producer's own output directory, `<root>/<notebook_id>`."""
        return self._directory

    @property
    def tables(self) -> tuple[ArtifactEntry, ...]:
        """Every table registered so far, in registration order."""
        return tuple(self._tables)

    @property
    def figures(self) -> tuple[ArtifactEntry, ...]:
        """Every figure registered so far, in registration order."""
        return tuple(self._figures)

    @property
    def metrics(self) -> tuple[MetricEntry, ...]:
        """Every metric registered so far, in registration order.

        Exposed so a producer can build an unregistered sidecar (for example
        `cleaning_report.py`'s `counts.json`) from the exact same list that
        `write_manifest()` emits, so the two can never drift apart.
        """
        return tuple(self._metrics)

    @property
    def notes(self) -> tuple[NoteEntry, ...]:
        """Every note registered so far, in registration order."""
        return tuple(self._notes)

    def _register_artifact_name(self, name: str) -> None:
        """Reserve `name` for a table/figure, raising on a repeat.

        Args:
            name: The bare artifact name (no extension).

        Raises:
            ValueError: `name` was already registered by this writer.
        """
        _validate_name(name, kind="artifact name")
        if name in self._registered_artifact_names:
            raise ValueError(f"Artifact name {name!r} is already registered.")
        self._registered_artifact_names.add(name)

    def add_table(
        self,
        frame: pd.DataFrame,
        *,
        name: str,
        title: str,
        description: str,
        sort_by: Sequence[str] | None = None,
    ) -> Path:
        """Write `frame` as `tables/<name>.csv` and register it in the manifest.

        Args:
            frame: The table to write. `index=False` is unconditional --
                callers that want a specific row order should either pass
                `sort_by` or call `sort_values()`/`reset_index()` themselves
                first.
            name: Bare name (no extension), validated and reserved before
                any filesystem call.
            title: Human-readable manifest title.
            description: Human-readable manifest description.
            sort_by: Optional column name(s) to sort ascending by, with
                `kind="stable"`, before writing.

        Returns:
            The absolute path the CSV was written to.

        Raises:
            ValueError: `name` is invalid or already registered.
        """
        self._register_artifact_name(name)
        target = self._tables_dir / f"{name}.csv"
        to_write = frame
        if sort_by is not None:
            to_write = frame.sort_values(by=list(sort_by), kind="stable")
        _atomic_write_bytes_from_csv(to_write, target)
        self._tables.append(
            ArtifactEntry(
                path=_relative_posix(target, directory=self._directory),
                title=title,
                type="csv",
                description=description,
            )
        )
        return target

    def add_json_table(
        self, payload: Any, *, name: str, title: str, description: str
    ) -> Path:
        """Write `payload` as `tables/<name>.json` and register it in the manifest.

        Args:
            payload: A JSON-serialisable object.
            name: Bare name (no extension), validated and reserved before
                any filesystem call.
            title: Human-readable manifest title.
            description: Human-readable manifest description.

        Returns:
            The absolute path the JSON file was written to.

        Raises:
            ValueError: `name` is invalid or already registered.
        """
        self._register_artifact_name(name)
        target = self._tables_dir / f"{name}.json"
        _atomic_write_json(payload, target)
        self._tables.append(
            ArtifactEntry(
                path=_relative_posix(target, directory=self._directory),
                title=title,
                type="json",
                description=description,
            )
        )
        return target

    def add_figure(
        self, figure: "Figure", *, name: str, title: str, description: str
    ) -> Path:
        """Write `figure` as `figures/<name>.png` and register it in the manifest.

        Args:
            figure: A matplotlib `Figure`.
            name: Bare name (no extension), validated and reserved before
                any filesystem call.
            title: Human-readable manifest title.
            description: Human-readable manifest description.

        Returns:
            The absolute path the PNG was written to.

        Raises:
            ValueError: `name` is invalid or already registered.
        """
        self._register_artifact_name(name)
        target = self._figures_dir / f"{name}.png"
        tmp = target.with_name(target.name + ".tmp")
        figure.savefig(
            tmp,
            format="png",
            dpi=150,
            metadata={"Software": None, "Creation Time": None},
        )
        os.replace(tmp, target)
        self._figures.append(
            ArtifactEntry(
                path=_relative_posix(target, directory=self._directory),
                title=title,
                type="png",
                description=description,
            )
        )
        return target

    def add_metric(self, name: str, value: int | float | str, description: str) -> None:
        """Register one `metrics[]` entry.

        Args:
            name: The metric's name, validated against
                `ARTIFACT_NAME_PATTERN` and reserved -- a metric name
                colliding with a previous one would silently drop a value
                from any `{name: value}` sidecar built from this list, so it
                raises instead.
            value: A number or a string.
            description: Human-readable manifest description.

        Raises:
            ValueError: `name` is invalid or already registered.
        """
        _validate_name(name, kind="metric name")
        if name in self._registered_metric_names:
            raise ValueError(f"Metric name {name!r} is already registered.")
        self._registered_metric_names.add(name)
        self._metrics.append(MetricEntry(name=name, value=value, description=description))

    def add_note(self, note_id: str, text: str) -> None:
        """Register one `notes[]` entry (**D2**).

        Args:
            note_id: The note's identifier, validated against
                `ARTIFACT_NAME_PATTERN` and reserved.
            text: The note's text.

        Raises:
            ValueError: `note_id` is invalid or already registered.
        """
        _validate_name(note_id, kind="note id")
        if note_id in self._registered_note_ids:
            raise ValueError(f"Note id {note_id!r} is already registered.")
        self._registered_note_ids.add(note_id)
        self._notes.append(NoteEntry(id=note_id, text=text))

    def write_json(self, payload: Any, *, name: str) -> Path:
        """Write an unregistered JSON sidecar directly under this notebook's folder.

        Unlike `add_json_table`, the result is never declared in
        `manifest.json` and is never pruned by `prune_undeclared` (which
        only inspects `tables/` and `figures/`) -- this is the mechanism
        `cleaning_report.py` uses for `counts.json`.

        Args:
            payload: A JSON-serialisable object.
            name: Bare name (no extension), validated before any filesystem
                call.

        Returns:
            The absolute path the JSON file was written to.

        Raises:
            ValueError: `name` is invalid.
        """
        _validate_name(name, kind="sidecar name")
        target = self._directory / f"{name}.json"
        _atomic_write_json(payload, target)
        return target

    def write_manifest(self) -> Path:
        """Write `manifest.json`, last, after confirming every declared path exists.

        Returns:
            The absolute path `manifest.json` was written to.

        Raises:
            FileNotFoundError: A declared table or figure path is missing on
                disk -- turns "the manifest lies" into a runtime guarantee
                rather than only a test assertion.
        """
        for entry in (*self._tables, *self._figures):
            candidate = self._directory / entry.path
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"Declared artifact missing on disk before writing the "
                    f"manifest: {candidate}"
                )

        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "notebook_id": self.notebook_id,
            "generated_at": self._generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tables": [asdict(entry) for entry in self._tables],
            "figures": [asdict(entry) for entry in self._figures],
            "metrics": [asdict(entry) for entry in self._metrics],
            "notes": [asdict(entry) for entry in self._notes],
        }
        target = self._directory / "manifest.json"
        _atomic_write_json(payload, target)
        self._manifest_written = True
        return target

    def _prune_undeclared_files(self) -> list[str]:
        """Delete orphaned registered-extension files under `tables/`/`figures/`.

        Returns:
            The relative POSIX path of every file removed, in sorted order.
        """
        declared = {entry.path for entry in (*self._tables, *self._figures)}
        pruned: list[str] = []
        for folder, extensions in (
            (self._tables_dir, _REGISTERED_TABLE_EXTENSIONS),
            (self._figures_dir, _REGISTERED_FIGURE_EXTENSIONS),
        ):
            if not folder.is_dir():
                continue
            for candidate in sorted(folder.iterdir()):
                if not candidate.is_file() or candidate.suffix not in extensions:
                    continue
                relative = _relative_posix(candidate, directory=self._directory)
                if relative in declared:
                    continue
                candidate.unlink()
                pruned.append(relative)
        return pruned

    def close(self) -> Path:
        """Write the manifest (if not already written) and prune orphaned files.

        Returns:
            This notebook's output directory.
        """
        # Always rewrite, even when `write_manifest()` was already called
        # explicitly: an artifact or metric registered after that call would
        # otherwise never reach disk, silently diverging the manifest from the
        # writer's own state.
        self.write_manifest()
        if self.prune_undeclared:
            pruned = self._prune_undeclared_files()
            if pruned:
                _logger.info(
                    "Pruned %d undeclared artifact(s) under %s: %s",
                    len(pruned),
                    self._directory,
                    pruned,
                )
        return self._directory

    def __enter__(self) -> "ResultsWriter":
        """Enter the writer's context.

        Returns:
            This writer, so artifacts can be registered inside the `with`
            block. The manifest is written by `__exit__` on a clean exit.
        """
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        """Write the manifest on a clean exit only -- never after an exception.

        Args:
            exc_type: The exception type raised in the `with` block, or
                `None` on a clean exit.
        """
        if exc_type is None:
            self.close()
