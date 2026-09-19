"""Pure, dependency-light logic for the results dashboard.

Discovers `results/<notebook_id>/manifest.json` producer folders, parses their
declared tables, figures, metrics and notes, resolves artifact paths safely, and
formats metric values for display. Every function here returns a status and a
human-readable message on failure and never raises, so a damaged or missing
`results/` tree can never crash the Streamlit page that calls into this module.

This module MUST NOT import `streamlit`; it is exercised directly by
`tests/test_dashboard.py` without a Streamlit runtime.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd

from nids import paths

SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({"1.0"})
"""Manifest `schema_version` values this dashboard understands fully."""

MODEL_COMPARISON_STEMS: Final[tuple[str, ...]] = ("model_comparison",)
"""Table path stems that identify notebook 3's model comparison table."""

DISCLOSURE_KEYWORDS: Final[tuple[str, ...]] = (
    "error floor",
    "error_floor",
    "label noise",
    "label_noise",
    "benchmark",
    "non-comparab",
    "noncomparab",
    "not comparable",
)
"""Case-insensitive substrings that mark a note as a mandatory honesty disclosure."""

SECTION_ORDER: Final[tuple[str, ...]] = (
    "data_cleaning",
    "eda_reduction_clustering",
)
"""Known producer ids, in display order; every other id follows, sorted."""

METRICS_PER_ROW: Final[int] = 4
"""Maximum metric cards rendered in a single `st.columns` row.

Four is the widest grid that still leaves each card room for a full value; beyond
that Streamlit truncates the number itself (``67,601`` collapsing to ``67…``),
which is the whole reason the metric grid wraps instead of using one long row.
"""

LABEL_ACRONYMS: Final[frozenset[str]] = frozenset(
    {
        "ari",
        "auc",
        "cv",
        "dbscan",
        "eps",
        "f1",
        "fn",
        "fp",
        "gp",
        "knn",
        "nmi",
        "pca",
        "pr",
        "roc",
        "svc",
        "ttl",
        "vif",
    }
)
"""Lower-case metric-name words rendered upper-case in a human-readable label."""

SHARE_NAME_KEYWORDS: Final[tuple[str, ...]] = (
    "share",
    "fraction",
    "proportion",
    "percent",
    "_pct",
)
"""Case-insensitive metric-name substrings that mark a value as a 0..1 share."""

SHARE_DESCRIPTION_KEYWORDS: Final[tuple[str, ...]] = (
    "share of",
    "fraction of",
    "proportion of",
    "as a share",
    "as a fraction",
)
"""Case-insensitive description substrings that mark a value as a 0..1 share.

Deliberately phrase-based rather than word-based: a description merely containing
"percentage" often reports a value that is *already* a percentage (82.1), which
must not be multiplied by 100 again. The `[-1, 1]` range guard in
`is_share_metric` is the second line of defence for exactly that case.
"""


class SectionStatus(StrEnum):
    """Outcome of loading one producer folder's `manifest.json`."""

    OK = "ok"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    MALFORMED = "malformed"
    UNREADABLE = "unreadable"
    MISSING_MANIFEST = "missing_manifest"


@dataclass(frozen=True, slots=True)
class Artifact:
    """A single declared table or figure, with its path resolved and checked."""

    path: str
    title: str
    type: str
    description: str
    resolved: Path
    exists: bool


@dataclass(frozen=True, slots=True)
class Metric:
    """A single declared scalar metric."""

    name: str
    value: int | float | str
    description: str


@dataclass(frozen=True, slots=True)
class HeadlineSpec:
    """One Overview headline slot: its label and where its value may be read from.

    `sources` is an ordered tuple of `(notebook_id, metric_name)` candidates; the
    first one present in the discovered manifests wins. Ordering matters: the
    producer that owns the number comes first, and an equivalent republished by a
    downstream notebook follows as a fallback, so a headline still fills in when
    one notebook has not run yet.
    """

    label: str
    sources: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class HeadlineMetric:
    """A `HeadlineSpec` resolved against a real manifest metric."""

    label: str
    notebook_id: str
    metric: Metric

    @property
    def source(self) -> str:
        """Provenance for the card tooltip, e.g. `classification / best_macro_f1_with_ttl`."""
        return f"{self.notebook_id} / {self.metric.name}"


HEADLINE_SPECS: Final[tuple[HeadlineSpec, ...]] = (
    HeadlineSpec(
        label="Best model",
        sources=(("classification", "best_model_with_ttl"),),
    ),
    HeadlineSpec(
        label="Best macro F1",
        sources=(("classification", "best_macro_f1_with_ttl"),),
    ),
    HeadlineSpec(
        label="Cleaned train rows",
        sources=(
            ("data_cleaning", "train_rows_final"),
            ("classification", "n_train_rows"),
        ),
    ),
    HeadlineSpec(
        label="Cleaned test rows",
        sources=(
            ("data_cleaning", "test_rows_final"),
            ("classification", "n_test_rows"),
        ),
    ),
    HeadlineSpec(
        label="Label-noise error floor",
        sources=(
            ("classification", "label_noise_error_floor_share"),
            ("data_cleaning", "test_contradictory_share_of_retained"),
        ),
    ),
)
"""The Overview headline slots, in display order.

Every slot names a manifest to read from; not one number lives here. A slot whose
sources are all absent is simply dropped, so the headline degrades to whatever the
notebooks that have actually run can support.
"""


@dataclass(frozen=True, slots=True)
class Note:
    """A single declared free-text note, possibly a mandatory disclosure."""

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class Section:
    """The parsed (or failed) contents of one `results/<notebook_id>/` folder."""

    notebook_id: str
    directory: Path
    status: SectionStatus
    problem: str | None
    schema_version: str | None
    generated_at: str | None
    tables: tuple[Artifact, ...]
    figures: tuple[Artifact, ...]
    metrics: tuple[Metric, ...]
    notes: tuple[Note, ...]


@dataclass(frozen=True, slots=True)
class TableLoad:
    """The result of reading one declared table artifact into a DataFrame."""

    frame: pd.DataFrame | None
    truncated: bool
    problem: str | None


def _empty_section(
    notebook_id: str,
    directory: Path,
    status: SectionStatus,
    problem: str,
) -> Section:
    """Build a `Section` for a folder whose manifest could not be parsed at all."""
    return Section(
        notebook_id=notebook_id,
        directory=directory,
        status=status,
        problem=problem,
        schema_version=None,
        generated_at=None,
        tables=(),
        figures=(),
        metrics=(),
        notes=(),
    )


def _resolve_artifact_path(directory: Path, path: str) -> Path | None:
    """Resolve `path` relative to `directory`, refusing absolute or escaping paths.

    Returns:
        The resolved `Path` when it is safely contained in `directory`, else `None`.
    """
    raw_path = Path(path)
    if raw_path.is_absolute():
        return None
    resolved = (directory / raw_path).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        return None
    return resolved


def _as_str(value: object, default: str = "") -> str:
    """Return `value` when it is a non-empty-typed `str`, else `default`."""
    return value if isinstance(value, str) else default


def _parse_artifacts(
    directory: Path,
    payload: dict[str, object],
    field_name: str,
    problems: list[str],
) -> tuple[Artifact, ...]:
    """Parse the `tables` or `figures` array of a manifest into `Artifact` entries."""
    raw_value = payload.get(field_name, [])
    if not isinstance(raw_value, list):
        problems.append(f"'{field_name}' is not a list; treated as empty.")
        return ()

    artifacts: list[Artifact] = []
    for entry in raw_value:
        if not isinstance(entry, dict):
            problems.append(f"skipped malformed entry in '{field_name}'")
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            problems.append(f"skipped entry in '{field_name}' with no 'path'")
            continue
        resolved = _resolve_artifact_path(directory, path)
        if resolved is None:
            problems.append(f"skipped unsafe path: {path}")
            continue
        artifacts.append(
            Artifact(
                path=path,
                title=_as_str(entry.get("title")),
                type=_as_str(entry.get("type")),
                description=_as_str(entry.get("description")),
                resolved=resolved,
                exists=resolved.is_file(),
            )
        )
    return tuple(artifacts)


def _parse_metrics(
    payload: dict[str, object],
    problems: list[str],
) -> tuple[Metric, ...]:
    """Parse the `metrics` array of a manifest into `Metric` entries."""
    raw_value = payload.get("metrics", [])
    if not isinstance(raw_value, list):
        problems.append("'metrics' is not a list; treated as empty.")
        return ()

    metrics: list[Metric] = []
    for entry in raw_value:
        if not isinstance(entry, dict):
            problems.append("skipped malformed entry in 'metrics'")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            problems.append("skipped entry in 'metrics' with no 'name'")
            continue
        value = entry.get("value", "")
        metrics.append(
            Metric(
                name=name,
                value=value if isinstance(value, (int, float, str)) else str(value),
                description=_as_str(entry.get("description")),
            )
        )
    return tuple(metrics)


def _parse_notes(
    payload: dict[str, object],
    problems: list[str],
) -> tuple[Note, ...]:
    """Parse the `notes` array of a manifest into `Note` entries."""
    raw_value = payload.get("notes", [])
    if not isinstance(raw_value, list):
        problems.append("'notes' is not a list; treated as empty.")
        return ()

    notes: list[Note] = []
    for entry in raw_value:
        if not isinstance(entry, dict):
            problems.append("skipped malformed entry in 'notes'")
            continue
        note_id = entry.get("id")
        if not isinstance(note_id, str) or not note_id:
            problems.append("skipped entry in 'notes' with no 'id'")
            continue
        notes.append(Note(id=note_id, text=_as_str(entry.get("text"))))
    return tuple(notes)


def load_section(directory: Path) -> Section:
    """Load and parse one producer folder's `manifest.json`, never raising.

    Args:
        directory: The `results/<notebook_id>/` folder to load.

    Returns:
        A `Section` describing the outcome. `Section.notebook_id` is always the
        folder name (the manifest's own `notebook_id` field is validated for
        presence but not used as identity, since it can be unreadable precisely
        when it would matter most).
    """
    notebook_id = directory.name
    manifest_path = directory / "manifest.json"

    if not manifest_path.is_file():
        return _empty_section(
            notebook_id,
            directory,
            SectionStatus.MISSING_MANIFEST,
            f"No manifest.json in results/{notebook_id}/ — this notebook has not run yet.",
        )

    try:
        text = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return _empty_section(
            notebook_id,
            directory,
            SectionStatus.UNREADABLE,
            f"Could not read manifest.json: {exc}",
        )

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return _empty_section(
            notebook_id,
            directory,
            SectionStatus.MALFORMED,
            f"manifest.json in results/{notebook_id}/ is not valid JSON "
            f"(line {exc.lineno}, column {exc.colno}): {exc.msg}",
        )

    if not isinstance(payload, dict):
        return _empty_section(
            notebook_id,
            directory,
            SectionStatus.MALFORMED,
            f"manifest.json in results/{notebook_id}/ does not contain a JSON object.",
        )

    manifest_notebook_id = payload.get("notebook_id")
    if not isinstance(manifest_notebook_id, str) or not manifest_notebook_id:
        return _empty_section(
            notebook_id,
            directory,
            SectionStatus.MALFORMED,
            f"manifest.json in results/{notebook_id}/ is missing a string 'notebook_id' field.",
        )

    status = SectionStatus.OK
    problems: list[str] = []

    raw_schema_version = payload.get("schema_version")
    if not isinstance(raw_schema_version, str) or raw_schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        status = SectionStatus.UNSUPPORTED_SCHEMA
        supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        problems.append(
            f"Unsupported schema_version {raw_schema_version!r}; supported: {supported}."
        )
    schema_version = raw_schema_version if isinstance(raw_schema_version, str) else None

    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str):
        generated_at = None

    tables = _parse_artifacts(directory, payload, "tables", problems)
    figures = _parse_artifacts(directory, payload, "figures", problems)
    metrics = _parse_metrics(payload, problems)
    notes = _parse_notes(payload, problems)

    return Section(
        notebook_id=notebook_id,
        directory=directory,
        status=status,
        problem="; ".join(problems) if problems else None,
        schema_version=schema_version,
        generated_at=generated_at,
        tables=tables,
        figures=figures,
        metrics=metrics,
        notes=notes,
    )


def _section_sort_key(section: Section) -> tuple[int, str]:
    """Sort key placing `SECTION_ORDER` ids first, in order, then the rest alphabetically."""
    try:
        rank = SECTION_ORDER.index(section.notebook_id)
    except ValueError:
        rank = len(SECTION_ORDER)
    return (rank, section.notebook_id)


def discover_sections(root: Path | None = None) -> tuple[Section, ...]:
    """Enumerate every producer folder under `root` (or `results_root()`).

    Args:
        root: The `results/` directory to scan. Defaults to `results_root()`.

    Returns:
        One `Section` per direct, non-hidden subdirectory of `root`, ordered by
        `SECTION_ORDER` then alphabetically. Returns `()` when `root` does not
        exist or cannot be listed; never raises.
    """
    base = root if root is not None else paths.results_root()
    if not base.is_dir():
        return ()

    try:
        children = sorted(base.iterdir())
    except OSError:
        return ()

    sections: list[Section] = []
    for child in children:
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if not is_dir or child.name.startswith("."):
            continue
        sections.append(load_section(child))

    return tuple(sorted(sections, key=_section_sort_key))


def load_table(artifact: Artifact, *, max_rows: int = 5000) -> TableLoad:
    """Read a declared CSV or JSON table artifact into a `pandas.DataFrame`.

    Args:
        artifact: The resolved table artifact to load.
        max_rows: Maximum number of rows to keep; longer tables are truncated.

    Returns:
        A `TableLoad` with either a populated `frame` or a `problem` message.
        Never raises.
    """
    if not artifact.exists:
        return TableLoad(
            frame=None,
            truncated=False,
            problem=f"Declared file is missing: {artifact.path}",
        )

    suffix = artifact.resolved.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(artifact.resolved, nrows=max_rows + 1)
        elif suffix == ".json":
            with artifact.resolved.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list) and payload and all(
                isinstance(item, dict) for item in payload
            ):
                frame = pd.DataFrame(payload[: max_rows + 1])
            else:
                try:
                    frame = pd.json_normalize(payload)
                except (TypeError, ValueError):
                    if isinstance(payload, dict):
                        frame = pd.DataFrame(
                            {"key": list(payload.keys()), "value": list(payload.values())}
                        )
                    else:
                        frame = pd.DataFrame({"key": ["value"], "value": [payload]})
        else:
            return TableLoad(
                frame=None,
                truncated=False,
                problem=f"Unsupported table format: {suffix or '(no extension)'}",
            )
    except (
        OSError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return TableLoad(frame=None, truncated=False, problem=str(exc))

    truncated = len(frame) > max_rows
    if truncated:
        frame = frame.iloc[:max_rows]
    return TableLoad(frame=frame, truncated=truncated, problem=None)


def format_metric_value(value: object) -> str:
    """Format a metric value for display, never raising for any input type.

    Args:
        value: The raw metric value from a manifest — typically `int`, `float`
            or `str`, but any type is handled.

    Returns:
        A human-readable string. `bool` is checked before `int` (a `bool` is an
        `int` subclass in Python). The strings `"inf"`/`"-inf"`/`"nan"` (and the
        `"infinity"` spellings, case-insensitively) render as `"∞"`/`"-∞"`/`"n/a"`,
        matching the float-infinity/NaN handling. Integers carry thousands
        separators; non-integral floats are rounded to three decimals.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"inf", "+inf", "infinity", "+infinity"}:
            return "∞"
        if normalized in {"-inf", "-infinity"}:
            return "-∞"
        if normalized == "nan":
            return "n/a"
        return value

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return f"{value:,}"

    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return "∞" if value > 0 else "-∞"
        if value.is_integer() and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:,.3f}"

    return str(value)


def metric_label(name: str) -> str:
    """Turn a technical metric name into a human-readable label.

    Args:
        name: The manifest metric name, e.g. `"train_duplicate_rows_dropped"`.

    Returns:
        A sentence-cased label, e.g. `"Train duplicate rows dropped"`. Words in
        `LABEL_ACRONYMS` are upper-cased wherever they appear, so
        `"kmeans_silhouette_at_selected_k_with_ttl"` keeps its `"TTL"`. The
        original `name` is returned unchanged when it has no word characters,
        so the caller always has something to show.
    """
    words = [word for word in name.replace("-", "_").split("_") if word]
    if not words:
        return name

    rendered = [
        word.upper() if word.lower() in LABEL_ACRONYMS else word.lower() for word in words
    ]
    first = rendered[0]
    if first.lower() not in LABEL_ACRONYMS:
        first = first.capitalize()
    return " ".join([first, *rendered[1:]])


def is_numeric_metric(metric: Metric) -> bool:
    """Return `True` when `metric.value` is a real number worth an `st.metric` card.

    `bool` is excluded on purpose: it is an `int` subclass in Python but reads as
    a flag, not a measurement, so it belongs in the text-badge branch alongside
    string values such as `"full_row"` or `"hist_gradient_boosting"`.
    """
    return isinstance(metric.value, (int, float)) and not isinstance(metric.value, bool)


def is_share_metric(metric: Metric) -> bool:
    """Return `True` when `metric` holds a 0..1 share that should render as a percentage.

    A metric qualifies only when both hold:

    1. its name or description signals a share/fraction/proportion, and
    2. its value is a real number inside `[-1, 1]`.

    The range guard is what keeps an already-percentage value (for example
    `expected_cost_reduction_with_ttl == 82.1`) from being multiplied by 100 a
    second time, and keeps `cost_ratio_fn_to_fp == 20.0` a plain ratio.
    """
    if not is_numeric_metric(metric):
        return False

    value = float(metric.value)
    if math.isnan(value) or math.isinf(value) or abs(value) > 1.0:
        return False

    lowered_name = metric.name.lower()
    if any(keyword in lowered_name for keyword in SHARE_NAME_KEYWORDS):
        return True

    lowered_description = metric.description.lower()
    return any(keyword in lowered_description for keyword in SHARE_DESCRIPTION_KEYWORDS)


def format_metric_display(metric: Metric) -> str:
    """Format `metric` for its card, applying the percentage rule then `format_metric_value`.

    Args:
        metric: The metric to display.

    Returns:
        `"5.50%"` for a share metric, otherwise whatever `format_metric_value`
        makes of the raw value: `"67,601"` for integers, `"0.405"` for floats,
        and the value itself for strings. Never raises.
    """
    if is_share_metric(metric):
        return f"{float(metric.value) * 100:,.2f}%"
    return format_metric_value(metric.value)


def metric_rows(
    metrics: Sequence[Metric],
    per_row: int = METRICS_PER_ROW,
) -> tuple[tuple[Metric, ...], ...]:
    """Chunk `metrics` into display rows of at most `per_row` entries, in manifest order.

    Args:
        metrics: The metrics to lay out.
        per_row: Maximum cards per row. Values below 1 are clamped to 1, so a
            caller can never produce an empty or negative `st.columns` spec.

    Returns:
        A tuple of rows; every row holds between 1 and `per_row` metrics, and
        concatenating the rows reproduces `metrics` exactly.
    """
    width = max(1, per_row)
    return tuple(
        tuple(metrics[start : start + width]) for start in range(0, len(metrics), width)
    )


def metrics_table(metrics: Sequence[Metric]) -> pd.DataFrame:
    """Build the full `name`/`value`/`description` frame behind the "All metrics" expander.

    Values are formatted exactly as the cards format them, so the expander is a
    complete, scrollable restatement of the grid rather than a second source of
    truth. Columns are always present, even when `metrics` is empty.
    """
    return pd.DataFrame(
        {
            "name": [metric.name for metric in metrics],
            "value": [format_metric_display(metric) for metric in metrics],
            "description": [metric.description for metric in metrics],
        },
        columns=["name", "value", "description"],
    )


def find_metric(
    sections: Sequence[Section],
    notebook_id: str,
    name: str,
) -> Metric | None:
    """Return the metric called `name` declared by section `notebook_id`, if any.

    Args:
        sections: Every discovered section to search.
        notebook_id: The producer folder name that should own the metric.
        name: The exact manifest metric name.

    Returns:
        The matching `Metric`, or `None` when that section is absent, failed to
        parse, or simply does not declare that metric. Never raises.
    """
    for section in sections:
        if section.notebook_id != notebook_id:
            continue
        for metric in section.metrics:
            if metric.name == name:
                return metric
    return None


def headline_metrics(
    sections: Sequence[Section],
    specs: Sequence[HeadlineSpec] = HEADLINE_SPECS,
) -> tuple[HeadlineMetric, ...]:
    """Resolve the Overview headline slots against the discovered manifests.

    Args:
        sections: Every discovered section.
        specs: The slots to resolve, in display order.

    Returns:
        One `HeadlineMetric` per slot whose sources could be satisfied, in `specs`
        order. Slots with no readable source are omitted rather than shown empty,
        so a half-run `results/` tree yields a shorter headline instead of a row
        of blanks. Returns `()` when nothing resolves. Never raises.
    """
    resolved: list[HeadlineMetric] = []
    for spec in specs:
        for notebook_id, name in spec.sources:
            metric = find_metric(sections, notebook_id, name)
            if metric is None:
                continue
            resolved.append(
                HeadlineMetric(label=spec.label, notebook_id=notebook_id, metric=metric)
            )
            break
    return tuple(resolved)


def headline_rows(
    headlines: Sequence[HeadlineMetric],
    per_row: int = METRICS_PER_ROW,
) -> tuple[tuple[HeadlineMetric, ...], ...]:
    """Chunk resolved headline slots into the same at-most-`per_row` grid as the sections."""
    width = max(1, per_row)
    return tuple(
        tuple(headlines[start : start + width])
        for start in range(0, len(headlines), width)
    )


def is_disclosure_note(note: Note) -> bool:
    """Return `True` when `note.text` matches a mandatory disclosure keyword."""
    lowered = note.text.lower()
    return any(keyword in lowered for keyword in DISCLOSURE_KEYWORDS)


def disclosure_notes(sections: Sequence[Section]) -> tuple[tuple[Section, Note], ...]:
    """Collect every mandatory disclosure note across `sections`, in order."""
    found: list[tuple[Section, Note]] = []
    for section in sections:
        for note in section.notes:
            if is_disclosure_note(note):
                found.append((section, note))
    return tuple(found)


def find_model_comparison(
    sections: Sequence[Section],
) -> tuple[Section, Artifact] | None:
    """Locate notebook 3's model comparison table by path stem across `sections`.

    Args:
        sections: Every discovered section to search.

    Returns:
        The `(Section, Artifact)` pair for the first matching table, or `None`
        when no section declares a table whose path stem is in
        `MODEL_COMPARISON_STEMS`.
    """
    for section in sections:
        for table in section.tables:
            if Path(table.path).stem in MODEL_COMPARISON_STEMS:
                return (section, table)
    return None


def section_label(section: Section) -> str:
    """Build a human-readable page label for `section` from its folder id."""
    return " ".join(word.capitalize() for word in section.notebook_id.split("_"))
