# Design: Results Dashboard

## Decisions

| # | Decision | Why | Rejected |
|---|---|---|---|
| D1 | Two modules: `dashboard/loading.py` (pure, no `streamlit` import) and `dashboard/app.py` (Streamlit only). | The smoke test imports and drives the whole logic layer with no Streamlit runtime involved. | One file with helpers inline — untestable without a server. |
| D2 | Every failure is a returned `SectionStatus` + `problem: str`, never an exception. | A traceback in the browser is the exact failure mode the proposal forbids. | `try/except` scattered in the page code. |
| D3 | Sections carry the folder name as identity; the manifest's `notebook_id` is advisory. | A folder whose manifest is unreadable still needs a name to display. | Keying by `notebook_id` — unavailable precisely when it matters. |
| D4 | Unknown `schema_version` renders best-effort behind a warning. | Only `"1.0"` exists today; a future producer must not blank the app. | Hard refusal. |
| D5 | Navigation is a `st.sidebar.radio` over a list built at runtime. | Works on every Streamlit version, no `pages/` directory, order is ours. | `st.navigation` / `pages/` — pins a Streamlit minimum and hardcodes page files. |
| D6 | The model comparison table is found by path **stem** match across all sections. | Notebook 3's folder name is not fixed yet; the table name is the stable part. | Hardcoding `results/classification/tables/model_comparison.csv`. |
| D7 | `pythonpath = ["."]` is added to `[tool.pytest.ini_options]`; `app.py` bootstraps `sys.path` from `Path(__file__).resolve().parent.parent`. | `dashboard` is not an installed package (`hatch` packages only `src/nids`). Both entry points then resolve `dashboard.loading` identically, with no absolute path. | Installing `dashboard` as a package — it is an app, not a library. |
| D8 | Tables are read with `pandas.read_csv(..., nrows=max_rows + 1)`; truncation is detected and flagged. | A 100k-row table would freeze the browser. | Unbounded read. |

## Module layout

```
dashboard/
  __init__.py      # empty; makes `dashboard` a package
  loading.py       # pure logic: discovery, parsing, resolution, formatting  (~220 lines)
  app.py           # Streamlit UI: navigation + 3 page renderers            (~180 lines)
tests/
  test_dashboard.py
```

`app.py` imports nothing from `src/nids` except `nids.paths.results_root`, and that only indirectly
through `loading.py`.

## `dashboard/loading.py` — public surface

```python
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({"1.0"})
MODEL_COMPARISON_STEMS: Final[tuple[str, ...]] = ("model_comparison",)
DISCLOSURE_KEYWORDS: Final[tuple[str, ...]] = (
    "error floor", "error_floor", "label noise", "label_noise",
    "benchmark", "non-comparab", "noncomparab", "not comparable",
)
SECTION_ORDER: Final[tuple[str, ...]] = (
    "data_cleaning", "eda_reduction_clustering",
)  # known ids first, in this order; every other id follows, sorted


class SectionStatus(StrEnum):
    OK = "ok"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    MALFORMED = "malformed"
    UNREADABLE = "unreadable"
    MISSING_MANIFEST = "missing_manifest"


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str            # as declared, POSIX, relative to the section folder
    title: str
    type: str            # "csv" | "json" | "png"
    description: str
    resolved: Path       # absolute, already confirmed inside the section folder
    exists: bool


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    value: int | float | str
    description: str


@dataclass(frozen=True, slots=True)
class Note:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class Section:
    notebook_id: str                 # folder name (D3)
    directory: Path
    status: SectionStatus
    problem: str | None              # human-readable, never a traceback
    schema_version: str | None
    generated_at: str | None
    tables: tuple[Artifact, ...]
    figures: tuple[Artifact, ...]
    metrics: tuple[Metric, ...]
    notes: tuple[Note, ...]


@dataclass(frozen=True, slots=True)
class TableLoad:
    frame: pd.DataFrame | None
    truncated: bool
    problem: str | None


def discover_sections(root: Path | None = None) -> tuple[Section, ...]: ...
def load_section(directory: Path) -> Section: ...
def load_table(artifact: Artifact, *, max_rows: int = 5000) -> TableLoad: ...
def format_metric_value(value: object) -> str: ...
def is_disclosure_note(note: Note) -> bool: ...
def disclosure_notes(sections: Sequence[Section]) -> tuple[tuple[Section, Note], ...]: ...
def find_model_comparison(
    sections: Sequence[Section],
) -> tuple[Section, Artifact] | None: ...
def section_label(section: Section) -> str: ...
```

## Discovery algorithm — `discover_sections(root)`

1. `base = root if root is not None else results_root()`.
2. If `base` is not an existing directory, return `()`.
3. For each `child` in `sorted(base.iterdir())` that is a directory and is not hidden:
   call `load_section(child)`; append the result. An `OSError` from `iterdir` returns `()`.
4. Sort: ids in `SECTION_ORDER` first in that order, all others alphabetically after.

`load_section(directory)` — one pass, no raise:

1. `manifest = directory / "manifest.json"`. Not a file → `MISSING_MANIFEST`, problem
   `"No manifest.json in results/<id>/ — this notebook has not run yet."`
2. Read text; `OSError`/`UnicodeDecodeError` → `UNREADABLE` with the OS message.
3. `json.loads`; `JSONDecodeError` → `MALFORMED` with line/column.
4. Not a `dict`, or no string `notebook_id` → `MALFORMED`.
5. `schema_version` absent or outside `SUPPORTED_SCHEMA_VERSIONS` → status `UNSUPPORTED_SCHEMA`,
   parsing continues.
6. Parse `tables`, `figures` with `_parse_artifacts`, `metrics`, `notes` with their parsers. A
   non-list value is treated as empty and appended to `problem`. An entry that is not a mapping, or
   lacks a string `path` / `name` / `id`, is skipped and counted in `problem`.
7. Per artifact: reject an absolute `path`, then `resolved = (directory / path).resolve()`; if not
   `resolved.is_relative_to(directory.resolve())` the entry is skipped and `problem` records
   `"skipped unsafe path"`. Otherwise `exists = resolved.is_file()`.
8. Missing optional fields default to `""`. Status stays `OK` unless step 5 changed it.

## Pages

| Page | Content |
|---|---|
| **Overview** | `st.title("UNSW-NB15 — Results Dashboard")`. `disclosure_notes(...)` rendered first as `st.warning`, each prefixed with its producer id. Then a `st.dataframe` of one row per section: id, status, `generated_at`, counts of tables/figures/metrics/notes. Then `st.caption` naming the resolved `results/` root. Empty discovery → `st.info("No results yet. Run a notebook, then reload this page.")` |
| **One page per discovered section** (dynamic, in `SECTION_ORDER`) | Status banner first when `status != OK`. Then `## Metrics` (`st.columns` of `st.metric`, `st.caption` for each description), `## Notes` (`st.warning` each), `## Tables` (per table: title, description, `st.dataframe`), `## Figures` (per figure: title, `st.image(..., caption=description)`). |
| **Model comparison** | `find_model_comparison(sections)`; found → producer id, `generated_at`, title, description, `st.dataframe`. Not found → `st.info("Notebook 3 has not produced a model comparison table yet.")` |

Sidebar: `st.sidebar.radio("Page", ["Overview", *section labels, "Model comparison"])`. Everything
Streamlit-facing lives inside `main()`; module level holds imports, constants and the `sys.path`
bootstrap only, so importing `dashboard.app` renders nothing.

## Rendering rules

- **Tables** — `load_table` dispatches on the resolved suffix: `.csv` → `pd.read_csv(path,
  nrows=max_rows + 1)`; `.json` → `json.load` then `pd.DataFrame(payload)` when it is a list of
  mappings, else `pd.json_normalize(payload)`, else a two-column key/value frame. Anything raising
  (`OSError`, `pd.errors.ParserError`, `pd.errors.EmptyDataError`, `ValueError`,
  `json.JSONDecodeError`) → `TableLoad(None, False, "<message>")`. More than `max_rows` rows → keep
  `max_rows`, `truncated=True`.
- **Figures** — `st.image(str(artifact.resolved), caption=artifact.description)` only when
  `artifact.exists`; otherwise `st.warning`.
- **Metrics** — `st.metric(label=metric.name, value=format_metric_value(metric.value))`.
- **Notes** — `st.warning(note.text)`.

`format_metric_value(value)`:

| Input | Output |
|---|---|
| `str` in `{"inf", "+inf", "infinity", "+infinity"}` (case-insensitive, stripped) | `"∞"` |
| `str` in `{"-inf", "-infinity"}` | `"-∞"` |
| `str` in `{"nan"}` | `"n/a"` |
| any other `str` | the string unchanged |
| `bool` | `"true"` / `"false"` (checked **before** `int`) |
| `int` | `f"{value:,}"` |
| `float` infinity / `-inf` / `nan` | `"∞"` / `"-∞"` / `"n/a"` |
| `float` that is integral and `abs < 1e15` | `f"{int(value):,}"` |
| any other `float` | `f"{value:,.4f}"` |
| anything else | `str(value)` |

## Graceful-degradation matrix

| Condition | Status | What the user sees |
|---|---|---|
| `results/` absent or empty | — | `st.info` "No results yet. Run a notebook, then reload this page." |
| Folder without `manifest.json` | `MISSING_MANIFEST` | Row in the overview + `st.info` "this notebook has not run yet" on its page |
| `manifest.json` unreadable (permissions, encoding) | `UNREADABLE` | `st.error` with the OS message |
| `manifest.json` not valid JSON | `MALFORMED` | `st.error` naming the folder, line and column |
| Top level not an object, or `notebook_id` missing | `MALFORMED` | `st.error` naming the missing field |
| `schema_version` unknown or absent | `UNSUPPORTED_SCHEMA` | `st.warning` naming the version; every parsable entry still renders |
| `tables`/`figures`/`metrics`/`notes` not a list | status unchanged | treated as empty, noted in the section's `problem` banner |
| One entry not a mapping, or missing `path`/`name`/`id` | status unchanged | entry skipped, count shown in the `problem` banner |
| Entry `path` escaping the section folder | status unchanged | entry skipped, `problem` says "unsafe path skipped" |
| Declared table/figure file missing | status unchanged, `exists=False` | `st.warning` "Declared file is missing: `<path>`" in place of the widget |
| Table exists but unparseable | — | `st.error` with the parser message, for that table only |
| Table longer than `max_rows` | — | first `max_rows` rows + `st.caption` "showing the first 5000 rows" |
| Metric value is `"inf"` | — | `∞` |
| No model comparison table anywhere | — | `st.info` "Notebook 3 has not produced a model comparison table yet." |

## Smoke test — `tests/test_dashboard.py`

Fixture `synthetic_results(tmp_path) -> Path` builds `tmp_path / "results"` with:

| Folder | Content | Asserted |
|---|---|---|
| `data_cleaning/` | valid manifest `schema_version "1.0"`; 1 CSV table, 1 PNG figure (a real 1×1 PNG written from bytes), metrics `[{"name":"rows","value":108000}, {"name":"ratio","value":"inf"}]`, notes with an error-floor and a non-comparability text | `OK`; counts; `format_metric_value("inf") == "∞"`; both notes classified as disclosures |
| `classification/` | valid manifest declaring `tables/model_comparison.csv` | `find_model_comparison` returns this section and artifact |
| `future_schema/` | valid JSON, `schema_version "9.9"`, one table present | `UNSUPPORTED_SCHEMA`, and the table still parsed |
| `broken_json/` | `manifest.json` containing `{not json` | `MALFORMED`, `problem` non-empty |
| `missing_file/` | manifest declaring `figures/absent.png`, file not created | figure `exists is False`, no raise |
| `no_manifest/` | empty directory | `MISSING_MANIFEST` |

Tests: discovery against that tree; discovery against a missing root returning `()`; discovery with
no argument using the existing `tmp_results_root` fixture from `tests/conftest.py`;
`load_table` on the CSV, on an absent file, and on a deliberately unparseable file; `format_metric_value`
parametrised over `"inf"`, `"-inf"`, `float("inf")`, `float("nan")`, `True`, `12345`, `1.5`, `"srcip_dstip"`;
a traversal entry (`"path": "../escape.csv"`) being skipped; `import dashboard.app` succeeding and
exposing `main` without starting a server; and an `ast`-based guard asserting `dashboard/` contains no
write-mode `open`, no `raw_data_dir`, and no `data/raw` string.

## Files

| File | Action |
|---|---|
| `pyproject.toml` | Modify — `streamlit` dependency (via `uv add`), `pythonpath = ["."]` under `[tool.pytest.ini_options]` |
| `uv.lock` | Modify — regenerated by `uv add` |
| `dashboard/__init__.py` | Create |
| `dashboard/loading.py` | Create |
| `dashboard/app.py` | Create |
| `tests/test_dashboard.py` | Create |
| `README.md` | Modify — one line: `uv run streamlit run dashboard/app.py` |
