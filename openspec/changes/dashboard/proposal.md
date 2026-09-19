# Proposal: Results Dashboard

## Intent

Add a read-only Streamlit app, `dashboard/app.py`, that discovers every `results/*/manifest.json` at
runtime and renders whatever exists: metrics, tables, figures and the manifest `notes[]`. It is the
first consumer of the `results-output-contract` capability delivered by `project-foundation`, and the
proof that the contract's promise — "a consumer enumerates every output by scanning for
`results/*/manifest.json`, with no global index" — actually holds.

## Why now

- `results/` is currently empty apart from `.gitkeep`. The dashboard must be usable **before** the
  notebooks land and must keep working as each one lands. Building it now forces that property into
  the design instead of retrofitting it.
- The two mandatory honesty disclosures (the retained label-noise **error floor**, and the
  **non-comparability** of our test metrics with published UNSW-NB15 benchmarks) currently live only
  inside `manifest.json`. Nobody reads a JSON file. They need a surface that shows them.
- A single place to view notebook 3's model comparison removes the "open five notebooks to compare
  models" step at review time.

## Scope

**In scope**

1. `dashboard/loading.py` — pure Python, no Streamlit import: discovery, manifest parsing, artifact
   resolution, metric formatting, disclosure-note classification.
2. `dashboard/app.py` — Streamlit UI only: sidebar navigation, Overview page, one dynamic page per
   discovered section, one dedicated Model Comparison page.
3. `tests/test_dashboard.py` — smoke test against a synthetic `results/` tree in `tmp_path`, with no
   Streamlit server started.
4. `uv add streamlit`, committing `pyproject.toml` and `uv.lock`, plus `pythonpath = ["."]` under
   `[tool.pytest.ini_options]` so `dashboard` is importable by pytest.

**Out of scope**

- Any write to `results/`, `data/raw/`, or anywhere else. The app is strictly read-only.
- Loading a dataset, importing a model, or recomputing any number. Every value shown is read from a
  manifest or a declared artifact file.
- Changes to `src/nids/**`, to `results.py`/`ResultsWriter`, or to any other change folder.
- Authentication, deployment, caching servers, theming beyond Streamlit defaults.

## Approach

`discover_sections(root)` globs `<root>/*/manifest.json`, parses each one defensively, and returns a
`Section` per folder carrying a `status` and an optional human-readable `problem` string. Every
failure mode — absent `results/`, folder without a manifest, invalid JSON, unknown `schema_version`,
declared file missing on disk, unparseable CSV — becomes a `Section.status` value plus a message the
UI renders with `st.info` / `st.warning` / `st.error`. **No failure mode may raise out of the
loading layer**, so no traceback can reach the browser.

All logic lives in plain, typed, docstringed functions in `dashboard/loading.py` that the Streamlit
page calls. Nothing meaningful runs at import time of `dashboard/app.py` beyond constants, so the
smoke test imports it and exercises the helpers directly.

## Constraints honoured

- Paths resolve through `nids.paths.results_root()`; no absolute path is hardcoded (AGENTS.md).
- Type hints, docstrings and pytest tests for every helper (AGENTS.md "Code").
- All app copy, labels and docs in English (AGENTS.md "Language").
- A metric `value` may be a number **or the string `"inf"`** (the notebooks emit infinities that way
  because `json.dumps` produces invalid `Infinity` for a float inf). Formatting handles both.

## Risks

| Risk | Mitigation |
|---|---|
| `streamlit` pulls a large dependency tree into `.venv` | Single `uv add streamlit` task, lockfile committed; no transitive pin is hand-edited. |
| Manifest schema evolves past `"1.0"` | Unknown `schema_version` renders best-effort behind a visible warning rather than failing. |
| A huge table freezes the browser | `load_table` reads at most `max_rows` (default 5000) and flags truncation in the UI. |
| Notebook 3's folder/table name is not yet fixed | Model Comparison page matches by table-path stem containing `model_comparison` across all sections, and shows a clear "not generated yet" state when absent. |

## Rollback

`git revert <sha>` removes `dashboard/*.py`, `tests/test_dashboard.py` and the `streamlit`
dependency. Nothing under `src/nids/`, `results/`, `data/raw/` or any other change folder is touched,
so no other capability is affected.
