# Tasks: Results Dashboard

**Delivery:** one slice, one PR, two commits (dependency change kept separate from the feature so a
lockfile conflict can be rebased on its own). Strategy `single-pr`.

**Authorized scope:** `dashboard/**`, `tests/test_dashboard.py`, `pyproject.toml`, `uv.lock`,
`README.md`. Nothing under `src/nids/**`, `data/**`, `results/**`, `notebooks/**`, or another
`openspec/changes/**` folder may be created, modified or deleted.

**TDD:** strict mode is enabled. Write `tests/test_dashboard.py` first, observe RED, then implement.

> **Apply-time correction (verified against `openspec/config.yaml`):** the file's
> `strict_tdd: false` (`strict_tdd_source: "explicit user correction"`) and
> `apply.tdd: false` override this header. Tests and implementation were written
> together, not RED-first; task 2.3's observed-RED step was skipped rather than
> fabricated. Tests remain required and were run green (task 5.1). No commit in
> this apply run (1.4 / 5.5 left unchecked): the orchestrator committing directly
> conflicted with a concurrent agent also writing to this tree, so `pyproject.toml`,
> `uv.lock`, `dashboard/**` and `tests/test_dashboard.py` are left staged/unstaged
> for the parent to commit. `README.md` task 4.5 was skipped: a concurrent agent
> was reported to own that file this run; it was left untouched to avoid a
> conflicting write.

---

## 1. Dependency and test path — commit 1

- [x] 1.1 Run `uv add streamlit` from the repository root. Spec: Req "importable and covered by a
      serverless smoke test". Outcome: `streamlit` appears under `[project].dependencies` in
      `pyproject.toml` and `uv.lock` is regenerated. Do not hand-edit either file.
- [x] 1.2 Add `pythonpath = ["."]` to `[tool.pytest.ini_options]` in `pyproject.toml` (design D7), so
      `import dashboard.loading` resolves under pytest. Leave `testpaths`, `markers` and `addopts`
      unchanged.
- [x] 1.3 Verify: `uv run python -c "import streamlit; print(streamlit.__version__)"` exits 0, and
      `uv run pytest` stays green (no new tests yet).
- [ ] 1.4 Commit `pyproject.toml` and `uv.lock` only:
      `chore(deps): add streamlit and enable repo-root imports for pytest`

## 2. Smoke test first — RED

- [x] 2.1 Create `tests/test_dashboard.py` with the `synthetic_results(tmp_path) -> Path` fixture
      exactly as the design's smoke-test table specifies: `data_cleaning/` (valid, 1 CSV, 1 real PNG,
      metrics including `{"name": "ratio", "value": "inf"}`, an error-floor note and a
      non-comparability note), `classification/` (declares `tables/model_comparison.csv`),
      `future_schema/` (`schema_version "9.9"`), `broken_json/` (`{not json`), `missing_file/`
      (declares `figures/absent.png`, never created), `no_manifest/` (empty directory).
- [x] 2.2 Write the tests: discovery over the fixture tree; discovery over an absent root returning
      `()`; discovery with no argument via the existing `tmp_results_root` fixture in
      `tests/conftest.py`; each of the five failure conditions observable in `Section.status` /
      `Section.problem`; `load_table` on a valid CSV, an absent file and an unparseable file;
      `format_metric_value` parametrised over `"inf"`, `"-inf"`, `float("inf")`, `float("nan")`,
      `True`, `12345`, `1.5`, `"srcip_dstip"`; a manifest entry with `"path": "../escape.csv"` being
      skipped; `is_disclosure_note` / `disclosure_notes` finding both mandatory notes;
      `find_model_comparison` locating the `classification/` table and returning `None` when absent;
      `import dashboard.app` succeeding and exposing `main`; and an `ast`-based guard over
      `dashboard/*.py` asserting no write-mode `open`, no `raw_data_dir`, no `data/raw` literal.
      No test may start a Streamlit server or call `subprocess`.
- [ ] 2.3 Observe RED: `uv run pytest tests/test_dashboard.py` fails on collection
      (`ModuleNotFoundError: dashboard`). Record the observed failure in the apply report.

## 3. Loading layer — GREEN

- [x] 3.1 Create `dashboard/__init__.py` (empty, package marker only).
- [x] 3.2 Create `dashboard/loading.py` with the constants, `SectionStatus`, the five frozen
      dataclasses and the nine public functions exactly as declared in the design's public-surface
      block, each with type hints and a docstring (AGENTS.md "Code"). Import `results_root` from
      `nids.paths`; no absolute path anywhere; no `streamlit` import in this module.
- [x] 3.3 Implement `discover_sections` and `load_section` following the design's eight numbered
      steps, including the `is_relative_to` containment check and the absolute-path rejection.
      Spec: Req "Runtime discovery", Req "never raises". Every failure path returns a status plus a
      message; no `raise` and no bare `except:`.
- [x] 3.4 Implement `load_table` (CSV and JSON dispatch, `nrows=max_rows + 1`, truncation flag,
      every documented exception converted to `TableLoad(None, False, msg)`) and
      `format_metric_value` exactly per the design's formatting table, `bool` checked before `int`.
      Spec: Req "A metric value may be a number or a string, including the string \"inf\"".
- [x] 3.5 Implement `is_disclosure_note`, `disclosure_notes`, `find_model_comparison` (stem match
      over `MODEL_COMPARISON_STEMS` across all sections, design D6) and `section_label`.
- [x] 3.6 Verify: `uv run pytest tests/test_dashboard.py` — every test except the
      `import dashboard.app` one passes.

## 4. Streamlit page

- [x] 4.1 Create `dashboard/app.py`: module level holds only imports, constants and the
      `sys.path` bootstrap from `Path(__file__).resolve().parent.parent` (design D7). Everything
      Streamlit-facing lives in `main()` and the render helpers it calls, so importing the module
      renders nothing. End with `if __name__ == "__main__": main()`.
- [x] 4.2 Implement `render_overview`, `render_section`, `render_model_comparison` and the
      `st.sidebar.radio` navigation per the design's Pages table. Disclosure notes render first on
      Overview, each prefixed with its producer id. Spec: Req "honesty disclosures are surfaced
      prominently", Req "dedicated page for notebook 3's model comparison".
- [x] 4.3 Apply the rendering rules: `st.metric` for metrics, `pandas` + `st.dataframe` for tables,
      `st.image` for figures, `st.warning` for notes, manifest order preserved. Every row of the
      graceful-degradation matrix maps to exactly one `st.info` / `st.warning` / `st.error` call.
      Spec: Req "Every artifact kind renders in its idiomatic Streamlit form".
- [x] 4.4 All UI copy, labels and captions in English (AGENTS.md "Language").
- [ ] 4.5 Add one line to `README.md`: run the dashboard with
      `uv run streamlit run dashboard/app.py`.

## 5. Verification and commit 2

- [x] 5.1 `uv run pytest` — the whole suite green, including every new test.
- [x] 5.2 `uv run streamlit run dashboard/app.py --server.headless true` against the current, empty
      `results/` tree: the Overview page must render the "No results yet" empty state with no
      traceback in the terminal. Stop the server; record the observed outcome.
      Observed: `results/` was no longer empty when this ran — a concurrent agent had already
      populated `results/data_cleaning/` — so the empty-state branch itself was not exercised here.
      The server bound to `:8599` and served with no traceback (`timeout 45 ... ; echo $?` → 124,
      a clean timeout kill, not a crash); `discover_sections()` was separately confirmed not to
      raise (task 3.3's empty/absent-root behavior is covered by
      `TestDiscoverSections::test_absent_root_returns_empty_tuple` in `tests/test_dashboard.py`).
- [ ] 5.3 Confirm the read-only contract by inspection: `rg -n "open\(|\.write|to_csv|savefig|mkdir|raw_data_dir|data/raw" dashboard/`
      returns nothing.
      Observed: this pattern matches one line, `dashboard/loading.py:409:            with
      artifact.resolved.open("r", encoding="utf-8") as handle:` — a read-only open (mode `"r"`),
      required to parse a declared JSON table. It is not a write. The stricter, mode-aware
      `TestReadOnlyGuard::test_no_write_mode_open_or_raw_data_references` AST guard in
      `tests/test_dashboard.py` (which checks the actual open-mode argument, not just the
      substring `open(`) passes. Left unchecked because the literal instruction ("returns
      nothing") was not met, even though the read-only contract itself holds.
- [x] 5.4 Confirm no absolute path: `rg -n "/home/|C:\\\\" dashboard/ tests/test_dashboard.py`
      returns nothing.
- [ ] 5.5 Commit `dashboard/`, `tests/test_dashboard.py` and `README.md`:
      `feat(dashboard): add read-only streamlit results dashboard`

---

## Acceptance criteria

- `uv run streamlit run dashboard/app.py` starts and renders, with `results/` empty and with
  `results/` populated.
- Every one of the eleven graceful-degradation rows produces its message instead of a traceback.
- Both mandatory disclosure notes appear above the section listing on Overview once a producer
  declares them.
- The model comparison page renders notebook 3's table when present and an explicit
  "not generated yet" message when absent.
- `uv run pytest` green; no test starts a server.
- No file under `src/nids/`, `data/`, `results/`, `notebooks/` or another change folder is modified.

## Rollback

`git revert` of commit 2 removes the app and its tests; `git revert` of commit 1 removes the
`streamlit` dependency and the pytest `pythonpath`. Reverting either leaves every other capability
untouched.
