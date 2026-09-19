# Results Dashboard Specification

## Purpose

Define a read-only Streamlit application that discovers, at runtime, every producer folder under
`results/` that follows the `results-output-contract`, and renders its metrics, tables, figures and
notes — degrading gracefully and visibly whenever a producer has not run or its output is damaged.

## Requirements

### Requirement: Runtime discovery of producer folders with no global index

The dashboard MUST enumerate its sections by scanning for `results/*/manifest.json` every time the
discovery function is called, resolving `results/` through `nids.paths.results_root()`. It MUST NOT
read a global index file, MUST NOT hardcode the set of producers, and MUST NOT contain any absolute
filesystem path.

#### Scenario: A producer that lands later appears without a code change

- GIVEN the dashboard's discovery function has been called against a `results/` tree containing one producer folder
- WHEN a second producer folder with a valid `manifest.json` is created and discovery is called again
- THEN the returned sections MUST include both producers
- AND no dashboard source file MAY need to change for the second producer to appear

#### Scenario: An empty or absent results tree is an empty state, not a failure

- GIVEN `results/` does not exist, or exists and contains no `manifest.json`
- WHEN discovery is called
- THEN it MUST return an empty result without raising
- AND the application MUST display an explicit message stating that no notebook has produced results yet

### Requirement: The loading layer never raises on damaged or incomplete output

Every function in the dashboard's loading layer MUST convert each failure mode into a returned
status plus a human-readable message. It MUST NEVER propagate an exception to the Streamlit page, so
a traceback can never be rendered in the browser.

#### Scenario: A folder without a manifest is reported, not skipped silently

- GIVEN a directory under `results/` containing no `manifest.json`
- WHEN discovery runs
- THEN that directory MUST be reported with a status meaning "manifest missing"
- AND the application MUST state that this section has not been generated yet

#### Scenario: A malformed manifest yields a message

- GIVEN a `manifest.json` that is not valid JSON, or whose top level is not a JSON object, or that lacks `notebook_id`
- WHEN that section is loaded
- THEN loading MUST return a status meaning "malformed" together with a message naming the folder and the problem
- AND loading MUST NOT raise

#### Scenario: An unknown schema_version degrades instead of failing

- GIVEN a `manifest.json` whose `schema_version` is not a supported version
- WHEN that section is loaded
- THEN loading MUST still return every entry it can parse
- AND the section MUST carry a status meaning "unsupported schema"
- AND the application MUST render the section behind a visible warning naming the unsupported version

#### Scenario: A declared file that is missing on disk is flagged where it would have been shown

- GIVEN a manifest declaring a table or figure whose `path` does not resolve to an existing file
- WHEN that section is rendered
- THEN that entry MUST be marked as missing rather than omitted
- AND the application MUST show a warning naming the declared path
- AND every other entry in the same section MUST still render

#### Scenario: An unreadable or unparseable table is an in-app error

- GIVEN a declared table file that exists but cannot be parsed as CSV or JSON
- WHEN that table is loaded
- THEN the loader MUST return no frame together with a message describing the failure
- AND the application MUST render that message as an error for that table only

#### Scenario: A path escaping its own producer folder is refused

- GIVEN a manifest entry whose `path` resolves outside that manifest's own folder
- WHEN that section is loaded
- THEN that entry MUST be excluded from the rendered entries
- AND the section MUST carry a message stating that an unsafe path was skipped

### Requirement: Every artifact kind renders in its idiomatic Streamlit form

For each successfully loaded section the application MUST render its metrics, its tables, its figures
and its notes, each with the `title` and `description` declared in the manifest.

#### Scenario: Each artifact kind uses its designated widget

- GIVEN a section whose manifest declares at least one metric, one table, one figure and one note, all resolving to existing files
- WHEN that section's page is rendered
- THEN each metric MUST be rendered with `st.metric`
- AND each table MUST be read with `pandas` and rendered with `st.dataframe`
- AND each figure MUST be rendered with `st.image`
- AND each note MUST be rendered with `st.warning`

#### Scenario: Manifest ordering is preserved

- GIVEN a manifest whose `tables`, `figures`, `metrics` and `notes` arrays are written in registration order
- WHEN the section is rendered
- THEN entries MUST appear in that same manifest order

### Requirement: A metric value may be a number or a string, including the string "inf"

Metric formatting MUST accept `int`, `float` and `str` values. The strings `"inf"`, `"-inf"`,
`"Infinity"` and `"-Infinity"` (case-insensitively) and the float infinities MUST render as an
explicit infinity symbol. A `NaN` value MUST render as an explicit not-available marker. Formatting
MUST NEVER raise for any value type present in a manifest.

#### Scenario: The string "inf" renders as infinity

- GIVEN a metrics entry whose `value` is the string `"inf"`
- WHEN that metric is formatted
- THEN the result MUST be the infinity symbol
- AND no exception MAY be raised

#### Scenario: Numeric and free-text values both render

- GIVEN one metrics entry with an integer value, one with a float value and one with a non-numeric string value such as a comparison key name
- WHEN each is formatted
- THEN each MUST produce a non-empty string
- AND the non-numeric string MUST be rendered unchanged

### Requirement: The honesty disclosures are surfaced prominently

The application MUST render every manifest note. It MUST additionally pin the label-noise error-floor
note and the benchmark non-comparability note to the top of the overview page, above the section
listing, so they are visible without navigating into a section.

#### Scenario: Both mandatory notes appear on the overview page

- GIVEN a discovered section whose manifest notes include an error-floor note and a benchmark non-comparability note
- WHEN the overview page is rendered
- THEN both notes MUST be rendered as warnings above the section listing
- AND each MUST name the producer it came from

#### Scenario: A note is never dropped

- GIVEN a section carrying a note that is neither the error-floor nor the non-comparability note
- WHEN that section's page is rendered
- THEN that note MUST still be rendered as a warning within the section

### Requirement: A dedicated page for notebook 3's model comparison

The application MUST provide a navigation entry that renders notebook 3's model comparison table
alone. It MUST locate that table by matching a declared table path stem against a known model
comparison name, across all discovered sections, rather than by a hardcoded folder path.

#### Scenario: The model comparison table renders on its own page

- GIVEN a discovered section declaring a table whose path stem identifies it as the model comparison
- WHEN the model comparison page is rendered
- THEN that table MUST be rendered with `st.dataframe` together with its manifest title and description
- AND the producer it came from MUST be named

#### Scenario: The page is informative before notebook 3 has run

- GIVEN no discovered section declares a model comparison table
- WHEN the model comparison page is rendered
- THEN the application MUST state that notebook 3 has not produced a model comparison yet
- AND it MUST NOT raise

### Requirement: The application is read-only and free of recomputation

The dashboard MUST read only from `results/`. It MUST NOT read `data/raw/`, MUST NOT import or load a
model, MUST NOT fit or transform any data, and MUST NOT create, modify or delete any file.

#### Scenario: No write and no raw-data access occurs

- GIVEN the dashboard's source files
- WHEN they are inspected
- THEN no call opening a file in a write mode MAY be present
- AND no reference to `raw_data_dir`, `data/raw`, or a model-loading API MAY be present
- AND the only filesystem root read MUST be the one returned by `results_root()`

### Requirement: The logic is importable and covered by a serverless smoke test

Discovery, loading and formatting MUST live in plain functions with type hints and docstrings,
separated from module-level Streamlit code, so they can be imported and called directly. The test
suite MUST exercise them against a synthetic `results/` tree built in a `tmp_path`, without starting
a Streamlit server.

#### Scenario: The smoke test drives the helpers without a server

- GIVEN a synthetic `results/` tree in `tmp_path` containing a valid section, a malformed manifest, an unsupported schema version, a section declaring a missing file, and a directory with no manifest
- WHEN the test imports the dashboard modules and calls the discovery, loading and formatting functions against that tree
- THEN every call MUST return without raising
- AND each of those five conditions MUST be observable in the returned statuses and messages
- AND no Streamlit server process MAY be started

#### Scenario: The application starts from the documented command

- GIVEN the project's virtual environment with the declared dependencies installed
- WHEN `uv run streamlit run dashboard/app.py` is executed from the repository root
- THEN the application MUST start and render its overview page
- AND it MUST do so whether `results/` is empty or populated
