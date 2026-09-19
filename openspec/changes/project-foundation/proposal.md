# Proposal: Project Foundation

Build the shared, installable foundation that three future notebooks, future skills and a future dashboard all depend on: an importable `src/` package, the agreed dataset cleaning steps, a leak-safe preprocessing Pipeline, a stratified subsample helper, a data validation script, a machine-readable output contract for `results/`, pytest coverage for everything in `src/`, and a README. **This change creates no notebook and trains no model.**

## Settled decisions

Both previously open product decisions are now resolved by the user. Nothing in this proposal blocks on a product decision.

| # | Decision | Value | Consequence carried into the design |
|---|---|---|---|
| 1 | Importable package name | **`nids`** | Short and domain-descriptive (Network Intrusion Detection System). Because it does not match the normalized `[project.name]` (`machine_learning_project`), `pyproject.toml` MUST carry an explicit `[tool.hatch.build.targets.wheel] packages = ["src/nids"]` line; hatchling auto-detection would otherwise fail silently. |
| 2 | Minimum-per-class subsample floor | **50** (default) | `floor=0` MUST still reproduce pure proportional sampling. The helper returns the realized per-class counts and a per-class boolean indicating whether the floor was applied, so every consumer can print the distortion instead of forgetting it. |

### What the floor of 50 buys and costs

Proportional stratification from the cleaned train partition down to 10,000 rows gives Worms roughly 12 rows. Twelve points cannot form a visible cluster in t-SNE, cannot produce a meaningful sub-cluster in spectral clustering, and make the Worms decision boundary of a Gaussian Process classifier essentially unlearnable. A floor of 50 makes the class visible and non-trivially represented in all three.

The cost is proportionality. Worms occupies 0.5% of the subsample against a true share near 0.118% — roughly a 4x over-representation. Any statement of the form "this subsample is representative of the dataset" becomes false for the floored classes, and class-conditional statistics read off the subsample (cluster sizes, class density, any prior estimated from sample counts) must not be extrapolated to the population. Mandatory disclosure via the returned per-class counts and flags is the mitigation.

## Intent

The repository has no code. `pyproject.toml` declares dependencies but has no `[build-system]` table, so nothing is installable and `import` from a notebook fails today. There is no `src/`, `tests/`, `notebooks/`, `results/`, `skills/`, `dashboard/`, or `README.md`.

Three notebooks are planned. Without a shared foundation, each would re-implement loading, cleaning, encoding and scaling in its own cells. That guarantees three things this project cannot afford:

1. **Divergent cleaning.** Three notebooks producing three different row counts from the same CSVs, with no way to tell which is right.
2. **Leakage by copy-paste.** AGENTS.md makes any `fit` or `fit_transform` on test data a blocking issue. A `StandardScaler` fitted on concatenated train+test inside one notebook cell is the single most likely way this project fails review, and it is exactly the kind of line that gets copied into the other two notebooks.
3. **Unreadable outputs.** A dashboard and skills that must each know the internal file naming of every notebook cannot be built incrementally; adding a notebook would mean editing the dashboard.

The dataset itself makes the foundation urgent, not merely convenient. Measured against the raw CSVs under the corrected cleaning order defined below: 38.6% of train rows are exact duplicates; 32.05% of test rows duplicate another test row; a further slice of test rows duplicates a train row outright; `proto` carries 133 values of which 130 fall below 1% of cleaned train rows; `is_ftp_login` is 100% identical to `ct_ftp_cmd` and contains impossible values 2 and 4; volumetric features are skewed from 3.3 to 76.3. None of these are things to discover independently three times.

Success looks like: a fresh clone runs `uv sync`, launches `uv run jupyter lab`, and the first cell of any notebook is `from nids.data import load_clean_partitions` — with the certainty that no test row ever reached a `.fit()`.

## The one item that needs a design review

> **Read this before the rest of the approach.** It is the single design consequence in this change that a reviewer should challenge on purpose.

The user's leakage rule compares the **full post-drop row** — the 39 feature columns plus `attack_cat` plus `label`. A test row is removed only when an identical train row exists including its label.

That rule retains a specific population: test rows whose **features exactly match a train row but whose label disagrees**.

| Leakage comparison key | Test rows removed | Test rows remaining |
|---|---|---|
| Full row — features + `attack_cat` + `label` (**the user's rule, the default**) | 4,248 | 78,084 |
| Features only | 8,542 | 73,790 |
| **Difference — feature-identical, label-contradictory rows the default RETAINS** | **4,294** | — |

Those 4,294 rows are **unanswerable by construction**. A model cannot distinguish them from their training twins using features alone, and because the twin is in the training set, a well-fitted model will predict the train label and be wrong on the test row. They form a near-certain error floor of roughly **5.5% of the retained 78,084-row test set**.

This is defensible. These rows are *label noise*, not *leakage* — removing them would delete evidence that the dataset's labelling is internally inconsistent, and that inconsistency is a real property worth measuring. But the consequence must be visible, not discovered later from a mysterious accuracy ceiling. Three requirements follow:

1. **The comparison key is an explicit parameter of the leakage function.** Default: the full-row rule. Alternative: features-only. Switching is a one-argument change, never a rewrite.
2. **The cleaning report states both numbers**: rows removed *and* rows retained-but-contradictory. The error floor is a reported metric, not a footnote.
3. **This is the top open item for the user's design review.** If the review concludes the error floor is unacceptable, flipping the parameter is the whole remedy.

## Scope

### In scope

| # | Deliverable |
|---|---|
| 1 | Repository structure: `src/nids/`, `tests/`, `notebooks/` (empty), `results/` (committed), `skills/` (empty), `dashboard/` (empty) |
| 2 | `pyproject.toml` gains `[build-system]` (hatchling) and an explicit wheel `packages` entry, making `src/nids` importable from notebook kernels |
| 3 | `src/nids`: path resolution, data loading, the agreed cleaning decisions in the corrected order, a preprocessing Pipeline factory, and the stratified subsample helper |
| 4 | A data validation script verifying `data/raw/` holds both CSVs with the expected row and column counts, printing manual download instructions when they are missing |
| 5 | An output contract for `results/<notebook_name>/`: folder layout, file naming, and a per-folder machine-readable `manifest.json` |
| 6 | A cleaning report written to `results/data_cleaning/`, computing every before/after count at run time |
| 7 | pytest tests for everything in `src/nids` |
| 8 | `README.md` with setup, data acquisition, and notebook launch instructions |

### Out of scope

- Any notebook. This change creates `notebooks/` empty.
- Any model training, evaluation, or metric.
- Any dashboard implementation. `dashboard/` is created empty.
- Any skill implementation. `skills/` is created empty.
- Any automated dataset download. The host forbids scripted download; the validation script prints instructions and exits.
- Any modification of `data/raw/`. Access is read-only, always.
- Any unauthorized commit. Each of the five commits below requires the user's explicit authorization.

## Capabilities

> Contract with `sdd-spec`. `openspec/specs/` is currently empty, so every capability here is new.

### New capabilities

- `project-packaging`: installable src-layout package, build backend configuration, and the documented notebook launch path that makes `import nids` work without a kernel registration step.
- `data-loading`: read-only access to the UNSW-NB15 partitions in `data/raw/`, deterministic dtypes, explicit column allow-lists, and the validation script with its manual-download guidance.
- `data-cleaning`: the dataset-level, one-time, unfitted operations in their mandated order — fixed column drops and the `service` mapping first, then train deduplication, the test-internal duplicate diagnostic, parameterized cross-partition leakage removal, and the label-noise diagnostics.
- `preprocessing-pipeline`: the fitted scikit-learn `Pipeline`/`ColumnTransformer` factory — rare-category grouping learned from train, one-hot encoding, `log1p` on skewed volumetric features, `StandardScaler`, and the `sttl`/`ct_state_ttl` exclusion toggle.
- `stratified-subsampling`: the seed-42 stratified subsample helper with a configurable minimum-per-class floor (default 50, `0` meaning pure proportional) and realized-count reporting.
- `results-output-contract`: the `results/<notebook_name>/` folder layout, file naming, and `manifest.json` schema that skills and the dashboard consume, plus the cleaning report as its first producer.

### Modified capabilities

None. `openspec/specs/` is empty.

## Approach

### 1. Package layout and importability

`src/nids/` src layout. `pyproject.toml` gains exactly three tables:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/nids"]
```

(plus `[tool.pytest.ini_options]`, see §7.)

The explicit `packages` line is required: hatchling auto-detects a single top-level package only when its directory name matches the normalized `[project.name]` (`machine-learning-project` → `machine_learning_project`). `nids` does not, so auto-detection would fail silently.

Once `[build-system]` exists, `uv sync` installs the project editable into `.venv` automatically — no `pip install -e .`, no `[tool.uv] package = true`.

**Notebook kernel:** the README documents `uv run jupyter lab` from the repository root as the required launch command. The running interpreter is then `.venv`'s Python, so `import nids` works with zero `ipykernel install` ceremony. A kernel registered from a different Python would not see the package; the README says so explicitly.

### 2. Cleaning step order (corrected)

The order below is mandatory. Column drops and the `service` mapping happen **before** any row comparison.

| Step | Operation | Partition | Mutates rows? |
|---|---|---|---|
| 1 | Drop `id`, `stcpb`, `dtcpb`, `is_ftp_login` → 41 columns remain | Both | No |
| 2 | Map `service` `"-"` → `"none"` | Both | No |
| 3 | Drop exact duplicate rows, comparing the full 41-column row | **Train only** | Yes |
| 4 | Count duplicate rows *within* the test partition | Test (diagnostic) | **No — they are kept** |
| 5 | Drop test rows matching a train row on the configured comparison key | **Test only** | Yes |
| 6 | Count feature combinations carrying more than one `attack_cat` | Train (diagnostic) | No |

**Comparison key for steps 3 and 5:** the final feature columns **plus `attack_cat` plus `label`** — i.e. the entire 41-column post-drop row. Step 5's key is parameterizable (see the design-review section above); step 3's is not.

#### Why drop first — and what it does not buy

The honest rationale is a principle, not a measured gain: **drop noise-bearing and derived columns before any row comparison, so the comparison key is exactly the data the project actually models.** Comparing rows on columns that will never enter a model means the comparison answers a question nobody asked.

The obvious hypothesis — that keeping `stcpb`/`dtcpb` during deduplication hides real duplicates — was tested directly against the raw CSVs and is **false for this dataset**:

| Deduplication run | Train duplicate rows found |
|---|---|
| With `stcpb`, `dtcpb`, `is_ftp_login` present | 67,601 |
| With those three columns dropped first | 67,601 |

Identical. The cause: 54.89% of train rows carry `stcpb == 0` and `dtcpb == 0`, and rows that duplicate each other already agree on those fields, so the extra columns never act as a discriminator.

**This null result is recorded deliberately.** The corrected order is still adopted — it is right on principle, and it genuinely does change *leakage* detection in step 5 — but the proposal must not claim it uncovers additional duplicates, and this measurement exists so that nobody rediscovers it in six months and files it as a finding.

#### Measured reference values

These figures are context for design decisions. They are **not** acceptance thresholds and **must not** appear as hardcoded assertions in `src/nids`, in `tests/`, or in the success criteria — the cleaning report computes them at run time (see §3).

| Quantity | Measured value |
|---|---|
| Train rows in → exact duplicates → remaining | 175,341 → 67,601 → **107,740** |
| Duplicate rows *within* test (kept) | **26,387** (32.05% of 82,332) |
| Test rows removed by the default full-row leakage key | **4,248** → 78,084 remain |
| Test rows removed by a features-only key (comparison) | 8,542 → 73,790 remain |
| Feature-identical, label-contradictory test rows retained | **4,294** |
| Train feature combinations with more than one `attack_cat` | **1,772** |
| Cleaned train `proto` values below the 1% threshold | **130 of 133** |

Post-clean train `attack_cat`: Normal 51,890 / Exploits 19,844 / Fuzzers 16,150 / Reconnaissance 7,522 / Generic 4,181 / DoS 3,806 / Analysis 1,594 / Backdoor 1,535 / Shellcode 1,091 / Worms 127.

Post-clean test `attack_cat`: Normal 35,806 / Generic 17,176 / Exploits 10,954 / Fuzzers 6,018 / DoS 3,887 / Reconnaissance 2,592 / Analysis 663 / Backdoor 579 / Shellcode 365 / Worms 44.

Post-clean train `state`: FIN 74,478 / INT 19,726 / CON 12,487 / REQ 953 / RST 83 / ECO 10 / PAR 1 / URN 1 / no 1. Deduplication reshapes this distribution substantially — `INT` falls from 82,275 raw rows to 19,726 — which is itself a finding the cleaning report should surface.

> **Historical reference only.** An earlier revision of this proposal quoted 67,601 / 8,541 / 107,740 / 73,791 as cleaned counts. Those were computed under the pre-correction rule (leakage on features only, drops after comparison) and are superseded by the table above. They are retained here solely so a reader of the old document can locate the change.

### 3. No hardcoded cleaned counts

Cleaned-partition row counts are **outputs of the cleaning code, never inputs to it**. Neither `src/nids` nor `tests/` may contain a literal cleaned row count.

The reason is practical: every count above shifted when the step order changed. A test asserting `len(train_clean) == 107740` does not verify correctness — it verifies that nobody has improved the cleaning logic, and it fails loudly for the wrong reason the next time a rule is refined.

What replaces them: **invariants**. The cleaning report computes and writes the actual numbers; tests assert relationships that hold regardless of the specific values. Examples, stated fully in the success criteria:

- The train row count strictly decreases across deduplication; the test row count is unchanged by it.
- The test-internal duplicate diagnostic changes no row count at all.
- `removed(full_row_key) + retained_contradictory(full_row_key) == removed(features_only_key)` — an arithmetic identity that holds on real data and on synthetic fixtures alike.
- A second run over the same inputs reproduces the report's counts exactly.

### 4. Module breakdown for `src/nids/`

| Module | Responsibility | Key public surface |
|---|---|---|
| `paths.py` | Resolve the repository root from the package location; expose `data/raw/`, `results/` paths. No hardcoded absolute paths anywhere. | `repo_root()`, `raw_data_dir()`, `results_dir(notebook_id)` |
| `columns.py` | Explicit allow-lists and constants: the 45 raw columns, the four dropped columns, target columns, categorical columns, skewed volumetric columns, the TTL shortcut pair, and the two leakage comparison keys. | `DROPPED_COLUMNS`, `TARGET_COLUMNS`, `CATEGORICAL_COLUMNS`, `SKEWED_COLUMNS`, `TTL_SHORTCUT_COLUMNS`, `feature_columns(include_ttl)`, `leakage_key_columns(key)` |
| `loading.py` | Read-only CSV loading with deterministic dtypes for the string columns. | `load_raw_train()`, `load_raw_test()` |
| `validation.py` | Verify `data/raw/` contents and shapes; on failure, print manual UNSW download instructions and exit non-zero. Runnable as a module. | `validate_raw_data()`, `main()` |
| `cleaning.py` | The unfitted dataset-level operations **in the corrected order**, plus every diagnostic they emit. | `drop_unused_columns()`, `normalize_service()`, `drop_train_duplicates()`, `count_test_internal_duplicates()`, `drop_test_leakage(train, test, *, comparison_key="full_row")`, `count_label_conflicts()`, `clean_partitions()` |
| `transformers.py` | The fitted `RareCategoryGrouper` estimator. | `RareCategoryGrouper` |
| `preprocessing.py` | The `ColumnTransformer`/`Pipeline` factory. | `build_preprocessor(include_ttl_features=True)` |
| `sampling.py` | Seed-42 stratified subsample with a minimum-per-class floor and realized-count reporting. | `stratified_subsample(df, max_rows=10_000, floor=50, seed=42)` |
| `results.py` | The output contract writer: folder creation, table/figure/metric registration, `manifest.json` emission. | `ResultsWriter` |
| `cleaning_report.py` | Runs the cleaning steps and writes `results/data_cleaning/` through `ResultsWriter`. Runnable as a module. | `build_cleaning_report()`, `main()` |

`cleaning.py` specifics required by the corrections:

- `clean_partitions()` executes steps 1–6 in exactly the order of the table in §2. The order is encoded in one function so no caller can reassemble it wrongly.
- `drop_test_leakage()` takes `comparison_key` as a keyword argument with the literal values `"full_row"` (default) and `"features_only"`. It returns the cleaned test frame **and** a diagnostics record carrying at minimum: `removed`, `retained_contradictory`, and the key that was used.
- `retained_contradictory` is a first-class diagnostic, not a derived afterthought: it counts test rows that match a train row on features but not on `attack_cat`/`label`. Under `"features_only"` it is zero by construction, which is itself a useful test.
- `count_test_internal_duplicates()` returns a count and mutates nothing. Test rows are never dropped for being duplicates of each other.

### 5. The cleaning / Pipeline boundary

This boundary is the whole point of the change. AGENTS.md makes *any* `fit` or `fit_transform` on test data a blocking review issue. That rule is only enforceable if every data-derived parameter lives on one side of a line that is obvious in the code. Left column: nothing is learned, so applying it to test is safe by construction. Right column: something is learned from train row statistics, so test may only ever be `.transform()`-ed.

| # | User decision | Side | Why it lands there |
|---|---|---|---|
| 1 | Drop `id`, `stcpb`, `dtcpb`, `is_ftp_login` | **Unfitted, dataset-level, BOTH — runs first** | A fixed column list. Dropping columns carries no leakage risk. Running it first makes every later row comparison use exactly the modelled columns. |
| 2 | `service` `"-"` → `"none"` | **Unfitted, dataset-level, BOTH — runs first** | A constant rename. The mapping never depends on row frequencies, so it is identical on any partition. This is the "never impute with the mode" rule made structural: there is no mode to compute. |
| 3 | Drop exact duplicate rows | **Unfitted, dataset-level, TRAIN ONLY** | Changes the row set, learns nothing. Applied to test it would silently alter the evaluation set. |
| 4 | Count duplicate rows inside test | **Diagnostic** | Reads only; mutates nothing. Removing them would shrink the evaluation set for a reason the evaluation never asked about. |
| 5 | Drop test rows matching train on the comparison key | **Unfitted, dataset-level, TEST ONLY** | Direction-critical. Reads train, mutates test. Must never run in reverse. Key is parameterized; default is the full row. |
| 6 | Keep conflicting labels, report the count | **Diagnostic** | Reads only; mutates nothing. Emitted to the cleaning report as the label-noise floor. |
| 7 | Group `state`/`proto` below 1% into `"other"` | **FITTED — `RareCategoryGrouper` inside the Pipeline** | The kept-category set is computed from *train row counts* at `fit()` and stored as `frequent_categories_`. Never a hardcoded list in `src/`. At `transform()`, both rare-in-train and unseen-in-train values map to `"other"`. |
| 8 | `log1p` on skewed volumetric features, then `StandardScaler` | **Pipeline** — `log1p` stateless, scaler **FITTED** | Scaler mean/std come from train only. `log1p` goes in the Pipeline via `FunctionTransformer` for composability: one `fit(X_train)` then `transform(X_test)`, not a manual pre-step a notebook could forget. |
| 9 | Optional exclusion of `sttl` / `ct_state_ttl` | **Pipeline factory parameter** | Changes which columns enter the transformer, not any fitted value. Two calls to `build_preprocessor()` produce the with/without pair AGENTS.md requires. |
| 10 | Stratified subsample, max 10,000, seed 42, floor 50 | **Standalone helper, outside the Pipeline** | Row selection for GP / spectral clustering / t-SNE, applied after cleaning and before modeling. Not a transformation of features. |

One-hot encoding is also **fitted** (`OneHotEncoder` categories from train only, `handle_unknown` set so unseen values never crash `transform`).

The enforceable consequence: `src/nids/cleaning.py` exposes no `fit`, and `src/nids/preprocessing.py` exposes no function that takes a test partition. A notebook physically cannot fit on test through this API without writing scikit-learn calls by hand — which is exactly what review looks for.

**Feature selection is an allow-list, never a set difference.** `feature_columns()` returns an explicit list. Building features as "all columns minus known targets" silently reintroduces `attack_cat` or `label` the moment a drop step is skipped. A test asserts neither target column is present in `X.columns`.

Note the deliberate asymmetry: `attack_cat` and `label` are part of the **row-comparison key** for cleaning, and are excluded from the **feature matrix** for modelling. These are different jobs on the same columns, and the code keeps them in different modules so they cannot be confused.

### 6. Output contract for `results/<notebook_name>/`

**`results/` is committed to the repository.** The dashboard must be able to read notebook outputs without rerunning any notebook, so the outputs live in git.

Verified state of `.gitignore` today: it holds `.venv/`, `data/raw/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.pyc`, `.atl/`. `results/` is therefore already not ignored, and **this change requires no `.gitignore` edit** — it only states the intent explicitly and adds `results/.gitkeep` so the directory exists before its first producer runs. `data/raw/` stays ignored.

The consequence to design around: committed outputs mean CSVs and PNGs enter git history permanently. The cleaning report must therefore stay **small and deterministic** — a handful of small tables, one figure, and scalar metrics. Two runs over the same inputs must produce byte-identical tables (no timestamps inside table files, stable row ordering), or every rerun becomes a spurious diff.

Layout, identical for every producer including the cleaning report:

```
results/<notebook_name>/
├── manifest.json
├── tables/*.csv        (or .json)
└── figures/*.png
```

One `manifest.json` per folder, never a global index. Each output folder is self-describing, no producer coordinates writes with another, and a dashboard or skill builds its own index at read time by globbing `results/*/manifest.json`. A shared global file would make every notebook run a write-conflict risk.

File naming: `snake_case`, descriptive, no timestamps in filenames (`generated_at` in the manifest carries that). Paths inside the manifest are relative to the manifest's own folder, so the whole `results/` tree stays relocatable.

Concrete example, `results/data_cleaning/manifest.json`. **Every numeric value below is computed at run time; the numbers shown are the measured reference run, not constants in the code:**

```json
{
  "schema_version": "1.0",
  "notebook_id": "data_cleaning",
  "generated_at": "2026-09-19T12:00:00Z",
  "tables": [
    {
      "path": "tables/row_counts_before_after.csv",
      "title": "Row counts per partition before and after cleaning",
      "type": "csv",
      "description": "Train and test row counts at each cleaning step, in execution order."
    },
    {
      "path": "tables/attack_cat_distribution.csv",
      "title": "attack_cat distribution before and after cleaning",
      "type": "csv",
      "description": "Per-class train and test counts and shares, before and after cleaning."
    },
    {
      "path": "tables/state_distribution.csv",
      "title": "state distribution before and after deduplication",
      "type": "csv",
      "description": "Train state counts before and after dropping exact duplicate rows."
    },
    {
      "path": "tables/rare_categories.csv",
      "title": "Rare state and proto categories grouped into 'other'",
      "type": "csv",
      "description": "Each category below the 1 percent cleaned-train threshold with its row count."
    }
  ],
  "figures": [
    {
      "path": "figures/class_balance_before_after.png",
      "title": "Class balance before and after cleaning",
      "type": "png",
      "description": "Grouped bar chart of attack_cat counts on a log scale."
    }
  ],
  "metrics": [
    {
      "name": "train_duplicate_rows_dropped",
      "value": 67601,
      "description": "Exact duplicate rows removed from train, compared on the full 41-column post-drop row."
    },
    {
      "name": "test_internal_duplicate_rows",
      "value": 26387,
      "description": "Duplicate rows within the test partition. Reported only; these rows are kept."
    },
    {
      "name": "test_leakage_rows_dropped",
      "value": 4248,
      "description": "Test rows identical to a train row on the configured comparison key."
    },
    {
      "name": "test_leakage_comparison_key",
      "value": "full_row",
      "description": "The comparison key used: features plus attack_cat plus label."
    },
    {
      "name": "test_contradictory_rows_retained",
      "value": 4294,
      "description": "Test rows matching a train row on features but carrying a different label. Retained by the full_row key; a near-certain error floor."
    },
    {
      "name": "label_noise_floor_combinations",
      "value": 1772,
      "description": "Train feature combinations carrying more than one attack_cat."
    }
  ]
}
```

`schema_version` guards forward compatibility. `metrics` is a flat list of `{name, value, description}` rather than free-form nested JSON, so consumers read every metric uniformly; anything richer lives in a referenced table. `value` may be a number or a string (the comparison key is a string), which consumers must tolerate.

#### Cleaning report contents

`results/data_cleaning/`, produced by `python -m nids.cleaning_report` — a script, not a notebook.

- `manifest.json` with `notebook_id: "data_cleaning"`, per the schema above.
- `tables/row_counts_before_after.csv` — the row count of each partition after each of the six ordered steps. Computed, never asserted against a literal.
- `tables/attack_cat_distribution.csv` — the ten classes before and after cleaning, counts and shares, for both partitions. This table is where the Generic collapse and the DoS collapse become visible; they are among the most consequential facts in the entire cleaning step and must not be buried.
- `tables/state_distribution.csv` — the `state` distribution before and after deduplication. `INT` dropping from 82,275 to 19,726 in the reference run means deduplication is not a neutral cleanup; it reshapes the protocol-state mix the models will see.
- `tables/rare_categories.csv` — every `state` and `proto` value below the 1% cleaned-train threshold with its count. Doubles as the human-readable record of the fitted grouping state.
- `counts.json` — headline scalars mirroring the manifest `metrics`: train duplicates dropped, test-internal duplicates (kept), leakage rows dropped, the comparison key used, contradictory rows retained, label-noise floor, NaN count, `is_ftp_login` / `ct_ftp_cmd` equality confirmation, `service == "-"` count.
- `figures/class_balance_before_after.png` — grouped bar chart, log scale (Worms at 127 against Normal at 51,890 is unreadable linearly).

The report MUST also carry two written notes:

1. **Benchmark non-comparability.** Removing leaking test rows means this project's test metrics are not directly comparable to published UNSW-NB15 results on the full 82,332-row test set.
2. **The retained error floor.** The count of feature-identical, label-contradictory test rows retained, expressed as a percentage of the retained test set, described as a near-certain minimum error rate that no model can eliminate.

### 7. Testing approach

TDD is disabled for this project (`strict_tdd: false`, explicit user correction). Ordinary functional checks apply: every public function in `src/nids` gets pytest coverage, written alongside the implementation rather than before it.

**Synthetic in-test mini-fixtures for all `src/nids` unit tests.** Hand-built 5–20 row DataFrames with the real 45-column shape and deliberate edge cases: a duplicate row, `service == "-"`, a category at exactly 1% and one just below, a value present only in test, a feature-identical pair with disagreeing labels, and a duplicate pair inside the test frame. This is the only way to control exact rare-category percentages precisely enough to test the fitted threshold, it runs in a clone with no `data/raw/`, and it sidesteps the UNSW-NB15 redistribution question that checking in real rows would raise.

Real-file tests are reserved for the validation script's happy path (raw row and column counts against the actual CSVs — these are properties of the *unmodified input*, not of the cleaning, so they are legitimate literals), marked `@pytest.mark.skipif(not raw_data_present())` so a fresh clone or CI without `data/raw/` still passes green.

Behavioral assertions worth naming now, all expressed as invariants:

- `clean_partitions` drops columns and maps `service` **before** any deduplication or leakage step, verified by a fixture whose duplicate status would differ under the wrong order.
- `drop_test_leakage` leaves the train row count **unchanged** and reduces only the test count.
- `removed + retained_contradictory` under `"full_row"` equals `removed` under `"features_only"` on the same fixture.
- `retained_contradictory` is zero under `"features_only"`.
- `count_test_internal_duplicates` returns a positive count on a fixture with duplicated test rows and leaves the test frame's length unchanged.
- No literal cleaned-partition row count appears anywhere in `src/nids` or `tests/`.
- `RareCategoryGrouper` maps both rare-in-train **and** unseen-in-train values to `"other"`.
- `RareCategoryGrouper.fit` learns from train counts; a grouper fitted on one frame and applied to another produces the first frame's category set.
- `feature_columns()` never contains `attack_cat` or `label`, with or without the TTL toggle.
- `stratified_subsample` with seed 42 returns identical row indices across repeated calls and retains every class including Worms; with `floor=0` it reproduces pure proportional allocation.
- `ResultsWriter` emits a `manifest.json` that round-trips through `json.load` and whose declared paths all exist on disk.

`pytest` configuration (`[tool.pytest.ini_options]` with `testpaths = ["tests"]`) is added to `pyproject.toml`; it is declared as a dev dependency today but has no configuration.

## Affected areas

| Area | Impact | Description |
|---|---|---|
| `pyproject.toml` | Modified | Add `[build-system]`, `[tool.hatch.build.targets.wheel]`, `[tool.pytest.ini_options]`. Dependencies unchanged. |
| `src/nids/` | New | Ten modules plus `__init__.py`. |
| `tests/` | New | One test module per `src/nids` module, plus shared synthetic fixtures in `conftest.py`. |
| `notebooks/` | New (empty) | `.gitkeep` only. |
| `results/` | New, **committed** | `.gitkeep`, then `data_cleaning/`. Already not covered by `.gitignore`. |
| `results/data_cleaning/` | New, **committed** | Generated by `python -m nids.cleaning_report`; small and deterministic by design. |
| `skills/` | New (empty) | `.gitkeep` only. |
| `dashboard/` | New (empty) | `.gitkeep` only. |
| `README.md` | New | Setup, manual data acquisition, `uv run jupyter lab`, validation and cleaning-report commands, the two mandatory caveats. |
| `.gitignore` | **Unchanged** | Already correct: `data/raw/` ignored, `results/` not ignored. |
| `data/raw/` | **Untouched** | Read-only access only. Never written, never modified. |

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| The retained feature-identical, label-contradictory test rows are mistaken for a modelling failure | **High** | Medium | Reported as an explicit metric with a written note in the cleaning report and the README. The comparison key is a one-argument switch if the design review rejects the tradeoff. Flagged as the top design-review item above. |
| Cleaning steps get reordered by a later edit, silently changing every count | Medium | High | The order lives in one function (`clean_partitions`) and is asserted by a fixture whose duplicate status differs under the wrong order. |
| A cleaned row count gets hardcoded into a test, freezing the cleaning logic | Medium | Medium | Explicit success criterion forbidding literal cleaned counts; invariant-based assertions replace them; the arithmetic identity between the two comparison keys gives a strong check with no magic numbers. |
| Leakage removal runs in the wrong direction, silently shrinking train | Low | High | `drop_test_leakage(train, test)` returns only a test frame; a regression test asserts the train row count is unchanged. |
| A fitted parameter gets hardcoded as a constant (rare-category list in `src/`) | Medium | High — blocking review issue | `RareCategoryGrouper` holds `frequent_categories_` as fitted state; a test asserts the set differs when fitted on different frames. No rare-value literal exists anywhere in `src/`. |
| `attack_cat` or `label` reaches the feature matrix | Low | High — blocking review issue | Explicit allow-list, never a set difference. A test asserts both are absent from `X.columns` under both TTL settings. The columns' legitimate use as a comparison key lives in a different module from `feature_columns()`. |
| Committed `results/` output churns git history on every rerun | Medium | Medium | The report is small and deterministic: stable row ordering, no timestamps inside table files, timestamps confined to `manifest.json`. |
| pandas 3.x loads string columns as `string[pyarrow]` rather than `object`, changing `"-"` equality and category grouping behavior | Medium | Medium | Explicit dtype handling in `loading.py` and a dtype-specific test, rather than an assumption carried over from pandas 1.x/2.x. `pyarrow` is already a declared dependency, so this is live, not hypothetical. |
| Unseen-at-transform `proto`/`state` values crash `transform()` | Medium | Medium | `RareCategoryGrouper` maps unseen values to `"other"`; `OneHotEncoder(handle_unknown=...)` set accordingly. Tested with a test-only category. |
| Subsample non-determinism despite seed 42, because upstream cleaning left a non-deterministic index | Medium | Medium | Cleaning steps reset the index deterministically before sampling; a test calls `stratified_subsample` twice and asserts identical indices. |
| The subsample floor's distortion is forgotten at the point of use | Medium | Medium | The helper returns realized per-class counts and a per-class floor-applied flag. Documented in the README and restated in the spec. |
| Test metrics are quietly compared to published UNSW-NB15 benchmarks | Medium | Medium | The cleaned test set is not the published 82,332-row set. Stated in the README and in the cleaning report. |
| `data/raw/` absent in a fresh clone breaks the test suite | Medium | Low | Real-file tests are `skipif`-guarded; the validation script prints manual download instructions and exits non-zero. |
| Notebook kernel points at a different Python, so `import nids` fails | Medium | Low | README mandates `uv run jupyter lab` from the repo root and explains why a separately registered kernel would not see the package. |

## Rollback plan

This change is **purely additive**. It creates new directories and new files, edits `pyproject.toml`, and touches no existing code because none exists. Nothing is migrated, no data is mutated, `data/raw/` is read-only throughout.

Because delivery is five sequential commits on `main` (see below), rollback is commit-scoped:

| Situation | Action |
|---|---|
| A single slice is wrong and later slices do not exist yet | `git revert <sha>` for that commit, or amend it before the next slice starts. |
| Several slices must go | `git revert` them in **reverse order** — each slice only imports from earlier ones, so the chain has no back-edges. |
| Everything must go and nothing is pushed | Reset `main` to `d761639`. |
| Everything must go and commits are pushed | Revert all five in reverse order rather than rewriting published history. |

Whole-change rollback, whichever route: `pyproject.toml` returns to its state at `d761639` (the three added tables removed; declared dependencies are unchanged by this proposal, so `uv.lock` stays valid), `src/`, `tests/`, `notebooks/`, `skills/`, `dashboard/`, `results/` and `README.md` are removed since none exists in history, and `uv sync` drops the editable install of `nids` from `.venv`. `data/raw/` is untouched by every step.

## Dependencies

- All runtime libraries are already declared in `pyproject.toml` (pandas 3.0.6+, scikit-learn 1.9.1+, numpy, matplotlib, seaborn, pyarrow, jupyterlab, ipykernel). **No new dependency is introduced.**
- `hatchling` is a build-time requirement resolved by `uv sync`, not a project dependency.
- The two UNSW-NB15 CSVs must be present in `data/raw/` to run the validation script and the cleaning report. They are gitignored and manually downloaded; unit tests do not require them.
- No open product decision blocks `sdd-apply`. The leakage comparison key is a design-review item, not a blocker — the default is settled and the switch is a parameter.

## Success criteria

Counts are invariants, never literals. No criterion below fixes a cleaned-partition row count.

### Packaging and environment

- [ ] `uv sync` succeeds and `uv run python -c "import nids"` works from a fresh clone.
- [ ] `uv run jupyter lab` launches a kernel where `import nids` resolves with no `ipykernel install` step.
- [ ] `uv run pytest` passes with `data/raw/` present **and** with `data/raw/` absent.

### Data access

- [ ] `python -m nids.validation` reports both raw CSVs at 175,341x45 and 82,332x45 (properties of the unmodified input), and prints manual download instructions with a non-zero exit when a file is missing.
- [ ] `data/raw/` is byte-identical before and after the change.

### Cleaning correctness

- [ ] `clean_partitions` drops the four fixed columns and maps `service` `"-"` → `"none"` **before** any deduplication or leakage comparison.
- [ ] After the drop step, exactly 41 columns remain and none of `id`, `stcpb`, `dtcpb`, `is_ftp_login` is among them.
- [ ] Deduplication and leakage comparison keys both include `attack_cat` and `label`.
- [ ] No `service` value equals `"-"` after cleaning, and the resulting `"none"` count equals the pre-mapping `"-"` count.
- [ ] The train row count strictly decreases across deduplication, and the test row count is unchanged by that step.
- [ ] `drop_test_leakage` leaves the train row count unchanged and strictly decreases the test row count under the default key.
- [ ] `count_test_internal_duplicates` returns a count and leaves the test row count unchanged.
- [ ] `removed + retained_contradictory` under `"full_row"` equals `removed` under `"features_only"` on the same inputs.
- [ ] `retained_contradictory` is zero under `"features_only"`.
- [ ] Two consecutive runs of `python -m nids.cleaning_report` on the same inputs produce identical computed counts and byte-identical table files.
- [ ] No literal cleaned-partition row count appears in `src/nids` or `tests/`.

### Leakage safety

- [ ] No `fit` or `fit_transform` anywhere in `src/nids` receives a test partition; `cleaning.py` exposes no `fit` at all.
- [ ] No rare-category literal appears anywhere in `src/nids`; the grouper's category set is fitted state.
- [ ] `feature_columns()` excludes `attack_cat` and `label` under both TTL settings.
- [ ] `build_preprocessor(include_ttl_features=False)` produces a matrix without `sttl` and `ct_state_ttl`.

### Sampling

- [ ] `stratified_subsample` returns identical indices across repeated calls and retains all ten classes at the default floor of 50.
- [ ] `stratified_subsample(..., floor=0)` reproduces pure proportional allocation.
- [ ] The helper returns realized per-class counts and a per-class floor-applied flag.

### Outputs and documentation

- [ ] `python -m nids.cleaning_report` produces `results/data_cleaning/` with `manifest.json`, `counts.json`, four tables and one figure; the manifest round-trips through `json.load` and every declared path exists.
- [ ] The report states the leakage comparison key used, the rows removed, **and** the contradictory rows retained with their share of the retained test set.
- [ ] The report carries the benchmark non-comparability note and the retained-error-floor note.
- [ ] `results/` is tracked by git; `.gitignore` is unchanged and still ignores `data/raw/`.
- [ ] `README.md` documents setup, manual data acquisition, the launch command, the benchmark non-comparability note, the retained error floor, and the subsample floor distortion.

### Delivery

- [ ] Each of the five commits is authorized by the user before it is made.
- [ ] Each commit uses a Conventional Commit message.

## Delivery: five sequential commits on `main`

**Estimated authored changed lines: ~1,650** (additions plus deletions, excluding `uv.lock` and generated `results/` output).

| Area | Estimate |
|---|---|
| `src/nids/` (11 files) | ~870 |
| `tests/` (10 files + `conftest.py`) | ~640 |
| `README.md` | ~120 |
| `pyproject.toml` edits, `.gitkeep` files | ~20 |

This exceeds the 400-line review budget roughly fourfold, so the work is sliced. **Delivery is five sequential commits on `main`, not pull requests.** The user reviews after each slice before the next begins, and an external reviewer (GGA) reviews each commit. Every commit requires the user's explicit authorization; nothing is committed without it.

| Slice | Contents | Est. lines | Conventional Commit subject |
|---|---|---|---|
| 1 | Packaging: `pyproject.toml` tables, `src/nids/__init__.py`, `paths.py`, `columns.py`, empty `notebooks/` `skills/` `dashboard/`, `results/.gitkeep`, README skeleton + tests | ~300 | `feat(packaging): add build system and package skeleton` |
| 2 | `loading.py`, `validation.py`, manual-download instructions, real-file skipif tests | ~250 | `feat(data): add raw partition loading and validation` |
| 3 | `cleaning.py` — the six ordered unfitted operations, the parameterized leakage key, all diagnostics + tests | ~360 | `feat(cleaning): add ordered dataset cleaning and diagnostics` |
| 4 | `transformers.py`, `preprocessing.py`, `sampling.py` — all fitted state and the subsample helper + tests | ~380 | `feat(preprocessing): add fitted pipeline factory and subsampling` |
| 5 | `results.py` output contract, `cleaning_report.py`, README completion + tests | ~340 | `feat(results): add output contract and cleaning report` |

Review guidance, in priority order:

- **Slice 4** is the compliance-critical one: every fitted parameter in the project lives there.
- **Slice 3** carries the corrected step order and the leakage key decision — review the order and the key before the numbers.
- Each slice imports only from earlier slices, so the chain is linear with no back-edges and reverse-order revert is always safe.

Session delivery strategy is `ask-on-risk`; this forecast is what triggered the question, and the answer above is the user's.
