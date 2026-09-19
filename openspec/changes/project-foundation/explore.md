# Exploration: project-foundation

## Current State

The repository has no code yet. Only `AGENTS.md`, `openspec/config.yaml`, `pyproject.toml`, `uv.lock`, `.gitignore`, `.venv/`, and `data/raw/` (two gitignored CSVs) exist. `pyproject.toml` declares dependencies but has **no `[build-system]` table** and is not installable — `import <anything from src>` will fail in a notebook kernel today. There is no `src/`, `tests/`, `notebooks/`, `results/`, `skills/`, `dashboard/`, or `README.md`.

Dataset ground truth (verified by the orchestrator against the raw CSVs, not re-derived here): train 175,341x45 / test 82,332x45; 67,601 exact duplicate rows in train (38.5%); 8,541 test rows (10.37%) duplicate a train feature vector (cross-partition leakage); 1,772 feature combinations carry conflicting `attack_cat` (label-noise floor); zero NaN; `service=="-"` in 53.71% of train rows; `proto` has 133 unique values, 128 below 1% of train rows; `is_ftp_login` is identical to `ct_ftp_cmd` with impossible values 2/4; several volumetric features are heavily right-skewed (skew 3.3-76.3).

## Affected Areas

- `pyproject.toml` — needs `[build-system]` + build backend config; currently not packaged, so nothing under `src/` is importable.
- `src/<package_name>/` — does not exist; will hold loading, cleaning, pipeline factory, subsample helper, validation script.
- `tests/` — does not exist; pytest is declared as a dev dependency but has zero config and zero tests.
- `results/data_cleaning/` — does not exist; needs to follow the same manifest contract as future notebook outputs.
- `notebooks/`, `skills/`, `dashboard/` — created empty in this change; only their presence and the output contract they depend on matter here.
- `README.md` — does not exist; must document setup, data acquisition, and the manual-download instructions the validation script also prints.

## Approaches

### 1. Package layout and importability

| Option | Description | Pros | Cons | Effort |
|---|---|---|---|---|
| A. Flat `src/*.py` | `src/` itself is the import root (`import src.cleaning`) | Fewer directories | `src` is not a valid/idiomatic distribution name; ambiguous imports; breaks if `src/` is ever renamed; not the convention hatchling expects | Low |
| B. `src/<package_name>/` (src layout) | Standard src layout with `__init__.py`, e.g. `src/nids/` | Matches Python/uv/hatchling conventions; isolates importable code from the repo root so accidental `sys.path` pickup of `notebooks/` never shadows it; unsurprising for future contributors | One more directory level | Low |

**Recommendation: B.** Concrete `pyproject.toml` edits required (none exist today):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/<package_name>"]
```

Why the explicit `packages` line: hatchling auto-detects a single top-level package only when its name matches the normalized `[project.name]` (`machine-learning-project` -> `machine_learning_project`). The package directory will not carry that exact name (see the naming decision below), so auto-detection fails unless `packages` is set explicitly. Alternative: uv's native build backend (`requires = ["uv_build"]`, `build-backend = "uv_build"`) works equally well and needs no extra Hatch table if the package name is aligned with the project name. Either is acceptable; hatchling is recommended for wider familiarity and its explicit `packages` list.

Once `[build-system]` exists, `uv sync` installs the project itself (editable) into `.venv` automatically — no separate `pip install -e .` step, and no need to force `[tool.uv] package = true`.

Jupyter kernel: `ipykernel` is already a dependency. Simplest, zero-registration path: always launch the notebook server as `uv run jupyter lab` from the repository root — the running interpreter is `.venv`'s Python, so `import <package_name>` works immediately with no manual `ipykernel install` step. Document this in the README as the required launch command; a kernel registered from a different Python would not see the package.

**Open decision (for propose):** the importable package name. The distribution name `machine-learning-project` normalizes to `machine_learning_project`, which is verbose in `import` statements. A short descriptive name (for example `nids`) is suggested; this is a naming choice, not a structural one.

### 2. Where cleaning ends and the Pipeline begins

The single biggest AGENTS.md compliance risk in this change is treating a **fitted** step as a **constant**.

**Dataset-level, one-time, deterministic (no fitted state):**
- Drop `id`, `stcpb`, `dtcpb`, `is_ftp_login` — a fixed column list dropped from both partitions; dropping columns carries no leakage risk.
- Drop exact duplicate rows — **train only**, per the user decision.
- Cross-partition leakage removal — **test only**, computed against train; must never touch train rows.
- `service` `"-"` -> `"none"` — a fixed value rename with no data-derived parameter; safe on any partition because the mapping never depends on row frequencies.
- Label-noise floor count — a diagnostic count on train; does not mutate data.

**MUST be fitted state inside a scikit-learn `Pipeline`/`ColumnTransformer`, fit on train only:**
- Rare `state`/`proto` grouping into `"other"` — **the set of categories kept as themselves is learned from train row counts (<1% threshold) at `fit()` time and stored as a fitted attribute** (for example `self.frequent_categories_`). This is explicitly not a constant list hardcoded in `src/`; any value rare in train, or unseen in train, must map to `"other"` at `transform()` time.
- One-hot encoding — categories fit on train only (`OneHotEncoder(handle_unknown=...)`), never on a train+test concatenation.
- `StandardScaler` — mean/std computed on train only.
- `log1p` on skewed volumetric features — technically stateless, so no fit-on-test risk on its own. Still recommended inside the Pipeline via `FunctionTransformer` for composability: one `Pipeline.fit(X_train).transform(X_test)` call instead of a separate manual pre-step.

Practical implication: `src/` needs a custom `RareCategoryGrouper` (or equivalent) transformer with real `fit`/`transform` methods, tested for (a) the 1% threshold computed from train counts, (b) correct grouping of both rare-in-train and unseen-in-train values, and (c) that fitting on test data is impossible by construction.

### 3. Stratified subsample with rare classes

Pure proportional stratification from 107,740 -> 10,000 gives Worms (127 rows, 0.118% of train) about 12 sampled rows.

| Option | Description | Pros | Cons | Effort |
|---|---|---|---|---|
| A. Pure proportional | `stratify=attack_cat`, exact proportional allocation | Statistically faithful class balance; a single `train_test_split` call | Worms about 12 rows, Backdoor about 143, Shellcode about 101 — too thin for t-SNE, and for a GP classifier 12 points makes that class's decision boundary essentially unlearnable; spectral clustering cannot form a meaningful sub-cluster from about 12 points | Low |
| B. Proportional with minimum-per-class floor | Allocate a floor (for example 50) to every class first, distribute the remaining budget proportionally, capped at 10,000 total | Guarantees every class, including Worms, has enough points to be distinguishable in t-SNE and to contribute non-trivially to GP/spectral clustering | Breaks true proportionality — floor classes become relatively over-represented (Worms 50/10000 = 0.5% vs a true 0.118%); any "representative sample" statement must be caveated for minority classes | Low-Medium |

t-SNE and spectral clustering are unsupervised — `attack_cat` is not used to compute the embedding, only to colour it afterwards, so the floor's value there is "enough points to see the cluster post hoc". A GP classifier is the case most sensitive to too few minority-class training points.

**Recommendation: B**, with the exact floor value (50 suggested as a starting point) confirmed as a product decision in `sdd-propose`.

### 4. Output contract / manifest schema

| Option | Description | Pros | Cons |
|---|---|---|---|
| A. One `manifest.json` per notebook folder (`results/<notebook_name>/manifest.json`) | Each output directory is fully self-describing | No write coordination between notebooks; each notebook writes only its own file; dashboard and skills discover outputs by globbing `results/*/manifest.json` | Consumers aggregate several small files instead of one read |
| B. Single global index (`results/manifest.json`) | One file references every notebook's outputs | One read for the dashboard | Every notebook run must update a shared file — write coordination risk, staleness, merge conflicts |

**Recommendation: A.** A dashboard or skill can build its own in-memory index at read time by scanning `results/*/manifest.json`; that is a consumption-time concern, not a producer-time one.

Minimum schema fields for `manifest.json`:

```json
{
  "schema_version": "1.0",
  "notebook_id": "data_cleaning",
  "generated_at": "2026-09-18T00:00:00Z",
  "tables": [{"path": "tables/attack_cat_before_after.csv", "title": "...", "type": "csv", "description": "..."}],
  "figures": [{"path": "figures/class_balance.png", "title": "...", "type": "png", "description": "..."}],
  "metrics": [{"name": "duplicate_rows_dropped", "value": 67601, "description": "..."}]
}
```

`schema_version` guards forward compatibility. `metrics` as a flat list of `{name, value, description}` (not free-form nested JSON) keeps machine consumption uniform; richer per-metric detail lives in a referenced `tables/*.csv`.

### 5. Cleaning report format

`results/data_cleaning/` should eat the same dog food as any future notebook:
- `manifest.json` with `notebook_id: "data_cleaning"`, following the schema above.
- `tables/*.csv`: before/after row counts per partition, `attack_cat` distribution before/after dedup, and the learned rare `state`/`proto` categories with their train row counts (this table doubles as a human-readable record of the fitted grouping state).
- `counts.json`: headline scalars — duplicate rows dropped, leakage rows dropped, label-noise floor, NaN count (0), `is_ftp_login`/`ct_ftp_cmd` equality confirmation.
- Figures are optional but natural (for example a before/after class-balance bar chart) and fit the `figures` manifest entry when produced.
- The report is produced by a script or function in `src/`, not a notebook — this change creates no notebook.

### 6. Testing strategy without real data in CI

`data/raw/` is gitignored and may be absent in a fresh clone or CI runner.

| Option | Description | Pros | Cons |
|---|---|---|---|
| A. Synthetic in-test mini-fixtures | Hand-built tiny in-memory DataFrames (5-20 rows, real 45-column shape) with deliberate edge cases: a duplicate row, `service=="-"`, a rare `proto` value, a conflicting-label pair | Runs anywhere, no data dependency, fast, fully deterministic; the only way to control exact rare-category percentages needed to test the fitted threshold precisely | Slightly more setup code per test |
| B. Checked-in tiny real-data sample | A literal slice of the real dataset in `tests/fixtures/` | Realistic | UNSW-NB15 redistribution/licensing needs checking before committing any real rows; cannot precisely control edge-case coverage; tension with the "raw data is never modified" rule |

**Recommendation: A** for all unit tests of `src/`. Reserve real-file tests for the data validation script's happy path (row/column counts against the actual CSVs), marked `@pytest.mark.skipif(not path.exists())` so CI without `data/raw/` still passes.

## Recommendation

`src/<package_name>/` layout with explicit hatchling `packages` config and `uv run jupyter lab` as the documented launch command; a custom fitted `RareCategoryGrouper` transformer inside the Pipeline (never a hardcoded rare-value list); one `manifest.json` per notebook output folder; synthetic mini-fixtures for all `src/` unit tests, with real-file tests skipped when `data/raw/` is absent. Two decisions are left open for `sdd-propose` because they are product choices, not structural ones: the importable package name, and the minimum-per-class subsample floor value.

## Risks

- **Leakage-removal direction**: the step must only ever drop TEST rows matching train, never the reverse. A regression test should assert the train row count is unchanged by this step and only the test row count changes.
- **Stratified-subsample determinism**: stratified sampling with seed 42 is reproducible only if the input DataFrame's row order/index is deterministic before sampling; reset or sort the index after upstream cleaning steps, or the exact sampled rows could silently differ across reruns despite the fixed seed.
- **`attack_cat`/`label` reaching features**: the feature list must be built from an explicit allow-list, not "all columns except known target columns" via set difference — the latter silently includes the target if a drop step is skipped. An explicit test should assert neither column is present in `X.columns` before the Pipeline touches it.
- **pandas 3.x behavioural differences**: pandas 3.0 enables copy-on-write by default and, with `pyarrow` already declared, string columns (`service`, `proto`, `state`) may load as `string[pyarrow]` rather than legacy `object`, which can affect the equality comparisons used for the `"-"` -> `"none"` mapping and rare-category grouping. Needs explicit dtype handling and a test, not an assumption carried over from pandas 1.x/2.x.
- **Unseen-at-transform categories**: any `proto`/`state` value present in test but entirely absent from train must also resolve to `"other"`, or `transform()` crashes on an unseen category.
- **Non-comparability with published benchmarks**: removing 8,541 leaking test rows means this project's test metrics are not directly comparable to published UNSW-NB15 results on the full 82,332-row test set. The README and cleaning report must state this explicitly.
- **Open decisions block apply**: if the package name and subsample floor are not resolved in `sdd-propose`, `sdd-apply` blocks on them.

## Ready for Proposal

Yes. Two product decisions should be resolved before task breakdown: (1) the importable package name (suggested: short, for example `nids`), and (2) the minimum-per-class subsample floor (suggested starting point: 50 rows per class). Everything else is a structural or technical recommendation the proposal can adopt directly.
