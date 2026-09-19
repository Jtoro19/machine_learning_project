# Design: skills

## Layout

```
src/nids/
  tabular.py          # schema detection, cleaning, generic pipeline fallback, feature-set
                       # computation, split-then-dedupe (round 3: moved from skills/_shared/)
skills/
  _shared/
    __init__.py
    common.py          # argparse plumbing, CSV loading, ResultsWriter wiring; re-exports
                        # nids.tabular's functions for each script
  eda-reduction-clustering/
    SKILL.md
    eda_reduction_clustering.py
  clustering-reduction/
    SKILL.md
    clustering_reduction.py
  classification/
    SKILL.md
    classification.py
tests/
  test_skills.py
  test_tabular.py       # direct tests of src/nids/tabular.py (round 3)
```

`skills/_shared/` is a plain importable package (not itself a "skill" — no `SKILL.md`). Each
script does `sys.path`-safe imports via `from skills._shared import common` when run as
`python skills/<name>/<script>.py` from the repo root (repo root is prepended to `sys.path` by
each script using `nids.paths.repo_root()`-style upward marker search, mirrored locally so the
scripts don't need `src/nids` to already be importable to find themselves).

**Round-3 review standards finding 3**: `is_unsw_raw_schema`, `clean_dataset_if_unsw`,
`compute_features`, `numeric_and_categorical_columns`, and `build_preprocessor_for` moved
from `skills/_shared/common.py` into `src/nids/tabular.py` (AGENTS.md: "Shared
preprocessing lives in `src/`" — this logic is dataset column/dtype-shaping, not
CLI/argparse/IO). `common.py` re-exports them under the same names, so every script call
site is unchanged; it keeps only argument parsing, CSV loading, and `ResultsWriter`
wiring as its own code.

## Decision 1 — argparse contract (shared across all three scripts)

```
--dataset PATH     required. CSV to read with pandas.read_csv.
--target NAME       required. Column name used as the (single) target.
--exclude NAME       repeatable AND/OR comma-separated (argparse action="append",
                     each value additionally split on ","). Defaults to [].
--output PATH       required. Directory the manifest/tables/figures are written under.
--seed INT          optional, default 42. Exposed for testing determinism, never used
                     to justify a non-42 default in real runs.
```

`--exclude` accepts both repeated flags and comma lists so a caller does not need to know which
style a given skill prefers.

`eda-reduction-clustering` and `clustering-reduction` additionally accept `--n-clusters
INT` (optional, default: chosen automatically — see Decision 5's "Choosing n_clusters,
never from the target" note, round-3 worth-fixing finding 7). Not added to this shared
contract since `classification` has no use for it; each of the two clustering scripts
adds it to the parser `build_arg_parser` returns.

## Decision 2 — schema detection, never assumption

`common.is_unsw_raw_schema(df) -> bool` returns `set(nids.columns.RAW_COLUMNS) <= set(df.columns)`.
This is the *only* place a UNSW-specific literal enters the routing decision, and it is a set
comparison against the frozen `nids.columns.RAW_COLUMNS`, not a re-typed list.

Routing rule, applied once per script after loading `--dataset` and computing the feature set
`features = all_columns - {target} - exclude`:

- IF `is_unsw_raw_schema(df)` AND `set(exclude) <= {"sttl", "ct_state_ttl"}` (i.e. the caller
  asked for nothing custom beyond the documented TTL toggle):
  use `nids.preprocessing.build_preprocessor(include_ttl_features=not ttl_excluded)` and
  `nids.columns.feature_columns(...)` for the feature list, ignoring `--target`/`--exclude`'s
  exact column set beyond validating it is a subset of the TTL pair — because `build_preprocessor`
  takes no data/column argument by design (see `src/nids/preprocessing.py` module docstring).
- ELSE: generic path (Decision 3). This covers every synthetic dataset in the test suite, and
  any real UNSW invocation with a non-trivial `--exclude`.

This is a deliberate narrowing: the frozen `nids` pipeline was designed with a fixed public
surface (`bool` in, no columns in), so "detect, never assume" here means detecting the one case
it actually supports, not attempting to bend it to arbitrary column subsets.

**Cleaning before fitting (review finding 2, round 2)**: schema detection runs once, on the
RAW loaded frame, via `common.clean_dataset_if_unsw(df)` (now `nids.tabular.
clean_dataset_if_unsw`, re-exported by `common`). When it detects the UNSW-NB15 raw
schema, it routes the frame through the relevant subset of `nids.cleaning`'s own step functions
before any fitting — reusing the project's cleaning implementation, never re-implementing it —
and returns `(cleaned_df, is_unsw_schema=True)`. Because cleaning removes `DROPPED_COLUMNS`, a
fresh `is_unsw_raw_schema` check on the cleaned frame would always return `False`; every
downstream function that needs the routing decision (`build_preprocessor_for`,
`assert_no_leakage`) therefore takes an explicit `is_unsw_schema` parameter instead of
re-detecting.

**Split before dedupe (review blocking finding 2, round 3)**: `clean_dataset_if_unsw` runs
ONLY the two COLUMN-level steps (`drop_unused_columns`, `normalize_service`) — it no longer
runs `drop_train_duplicates`. That step's own docstring restricts it to "the training
partition only", so it MUST run after `train_test_split`, on the resulting TRAIN split alone,
never on the full frame before splitting (deduplicating first would also remove duplicate rows
that happened to land in the held-out split, silently reshaping it — the round-3 blocking
bug). Every script now calls `nids.tabular.split_and_dedupe(df, target, test_size=..., seed=...,
is_unsw_schema=...)` — a single call site that splits the FULL frame first, then applies
`dedupe_train_split` (which itself calls `nids.cleaning.drop_train_duplicates`, UNSW-schema
only) to the TRAIN partition only, returning `(train_df, test_df)` with `test_df` byte-for-byte
whatever `train_test_split` produced. `RareCategoryGrouper` is still fitted on the CLEANED
(and, for the UNSW path, now also DEDUPLICATED-TRAIN-ONLY) partition, per its own module
docstring.

**Tolerating an already-dropped `--exclude` name (review blocking finding 1, round 3)**:
`compute_features(df, target, exclude, original_columns=None)` validates `--exclude` names
against `original_columns` (the dataset's columns BEFORE `clean_dataset_if_unsw` ran) when the
caller supplies it, instead of always validating against the (possibly already-cleaned)
`df.columns`. Every script captures `original_columns = list(raw_df.columns)` immediately
after `load_dataset`, before calling `clean_dataset_if_unsw`, and threads it through. This
tolerates `--exclude id` (documented as the standard usage example) even though `id` is a
`DROPPED_COLUMNS` member cleaning already removed, while a genuine typo (a name absent from
BOTH the pre- and post-cleaning column sets) still raises `ValueError`.

## Decision 3 — generic preprocessing fallback

`nids.preprocessing.build_generic_preprocessor(numeric_cols, categorical_cols) -> sklearn
Pipeline` (PUBLIC as of round 3 — a previous revision kept it underscore-private,
re-exported under this name only at its one caller, purely to satisfy a blunt AST guard
in `tests/test_leak_safety.py`; the guard is now precise about rejecting data-shaped
parameters while permitting `bool`/`list[str]`, so this function no longer needs to hide.
`nids.tabular.build_preprocessor_for`, re-exported by `common.build_preprocessor_for`,
calls it directly):

- numeric branch: `SimpleImputer(strategy="median")` -> `StandardScaler()`
- categorical branch: `SimpleImputer(strategy="constant", fill_value="missing")` ->
  `nids.transformers.RareCategoryGrouper(threshold=nids.columns.RARE_CATEGORY_THRESHOLD)` ->
  `OneHotEncoder(handle_unknown="ignore")` (round-3 standards finding 6: reads the shared
  constant instead of a hardcoded `0.01` literal, so it cannot drift from
  `build_preprocessor`'s own `RareCategoryGrouper()` default)
- `ColumnTransformer(..., remainder="drop")`, so anything outside the two detected column sets
  (including target/excluded columns, defensively) never reaches the model even if a caller
  passes the full frame.

`RareCategoryGrouper` is reused as-is (`src/nids/transformers.py`): its public contract takes a
`DataFrame` and a `threshold`, with no UNSW-specific column name baked in, so it is safe to reuse
for arbitrary categorical columns.

Numeric/categorical column detection: `df[features].select_dtypes(include="number")` for
numeric, everything else (object/category/string/bool) for categorical.

## Decision 4 — leakage assertion

Every script builds `X = df[features]` and asserts
`target not in features and set(exclude).isdisjoint(features)` immediately before constructing
`X`, and again asserts `target not in X.columns and set(exclude).isdisjoint(X.columns)` right
before `pipeline.fit`/`pipeline.transform`. Both checks raise `ValueError` (round 2: changed
from a bare `assert`, which vanishes under `python -O`) with the offending column names — this
is the "review checklist: any feature derived from the target is a blocking issue" rule made
mechanical.

## Decision 5 — per-skill behavior

**eda-reduction-clustering**: `nids.tabular.split_and_dedupe(df, target, test_size=0.3,
seed=42, is_unsw_schema=...)` (split FIRST, dedupe TRAIN split only — round-3 blocking
finding 2, see the "Cleaning before fitting" / "Split before dedupe" notes above); fit
preprocessor on train; PCA(n_components=2, random_state=42) on transformed train; KMeans on
the PCA output, `n_clusters` from `--n-clusters` or the silhouette sweep (see "Choosing
n_clusters" below). Outputs: `tables/summary_statistics.csv` (numeric describe()),
`tables/target_distribution.csv`, `tables/pca_explained_variance.csv`,
`tables/cluster_target_crosstab.csv`, `figures/pca_scatter_by_target.png`,
`figures/pca_scatter_by_cluster.png`. Metrics: `pca_explained_variance_ratio_sum`,
`n_clusters`, `n_clusters_selection_method`.

**clustering-reduction**: same preprocessing entry as above but adds `nids.sampling.
stratified_subsample(df, stratify_by=target, max_rows=10_000, seed=42)` before the split
(bounding runtime on large inputs, per AGENTS.md's subsample rule generalized to any target
column), then `split_and_dedupe` on the subsample (same split-first-dedupe-train-only
ordering); fit preprocessor on train; PCA on transformed train; KMeans vs
AgglomerativeClustering (same `n_clusters` for both) compared on that same PCA(2D) embedding
via `sklearn.metrics.silhouette_score` and `adjusted_rand_score` against the
(label-encoded) target. Outputs: `tables/subsample_allocation.csv` (from
`SubsampleResult.to_frame()`), `tables/clustering_comparison.csv` (algorithm, silhouette,
ARI), `figures/clusters_kmeans.png`, `figures/clusters_agglomerative.png`. Metrics:
`best_silhouette_score`, `best_algorithm`, `n_clusters`, `n_clusters_selection_method`.

**classification**: `split_and_dedupe(..., is_unsw_schema=...)` (same split-first-dedupe-
train-only ordering, called once per TTL variant inside `_run_variant`, deterministically
reproducing the same split each time since `df`/`target`/`seed` are unchanged); fit
`RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)` (fast, no GPU/heavy
tuning, matches "modest model settings"); when `sttl` and `ct_state_ttl` are BOTH present in
`features` (i.e. UNSW-shaped input with the TTL pair not already excluded), train and report
twice — with and without those two columns — otherwise once. Reports macro F1 and balanced
accuracy (never accuracy alone, per AGENTS.md). Outputs: `tables/classification_report.csv`
(precision/recall/f1 per class, one row set per TTL variant when applicable),
`figures/confusion_matrix.png` (per variant). Metrics: `macro_f1`, `balanced_accuracy` (suffixed
`_without_ttl` for the second variant when present).

### Choosing `n_clusters`, never from the target (review worth-fixing finding 7, round 3)

Both clustering skills previously set `n_clusters = max(2, min(y_train.nunique(), 10))` — a
hyperparameter read directly from the target's class count — then scored that same
clustering against that same target (silhouette AND adjusted Rand index), a supervision
leak into an otherwise unsupervised analysis step. Both skills now accept an explicit
`--n-clusters INT` CLI flag; when omitted, `skills._shared.common.choose_n_clusters(X, seed,
k_min=2, k_max=10)` fits `KMeans(n_clusters=k, random_state=seed, n_init=10)` for every
candidate `k` (clipped so no candidate exceeds `len(X) - 1`) and keeps the `k` with the
highest `silhouette_score(X, labels)` — computed purely from `X`'s own point geometry,
never the target. Both the chosen `n_clusters` and how it was chosen
(`n_clusters_selection_method`: `user_specified` / `silhouette_sweep` / `fixed_minimum`) are
recorded as metrics.

## Decision 6 — results wiring

`--output` is the exact folder the manifest lives in. `notebook_id = Path(output).name` (must
match `nids.paths.NOTEBOOK_ID_PATTERN`), `root = Path(output).parent`. `common.
resolve_results_writer` validates `notebook_id` against `NOTEBOOK_ID_PATTERN` explicitly and
raises a clear, pattern-naming `ValueError` BEFORE `root.mkdir(...)` runs (round-3
worth-fixing finding 8 — rather than relying solely on `ResultsWriter.__init__`'s own
validation, which raises only after `root` already exists on disk). Scripts call
`ResultsWriter(notebook_id, root=root)` and use the `with` context manager so a mid-run exception
never leaves a manifest claiming artifacts that were not actually written.

## Decision 7 — test strategy

`tests/test_skills.py` builds one synthetic `DataFrame` fixture (a few hundred rows, 4 numeric +
2 categorical feature columns, one multiclass target with a rare class of 2 rows), writes it to
`tmp_path/dataset.csv`, and calls each script's `main(argv)` directly (no subprocess) with
`--output tmp_path/out_<skill>`. Each test asserts: exit is `None`/0, `manifest.json` exists and
parses, every declared table/figure file exists, and neither the target column name nor any
excluded column name appears in any column of the fitted preprocessor's
`get_feature_names_out()` (categorical one-hot names are prefixed with the source column, so a
substring check anchored on the original column name plus `_`/exact match is used).

A `@pytest.mark.slow` test per script additionally runs against
`data/raw/UNSW_NB15_training-set.csv` (or whatever the committed raw filename is, resolved via
`nids.paths.raw_data_dir()`), `skipif` the file is absent, with `--target attack_cat --exclude
label` and a small forced subsample so it stays fast.

## Out of scope / deferred

- No shared CLI entry-point installer (`pyproject.toml` `[project.scripts]`) — scripts are
  invoked directly with `uv run python skills/<name>/<script>.py`, matching how the notebooks
  already invoke `nids`.
- No skill covers hyperparameter search; all model settings are fixed and modest by design.
