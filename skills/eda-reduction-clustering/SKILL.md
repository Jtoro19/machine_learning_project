---
name: eda-reduction-clustering
description: >-
  Use when the user asks for exploratory data analysis, PCA dimensionality reduction, or
  a first-pass KMeans clustering report over a tabular CSV dataset with a target column
  (e.g. "run EDA and clustering on this dataset", "show me a PCA projection of my data").
  Dataset-agnostic; not specific to the UNSW-NB15 project data.
---

# EDA + Reduction + Clustering

Runs exploratory summary statistics, a 2D PCA projection, and a KMeans clustering pass
over any tabular CSV dataset with a target column, and writes a
[results-output-contract](../../openspec/changes/project-foundation/specs/results-output-contract/spec.md)
folder (`manifest.json` + `tables/` + `figures/`).

## When to use

Invoke this skill when the user wants a quick EDA + reduction + clustering pass over a
CSV file — for example to sanity-check a new dataset, or to visualize class separability
before building a classifier.

## Usage

```bash
uv run python skills/eda-reduction-clustering/eda_reduction_clustering.py \
  --dataset path/to/data.csv \
  --target target_column_name \
  --exclude id \
  --output results/my_eda_run
```

- `--dataset` (required): path to a CSV file.
- `--target` (required): the target/label column name.
- `--exclude` (repeatable, or comma-separated): extra columns to drop from the feature
  set (e.g. an id column, a second target).
- `--output` (required): output folder. Its basename becomes the manifest's
  `notebook_id` and must be lowercase `snake_case` (e.g. `results/eda_run`, not
  `results/EDA-Run`).
- `--seed` (optional, default `42`).
- `--n-clusters` (optional, default: chosen automatically). Fixes the KMeans cluster
  count. When omitted, `k` is chosen by an unsupervised silhouette-score sweep over
  `k=2..10` — never derived from the target (see "Choosing `n_clusters`" below).

Run `--help` for the full flag list without needing a dataset.

**Only the UNSW-NB15 schema is detected and cleaned automatically.** For any other
dataset, identifier-like columns (row IDs, primary keys, etc.) are NOT dropped
automatically — the caller is responsible for naming them via `--exclude`, or they
will reach the feature matrix (and, for a generic dataset, be treated as an ordinary
numeric or categorical feature).

## What it does

1. When the input matches the UNSW-NB15 raw schema, cleans it first through the
   project's own `nids.cleaning` steps (drop the fixed unused columns, normalize
   `service`) — the same column-level cleaning the canonical `nids` pipeline requires
   before fitting `RareCategoryGrouper`.
2. Splits the (possibly cleaned) input with `train_test_split` (seed 42, stratified by
   the target when every class has at least 2 rows, otherwise unstratified), THEN drops
   exact-duplicate rows from the resulting TRAIN split only (see "Note on splitting and
   deduplication ordering" below), and fits preprocessing on that TRAIN split only
   (reuses `nids.preprocessing.build_preprocessor` for a UNSW-shaped frame; otherwise a
   generic median-impute+scale / rare-group+one-hot pipeline built from the dataset's own
   detected column dtypes).
3. Fits PCA(2D) on the transformed train split.
4. Fits KMeans on the PCA embedding (`n_clusters` from `--n-clusters`, or an unsupervised
   silhouette sweep — see below).
5. Writes: `summary_statistics`, `target_distribution`, `pca_explained_variance`,
   `cluster_target_crosstab` tables; `pca_scatter_by_target`, `pca_scatter_by_cluster`
   figures; `pca_explained_variance_ratio_sum`, `n_clusters`, and
   `n_clusters_selection_method` metrics.

Neither the target column nor any `--exclude` column ever reaches the feature matrix —
this is asserted twice (after feature-set computation, and immediately before
fit/transform).

### Note on splitting and deduplication ordering

The preprocessing pipeline is fitted on the training split only, in line with AGENTS.md's
split-first rule, even though this skill fits no supervised model afterward: the rule
governs the fit/fit_transform OPERATION, not whichever model consumes its output.
`clustering-reduction` (this skill's sibling) follows the same split-then-fit-on-train
approach. `classification` (which DOES fit a supervised RandomForest) also splits its data
into train/test.

Exact-duplicate-row removal (for a UNSW-shaped input) runs AFTER the split, on the TRAIN
split only, never before the split and never on the held-out split — deduplicating before
splitting would let held-out rows be silently removed too, shrinking a split that is
supposed to stay untouched (`nids.cleaning.drop_train_duplicates`'s own contract is
"training partition only").

### Choosing `n_clusters`

`n_clusters` is never derived from the target's class count (a supervision leak into an
otherwise unsupervised analysis). By default it is chosen by fitting `KMeans` for every
`k` in `2..10` and keeping the `k` with the highest `silhouette_score` computed purely
from the PCA-embedded points — pass `--n-clusters` to fix it explicitly instead. The
`n_clusters_selection_method` metric records which path was taken
(`user_specified`/`silhouette_sweep`/`fixed_minimum`).

Each `silhouette_score` call is capped at a 10,000-row random subsample
(`sample_size=10_000, random_state=seed`), matching AGENTS.md's 10,000-row cap for
Gaussian Process, spectral clustering, and t-SNE — `silhouette_score` is O(n²) in the
number of rows, evaluated once per candidate `k` (up to 9 times), so scoring it over
this skill's full train split (unlike `clustering-reduction`, which already subsamples
before clustering) would otherwise be evaluated at full UNSW-NB15 scale.
