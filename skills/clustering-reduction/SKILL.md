---
name: clustering-reduction
description: >-
  Use when the user wants to compare clustering algorithms (KMeans vs Agglomerative) on
  a dimensionality-reduced, stratified subsample of a tabular CSV dataset (e.g.
  "compare clustering algorithms on this data", "which clustering method separates my
  classes best on a subsample"). Dataset-agnostic; not specific to the UNSW-NB15
  project data.
---

# Clustering + Reduction Comparison

Draws a seeded stratified subsample (capped at 10,000 rows), reduces it to a 2D PCA
embedding, then compares KMeans and Agglomerative clustering on that embedding using
silhouette score and adjusted Rand index against the target column. Writes a
[results-output-contract](../../openspec/changes/project-foundation/specs/results-output-contract/spec.md)
folder (`manifest.json` + `tables/` + `figures/`).

## When to use

Invoke this skill when the user wants to know which clustering algorithm best recovers
the target's structure on a dataset too large (or just inconveniently large) to cluster
directly, or wants a quick KMeans-vs-Agglomerative comparison.

## Usage

```bash
uv run python skills/clustering-reduction/clustering_reduction.py \
  --dataset path/to/data.csv \
  --target target_column_name \
  --exclude id \
  --output results/my_clustering_run
```

Same `--dataset`/`--target`/`--exclude`/`--output`/`--seed` contract as
`eda-reduction-clustering` (see that skill's `SKILL.md`), plus an optional
`--n-clusters` (fixed cluster count for both compared algorithms; when omitted, chosen
by an unsupervised silhouette sweep over `k=2..10` — see "Choosing `n_clusters`" below).
Run `--help` for the full flag list without needing a dataset.

**Only the UNSW-NB15 schema is detected and cleaned automatically.** For any other
dataset, identifier-like columns are NOT dropped automatically — the caller is
responsible for naming them via `--exclude`.

## What it does

1. When the input matches the UNSW-NB15 raw schema, cleans it first through the
   project's own `nids.cleaning` steps (drop the fixed unused columns, normalize
   `service`) — the same column-level cleaning the canonical `nids` pipeline requires
   before fitting `RareCategoryGrouper`.
2. `nids.sampling.stratified_subsample(df, stratify_by=target, max_rows=10_000,
   seed=42)` — returns the (possibly cleaned) input unchanged when it is already at or
   below 10,000 rows.
3. Splits the subsample with `train_test_split` (seed 42, stratified by the target when
   every class has at least 2 rows, otherwise unstratified), THEN drops exact-duplicate
   rows from the resulting TRAIN split only (see "Note on splitting and deduplication
   ordering" below), and fits preprocessing (UNSW-detected pipeline or generic fallback,
   same routing as `eda-reduction-clustering`) and PCA(2D) on that TRAIN split only.
4. Fits KMeans and AgglomerativeClustering (`n_clusters` from `--n-clusters`, or an
   unsupervised silhouette sweep — see below) on the PCA embedding; scores both with
   `silhouette_score` and `adjusted_rand_score` against the (label-encoded) target.
5. Writes: `subsample_allocation`, `clustering_comparison` tables; `clusters_kmeans`,
   `clusters_agglomerative` figures; `best_silhouette_score`, `best_algorithm`,
   `n_clusters`, `n_clusters_selection_method` metrics.

Neither the target column nor any `--exclude` column ever reaches the feature matrix —
asserted twice, exactly as in `eda-reduction-clustering`.

### Note on splitting and deduplication ordering

The preprocessing pipeline is fitted on the training split only, in line with AGENTS.md's
split-first rule, even though this skill fits no supervised model afterward: the rule
governs the fit/fit_transform OPERATION, not whichever model consumes its output.
`eda-reduction-clustering` (this skill's sibling) follows the same split-then-fit-on-train
approach. `classification` (which DOES fit a supervised RandomForest) also splits its data
into train/test.

Exact-duplicate-row removal (for a UNSW-shaped input, applied to the stratified subsample)
runs AFTER the split, on the TRAIN split only, never before the split and never on the
held-out split — see `eda-reduction-clustering/SKILL.md`'s identical note for the full
rationale.

### Choosing `n_clusters`

`n_clusters` is never derived from the target's class count. By default both algorithms
share a `k` chosen by fitting `KMeans` for every `k` in `2..10` and keeping the `k` with
the highest `silhouette_score` computed purely from the PCA-embedded points — pass
`--n-clusters` to fix it explicitly instead.
