# Cluster-then-Reduce Specification

## Purpose

Define the auditable reporting contract for clustering the cleaned UNSW-NB15 **training** partition in
its full preprocessed feature space and projecting the result to 2D afterwards. This capability is a
consumer of `results-output-contract`, `preprocessing-pipeline`, `stratified-subsampling`,
`data-cleaning` and `data-loading`; it restates none of their internals and adds no requirement to them.

## Requirements

### Requirement: The notebook consumes the `nids` boundary and never re-implements cleaning or preprocessing

The notebook MUST obtain data-touching behaviour exclusively from `nids.columns`, `nids.data`,
`nids.preprocessing`, `nids.sampling`, `nids.paths` and `nids.results`. It MUST NOT re-implement raw
CSV loading, column dropping, `"-"`-to-category mapping, rare-category thresholding, scaling, one-hot
encoding or `log1p`. Every export MUST go through `ResultsWriter`; the notebook MUST NOT call `to_csv`
or `savefig` directly.

#### Scenario: No re-implementation appears in the executed notebook

- GIVEN the executed `notebooks/02_clustering_reduction.ipynb`
- WHEN its code cells are searched for `read_csv`, `OneHotEncoder(`, `StandardScaler(`, `.to_csv(`, `savefig(`, `drop_duplicates(`
- THEN no match MUST appear outside an argument passed to a `nids` function

### Requirement: Only the training partition is used and the test partition is never bound

The notebook MUST derive its data from the cleaned training partition alone. No name in the notebook's
global namespace MAY refer to the cleaned testing partition at any point after the load cell.

#### Scenario: A runtime assertion proves the test partition is absent

- GIVEN the notebook has executed its load cell
- WHEN the assertion cell runs
- THEN it MUST assert that no global name is bound to the `CleaningResult` or to its `test` frame
- AND execution MUST fail loudly if that assertion does not hold

### Requirement: `attack_cat` and `label` never enter a fit

Neither target MAY be passed to any `fit`, `fit_transform` or `fit_predict`. Both MAY be used only for
subsample stratification, figure colouring, and post-hoc external validation.

#### Scenario: A runtime assertion proves the label boundary

- GIVEN a fitted preprocessing pipeline for each variant
- WHEN the assertion cell runs
- THEN it MUST assert `"attack_cat"` and `"label"` are absent from `feature_columns(...)` for both variants
- AND it MUST assert both names are absent from the fitted pipeline's `get_feature_names_out()`
- AND it MUST assert every clustering estimator was called with the design matrix as its only positional argument

### Requirement: Clustering happens in the full preprocessed space, before any reduction

Every clustering estimator MUST be fitted on the output of the preprocessing pipeline with no PCA,
t-SNE, UMAP or other reduction applied first. Any 2D projection MUST be computed after the cluster
labels exist and MUST NOT feed back into any clustering fit.

#### Scenario: Projections are downstream of labels

- GIVEN the notebook's execution order
- WHEN the projection cells run
- THEN every cluster label array they colour MUST already have been produced by a fit on the full preprocessed matrix

### Requirement: Four algorithms with three internal metrics, both TTL variants

The notebook MUST report K-Means, Ward agglomerative, DBSCAN and spectral clustering, each with
silhouette, Calinski-Harabasz and Davies-Bouldin, for both the `with_ttl` and `without_ttl` variants.
K-Means and Ward MUST be swept over a stated `k` range whose choice is justified in the notebook.

#### Scenario: The internal metric table is complete

- GIVEN `results/clustering_reduction/tables/internal_metrics_by_method.csv`
- WHEN it is read
- THEN it MUST contain one row per (variant, method) for all four methods and both variants
- AND each row MUST carry `silhouette`, `calinski_harabasz`, `davies_bouldin`, `n_clusters` and `n_rows_scored`

### Requirement: Stated and enforced compute caps

The notebook MUST state its compute caps in markdown and enforce them in code: all clustering on the
seed-42 stratified subsample of at most 10,000 rows with floor 50; spectral clustering on at most
5,000 rows; t-SNE and UMAP on the 10,000-row subsample; `n_jobs=-1` wherever the estimator supports it;
a stated bootstrap repeat count between 3 and 5.

#### Scenario: No estimator is fitted on the full partition

- GIVEN the executed notebook
- WHEN every clustering, projection and metric call is inspected
- THEN each MUST receive at most the subsample row count declared for it
- AND the spectral call MUST receive at most 5,000 rows

### Requirement: External validation and stability are reported

The notebook MUST report ARI and NMI of each method's labels against both `attack_cat` and `label`, a
cluster-by-`attack_cat` contingency table for the chosen method, and pairwise ARI between bootstrap
reruns of the chosen method.

#### Scenario: External validation covers both references

- GIVEN `tables/external_validation_metrics.csv`
- WHEN it is read
- THEN it MUST contain rows for `reference` values `attack_cat` and `label` for every (variant, method)

#### Scenario: Stability is reported as a distribution, not a single number

- GIVEN `tables/stability_pairwise_ari.csv`
- WHEN it is read
- THEN it MUST contain one row per rerun pair with the shared-row count and the ARI
- AND `tables/stability_summary.csv` MUST carry the repeat count and the mean, minimum and maximum ARI

### Requirement: The notebook-01 comparison degrades gracefully

The notebook MUST attempt to read `results/eda_reduction_clustering/manifest.json` and populate a
comparison table of reduce-then-cluster against cluster-then-reduce. If that file, or any metric it
needs, is absent, the notebook MUST continue, MUST mark the missing values as unavailable, MUST
register a note explaining why, and MUST NOT raise. It MUST NEVER write into notebook 01's folder.

#### Scenario: Notebook 01 has not run

- GIVEN `results/eda_reduction_clustering/manifest.json` does not exist
- WHEN the comparison cell runs
- THEN the notebook MUST execute to completion
- AND `tables/pipeline_comparison.csv` MUST exist with the notebook-01 values marked unavailable
- AND the manifest MUST contain a note stating the comparison was not available and how to obtain it

#### Scenario: The comparison is disclosed as uncontrolled

- GIVEN the comparison table is populated
- WHEN the manifest notes are read
- THEN a note MUST state that the two notebooks cluster different populations in different spaces, so the comparison is directional rather than a controlled experiment

### Requirement: Output contract compliance with a digit-free, finite-safe manifest

All output MUST go to `results/clustering_reduction/` with `notebook_id` `clustering_reduction`. Every
table, figure and metric name MUST match `^[a-z][a-z0-9_]{0,63}$`. Any non-finite metric value MUST be
registered as the string `"inf"` or `"-inf"`, never as a floating-point infinity.

#### Scenario: The manifest parses and every path resolves

- GIVEN the notebook has executed to completion
- WHEN `results/clustering_reduction/manifest.json` is parsed as JSON
- THEN parsing MUST succeed with no `Infinity` or `NaN` token present
- AND every `path` under `tables` and `figures` MUST resolve to a file inside that folder

### Requirement: English-only artefacts and a single executed pass

Every markdown cell, chart title, axis label, legend label, table column and manifest string MUST be in
English. The notebook MUST execute end to end in one `nbconvert` pass with no manual cell reordering.

#### Scenario: The notebook executes from a cold kernel

- GIVEN a clean environment with the raw partitions present
- WHEN `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_clustering_reduction.ipynb` runs
- THEN it MUST complete without error
- AND `results/clustering_reduction/manifest.json` MUST be present and complete
