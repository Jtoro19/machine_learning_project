# EDA, Dimensionality Reduction and Clustering Specification

> **Q1 status: PROVISIONAL, awaiting the user's ruling.** Every requirement below that
> references `results/eda_reduction_clustering/` resolves the notebook's results folder
> name under **proposal option A**: the notebook file stays
> `notebooks/01_eda_reduction_clustering.ipynb`; the results folder drops the numeric
> prefix because `src/nids/paths.py`'s `NOTEBOOK_ID_PATTERN = ^[a-z][a-z0-9_]{0,63}$`
> rejects a leading digit. If the user rules for option B or C instead, every occurrence
> of `eda_reduction_clustering` as a `notebook_id` in this spec MUST be updated to match
> before design/apply proceed.

## Purpose

Define the reporting contract for the unsupervised exploratory analysis of the cleaned
UNSW-NB15 training partition: which analyses MUST be reported, the mandatory dual
with/without-TTL-shortcut reporting, the boundary on how `attack_cat` may be used, and
the mandatory disclosures that silhouette and DBSCAN results are properties of a
stratified subsample rather than the full partition. This capability makes the notebook
auditable rather than decorative — every requirement below is checkable against the
executed notebook and its exported `results/` folder, not against prose intent.

This capability is a **consumer** of `results-output-contract`, `preprocessing-pipeline`,
`stratified-subsampling`, `data-cleaning` and `data-loading`. It restates none of their
internal behavior and adds no requirement to them; where a scenario below depends on
their guarantees (for example, that `feature_columns()` excludes `attack_cat` and
`label`), that dependency is consumed as given, not re-verified here.

## Requirements

### Requirement: The notebook consumes the `nids` package boundary and never re-implements cleaning or preprocessing

The notebook MUST import data-touching functionality exclusively from `nids.columns`,
`nids.data`, `nids.preprocessing`, `nids.sampling` and `nids.results`. It MUST NOT
contain any re-implementation of logic those modules already own: raw CSV loading,
duplicate detection, column dropping, scaling, one-hot encoding, `log1p` transforms on
partition columns, `"-"`-to-category mapping, or rare-category threshold logic. Every
export to disk MUST go through `ResultsWriter`; the notebook MUST NOT call `to_csv` or a
matplotlib save function outside of it.

#### Scenario: No re-implementation patterns appear in the executed notebook

- GIVEN the executed notebook `notebooks/01_eda_reduction_clustering.ipynb`
- WHEN its code cells are searched for `read_csv`, `duplicated(`, `StandardScaler(`, `OneHotEncoder(`, `drop(columns=`, `to_csv(`, or `savefig(`
- THEN no code cell MUST contain any of these patterns

#### Scenario: Every data-touching import resolves to `nids`

- GIVEN the notebook's import cell
- WHEN the imported names that read, clean, transform, sample, or export data are enumerated
- THEN every one of them MUST originate from `nids.columns`, `nids.data`, `nids.preprocessing`, `nids.sampling`, or `nids.results`

### Requirement: Only the cleaned training partition is used; the test partition is never read or referenced

The notebook MUST call `load_clean_partitions()` exactly once, with default arguments,
and MUST bind only the returned `.train` attribute to a name. The returned `.test`
attribute MUST NOT be bound to a name, printed, described, plotted, or used as an input
to any other call, anywhere in the notebook.

#### Scenario: No test-partition access pattern appears in a code cell

- GIVEN the executed notebook
- WHEN its code cells are searched for `load_raw_test`, `.test\b`, `test_df`, or `X_test`
- THEN no code cell MUST contain a match

#### Scenario: `load_clean_partitions` is called exactly once with defaults

- GIVEN the notebook's setup section
- WHEN calls to `load_clean_partitions` are counted
- THEN there MUST be exactly one call, with no arguments overriding `comparison_key` or `raw_dir`
- AND only its `.train` attribute MUST be assigned to a variable name

### Requirement: `attack_cat` and `label` never enter any model fit; `attack_cat` MAY be used post-hoc only

`attack_cat` and `label` MUST NOT be members of the feature allow-list consumed by any
fitting call, and no cell in the notebook MUST call `.fit(X, y)` with either column as
`y`: PCA, K-Means and DBSCAN are all fitted on the feature matrix alone. `attack_cat` MAY
be used only in the following contexts: descriptive statistics and class-distribution
reporting (Section 1); as the `stratify_by` key for `stratified_subsample` (Sections 0
and 4); as a post-hoc coloring variable on an already-fitted PCA projection (Section 3);
and as a post-hoc cross-tabulation against already-assigned cluster labels (Section 5).
The notebook MUST prove the feature/label boundary at run time with two assertion cells
in its setup section, not assert it only in markdown prose.

#### Scenario: The label-boundary assertions are present and pass

- GIVEN the notebook's setup section
- WHEN the executed notebook is inspected
- THEN a cell MUST assert that `"attack_cat" not in feature_columns(True)` and `"label" not in feature_columns(True)`
- AND a cell MUST assert that `{"attack_cat", "label"}` do not intersect `pipeline.get_feature_names_out()` for at least one fitted pipeline
- AND both assertions MUST have executed without raising an error in the notebook's saved outputs

#### Scenario: No cell fits a model against `attack_cat` or `label`

- GIVEN the executed notebook
- WHEN every `.fit(` and `.fit_predict(` call is enumerated
- THEN none MUST pass `attack_cat` or `label` (or a frame/series derived only from them) as a target argument

#### Scenario: `attack_cat` coloring on the PCA scatter is post-hoc

- GIVEN the PC1/PC2 scatter figure in Section 3
- WHEN the cell producing it is inspected
- THEN the `PCA` fit call MUST precede the assignment of `attack_cat` to a color/hue argument
- AND `attack_cat` MUST NOT appear anywhere in the matrix passed to `PCA().fit(...)`

### Requirement: The notebook respects the AGENTS.md data rules it consumes but does not implement

Because this notebook consumes cleaning and preprocessing rather than implementing them,
it MUST never act in a way that contradicts the project's data rules: the `id` column
MUST NOT appear as a feature; `attack_cat` and `label` MUST NOT be used as input
features (see the requirement above); no imputer, encoder or scaler MUST be fitted
outside a pipeline fitted on training data only (guaranteed by consuming
`build_preprocessor` unfitted and calling `.fit` once on the training frame); the `"-"`
category in `service` MUST NOT be remapped or imputed by this notebook (it is preserved
as delivered by `nids.data`); rare `proto` categories MUST NOT be regrouped by this
notebook (grouping under 1% into `"other"` is `build_preprocessor`'s responsibility, not
the notebook's); every stochastic operation (subsampling, PCA, K-Means) MUST use seed or
`random_state` 42; and silhouette/DBSCAN, as clustering-adjacent unsupervised analyses
requiring a subsample, MUST respect the project's stratified-subsample cap of at most
10,000 rows.

#### Scenario: No `id`, `attack_cat`, or `label` column reaches a fitted pipeline's input frame

- GIVEN either fitted preprocessing pipeline (with-TTL or without-TTL)
- WHEN the frame passed to its `.fit(...)` call is inspected
- THEN it MUST contain only columns drawn from `feature_columns(include_ttl=...)`
- AND `id`, `attack_cat`, and `label` MUST NOT be present among its columns

#### Scenario: No seed literal other than 42 appears

- GIVEN the executed notebook's code cells
- WHEN every `seed=`, `random_state=` argument is enumerated
- THEN every value MUST be `42`
- AND no other integer literal MUST be passed as a seed or random_state argument

### Requirement: The results folder identity is `eda_reduction_clustering` (provisional pending Q1 ruling)

`ResultsWriter` MUST be constructed with `notebook_id="eda_reduction_clustering"`,
producing `results/eda_reduction_clustering/` as the producer folder, while the notebook
file itself remains named `notebooks/01_eda_reduction_clustering.ipynb`. No table,
figure, or metric name registered through `ResultsWriter` MUST begin with a digit,
because `ResultsWriter` validates artifact names against the same
`^[a-z][a-z0-9_]{0,63}$` pattern used for `notebook_id`.

#### Scenario: `ResultsWriter` is constructed with the resolved notebook id

- GIVEN the notebook's setup section
- WHEN the `ResultsWriter(...)` call is inspected
- THEN its `notebook_id` argument MUST equal `"eda_reduction_clustering"`

#### Scenario: No exported artifact name begins with a digit

- GIVEN every `add_table`, `add_json_table`, `add_figure`, and `add_metric` call in the notebook
- WHEN each call's `name` argument is inspected
- THEN none MUST start with a digit

### Requirement: Descriptive statistics are reported for the cleaned training partition

Section 1 MUST report, through `ResultsWriter`: a column inventory with dtype,
non-null count and feature/target role (`partition_shape_and_dtypes`); a
central-tendency comparison of mean against median with skew, per numeric feature
(`numeric_summary_mean_vs_median`); categorical frequency counts for `proto`,
`service`, and `state` (`categorical_value_counts`); and the class balance of
`attack_cat` (`attack_cat_distribution`). It MUST also produce, as figures: sorted
`attack_cat` bars on a log x-axis; histograms of key volumetric features in both raw
and `log1p` form; a paired ECDF overlay of raw versus `log1p`; and boxplots of key
features grouped by `attack_cat` on a log y-axis. No row count in any of these outputs
MUST be a literal hardcoded in the notebook; every count MUST be computed at run time
from the loaded training frame.

#### Scenario: All four descriptive tables and five descriptive figures are registered

- GIVEN the completed notebook run
- WHEN `results/eda_reduction_clustering/manifest.json` is inspected
- THEN it MUST list `partition_shape_and_dtypes`, `numeric_summary_mean_vs_median`, `categorical_value_counts`, and `attack_cat_distribution` under `tables`
- AND it MUST list `attack_cat_distribution_bars`, `key_feature_histograms_raw`, `key_feature_histograms_log1p`, `key_feature_ecdf_raw_vs_log1p`, and `key_feature_boxplots_by_attack_cat` under `figures`

#### Scenario: No count is hardcoded

- GIVEN the notebook's descriptive-statistics cells
- WHEN the values feeding `partition_shape_and_dtypes` and `attack_cat_distribution` are traced
- THEN each value MUST derive from an expression evaluated against the loaded training frame at run time, not from a literal integer assigned directly to a reported field

### Requirement: Correlation analysis and multicollinearity (VIF) are reported

Section 2 MUST report Pearson and Spearman correlation matrices over the numeric
features, each as a table and as a heatmap figure using a diverging color palette
centered at zero with `vmin=-1`, `vmax=1`. It MUST report every feature pair with
`|r| > 0.9` under either method (`high_correlation_pairs`). Correlation matrices and
their heatmaps MUST be produced once only, covering the with-TTL feature set; the
notebook MUST state in markdown why the without-TTL matrix is not duplicated (it is the
with-TTL matrix with the `sttl`/`ct_state_ttl` rows and columns removed, and every
remaining cell is numerically identical because Pearson and Spearman are pairwise).

The Variance Inflation Factor MUST be computed with an inline helper implementing
`1 / (1 - R²)` via `sklearn.linear_model.LinearRegression`, with no new project
dependency added. VIF MUST be computed on the 36 numeric features in their modelled
form (`log1p` applied to `SKEWED_COLUMNS`, prior to any scaling) and MUST exclude
one-hot-encoded columns. VIF MUST be reported for both the with-TTL and without-TTL
feature sets, because VIF is a multivariate quantity and removing `sttl` /
`ct_state_ttl` changes every remaining feature's VIF. Two guards MUST hold: a feature
whose regression `R² >= 1.0` MUST report `vif = inf` and MUST NOT trigger a division;
and any zero-variance column MUST be dropped from the regression set before VIF is
computed, and MUST still appear in the VIF table with `vif = NaN` and a stated reason.

#### Scenario: Correlation heatmaps use a diverging palette centered at zero

- GIVEN `pearson_correlation_heatmap` and `spearman_correlation_heatmap`
- WHEN their color-mapping arguments are inspected
- THEN both MUST use a diverging colormap
- AND both MUST set `vmin=-1` and `vmax=1`

#### Scenario: A perfectly collinear feature reports infinity, not a division error

- GIVEN a numeric feature whose regression against all other numeric features yields `R² >= 1.0`
- WHEN its VIF is computed
- THEN the reported `vif` value MUST be `inf`
- AND the computation MUST NOT raise a `ZeroDivisionError` or equivalent

#### Scenario: A zero-variance column is dropped and flagged, not silently regressed

- GIVEN a numeric feature with zero variance in the training partition (for example `is_sm_ips_ports`)
- WHEN the VIF table is produced
- THEN that feature MUST NOT be used as a regressor for any other feature's VIF
- AND that feature's own row in `variance_inflation_factors` MUST have `vif = NaN` and a non-empty reason string

#### Scenario: VIF is reported for both TTL variants; the correlation matrix is not

- GIVEN the completed notebook run
- WHEN `variance_inflation_factors` is inspected
- THEN its `variant` column MUST contain both `with_ttl` and `without_ttl` rows
- AND `pearson_correlation_matrix` and `spearman_correlation_matrix` MUST each appear exactly once in the manifest, with no `_with_ttl` / `_without_ttl` counterpart

### Requirement: Principal Component Analysis quantifies redundancy in the feature space

Section 3 MUST fit `PCA(svd_solver="full", random_state=42)` on each of the two
preprocessed feature matrices (with-TTL and without-TTL) and MUST report, for both
variants: explained and cumulative variance per component
(`pca_explained_variance`); the number of components reaching 90% and 95% cumulative
variance (`pca_components_for_variance`); the top-loading features for PC1 and PC2 by
absolute value (`pca_top_loadings_pc1_pc2`); and the share of total variance
contributed by the numeric block versus the one-hot block
(`pca_variance_by_feature_block`). It MUST produce a scree plot and a cumulative
variance plot with 90%/95% reference lines, a top-loadings bar figure, and a 2D
scatter of PC1 against PC2 colored post-hoc by `attack_cat` for reference only. Because
`svd_solver="full"` uses scikit-learn's deterministic `svd_flip` sign convention,
component signs and loadings MUST NOT change between reruns over unchanged input data.

#### Scenario: Components reaching 90% and 95% variance are reported for both variants

- GIVEN the completed notebook run
- WHEN `pca_components_for_variance` is inspected
- THEN it MUST contain a row for `threshold=0.90` and a row for `threshold=0.95`, for both `with_ttl` and `without_ttl`
- AND each `n_components` value MUST be a positive integer no greater than the number of input columns for that variant

#### Scenario: The PCA scatter coloring is post-hoc and does not affect the fit

- GIVEN the `pca_scatter_pc1_pc2_by_attack_cat` figure
- WHEN the cell producing it is traced
- THEN the `PCA().fit(...)` call MUST precede any reference to `attack_cat` in that cell
- AND `attack_cat` MUST NOT be a column of the matrix passed to `.fit(...)`

#### Scenario: PCA loadings are stable across reruns

- GIVEN two consecutive `nbconvert --execute` runs over unchanged input data
- WHEN `pca_top_loadings_pc1_pc2` from each run is compared
- THEN every loading value and every sign MUST be identical between the two runs

### Requirement: K-Means and DBSCAN clustering are reported on the 90%-variance PCA space

Section 4 MUST fit `KMeans(n_clusters=k, n_init=10, random_state=42)` for
`k = 2..12` on the components reaching 90% cumulative variance, for both TTL variants,
on the **full** cleaned training partition (not the subsample), and MUST report
inertia, silhouette score, and Calinski-Harabasz score per `k`
(`kmeans_k_sweep_metrics`), plus a justified selection of `k` stated in markdown. It
MUST produce elbow (inertia), silhouette, and Calinski-Harabasz line figures across the
sweep for both variants.

Silhouette scoring MUST be computed on `stratified_subsample(cleaned.train,
stratify_by="attack_cat", max_rows=10_000, floor=50, seed=42)`, using cluster labels
assigned by the full-partition K-Means fit, and MUST NOT be computed on the full
partition. At the selected `k` only, silhouette MUST additionally be computed on a
`floor=0` subsample as a population-representative sensitivity check, and both values
MUST be reported as separate metrics.

DBSCAN MUST run on the same stratified subsample, in the same PCA space, with
`min_samples = 2 × n_components_90` and `eps` selected by a deterministic rule: compute
the `min_samples`-th nearest-neighbor distance for every point, sort ascending, and
select the point of maximum perpendicular distance from the chord joining the first and
last points of the sorted curve. The selected `eps` MUST be drawn on
`dbscan_k_distance_plot`. DBSCAN's cluster sizes and noise fraction
(`dbscan_cluster_summary`) MUST be reported per TTL variant. The subsample's allocation
(available vs. allocated rows per class, per sampler configuration) MUST be reported in
`clustering_sample_allocation`. The notebook MUST include a disclosure note stating that
silhouette is a subsample estimate, a disclosure note stating that the DBSCAN noise
fraction is a property of the subsample rather than the full partition, and a disclosure
note stating that the selected `eps` is not transferable to a differently sized
partition.

#### Scenario: K-Means is fitted on the full partition; silhouette is disclosed as a subsample estimate

- GIVEN the completed notebook run
- WHEN the K-Means fit call for a given `k` and TTL variant is traced
- THEN the frame passed to `.fit(...)` MUST have row count equal to the full cleaned training partition, not the subsample
- AND `results/eda_reduction_clustering/manifest.json`'s `notes` array MUST contain a note whose id is `silhouette_is_subsample_estimate`

#### Scenario: `eps` is selected deterministically and marked on the k-distance plot

- GIVEN the `dbscan_k_distance_plot` figure
- WHEN the selected `eps` value used to fit `DBSCAN` is compared against the value annotated on that figure
- THEN the two values MUST be equal
- AND rerunning the `eps`-selection computation over unchanged input data MUST produce the same value

#### Scenario: DBSCAN's noise fraction is disclosed as a subsample property

- GIVEN the completed notebook run
- WHEN `manifest.json`'s `notes` array is inspected
- THEN it MUST contain a note whose id is `dbscan_is_subsample_property`
- AND a note whose id is `dbscan_eps_not_transferable`

#### Scenario: The floor=0 sensitivity silhouette is reported alongside the default-floor silhouette

- GIVEN the selected `k` for the with-TTL variant
- WHEN the metrics array is inspected
- THEN it MUST contain both `kmeans_silhouette_at_selected_k_with_ttl` and `kmeans_silhouette_floor0_sensitivity_with_ttl`

### Requirement: Cluster profiling reports original-scale feature means and class composition, excluding ARI/NMI

Section 5 MUST report, for both TTL variants: the mean of each **original, unscaled**
feature per cluster against the global mean, with ratio and standard-deviations-from-mean
(`cluster_profile_original_units`); and the `attack_cat` composition of each cluster,
with both share-of-cluster and share-of-class (`cluster_attack_cat_composition`). This
section MUST NOT compute or report the Adjusted Rand Index, Normalized Mutual
Information, or any other external cluster-validation metric that compares cluster
assignments to `attack_cat` as ground truth.

#### Scenario: Cluster profiling uses unscaled original-unit features

- GIVEN `cluster_profile_original_units`
- WHEN its `cluster_mean` and `global_mean` values are compared against the standardized (scaled) feature matrix
- THEN the reported means MUST match the original training-partition feature scale, not the standardized scale used for clustering

#### Scenario: No external cluster-validation metric is present

- GIVEN the executed notebook and its manifest
- WHEN both are searched for `adjusted_rand_score`, `normalized_mutual_info_score`, `ARI`, or `NMI`
- THEN no match MUST be found in any code cell, table, or metric name

### Requirement: Dual with/without-TTL reporting applies wherever the toggle can change a conclusion

The following outputs MUST report both `with_ttl` and `without_ttl` values, distinguished
by a `variant` column in a single long-format table: `variance_inflation_factors`,
`pca_explained_variance`, `pca_components_for_variance`, `pca_top_loadings_pc1_pc2`,
`kmeans_k_sweep_metrics`, `dbscan_cluster_summary`, `cluster_profile_original_units`,
and `cluster_attack_cat_composition`. The correlation matrices and their heatmaps are
the sole documented exception (see the correlation requirement above). Section 6 MUST
state in prose, for each dual-reported result, whether removing `sttl` and
`ct_state_ttl` changes the conclusion drawn from it, and MUST NOT report a dual result
without also stating whether it differs.

#### Scenario: Every dual-reported table contains both variant values

- GIVEN each table named in this requirement
- WHEN its `variant` column is inspected
- THEN it MUST contain at least one row with `with_ttl` and at least one row with `without_ttl`

#### Scenario: Section 6 states where TTL removal changes conclusions

- GIVEN the notebook's conclusions section
- WHEN its markdown content is inspected
- THEN it MUST explicitly state, for VIF, PCA, and clustering results, whether removing the TTL shortcut pair changed the reported conclusion

### Requirement: All exported artifacts follow the results-output-contract and are reproducible

`results/eda_reduction_clustering/` MUST contain `manifest.json` at its root, `tables/`
and `figures/` subfolders, and MUST be interpretable in isolation, per the
`results-output-contract` capability this notebook consumes. Every table, figure,
metric, and disclosure note MUST be registered through `ResultsWriter`'s `add_table`,
`add_json_table`, `add_figure`, `add_metric`, and `add_note` methods, used as a context
manager so the manifest is written only on a clean exit. A `counts.json` sidecar MUST
mirror the registered metrics via `write_json`. Every figure title, axis label, legend
entry, table column name, and markdown cell MUST be in English. Two consecutive
`nbconvert --execute` runs over unchanged input data MUST produce byte-identical table
files under `tables/`, and `manifest.json` MUST differ between the two runs only in its
`generated_at` field.

#### Scenario: The manifest is complete and self-describing

- GIVEN the completed notebook run
- WHEN `results/eda_reduction_clustering/manifest.json` is parsed
- THEN parsing MUST succeed
- AND every path listed under `tables` and `figures` MUST resolve to an existing file within `results/eda_reduction_clustering/`
- AND the manifest's `notes` array MUST contain all six disclosure notes: `silhouette_is_subsample_estimate`, `dbscan_is_subsample_property`, `dbscan_eps_not_transferable`, `attack_cat_is_post_hoc_only`, `onehot_variance_asymmetry`, `ttl_shortcut_effect`

#### Scenario: Table files are byte-identical across reruns

- GIVEN the notebook executed twice, consecutively, over unchanged input data
- WHEN each run's `tables/*.csv` files are compared byte-for-byte between the two runs
- THEN every table file MUST be byte-identical across the two runs

#### Scenario: All notebook-facing text is in English

- GIVEN every figure and every markdown cell in the executed notebook
- WHEN titles, axis labels, legend entries, and table column names are inspected
- THEN none MUST contain non-English text

### Requirement: The notebook executes end to end in a single, verifiable pass

The notebook MUST execute to completion with `uv run jupyter nbconvert --to notebook
--execute --inplace notebooks/01_eda_reduction_clustering.ipynb`, exiting with status
`0` and producing no cell with an `error` output. A cell near the end of the notebook
MUST re-verify, at run time, that `results/eda_reduction_clustering/manifest.json`
parses and that every path it declares resolves to an existing file, so the manifest's
completeness is evidenced by the executed notebook itself rather than asserted only in
this spec.

#### Scenario: `nbconvert --execute` completes without error

- GIVEN a clean kernel and present raw data under `data/raw/`
- WHEN `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_reduction_clustering.ipynb` is run
- THEN the process MUST exit with status `0`
- AND no cell in the resulting notebook MUST contain an output of type `error`

#### Scenario: The manifest self-check cell passes during execution

- GIVEN the notebook's final manifest-verification cell
- WHEN the notebook is executed via `nbconvert`
- THEN that cell MUST complete without raising an exception
- AND its saved output MUST confirm that every declared table and figure path exists
