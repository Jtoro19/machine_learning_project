# Proposal: Notebook 01 — EDA, Dimensionality Reduction and Clustering

> **Read first.** This change adds exactly one executed notebook and one `results/` producer folder.
> It writes no Python source, changes no dependency, and never touches the testing partition.
> Four decisions are made here that the user has not previously ruled on — they are marked
> **Q1**–**Q4** and collected in [Decisions returned to the orchestrator](#decisions-returned-to-the-orchestrator).
> **Q1 is blocking for the apply phase**: the requested results folder name is rejected by the
> foundation's own validator.

---

## Intent

### The question this notebook answers

After `project-foundation` cleans the UNSW-NB15 training partition, we hold roughly 108,000 rows
across 39 features and ten `attack_cat` classes, and we know nothing about its shape. Before any
supervised model is fitted, three things must be established:

1. **What the data looks like.** Distributions, skew, categorical balance, class imbalance.
2. **How much of the feature space is redundant.** 39 features that are heavily intercorrelated are
   not 39 degrees of freedom; PCA and VIF quantify how many there actually are.
3. **Whether unsupervised structure exists at all.** If K-Means on the reduced space recovers groups
   that align with attack families without ever seeing a label, that is evidence the classes are
   separable. If it does not, that is equally valuable — it tells us supervised performance will come
   from fine boundaries, not from gross geometric separation.

### Why now

- The foundation delivers the cleaned partition, the preprocessing pipeline, the subsampler and the
  results writer. Consuming them is the first proof that the API is usable from a notebook.
- The later supervised notebooks need the answers this one produces: how many components carry 90%
  of the variance, which feature pairs are collinear, and how much of the apparent structure
  disappears when `sttl` / `ct_state_ttl` are removed.
- `results/data_cleaning/` is currently the only producer under the output contract. A second,
  independently-written producer is the first real test that the contract generalises.

### What success looks like

One notebook that executes clean from a cold kernel, imports every non-trivial operation from
`nids`, exports a complete and self-describing `results/` folder, reports every conclusion twice
(with and without the TTL shortcut pair), and states plainly where its numbers are subsample
estimates rather than population facts.

---

## Scope

### In scope

| # | Deliverable |
|---|---|
| 1 | `notebooks/01_eda_reduction_clustering.ipynb`, committed **with outputs**, executed end to end in a single `nbconvert` pass. |
| 2 | Six analysis sections: descriptive statistics, correlation, PCA, clustering on the reduced space, cluster profiling, plain-language conclusions. |
| 3 | A `results/` producer folder (see **Q1** for its name) with ~17 tables, ~16 figures, ~20 metrics and ~6 disclosure notes, all registered through `ResultsWriter`. |
| 4 | Dual reporting with and without `sttl` / `ct_state_ttl` for every result where the toggle can change the conclusion (VIF, PCA, K-Means, cluster composition). |
| 5 | A `variance_inflation_factor` helper written inline in the notebook from `1 / (1 - R²)` — no new dependency (see **F1**). |
| 6 | Explicit assertion cells proving the label boundary and the no-test-partition rule at run time. |

### Out of scope

| Excluded | Why |
|---|---|
| Any supervised model, any fit on `attack_cat` / `label` | Notebook 2's job. This notebook is unsupervised end to end. |
| ARI / NMI or any external cluster-validation metric | Explicitly reserved for notebook 2 by the user. |
| t-SNE and UMAP | Not argued for. PCA answers the "how many dimensions" question; a nonlinear embedding would produce a picture that feeds nothing downstream here and cannot be used for the clustering step (no out-of-sample transform for t-SNE). `umap-learn` stays declared and unused until a change needs it. |
| Any edit to `src/nids/`, `pyproject.toml`, `tests/` | The `nids` API is frozen by the foundation design. If this notebook needs something the API does not offer, that is a finding to report, not a patch to apply. **Concurrent-work boundary**: `src/nids/` and `openspec/changes/project-foundation/` are being edited by another agent. |
| Any read of the testing partition | Hard rule. Verified mechanically (see Success Criteria #4). |
| Any modification under `data/raw/` | Read-only, via `nids.loading` only. |
| A `.gitattributes` notebook diff driver | Would be a repository-wide policy change riding on a notebook change. Flagged as a follow-up in **F4**. |

---

## Capabilities

> Contract with `sdd-spec`. `openspec/specs/` is empty — `project-foundation` has not been archived —
> so nothing existing can be modified.

### New capabilities

- `eda-reduction-clustering`: the reporting contract for the unsupervised exploratory analysis of the
  cleaned training partition. Owns the requirements that make the notebook auditable rather than
  decorative: which analyses MUST be reported, the mandatory dual TTL reporting, the label-usage
  boundary (`attack_cat` for stratification, colouring and profiling only — never as an input
  feature and never as a fitting target), and the mandatory disclosure that silhouette and DBSCAN
  results are properties of a stratified subsample.

### Modified capabilities

None. This change is a **consumer** of `results-output-contract`, `preprocessing-pipeline`,
`stratified-subsampling`, `data-cleaning` and `data-loading`. It adds no requirement to any of them
and restates none of their behaviour.

---

## Approach

### 1. The `nids` API surface this notebook consumes

Every signature below is taken verbatim from `openspec/changes/project-foundation/design.md`. The
notebook calls these and nothing else that touches data. It does not restate their behaviour, and it
does not re-implement any of it.

```python
from nids.columns import feature_columns, TTL_SHORTCUT_COLUMNS, CATEGORICAL_COLUMNS, SKEWED_COLUMNS
from nids.data        import load_clean_partitions   # -> CleaningResult          (slice 3)
from nids.preprocessing import build_preprocessor    # (include_ttl_features: bool) -> Pipeline  (slice 4)
from nids.sampling   import stratified_subsample     # -> SubsampleResult          (slice 4)
from nids.results    import ResultsWriter            #                             (slice 5)
```

| Call | Exact signature | How the notebook uses it |
|---|---|---|
| `load_clean_partitions` | `(*, comparison_key: LeakageKey = "full_row", raw_dir: Path \| None = None) -> CleaningResult` | Called **once**, defaults. Only `.train` is bound to a name. `.test` is never referenced. |
| `feature_columns` | `(include_ttl: bool = True) -> list[str]` | The only source of the feature allow-list. Called with `True` and `False`. |
| `build_preprocessor` | `(include_ttl_features: bool = True) -> Pipeline` | Two unfitted pipelines. Each is `.fit(cleaned.train[feature_columns(v)])` — the frame is named at the call site, which is exactly what the AGENTS.md review checklist greps for. |
| `stratified_subsample` | `(df, *, stratify_by="attack_cat", max_rows=10_000, floor=50, seed=42) -> SubsampleResult` | Row selection for silhouette, DBSCAN and the PCA scatter. Called twice: default floor, and `floor=0` for the sensitivity check. |
| `ResultsWriter` | `(notebook_id, *, root=None, generated_at=None, prune_undeclared=True)`; `.add_table(frame, *, name, title, description, sort_by=None)`, `.add_json_table`, `.add_figure(figure, *, name, title, description)`, `.add_metric(name, value, description)`, `.add_note(note_id, text)`, `.write_json`, `.close()` | Every export. Used as a context manager so the manifest is written last, on clean exit only. |

**What the notebook is forbidden to contain**, because `nids` already owns it: `pd.read_csv`,
`.duplicated(`, `.drop(columns=`, `StandardScaler(`, `OneHotEncoder(`, `np.log1p` on a partition
column, any `"-"` → `"none"` mapping, any rare-category threshold, any `to_csv` / `savefig` call
outside `ResultsWriter`. Success Criterion #5 checks this with `rg`.

### 2. Notebook structure, mapped to the six requested sections

| § | Section | What it computes | TTL variants |
|---|---|---|---|
| 0 | Setup and guards | Imports, `load_clean_partitions()`, both fitted pipelines, both subsamples, the label-boundary assertions | — |
| 1 | Descriptive statistics | Shape, dtypes, `describe` with median and skew side by side, categorical `value_counts`, sorted `attack_cat` bars on a log scale, histograms and ECDFs of key volumetric features before and after `log1p`, boxplots by `attack_cat` | Not applicable — descriptive stats describe all 39 columns; the TTL pair is shown as two of them |
| 2 | Correlation and collinearity | Pearson and Spearman matrices, diverging palette centred at 0, pairs with \|r\| > 0.9, VIF per numeric feature | **VIF: both.** Matrices: one only (see below) |
| 3 | Dimensionality reduction | PCA on the preprocessed matrix, scree and cumulative variance, components reaching 90% and 95%, top PC1/PC2 loadings, 2D scatter coloured by `attack_cat` **for reference only** | **Both** |
| 4 | Clustering on the reduced space | K-Means k = 2…12 on the 90%-variance components: inertia, silhouette, Calinski-Harabasz; a justified k; DBSCAN with `eps` from a k-distance curve; noise fraction | **Both** |
| 5 | Cluster profiling | Per-cluster mean of the **original, unscaled** features against the global mean; `attack_cat` composition per cluster | **Both** |
| 6 | Conclusions | What structure exists, what the clusters appear to represent, limitations | — |

**Why the correlation heatmaps are not duplicated per variant, but VIF is.** The without-TTL
correlation matrix is the with-TTL matrix with two rows and two columns deleted — every remaining
cell is numerically identical, because Pearson and Spearman are pairwise. A second heatmap would
carry zero new information. VIF is the opposite case: it is a *multivariate* quantity, `1 / (1 - R²)`
of each feature regressed on **all the others**, so removing `sttl` and `ct_state_ttl` changes every
other feature's VIF. That asymmetry is itself worth stating in the notebook, and it is the honest
answer to "run everything with and without the TTL features": run it where it can change, say why it
cannot change where it cannot.

The TTL pair are both plain-numeric features (foundation design, §`preprocessing.py`), so excluding
them simply removes two columns from the `numeric` branch — no fitted value changes shape or
meaning. The resulting PCA feature space differs by two columns, which is enough to change every
component, every loading and every cluster assignment.

### 3. Exported artifact inventory

All names satisfy `ARTIFACT_NAME_PATTERN` = `^[a-z][a-z0-9_]{0,63}$`. Every table that has a
with/without-TTL form is emitted as **one long-format CSV with a `variant` column**, not two files —
this halves the manifest, and it puts the comparison the user asked for inside a single table a
reader can sort.

**Tables (17)**

| § | `name` | Shape / key | Manifest `description` (abridged) |
|---|---|---|---|
| 1 | `partition_shape_and_dtypes` | column, dtype, non_null, role | Column inventory of the cleaned training partition with feature/target role |
| 1 | `numeric_summary_mean_vs_median` | feature × {count, mean, median, std, min, q25, q75, max, skew, mean_median_ratio} | Central-tendency contrast exposing right skew |
| 1 | `categorical_value_counts` | column, category, rows, share | `proto` / `service` / `state` frequencies |
| 1 | `attack_cat_distribution` | attack_cat, rows, share | Class balance of the cleaned training partition |
| 2 | `pearson_correlation_matrix` | 36 × 36 | Linear correlation between numeric features |
| 2 | `spearman_correlation_matrix` | 36 × 36 | Rank correlation between numeric features |
| 2 | `high_correlation_pairs` | feature_a, feature_b, pearson, spearman, method_flagged | Pairs exceeding \|r\| > 0.9 under either method |
| 2 | `variance_inflation_factors` | variant, feature, r_squared, vif | Multicollinearity per feature, both TTL variants |
| 3 | `pca_explained_variance` | variant, component, explained, cumulative | Scree data for both variants |
| 3 | `pca_components_for_variance` | variant, threshold, n_components | Components reaching 90% and 95% |
| 3 | `pca_top_loadings_pc1_pc2` | variant, component, feature, loading, abs_loading | Top loadings by absolute value |
| 3 | `pca_variance_by_feature_block` | variant, block, share_of_total_variance | Numeric vs one-hot contribution (see **F6**) |
| 4 | `kmeans_k_sweep_metrics` | variant, k, inertia, silhouette, calinski_harabasz | The k = 2…12 sweep |
| 4 | `dbscan_cluster_summary` | variant, cluster, rows, share | Cluster sizes including the `-1` noise label |
| 4 | `clustering_sample_allocation` | sampler, attack_cat, available, allocated, floor_applied | `SubsampleResult.to_frame()` for both samplers — the disclosure the subsampling spec mandates |
| 5 | `cluster_profile_original_units` | variant, cluster, feature, cluster_mean, global_mean, ratio, std_deviations | Unscaled per-cluster means against the global mean |
| 5 | `cluster_attack_cat_composition` | variant, cluster, attack_cat, rows, share_of_cluster, share_of_class | Post-hoc composition — **profiling only** |

**Figures (16)** — every one carries an English title and English axis labels.

| § | `name` | Kind |
|---|---|---|
| 1 | `attack_cat_distribution_bars` | Sorted horizontal bars, log x-axis |
| 1 | `key_feature_histograms_raw` | Histogram grid, raw units |
| 1 | `key_feature_histograms_log1p` | Same grid after `log1p` |
| 1 | `key_feature_ecdf_raw_vs_log1p` | Paired ECDF overlay |
| 1 | `key_feature_boxplots_by_attack_cat` | Boxplots per class, log y-axis |
| 2 | `pearson_correlation_heatmap` | Diverging palette centred at 0, `vmin=-1`, `vmax=1` |
| 2 | `spearman_correlation_heatmap` | Same palette and limits |
| 2 | `variance_inflation_factors_bars` | VIF per feature, both variants, log y-axis |
| 3 | `pca_scree_plot` | Per-component explained variance, both variants overlaid |
| 3 | `pca_cumulative_variance` | Cumulative curve with 90% and 95% reference lines |
| 3 | `pca_top_loadings_pc1_pc2` | Horizontal loading bars |
| 3 | `pca_scatter_pc1_pc2_by_attack_cat` | Scatter on the subsample, coloured post hoc |
| 4 | `kmeans_elbow_inertia` | Inertia vs k, both variants |
| 4 | `kmeans_silhouette_by_k` | Silhouette vs k, both variants |
| 4 | `kmeans_calinski_harabasz_by_k` | CH vs k, both variants |
| 4 | `dbscan_k_distance_plot` | Sorted k-th nearest-neighbour distance with the selected `eps` marked |

**Metrics (~20)**, names fixed so a future dashboard can rely on them:
`train_rows`, `feature_columns_with_ttl`, `feature_columns_without_ttl`, `attack_cat_classes`,
`correlated_pairs_above_0_9`, `max_vif_with_ttl`, `max_vif_without_ttl`,
`pca_components_90_with_ttl`, `pca_components_90_without_ttl`, `pca_components_95_with_ttl`,
`pca_components_95_without_ttl`, `onehot_share_of_total_variance_with_ttl`,
`kmeans_selected_k_with_ttl`, `kmeans_selected_k_without_ttl`,
`kmeans_silhouette_at_selected_k_with_ttl`, `kmeans_silhouette_floor0_sensitivity_with_ttl`,
`silhouette_sample_rows`, `dbscan_eps_with_ttl`, `dbscan_min_samples`,
`dbscan_noise_fraction_with_ttl`, `dbscan_cluster_count_with_ttl`.

**Notes (6)**, each a mandatory disclosure:
`silhouette_is_subsample_estimate`, `dbscan_is_subsample_property`,
`dbscan_eps_not_transferable`, `attack_cat_is_post_hoc_only`,
`onehot_variance_asymmetry`, `ttl_shortcut_effect`.

Plus a `counts.json` sidecar via `write_json`, mirroring the metric list — the same convention
`cleaning_report.py` uses, so the two producer folders read the same way.

### 4. Determinism and reproducibility

| Source of drift | How it is pinned |
|---|---|
| Subsample selection | `stratified_subsample(..., seed=42)`, on a frame whose positional order is fixed by `clean_partitions`' terminal `reset_index(drop=True)` |
| PCA | `PCA(svd_solver="full", random_state=42)`. `"full"` rather than `"auto"`: `"auto"` may select the randomized solver on larger inputs, and `"full"` is exactly deterministic. scikit-learn's `svd_flip` fixes component signs, so loadings do not flip between runs |
| K-Means | `KMeans(n_clusters=k, n_init=10, random_state=42)` |
| DBSCAN | Deterministic given the input and `eps`; border-point assignment depends on row order, which is fixed |
| `eps` selection | A stated rule, not an eyeballed knee — see **F3** |
| Table / JSON / PNG bytes under `results/` | `ResultsWriter`'s contract: `lineterminator="\n"`, `float_format="%.6f"`, explicit `sort_by`, PNG `metadata={"Software": None, "Creation Time": None}` |
| Wall-clock churn | One `generated_at`, in `manifest.json` only |
| Notebook execution counts | A single `nbconvert --execute` pass always renumbers 1…N — see **F4** |

---

## Technical findings and decisions

### F1 — VIF: compute it, do not add `statsmodels`

**Decision: compute VIF in the notebook, no new dependency.**

`statsmodels` is not in the declared dependency set (`ipykernel`, `jupyterlab`, `matplotlib`,
`numpy`, `pandas`, `pyarrow`, `scikit-learn`, `scipy`, `seaborn`, `umap-learn`; dev: `pytest`).
Adding it would pull in `patsy` and a substantial compiled surface, permanently, to obtain one
column of numbers that is five lines of arithmetic:

```python
def variance_inflation_factors(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in X.columns:
        r2 = LinearRegression().fit(X.drop(columns=[column]), X[column]).score(
            X.drop(columns=[column]), X[column]
        )
        rows.append({"feature": column, "r_squared": r2,
                     "vif": float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)})
    return pd.DataFrame(rows)
```

This is not an approximation of `statsmodels.stats.outliers_influence.variance_inflation_factor` —
it is the same quantity by the same definition. `LinearRegression` fits an intercept by default,
which is the constant column that `statsmodels`' helper makes the caller add manually and that
callers routinely forget, producing silently inflated values. Two guards the notebook must carry:

- **`r2 >= 1.0` → report `inf`, never divide.** Perfectly collinear columns exist in this dataset's
  neighbourhood (`swin` / `dwin`, the `ct_*` family), and a `ZeroDivisionError` mid-notebook is a
  worse outcome than an honest `inf` in the table.
- **Drop zero-variance columns first** and list them in the table with `vif = NaN` and a reason.
  `is_sm_ips_ports` is near-constant on this partition; regressing on a constant column produces a
  meaningless `R²`.

**Which matrix VIF runs on.** The 36 numeric features (15 skewed + 21 plain) in their *modelled*
form: `log1p` applied to `SKEWED_COLUMNS`, then standardised. VIF is scale-invariant, so the scaler
is irrelevant, but `log1p` is not — it materially changes linear relationships among the volumetric
features, and the modelled form is the space PCA and the clustering actually see. One-hot columns are
excluded: VIF over a dummy block is dominated by the mutual-exclusivity constraint and reports
structural, not substantive, collinearity.

Reported for both TTL variants, because VIF is multivariate (see Approach §2).

### F2 — Silhouette runs on the stratified subsample, and says so

Silhouette is O(n²) in pairwise distances. On ~108,000 rows that is ~1.2 × 10¹⁰ pairs — not viable,
and `sklearn.metrics.silhouette_score` chunking only bounds the memory, not the time.

**Decision: silhouette is computed on `stratified_subsample(cleaned.train, max_rows=10_000,
floor=50, seed=42)` — at most 10,000 rows**, matching the AGENTS.md subsample cap and the
`stratified-subsampling` capability. K-Means itself is still fitted on the **full** cleaned
partition; only the metric is sampled, and the sampled rows' cluster labels come from that
full-partition fit.

**The honest caveat, which the notebook must state and `clustering_sample_allocation` must evidence:**
the default `floor=50` deliberately over-represents minority `attack_cat` classes relative to the
population. The silhouette computed on that sample is therefore **not an unbiased estimate of the
full-partition silhouette** — it is the silhouette of a class-rebalanced sample. Mitigation, at the
cost of one extra computation rather than eleven: at the **selected k only**, silhouette is also
computed on a `floor=0` subsample, which is pure proportional allocation and therefore
population-representative in class mix. Both numbers are reported
(`kmeans_silhouette_at_selected_k_with_ttl`, `kmeans_silhouette_floor0_sensitivity_with_ttl`). If
they disagree materially, that disagreement is a section 6 finding.

Note `silhouette_is_subsample_estimate` states the sample size, the stratification key, the floor,
and that the value is an estimate.

### F3 — DBSCAN runs on the subsample, and `eps` is chosen by a stated rule

**Decision: DBSCAN runs on the same ≤10,000-row stratified subsample, in the same PCA space.**

Rationale. On ~108,000 rows in ~10 PCA dimensions, `DBSCAN`'s tree-based neighbour search degrades
(ball-tree performance falls off well before ten dimensions) and, worse, memory is driven by
neighbourhood *size*, not row count: this partition has large near-duplicate dense regions, so an
`eps` even slightly too large produces neighbour lists that can exhaust memory mid-notebook. That
failure mode is unbounded and would take the whole run down. A 10,000-row bound makes the cost
predictable and the notebook reliable.

**What this costs, stated plainly:** the reported noise fraction, cluster count and cluster sizes are
properties **of that subsample**, not of the cleaned training partition. Two consequences the notebook
must disclose:

- `dbscan_is_subsample_property`: the noise fraction is a subsample statistic. It is not a claim
  about the 108,000-row partition.
- `dbscan_eps_not_transferable`: local density scales with n. An `eps` calibrated on a 10,000-row
  sample is **not** the right `eps` for the full partition — it will be too large there. Nobody
  should lift this number into another notebook.

**`eps` selection rule**, so the result is reproducible rather than eyeballed: compute the k-th
nearest-neighbour distance for every point with `k = min_samples`, sort ascending, and select the
point of **maximum perpendicular distance from the chord joining the first and last points of the
sorted curve**. Deterministic, ~10 lines, no new dependency, and it is the same geometric idea the
"kneedle" method formalises. The selected value is drawn on `dbscan_k_distance_plot` so a reader can
disagree with it from the picture.

`min_samples = 2 × n_components_90` (the Sander et al. heuristic for d-dimensional data), reported as
a metric rather than buried in a call.

### F4 — Committed notebook outputs: what actually churns, and what does not

**The repository cost, estimated.** A 10 × 6 inch matplotlib PNG at the inline default of 100 dpi
runs roughly 100–400 KB for a dense scatter or an annotated 36 × 36 heatmap. Sixteen figures inline
is therefore roughly **2–6 MB of base64 inside the `.ipynb`**, and the same sixteen figures written
again under `results/.../figures/` at `ResultsWriter`'s 150 dpi is roughly **4–9 MB more**. Total
added weight: **on the order of 10 MB**. That is an estimate from figure geometry, not a measurement;
the apply phase must report the measured `du -sh` of both the notebook and the results folder, and if
either exceeds twice this estimate that is a finding, not a rounding error.

**What keeps reruns from producing gratuitous diffs — concretely:**

1. **The numbers do not move.** Seed 42 on the subsample, `svd_solver="full"` with `svd_flip` on PCA,
   `random_state=42` with `n_init=10` on K-Means, a deterministic `eps` rule, and a fixed positional
   row order inherited from `clean_partitions`' terminal `reset_index(drop=True)`. Identical inputs
   produce identical values, so every text output and every table is byte-stable.
2. **`results/` is byte-stable by contract, not by luck.** `ResultsWriter` writes CSV with
   `lineterminator="\n"` and `float_format="%.6f"`, JSON with fixed indent and a trailing newline,
   and PNG with `metadata={"Software": None, "Creation Time": None}` — without that last override
   matplotlib stamps its version into a PNG text chunk and every rerun is a binary diff. Every table
   gets an explicit `sort_by`, so pandas tie-ordering cannot reshuffle rows.
3. **One timestamp, one file.** `generated_at` lives only in `manifest.json`. A rerun over unchanged
   inputs produces a **one-line diff in exactly one file**, and `SOURCE_DATE_EPOCH` makes even that
   disappear.
4. **Execution counts are renumbered in one pass.** Interactive re-running leaves arbitrary,
   non-sequential `execution_count` values, and every partial re-run rewrites a scattered subset of
   cells. Executing only via
   `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_reduction_clustering.ipynb`
   always yields 1…N in order. This is a **process rule for the apply phase**, and it is the single
   largest source of avoidable notebook churn.
5. **Cell IDs are stable.** nbformat v4.5 `id` fields are preserved on re-execution, not regenerated,
   so they do not contribute churn.
6. **No stray reprs.** Every plotting cell ends in a statement that does not echo the figure object
   (no bare `fig` as the last expression), and `pd.set_option` display limits are set once in the
   setup cell so DataFrame HTML reprs have a fixed shape.

**What is honestly not fixed.** The inline PNG bytes are produced by the IPython inline backend,
which does **not** apply `ResultsWriter`'s metadata suppression; those blobs can differ across
matplotlib versions even when the plotted data is identical. And any re-execution rewrites every
base64 blob in the file regardless. The mitigation is procedural — execute once, deliberately, as
the last step before commit — not technical. A `.gitattributes` notebook diff driver would improve
review ergonomics but is a repository-wide policy change; recorded as a follow-up, not smuggled in
here.

### F5 — `attack_cat` is a label. Where it may appear, and where it may not

This is the AGENTS.md blocking rule ("any feature derived from the target is a blocking issue"), and
the boundary must be unambiguous because `attack_cat` legitimately appears four times in this
notebook.

| Use | Section | Legitimate? | Why |
|---|---|---|---|
| Class-distribution table, bars, boxplot grouping | 1 | **Yes** | Describing the label is the point of a descriptive section. It is not fed to anything. |
| `stratify_by="attack_cat"` in `stratified_subsample` | 0, 4 | **Yes, with disclosure** | The label selects *which rows are evaluated*, never what the clustering learns. See the caveat below. |
| Colouring the PC1/PC2 scatter | 3 | **Yes** | Post-hoc colouring of an already-computed projection. The PCA was fitted before the colour was chosen. |
| Cluster composition profiling | 5 | **Yes** | Post-hoc cross-tabulation of cluster labels against true classes. No metric is optimised against it; ARI/NMI are explicitly notebook 2's. |

| Forbidden use | Structural guarantee |
|---|---|
| As an input feature | `feature_columns()` is an explicit **allow-list** of 39 names, not `set(all) − set(targets)`. A skipped drop cannot reintroduce it. |
| Reaching the design matrix by accident | `ColumnTransformer(remainder="drop")` drops `attack_cat` and `label` even if the full 41-column frame is passed to `fit`. |
| As a fitting target | Nothing in this notebook calls `.fit(X, y)`. PCA, K-Means and DBSCAN are all fitted on `X` alone. |

**The notebook proves this at run time**, in the setup cell, rather than asserting it in prose:

```python
assert "attack_cat" not in feature_columns(True) and "label" not in feature_columns(True)
assert not ({"attack_cat", "label"} & set(pipeline.get_feature_names_out()))
```

**The one caveat that must not be glossed.** Because the subsample is stratified by `attack_cat` with
a per-class floor, the rows DBSCAN clusters were *selected* using label information. The clustering
is still unsupervised — no label enters any `fit` — but the sample it runs on is class-rebalanced,
so minority attack families are over-represented relative to the population and the noise fraction is
conditioned on that. The `floor=0` sensitivity run (**F2**) is the label-neutral comparison, and it
reuses the same tested helper rather than introducing a second, untested sampler.

### F6 — One-hot columns are unscaled, so PCA is dominated by the numeric block

Not one of the five findings handed down, but it changes how section 3 must be read, so it is
recorded here rather than discovered during apply.

`build_preprocessor`'s categorical branch is `RareCategoryGrouper → OneHotEncoder`, with **no
scaler**. The numeric and skewed branches both end in `StandardScaler`. So in the ~52-column design
matrix, numeric columns have variance 1.0 while a one-hot column has variance `p(1−p) ≤ 0.25` — and
for a rare category, far less. PCA maximises variance, so the components are dominated by the numeric
block **by construction**, and PC1/PC2 loadings will be numeric-heavy for a structural reason, not a
substantive one.

Scaling the one-hot block would require changing `build_preprocessor`, which is out of scope. So the
notebook **measures** the asymmetry instead of asserting it: `pca_variance_by_feature_block` reports
what share of total variance the one-hot block contributes against the numeric block, and the
`onehot_variance_asymmetry` note explains what that means for loading interpretation. A measured
number a reader can check beats a caveat a reader must trust.

---

## Affected Areas

| Area | Impact | Description |
|---|---|---|
| `notebooks/01_eda_reduction_clustering.ipynb` | New | The deliverable, committed executed with outputs |
| `results/<notebook_id>/` (name per **Q1**) | New (generated, committed) | `manifest.json`, `counts.json`, `tables/*.csv` (17), `figures/*.png` (16) |
| `openspec/changes/notebook-01-eda-reduction-clustering/` | New | This proposal, plus the spec, design and tasks that follow |
| `src/nids/**` | **Untouched** | Frozen API; concurrently edited by another agent |
| `pyproject.toml` | **Untouched** | No dependency added — see **F1** |
| `data/raw/**` | **Untouched** | Read-only, through `nids.loading` |
| `results/data_cleaning/**` | **Untouched** | The output contract forbids one producer writing into another's folder |
| `tests/**` | **Untouched** | `openspec/config.yaml` `rules.tasks`: tests are required for `src/` and skill scripts; notebooks are exploratory |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Q1 unresolved**: `results_dir("01_eda_reduction_clustering")` raises `ValueError` — the requested folder name is invalid | **Certain** if unaddressed | Apply phase fails on the first `ResultsWriter` call | Verified against the committed `src/nids/paths.py:19`. Recommended answer in **Q1**; needs a ruling before apply |
| `project-foundation` slices 3–5 not yet committed | Certain today | Apply cannot start | Explicit dependency below. Planning proceeds against the frozen design |
| The frozen API shifts during the concurrent foundation work | Medium | Notebook calls a signature that changed | Re-read `design.md` at the start of apply and diff the five consumed signatures before writing a cell |
| DBSCAN memory blow-up | Low, after **F3** | Notebook run dies | Bounded to ≤10,000 rows; `eps` from a stated rule rather than a guess |
| K-Means sweep is slower than expected (22 fits × `n_init=10` on ~108k × ~10) | Medium | Long execution, not failure | Acceptable for a one-shot `nbconvert` run. If it exceeds ~15 minutes, report it; do **not** silently switch to `MiniBatchKMeans`, which would change the reported inertia |
| Notebook + results push ~10 MB into the repository | Certain | Repository weight, diff churn | Quantified and bounded in **F4**; measured size reported at apply |
| Line count far exceeds the 400-line review budget | Certain | Reviewer load | The user has already chosen a **single slice, one commit**. Recorded as an accepted `size:exception`; the tasks phase must state the guard lines explicitly |
| Section 1 partly duplicates `results/data_cleaning/` content | Certain | Apparent redundancy | Intentional and contract-compliant: the output contract requires each producer folder to be interpretable in isolation. The cleaning report shows *before vs after cleaning*; this notebook describes the *cleaned train partition only*. The notebook says so |
| An estimate is read as a population fact | Medium | Wrong conclusions downstream | Three mandatory notes (`silhouette_is_subsample_estimate`, `dbscan_is_subsample_property`, `dbscan_eps_not_transferable`) plus the allocation table as evidence |

---

## Rollback Plan

The change is one commit adding files and modifying none.

1. **Full revert**: `git revert <sha>`. It removes `notebooks/01_eda_reduction_clustering.ipynb` and
   the entire `results/<notebook_id>/` folder. Nothing else in the repository is affected: no source
   module, no dependency, no test, no other producer's results folder. There is no migration to undo
   and no schema to downgrade.
2. **Outputs bad, notebook source fine**: delete `results/<notebook_id>/` and re-run the single
   `nbconvert` command. `ResultsWriter(prune_undeclared=True)` removes orphans from a previous schema
   on `close()`, so a partial rewrite cannot leave stale files the manifest does not declare.
3. **Run aborted mid-way**: `ResultsWriter` writes every artifact to `<path>.tmp` then `os.replace`s
   it, and writes `manifest.json` **last**. A crashed run leaves no manifest, which is the
   unambiguous signal that the folder is incomplete — no half-declared state to reason about.
4. **Blast radius if the notebook is simply wrong**: zero outside its own two paths. No downstream
   artifact consumes this folder yet.

---

## Dependencies

### Blocking — `project-foundation` slices 3, 4 and 5

`project-foundation` is in flight, delivered as five sequential commits. Committed today: **slice 1
only** (`d71e306` — `paths.py`, `columns.py`). Slice 2 (`loading.py`, `validation.py`) exists in the
worktree but is not committed.

| Needed | Delivered by | Status |
|---|---|---|
| `nids.data.load_clean_partitions` | slice 3 | **Not committed** |
| `nids.preprocessing.build_preprocessor` | slice 4 | **Not committed** |
| `nids.sampling.stratified_subsample` | slice 4 | **Not committed** |
| `nids.results.ResultsWriter` | slice 5 | **Not committed** |
| `nids.columns.feature_columns`, `nids.paths` | slice 1 | Committed (`d71e306`) |

**This change is therefore blocked on `project-foundation` slices 3–5 before its apply phase can
run.** Planning — proposal, spec, design, tasks — proceeds now against the API frozen in
`openspec/changes/project-foundation/design.md`, which is exactly what that document exists for. The
apply phase must re-verify the five consumed signatures against the committed source before writing
its first cell.

### Non-blocking

- **Raw data is present.** `data/raw/UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` both
  exist. The setup cell still calls `nids.validation.validate_raw_data()` first, so a truncated or
  replaced file fails loudly instead of halfway through section 3.
- **`notebooks/` and `results/` exist** (both carry `.gitkeep` from slice 1). No directory
  bootstrapping is required.
- **Jupyter and the kernel** are declared (`jupyterlab`, `ipykernel`) and the package is installed
  editable via `[build-system]` hatchling, so `import nids` resolves from a notebook without kernel
  registration.

---

## Success Criteria

Each is a check someone else can run, not a judgement.

- [ ] **Executes clean.** `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_reduction_clustering.ipynb` exits 0, and no cell carries an `error` output.
- [ ] **Manifest is complete.** `results/<notebook_id>/manifest.json` parses, and every `tables[].path` and `figures[].path` resolves to an existing file within that folder. (`ResultsWriter.write_manifest` enforces this at write time; a final notebook cell re-checks it so the notebook itself is the evidence.)
- [ ] **Inventory matches.** The manifest declares **17 tables and 16 figures**, or the inventory in this proposal has been updated with a stated reason.
- [ ] **No test-partition access.** `rg -n 'load_raw_test|\.test\b|test_df|X_test' notebooks/01_eda_reduction_clustering.ipynb` returns no match in a code cell. `.test` of the `CleaningResult` is never bound to a name.
- [ ] **No re-implementation.** `rg -n 'read_csv|duplicated\(|StandardScaler\(|OneHotEncoder\(|drop\(columns=|to_csv\(|savefig\(' notebooks/01_eda_reduction_clustering.ipynb` returns no match. Every export goes through `ResultsWriter`.
- [ ] **Label boundary proven at run time.** The two setup assertions (**F5**) are present and passed during the executed run.
- [ ] **Seed 42 everywhere.** `stratified_subsample` at its default seed; `PCA(random_state=42)`; `KMeans(random_state=42)`. No other seed literal appears.
- [ ] **Both TTL variants reported.** `variance_inflation_factors`, `pca_explained_variance`, `pca_components_for_variance`, `kmeans_k_sweep_metrics`, `cluster_profile_original_units` and `cluster_attack_cat_composition` each contain both `with_ttl` and `without_ttl` in their `variant` column, and section 6 states in prose whether the conclusions differ.
- [ ] **All six disclosure notes present** in `manifest.json`'s `notes` array.
- [ ] **English everywhere.** Every figure title, axis label, legend entry, table column name and markdown cell is in English.
- [ ] **Deterministic tables.** Re-running the notebook leaves `results/<notebook_id>/tables/*.csv` byte-identical; `manifest.json` differs only in `generated_at`.
- [ ] **Size reported.** The apply phase reports the measured size of the executed `.ipynb` and of `results/<notebook_id>/`.

---

## Effort Estimate

**Authored changed lines: 1,600 – 2,200; plan for ~1,900.**

Counted as `.ipynb` JSON source lines — code, markdown and per-cell scaffolding. Base64 output blobs
are excluded (each image is a single JSON string, so it contributes ~1 line and a lot of bytes), and
generated `results/` files are excluded as generated artifacts per the review-workload guard.

| Driver | Lines |
|---|---|
| ~70 cells × ~9 lines of nbformat scaffolding (`cell_type`, `id`, `metadata`, `outputs`, source brackets) | ~630 |
| 16 figures × ~18 lines each — English title, both axis labels, legend, log scaling, palette centring | ~290 |
| 17 tables × ~11 lines — assembly, long-format reshape, `sort_by` | ~190 |
| 33 `ResultsWriter` registrations × ~5 lines — `name`, `title`, `description` are required arguments, not optional decoration | ~165 |
| Narrative markdown — section 6 is a required deliverable, and every disclosure is prose | ~250 |
| Setup, guards, assertions, both pipelines, both subsamples | ~90 |
| K-Means sweep × 2 variants, DBSCAN, knee rule, VIF helper | ~110 |
| Cluster profiling against global means, both variants | ~60 |
| Metrics and notes registration (~26 calls) | ~80 |

**Why conservative.** The parent measured `project-foundation` slice 1 at roughly **2.4× its original
estimate**. The same pressures apply here and then some: (a) every result is produced twice for the
two TTL variants, and a loop over a variant dict is more lines than a single path, not fewer;
(b) every figure needs explicit English labelling, which is roughly a third of each plotting cell;
(c) `ResultsWriter`'s keyword-only `title`/`description` mean 33 artifacts cost ~165 lines of pure
registration before any analysis; (d) `.ipynb` is JSON, so the per-cell overhead is real and
unavoidable; (e) honest disclosure prose is a deliverable of this change, not padding.

**Review-budget consequence.** ~1,900 authored lines against a 400-line budget is roughly 4.75×. The
user has already chosen a **single slice, one commit** for this change, so this is an accepted
`size:exception` rather than an open question. The tasks phase must still emit the guard lines
explicitly, and should note that a notebook is reviewed section by section rather than line by line,
which is what makes a single slice tolerable here.

---

## Decisions returned to the orchestrator

These are product/naming decisions. They are not assumed, and **Q1 blocks apply.**

### Q1 — BLOCKING: the requested results folder name is invalid

`results/01_eda_reduction_clustering/` **cannot be created**. `src/nids/paths.py:19`, already
committed at `d71e306`, defines:

```python
NOTEBOOK_ID_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
```

and `results_dir()` raises `ValueError` before any filesystem call when the id does not match. A
leading digit is rejected. `ResultsWriter` validates artifact names against the same pattern, so
table and figure names may not start with a digit either.

Three options:

| Option | Consequence |
|---|---|
| **A (recommended)** — notebook file stays `notebooks/01_eda_reduction_clustering.ipynb`; results folder becomes `results/eda_reduction_clustering/` | Filename ordering preserved; consistent with the existing `data_cleaning` producer, which also carries no numeric prefix. No source change. |
| B — relax the pattern to `^[a-z0-9][a-z0-9_]{0,63}$` | Requires editing `src/nids/paths.py` and `src/nids/results.py`, both out of scope here and concurrently owned by `project-foundation`. Would have to be raised as a foundation change. |
| C — drop the numeric prefix from the notebook filename too | Loses the execution-order signal in `notebooks/`. |

**Recommendation: A.** Every occurrence of `<notebook_id>` in this proposal resolves to
`eda_reduction_clustering` under that answer.

### Q2 — t-SNE / UMAP stay out

The user allowed an argument for inclusion. **This proposal does not make one.** PCA answers the
dimensionality question and produces the space the clustering runs in; t-SNE has no out-of-sample
transform and would produce a picture feeding nothing downstream. Recorded so the omission is a
decision, not an oversight. Reverse it only if the orchestrator wants a visual-structure figure for
its own sake.

### Q3 — `statsmodels` is not added

See **F1**. VIF is computed from `1 / (1 - R²)` using `sklearn.linear_model.LinearRegression`. Flagged
because adding a dependency is a decision with a permanent cost and the user may prefer the canonical
library anyway.

### Q4 — Section 5 profiles both TTL variants

The user's section 5 did not specify whether profiling runs for both variants. This proposal runs
both, because "does removing the TTL shortcut change what the clusters represent" is precisely the
question the with/without requirement exists to answer, and answering it costs two extra columns in
two CSVs rather than two extra notebooks. Flagged in case the intent was with-TTL only.
