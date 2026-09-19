# Tasks: Notebook 02 — Cluster-then-Reduce

> Unattended run. No user review checkpoint before `sdd-apply`. **One delivery slice, one commit**,
> accepted `size:exception` (a notebook is reviewed section by section). Strict TDD does not apply:
> `openspec/config.yaml` marks notebooks exploratory, no pytest tests required. `src/nids/**`,
> `pyproject.toml` and every other change folder are owned by concurrent agents — read-only here.
> `results/eda_reduction_clustering/` is read-only by output-contract rule.

## Review Workload Forecast

| Field | Value |
|---|---|
| Estimated changed lines | ~1,500 authored lines (notebook JSON + generated results) |
| 400-line budget risk | High — accepted |
| Chained PRs recommended | No |
| Delivery strategy | single-pr, `size:exception` (already settled) |

Decision needed before apply: No

## Authorized edit roots

- `notebooks/`
- `results/clustering_reduction/`
- `openspec/changes/notebook-02-clustering-reduction/`

## Checklist

### 1. Scaffold
- [ ] 1.1 Create `notebooks/02_clustering_reduction.ipynb` as nbformat v4.5 JSON with hand-assigned cell ids from the design outline (no jupytext).
- [ ] 1.2 Cells `s0_00`–`s0_02`: title, the five questions, re-run command, compute-budget markdown table, imports, constants (`SEED=42`, `MAX_ROWS=10_000`, `FLOOR=50`, `SPECTRAL_MAX_ROWS=5_000`, `K_RANGE=range(2,13)`, `BOOTSTRAP_REPEATS=5`, `BOOTSTRAP_FRACTION=0.8`, `TRUSTWORTHINESS_ROWS=2_000`).

### 2. Data and guards
- [ ] 2.1 Cells `s0_03`–`s0_08`: `load_clean_partitions().train` in one expression, `stratified_subsample`, `index_frame`, targets, `build_preprocessor_pair()` fitted per variant on `SUB[feature_columns(include_ttl)]`.
- [ ] 2.2 Cell `s0_09_assertions`: the five runtime assertions (no test partition bound, targets absent from `feature_columns`, targets absent from `get_feature_names_out`, shape and finiteness, subsample parameters).
- [ ] 2.3 Cell `s0_10_helpers`: `record_metric`, `record_note`, `finite` (non-finite → string `"inf"`/`"-inf"`/`"nan"`), `internal_metrics`, `new_figure` + `CLASS_COLOURS`, `select_eps`.
- [ ] 2.4 Cell `s0_11_writer`: `ResultsWriter("clustering_reduction")`, tables `subsample_allocation`, `attack_cat_class_sizes`, `feature_space_shape`, and the four shape metrics.

### 3. Clustering in the full feature space
- [ ] 3.1 K-Means sweep over `K_RANGE` × 2 variants; `kmeans_k_sweep_metrics`; `kmeans_internal_metrics_by_k`; `k_star` by argmax silhouette (ties → smaller k); both `kmeans_*` metrics.
- [ ] 3.2 Ward: one `scipy` linkage per variant, `fcluster` per k, `MemoryError` fallback to 5,000 rows with note; `ward_k_sweep_metrics`; `ward_internal_metrics_by_k`; both `ward_*` metrics.
- [ ] 3.3 DBSCAN: `min_samples = min(2*d, 50)`, `select_eps` with its three guards; `dbscan_parameters`, `dbscan_cluster_summary`, `dbscan_k_distance_plot`, four `dbscan_*` metrics, note `dbscan_eps_not_transferable`.
- [ ] 3.4 Spectral on the nested 5,000-row subsample; `spectral_cluster_summary`, `spectral_cluster_sizes`, `spectral_rows`, `spectral_silhouette_{v}`, note `spectral_is_five_thousand_rows`.
- [ ] 3.5 Combined `internal_metrics_by_method` table and figure; `best_method_{v}` by argmax silhouette.

### 4. Projections
- [ ] 4.1 PCA, t-SNE and UMAP per variant, each embedding computed once and reused for both colourings.
- [ ] 4.2 Six scatter figures (`{pca,tsne,umap}_projection_by_{cluster,attack_cat}`), English titles, axis and legend labels.
- [ ] 4.3 `projection_trustworthiness` on the 2,000-row nested slice.

### 5. External validation and stability
- [ ] 5.1 ARI and NMI per (variant, method) against `attack_cat` and `label`; `external_validation_metrics`; `external_validation_by_method`; the three `best_*` metrics.
- [ ] 5.2 `cluster_attack_cat_contingency`, `cluster_attack_cat_purity`, `cluster_attack_cat_heatmap` for the chosen method.
- [ ] 5.3 Five seeded 80% resamples of K-Means at `k_star`; `stability_pairwise_ari` (10 pairs with shared-row counts), `stability_summary`, `stability_ari_distribution`, `stability_repeats`, `stability_mean_ari_{v}`.

### 6. Comparison and conclusions
- [ ] 6.1 Read `results/eda_reduction_clustering/manifest.json` read-only inside `try/except`; populate `pipeline_comparison` and `pipeline_comparison_metrics`; missing values become `"not available"`, never an exception.
- [ ] 6.2 Register `notebook_01_comparison_available`, note `comparison_is_not_controlled`, and conditionally `notebook_01_comparison_unavailable` with the explicit re-run instruction.
- [ ] 6.3 TTL delta section naming every place where dropping `sttl`/`ct_state_ttl` changes a conclusion, plus plain-language conclusions and limitations; remaining notes; `runtime_seconds_total`.
- [ ] 6.4 Close: `counts.json` sidecar, `writer.close()`, print manifest path and pruned files.

### 7. Execute and verify
- [ ] 7.1 Run `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_clustering_reduction.ipynb`; record wall-clock time.
- [ ] 7.2 Confirm wall-clock under ~15 minutes; if exceeded, report the measured time and the dominant step rather than silently trimming scope.
- [ ] 7.3 Verify `results/clustering_reduction/manifest.json`: parses as JSON, contains no `Infinity`/`NaN` token, `notebook_id == "clustering_reduction"`, 16 tables + 15 figures present, every declared path resolves.
- [ ] 7.4 Verify no name matches `^[0-9]` among artifact names and every name matches `^[a-z][a-z0-9_]{0,63}$`.
- [ ] 7.5 Confirm both assertion cells executed without error in the stored outputs.
- [ ] 7.6 Confirm `git status` shows no change under `src/nids/`, `pyproject.toml`, `results/eda_reduction_clustering/`, or any other change folder.
- [ ] 7.7 Confirm every markdown cell, chart title, axis label and manifest string is English.

### 8. Deliver
- [ ] 8.1 One commit on a feature branch: `feat(notebooks): add cluster-then-reduce notebook 02 and its results folder`.

## Acceptance criteria

1. The notebook executes end to end from a cold kernel in one pass, under ~15 minutes on 16 threads.
2. No fit, `fit_transform` or `fit_predict` ever receives `attack_cat` or `label`; the assertion cell proves it.
3. The test partition is never bound to a name; the assertion cell proves it.
4. Every clustering call receives at most its declared cap (10,000 rows, spectral 5,000).
5. `results/clustering_reduction/` is complete, self-describing, and its manifest parses with no non-finite token.
6. The notebook completes successfully whether or not notebook 01 has produced its manifest.
