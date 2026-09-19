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
- [x] 1.1 Create `notebooks/02_clustering_reduction.ipynb` as nbformat v4.5 JSON with hand-assigned cell ids from the design outline (no jupytext).
- [x] 1.2 Cells `s0_00`–`s0_02`: title, the five questions, re-run command, compute-budget markdown table, imports, constants (`SEED=42`, `MAX_ROWS=10_000`, `FLOOR=50`, `SPECTRAL_MAX_ROWS=5_000`, `K_RANGE=range(2,13)`, `BOOTSTRAP_REPEATS=5`, `BOOTSTRAP_FRACTION=0.8`, `TRUSTWORTHINESS_ROWS=2_000`).

### 2. Data and guards
- [x] 2.1 Cells `s0_03`–`s0_08`: `load_clean_partitions().train` in one expression, `stratified_subsample`, `index_frame`, targets, `build_preprocessor_pair()` fitted per variant on `SUB[feature_columns(include_ttl)]`.
- [x] 2.2 Cell `s0_09_assertions`: the five runtime assertions (no test partition bound, targets absent from `feature_columns`, targets absent from `get_feature_names_out`, shape and finiteness, subsample parameters).
- [x] 2.3 Cell `s0_10_helpers`: `record_metric`, `record_note`, `finite` (non-finite → string `"inf"`/`"-inf"`/`"nan"`), `internal_metrics`, `new_figure` + `CLASS_COLOURS`, `select_eps`.
- [x] 2.4 Cell `s0_11_writer`: `ResultsWriter("clustering_reduction")`, tables `subsample_allocation`, `attack_cat_class_sizes`, `feature_space_shape`, and the four shape metrics.

### 3. Clustering in the full feature space
- [x] 3.1 K-Means sweep over `K_RANGE` × 2 variants; `kmeans_k_sweep_metrics`; `kmeans_internal_metrics_by_k`; `k_star` by argmax silhouette (ties → smaller k); both `kmeans_*` metrics.
- [x] 3.2 Ward: one `scipy` linkage per variant, `fcluster` per k, `MemoryError` fallback to 5,000 rows with note; `ward_k_sweep_metrics`; `ward_internal_metrics_by_k`; both `ward_*` metrics. (Fallback did not trigger this run — 10,000×~50 dims fit in memory.)
- [x] 3.3 DBSCAN: `min_samples = min(2*d, 50)`, `select_eps` with its three guards; `dbscan_parameters`, `dbscan_cluster_summary`, `dbscan_k_distance_plot`, four `dbscan_*` metrics, note `dbscan_eps_not_transferable`.
- [x] 3.4 Spectral on the nested 5,000-row subsample; `spectral_cluster_summary`, `spectral_cluster_sizes`, `spectral_rows`, `spectral_silhouette_{v}`, note `spectral_is_five_thousand_rows`.
- [x] 3.5 Combined `internal_metrics_by_method` table and figure; `best_method_{v}` by argmax silhouette. **Deviation**: the design's artifact inventory names both the table and the figure `internal_metrics_by_method`, but the committed `ResultsWriter._register_artifact_name` (src/nids/results.py) uses one shared name set across tables and figures, so a same-name table+figure pair raises `ValueError: Artifact name already registered` (confirmed by execution). Per "follow committed signatures over any design snippet if they differ", the figure was renamed to `internal_metrics_by_method_figure`; the table keeps the design's name `internal_metrics_by_method`. Manifest counts (16 tables, 15 figures) are unaffected.

### 4. Projections
- [x] 4.1 PCA, t-SNE and UMAP per variant, each embedding computed once and reused for both colourings.
- [x] 4.2 Six scatter figures (`{pca,tsne,umap}_projection_by_{cluster,attack_cat}`), English titles, axis and legend labels. **Note**: "by cluster" panels use the K-Means partition at `k*` (documented in markdown cell `s4_00_intro`) because it is the only method's labels defined over all 10,000 rows the projections were computed on; spectral clustering only labels its own nested 5,000-row subset and cannot colour the full projection without a shape mismatch.
- [x] 4.3 `projection_trustworthiness` on the 2,000-row nested slice.

### 5. External validation and stability
- [x] 5.1 ARI and NMI per (variant, method) against `attack_cat` and `label`; `external_validation_metrics`; `external_validation_by_method`; the three `best_*` metrics.
- [x] 5.2 `cluster_attack_cat_contingency`, `cluster_attack_cat_purity`, `cluster_attack_cat_heatmap` for the chosen method.
- [x] 5.3 Five seeded 80% resamples of K-Means at `k_star`; `stability_pairwise_ari` (10 pairs with shared-row counts), `stability_summary`, `stability_ari_distribution`, `stability_repeats`, `stability_mean_ari_{v}`.

### 6. Comparison and conclusions
- [x] 6.1 Read `results/eda_reduction_clustering/manifest.json` read-only inside `try/except`; populate `pipeline_comparison` and `pipeline_comparison_metrics`; missing values become `"not available"`, never an exception. (Notebook 01's manifest became available mid-run from the concurrent sibling agent, so the comparison table populated with real values; the `try/except` degradation path was exercised and verified in an earlier attempt before notebook 01 finished.)
- [x] 6.2 Register `notebook_01_comparison_available`, note `comparison_is_not_controlled`, and conditionally `notebook_01_comparison_unavailable` with the explicit re-run instruction. (Conditional note did not fire this run since notebook 01 was available; logic verified directly.)
- [x] 6.3 TTL delta section naming every place where dropping `sttl`/`ct_state_ttl` changes a conclusion, plus plain-language conclusions and limitations; remaining notes; `runtime_seconds_total`.
- [x] 6.4 Close: `counts.json` sidecar, `writer.close()`, print manifest path. (Pruned-file list is surfaced via the `logging.INFO` pruning log line configured at notebook start, not a separate print statement — `close()` does not return the pruned list.)

### 7. Execute and verify
- [x] 7.1 Run `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_clustering_reduction.ipynb`; record wall-clock time. **1m42s wall clock** (well under budget; two earlier attempts failed fast on an assertion bug and a naming collision, both fixed).
- [x] 7.2 Confirm wall-clock under ~15 minutes. Confirmed: 102s, ~9x under budget.
- [x] 7.3 Verify `results/clustering_reduction/manifest.json`: parses as JSON, contains no bare `Infinity`/`NaN` token, `notebook_id == "clustering_reduction"`, 16 tables + 15 figures present, every declared path resolves. All confirmed programmatically.
- [x] 7.4 Verify no name matches `^[0-9]` among artifact names and every name matches `^[a-z][a-z0-9_]{0,63}$`. Confirmed programmatically, zero violations.
- [x] 7.5 Confirm both assertion cells executed without error in the stored outputs. Confirmed — `s0_09_assertions` printed its pass message; no CellExecutionError in the final run.
- [x] 7.6 Confirm `git status` shows no change under `src/nids/`, `pyproject.toml`, `results/eda_reduction_clustering/`, or any other change folder. Confirmed — only `notebooks/02_clustering_reduction.ipynb` and `results/clustering_reduction/` are attributable to this change; other untracked paths (`notebooks/01_...`, `results/eda_reduction_clustering/`, `skills/`, `openspec/changes/skills/`) belong to concurrent sibling agents and were not touched.
- [x] 7.7 Confirm every markdown cell, chart title, axis label and manifest string is English. Confirmed by construction and spot-check.

### 8. Deliver
- [ ] 8.1 One commit on a feature branch: `feat(notebooks): add cluster-then-reduce notebook 02 and its results folder`. **Not performed by this agent** — HARD SCOPE for this run forbids git mutations; the parent orchestrator commits.

## Acceptance criteria

1. The notebook executes end to end from a cold kernel in one pass, under ~15 minutes on 16 threads.
2. No fit, `fit_transform` or `fit_predict` ever receives `attack_cat` or `label`; the assertion cell proves it.
3. The test partition is never bound to a name; the assertion cell proves it.
4. Every clustering call receives at most its declared cap (10,000 rows, spectral 5,000).
5. `results/clustering_reduction/` is complete, self-describing, and its manifest parses with no non-finite token.
6. The notebook completes successfully whether or not notebook 01 has produced its manifest.
