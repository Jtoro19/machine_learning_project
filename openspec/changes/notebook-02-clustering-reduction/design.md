# Design: Notebook 02 — Cluster-then-Reduce

`NOTEBOOK_ID = "clustering_reduction"` → `results/clustering_reduction/`.
File: `notebooks/02_clustering_reduction.ipynb`, nbformat v4.5 JSON, hand-assigned cell ids, ~64 cells.
Run: `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_clustering_reduction.ipynb`

## Constants (cell `s0_02_config`)

```python
SEED = 42; MAX_ROWS = 10_000; FLOOR = 50; SPECTRAL_MAX_ROWS = 5_000
K_RANGE = range(2, 13)          # own choice: brackets the 10 attack_cat classes; matches nb01's grid
BOOTSTRAP_REPEATS = 5; BOOTSTRAP_FRACTION = 0.8; TRUSTWORTHINESS_ROWS = 2_000
VARIANTS = ("with_ttl", "without_ttl")
```

## Compute budget (stated in markdown cell `s0_01_budget` and enforced in code)

| Step | Rows | Cap mechanism | Est. |
|---|---|---|---|
| Preprocess ×2 variants | 10,000 | subsample | <5 s |
| K-Means sweep 11 k × 2 | 10,000 | `n_init=10, random_state=42` (KMeans has no `n_jobs` in sklearn 1.9; threads come from BLAS/OMP) | ~30 s |
| Ward | 10,000 | ONE `scipy.cluster.hierarchy.linkage(X, method="ward")` per variant, `fcluster(..., criterion="maxclust")` per k | ~2 min |
| DBSCAN + k-distance | 10,000 | `NearestNeighbors(n_jobs=-1)`, `DBSCAN(n_jobs=-1)` | ~30 s |
| Spectral | **5,000** | nested `stratified_subsample`, `affinity="nearest_neighbors"`, `n_jobs=-1` | ~2 min |
| PCA / t-SNE / UMAP ×2 | 10,000 | `TSNE(n_jobs=-1, init="pca", learning_rate="auto")`; UMAP `random_state=42` (single-threaded by design, accepted for determinism) | ~4 min |
| Silhouette / CH / DB | ≤10,000 | direct; 1e8 pairs is seconds | ~40 s |
| Trustworthiness | **2,000** | nested slice of the subsample | ~15 s |
| Bootstrap stability | 5 × 8,000 | K-Means only | ~30 s |

Target: **< 15 min on 16 threads.** `n_jobs=-1` is passed wherever the estimator accepts it; the table
names the two that do not (`KMeans`, `AgglomerativeClustering`/`linkage`) so their absence is not a bug.

## Data path (cells `s0_03`–`s0_08`)

```python
train = load_clean_partitions().train.reset_index(drop=True)   # no name ever binds CleaningResult or .test
sub   = stratified_subsample(train, stratify_by="attack_cat", max_rows=MAX_ROWS, floor=FLOOR, seed=SEED)
SUB   = sub.data
y_cat = SUB["attack_cat"].to_numpy(); y_bin = SUB["label"].to_numpy()   # post-hoc only
index_frame = SUB[["attack_cat", "label"]].assign(position=np.arange(len(SUB)))
pipes = build_preprocessor_pair()                              # {"with_ttl": Pipeline, "without_ttl": Pipeline}
X = {}
for v in VARIANTS:
    cols = feature_columns(include_ttl=(v == "with_ttl"))
    pipes[v].fit(SUB[cols])                                    # frame named at the call site
    X[v] = pipes[v].transform(SUB[cols])
```

`remainder="drop"` plus the explicit `cols` slice is the double guarantee that no target reaches a fit.
The subsample **is** the analysis population (note `subsample_is_the_analysis_population`); nothing here
is an estimate of a full-partition quantity.

### Assertion cell `s0_09_assertions`

1. `assert "cleaned" not in globals() and "test" not in globals()`; assert no global `DataFrame` other
   than `train`, `SUB`, `index_frame` and the exported tables.
2. For both variants: `assert set(cols).isdisjoint({"attack_cat", "label"})`.
3. For both variants: `assert not ({"attack_cat", "label"} & set(pipes[v].get_feature_names_out()))`.
4. `assert X[v].shape[0] == len(SUB)` and `np.isfinite(X[v]).all()`.
5. `assert sub.seed == SEED and sub.requested_max_rows == MAX_ROWS and sub.floor == FLOOR`.

## Helpers (cell `s0_10_helpers`)

- `record_metric(name, value, description)` / `record_note(note_id, text)` — write to one ordered dict
  **and** to `ResultsWriter`, so `counts.json` (`writer.write_json`, unregistered sidecar) cannot drift.
- `finite(value)` → `float(value)` when `np.isfinite`, else `"inf"` / `"-inf"` / `"nan"` as a **string**.
  Applied to every `add_metric` call and to every metric column before `add_table` (D9).
- `internal_metrics(matrix, labels)` → `dict(silhouette, calinski_harabasz, davies_bouldin, n_clusters,
  n_rows_scored)`; returns all-`"nan"` when `len(np.unique(labels[labels >= 0])) < 2`.
- `new_figure(...)` → `(fig, ax)` with the shared style; `CLASS_COLOURS` built once from
  `sorted(SUB["attack_cat"].unique())` with `tab10` (guard: fall back to `tab20` above 10 classes) so a
  class keeps one colour in every figure. Clusters use `viridis` sampled at `n_clusters`.
- `select_eps(sample, min_samples)` — see below.

## DBSCAN `eps` rule (deterministic, cell `s3_01_eps`)

`min_samples = min(2 * X[v].shape[1], 50)` — the classic `2·d` rule, capped at 50 so a ~55-dimensional
one-hot space does not demand a neighbourhood larger than any real density pocket. Registered per variant.

```python
k = min_samples
distances, _ = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(sample).kneighbors(sample)
kdist = np.sort(distances[:, k - 1]); m = kdist.size
x = np.arange(m, dtype=np.float64) / (m - 1)
span = kdist[-1] - kdist[0]
if span == 0: return float(np.median(kdist)), "degenerate_flat_curve"
y = (kdist - kdist[0]) / span
dx, dy = x[-1] - x[0], y[-1] - y[0]
perp = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / np.hypot(dx, dy)
eps = float(kdist[int(np.argmax(perp))])                       # first maximum: deterministic tie-break
if eps <= 0:
    positive = kdist[kdist > 0]
    if positive.size == 0: return 0.0, "no_positive_distance"   # DBSCAN skipped, metrics = "nan"
    return float(positive.min()), "degenerate_zero_knee"
return eps, "chord"
```

Both axes are normalised to `[0, 1]` first, otherwise the chord is nearly horizontal and the rule
silently becomes "largest vertical gap". `rule_used` is exported in `dbscan_parameters`.

## Ward, spectral, stability

- **Ward**: `Z[v] = scipy.cluster.hierarchy.linkage(X[v], method="ward")` once per variant; each `k` is
  `fcluster(Z[v], t=k, criterion="maxclust")`. Guard: `except MemoryError` → recompute on a nested
  5,000-row `stratified_subsample(index_frame, max_rows=5000, floor=FLOOR, seed=SEED)` and register note
  `ward_reduced_to_five_thousand_rows`.
- **Spectral**: `spec_pos = stratified_subsample(index_frame, max_rows=SPECTRAL_MAX_ROWS, floor=FLOOR,
  seed=SEED).data["position"].to_numpy()`; `SpectralClustering(n_clusters=k_star[v], affinity=
  "nearest_neighbors", n_neighbors=10, assign_labels="kmeans", random_state=SEED, n_jobs=-1).fit_predict(X[v][spec_pos])`.
  Its metrics are scored on those 5,000 rows only; `n_rows_scored` records it and note
  `spectral_is_five_thousand_rows` states it.
- **Chosen method** per variant: `argmax silhouette` over the four methods, ties broken by method name
  ascending; registered as `best_method_{v}`. `k_star[v]` = `argmax silhouette` over `K_RANGE` for K-Means,
  ties broken by smaller `k`.
- **Stability**: for `r in range(BOOTSTRAP_REPEATS)` draw `rng = np.random.default_rng(SEED + r)`,
  `rows_r = rng.choice(len(SUB), size=int(0.8*len(SUB)), replace=False)`, fit
  `KMeans(n_clusters=k_star[v], n_init=10, random_state=SEED)`; for each of the 10 unordered pairs compute
  `adjusted_rand_score` on `np.intersect1d(rows_a, rows_b)`. Resampling without replacement on a fixed
  fraction is what makes the intersection large and the ARI meaningful; stated in the markdown.

## Notebook-01 comparison (cells `s6_01`–`s6_03`) — graceful degradation

```python
nb1 = paths.results_root() / "eda_reduction_clustering" / "manifest.json"   # read-only, never written
available = nb1.is_file()
metrics_nb1 = {}
if available:
    try:
        payload = json.loads(nb1.read_text(encoding="utf-8"))
        metrics_nb1 = {entry["name"]: entry["value"] for entry in payload.get("metrics", [])}
    except (json.JSONDecodeError, KeyError, OSError) as error:
        available = False; reason = f"{type(error).__name__}: {error}"
```

Wanted keys per variant: `kmeans_selected_k_{v}`, `kmeans_silhouette_at_selected_k_{v}`. Any missing key
yields the string `"not available"` in `pipeline_comparison`, never a `KeyError`. When `available` is
false the notebook prints a clear message — *"Notebook 01 has not produced results/eda_reduction_clustering/manifest.json yet; run notebook 01 and re-execute this notebook to populate the comparison."* — registers
metric `notebook_01_comparison_available = "no"` and note `notebook_01_comparison_unavailable`, and
continues. The comparison **never** raises and never mutates notebook 01's folder.

Note `comparison_is_not_controlled` (always registered): notebook 01 clusters the **full** partition in
PCA-90 space with silhouette measured on a subsample; this notebook clusters a **10,000-row subsample**
in the full feature space with silhouette measured on the same rows. Different populations, different
spaces — the comparison is directional evidence, not a controlled experiment.

## Two-variant TTL structure

Every table carries a `variant` column; every swept figure overlays the two variants as two series with
one name and no suffix; every scalar metric carries a `_{v}` suffix, symmetrically for both variants even
when only one is interesting. Section 7 computes the deltas (`silhouette`, `ari_attack_cat`,
`nmi_attack_cat`, chosen `k`, chosen method) and names in plain English each place where dropping
`sttl`/`ct_state_ttl` changes the conclusion rather than only the number.

## Cell outline

| Cell id | Kind | Content |
|---|---|---|
| `s0_00_title` … `s0_02_config` | md/code | Title, the five questions, how to re-run, compute budget; imports; constants |
| `s0_03`–`s0_08` | code | Load train, subsample, `index_frame`, targets, fit both pipelines, shapes |
| `s0_09_assertions` | code | The five assertions above |
| `s0_10_helpers` | code | `record_metric`, `record_note`, `finite`, `internal_metrics`, `new_figure`, `select_eps` |
| `s0_11_writer` | code | `writer = ResultsWriter(NOTEBOOK_ID)`; print `writer.directory`; tables `subsample_allocation` (`sub.to_frame()`), `attack_cat_class_sizes`, `feature_space_shape`; metrics `subsample_rows`, `subsample_classes`, `features_with_ttl`, `features_without_ttl` |
| `s1_00`–`s1_05` | md/code | K-Means sweep over `K_RANGE` × 2 variants; table `kmeans_k_sweep_metrics`; figure `kmeans_internal_metrics_by_k`; select `k_star`; metrics `kmeans_selected_k_{v}`, `kmeans_silhouette_at_selected_k_{v}` |
| `s2_00`–`s2_03` | md/code | Ward linkage once per variant, cut per `k`; table `ward_k_sweep_metrics`; figure `ward_internal_metrics_by_k`; metrics `ward_selected_k_{v}`, `ward_silhouette_at_selected_k_{v}` |
| `s3_00`–`s3_04` | md/code | `select_eps`; tables `dbscan_parameters`, `dbscan_cluster_summary`; figure `dbscan_k_distance_plot`; metrics `dbscan_eps_{v}`, `dbscan_min_samples_{v}`, `dbscan_cluster_count_{v}`, `dbscan_noise_fraction_{v}`; note `dbscan_eps_not_transferable` |
| `s3_05`–`s3_07` | md/code | Spectral on 5,000 rows; table `spectral_cluster_summary`; figure `spectral_cluster_sizes`; metrics `spectral_rows`, `spectral_silhouette_{v}`; note `spectral_is_five_thousand_rows` |
| `s3_08`–`s3_09` | code | Table `internal_metrics_by_method`; figure `internal_metrics_by_method`; metric `best_method_{v}` |
| `s4_00`–`s4_06` | md/code | PCA / t-SNE / UMAP per variant, each embedding computed **once** and reused for both colourings; six scatter figures; table `projection_trustworthiness` |
| `s5_00`–`s5_05` | md/code | ARI/NMI vs `attack_cat` and `label`; tables `external_validation_metrics`, `cluster_attack_cat_contingency`, `cluster_attack_cat_purity`; figures `external_validation_by_method`, `cluster_attack_cat_heatmap`; metrics `best_ari_attack_cat_{v}`, `best_nmi_attack_cat_{v}`, `best_ari_label_{v}` |
| `s5_06`–`s5_08` | md/code | Bootstrap stability; tables `stability_pairwise_ari`, `stability_summary`; figure `stability_ari_distribution`; metrics `stability_repeats`, `stability_mean_ari_{v}` |
| `s6_01`–`s6_03` | md/code | Notebook-01 manifest read, table `pipeline_comparison`, figure `pipeline_comparison_metrics`; metric `notebook_01_comparison_available`; notes `comparison_is_not_controlled`, conditional `notebook_01_comparison_unavailable` |
| `s7_00`–`s7_03` | md/code | TTL deltas, plain-language conclusions, limitations; notes `targets_never_fitted`, `test_partition_never_loaded`, `subsample_is_the_analysis_population`, `infinite_metrics_as_string`; metric `runtime_seconds_total` |
| `s8_00_close` | code | `counts.json` via `writer.write_json`; `writer.close()`; print the manifest path and the pruned list |

## ResultsWriter inventory

**Tables** (`add_table`, `sort_by` in brackets): `subsample_allocation` "Stratified subsample allocation per attack category" [`label`] · `attack_cat_class_sizes` "Attack category sizes in the subsample" [`attack_cat`] · `feature_space_shape` "Preprocessed feature space shape per variant" [`variant`] · `kmeans_k_sweep_metrics` "K-Means internal metrics by number of clusters" [`variant`,`k`] · `ward_k_sweep_metrics` "Ward agglomerative internal metrics by number of clusters" [`variant`,`k`] · `dbscan_parameters` "DBSCAN neighbourhood parameters and the rule that selected them" [`variant`] · `dbscan_cluster_summary` "DBSCAN cluster sizes including the noise label" [`variant`,`cluster`] · `spectral_cluster_summary` "Spectral cluster sizes on the 5,000-row subsample" [`variant`,`cluster`] · `internal_metrics_by_method` "Internal validation metrics for all four methods" [`variant`,`method`] · `external_validation_metrics` "Adjusted Rand Index and Normalised Mutual Information against both references" [`variant`,`method`,`reference`] · `cluster_attack_cat_contingency` "Cluster by attack category contingency table for the chosen method" [`variant`,`cluster`] · `cluster_attack_cat_purity` "Dominant attack category and purity per cluster" [`variant`,`cluster`] · `stability_pairwise_ari` "Pairwise Adjusted Rand Index between bootstrap reruns" [`variant`,`run_a`,`run_b`] · `stability_summary` "Bootstrap stability summary per variant" [`variant`] · `projection_trustworthiness` "Trustworthiness of each 2D projection" [`variant`,`projection`] · `pipeline_comparison` "Reduce-then-cluster against cluster-then-reduce on shared metrics" [`pipeline`,`variant`,`metric`].

**Figures** (`add_figure`, all titles/labels English, dpi 150):

| Name | Type | x | y | Colour | Scale |
|---|---|---|---|---|---|
| `kmeans_internal_metrics_by_k` | 3 stacked line+marker panels | Number of clusters k | Silhouette / Calinski-Harabasz / Davies-Bouldin | two-colour variant pair, selected k ringed | linear |
| `ward_internal_metrics_by_k` | 3 stacked line+marker panels | Number of clusters k | same three | two-colour variant pair | linear |
| `dbscan_k_distance_plot` | 2 sorted curves + dashed eps lines | Points sorted by k-distance | Distance to the k-th nearest neighbour | two-colour variant pair | linear |
| `spectral_cluster_sizes` | grouped bars | Cluster | Rows | two-colour variant pair | linear |
| `internal_metrics_by_method` | 3 grouped-bar panels | Clustering method | Silhouette / Calinski-Harabasz / Davies-Bouldin | two-colour variant pair | linear; CH panel log if max/min > 100 |
| `pca_projection_by_cluster` | 2 scatter panels (one per variant) | First principal component | Second principal component | viridis by cluster | linear |
| `pca_projection_by_attack_cat` | 2 scatter panels | First principal component | Second principal component | `CLASS_COLOURS` by attack category | linear |
| `tsne_projection_by_cluster` | 2 scatter panels | t-SNE dimension 1 | t-SNE dimension 2 | viridis by cluster | linear |
| `tsne_projection_by_attack_cat` | 2 scatter panels | t-SNE dimension 1 | t-SNE dimension 2 | `CLASS_COLOURS` | linear |
| `umap_projection_by_cluster` | 2 scatter panels | UMAP dimension 1 | UMAP dimension 2 | viridis by cluster | linear |
| `umap_projection_by_attack_cat` | 2 scatter panels | UMAP dimension 1 | UMAP dimension 2 | `CLASS_COLOURS` | linear |
| `external_validation_by_method` | 4 grouped-bar panels (method × reference) | Clustering method | Score | ARI and NMI as two bar series | linear, 0–1 |
| `cluster_attack_cat_heatmap` | row-normalised heatmap, 2 panels | Attack category | Cluster | sequential `magma`, annotated | 0–1 |
| `stability_ari_distribution` | box + jittered strip | Variant | Pairwise Adjusted Rand Index | two-colour variant pair | linear, 0–1 |
| `pipeline_comparison_metrics` | grouped bars, or one centred "Notebook 01 results not available" annotation | Metric | Value | reduce-then-cluster vs cluster-then-reduce | linear |

**Metrics** (all through `finite`): `subsample_rows`, `subsample_classes`, `features_with_ttl`, `features_without_ttl`, `spectral_rows`, `stability_repeats`, `notebook_01_comparison_available`, `runtime_seconds_total`, and per variant `kmeans_selected_k_{v}`, `kmeans_silhouette_at_selected_k_{v}`, `ward_selected_k_{v}`, `ward_silhouette_at_selected_k_{v}`, `dbscan_eps_{v}`, `dbscan_min_samples_{v}`, `dbscan_cluster_count_{v}`, `dbscan_noise_fraction_{v}`, `spectral_silhouette_{v}`, `best_method_{v}`, `best_ari_attack_cat_{v}`, `best_nmi_attack_cat_{v}`, `best_ari_label_{v}`, `stability_mean_ari_{v}`.

**Notes**: `targets_never_fitted` · `test_partition_never_loaded` · `subsample_is_the_analysis_population` · `dbscan_eps_not_transferable` · `spectral_is_five_thousand_rows` · `comparison_is_not_controlled` · `infinite_metrics_as_string` · conditional `notebook_01_comparison_unavailable`, `ward_reduced_to_five_thousand_rows`.

## Reproducibility

`results/clustering_reduction/tables/*.csv`, `figures/*.png` and `counts.json` are byte-identical across
reruns; `manifest.json` differs only in `generated_at`. The `.ipynb` itself is expected to differ in its
base64 output blobs — stated, not implied. UMAP with `random_state=42` is deterministic at the cost of
single-threading; t-SNE, K-Means, spectral and every subsample are seeded at 42.

## Files

| Path | Action |
|---|---|
| `notebooks/02_clustering_reduction.ipynb` | Create — the deliverable, committed executed |
| `results/clustering_reduction/**` | Create (generated, committed) — 16 tables, 15 figures, manifest, `counts.json` |
| `src/nids/**`, `pyproject.toml`, `results/eda_reduction_clustering/**`, other change folders | **Untouched** |
