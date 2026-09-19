# Proposal: Notebook 02 — Cluster-then-Reduce

## Intent

Notebook 01 reduces first (PCA to 90% variance) and clusters the reduced space. This notebook does
the mirror: it clusters the **full preprocessed feature space** of the cleaned training partition and
only then projects to 2D for visual inspection. The pair answers one question that neither notebook
answers alone: **is the unsupervised structure of UNSW-NB15 a property of the data, or an artefact of
the reduction that preceded the clustering?**

Concretely it establishes, without ever fitting on a target:

1. Which of four algorithms (K-Means, Ward agglomerative, DBSCAN, spectral) recovers structure in the
   untouched feature space, scored by silhouette, Calinski-Harabasz and Davies-Bouldin.
2. How far that structure aligns with `attack_cat` and `label`, by ARI, NMI and a contingency table.
3. Whether the chosen partition is stable under resampling (pairwise ARI over bootstrap reruns).
4. Whether cluster-then-reduce and reduce-then-cluster reach the same conclusion on the same metrics.
5. Whether removing `sttl` / `ct_state_ttl` changes any of the above.

## Scope

### In scope

| # | Deliverable |
|---|---|
| 1 | `notebooks/02_clustering_reduction.ipynb`, nbformat v4.5 JSON with fixed cell ids, committed executed. |
| 2 | Clustering on the full preprocessed space of the seed-42 stratified subsample (max 10,000 rows, floor 50). |
| 3 | PCA / t-SNE / UMAP 2D projections, coloured by cluster and by `attack_cat`. |
| 4 | External validation (ARI, NMI, contingency, purity) and bootstrap stability (5 reruns). |
| 5 | A notebook-01 comparison table that degrades gracefully when notebook 01 has not run yet. |
| 6 | `results/clustering_reduction/` — 16 tables, 15 figures, ~22 metrics, ~8 notes, all via `ResultsWriter`. |
| 7 | Two variants (`with_ttl`, `without_ttl`) for every metric where the TTL shortcut can change a conclusion. |
| 8 | Runtime assertion cells proving the test partition is never bound and no target ever enters a fit. |

### Out of scope

- Any supervised model, any use of `cleaned.test`, any change to `src/nids/**`, `pyproject.toml`, or
  any other change folder. No new dependency (`statsmodels` is not installed and is not used).
- Rewriting notebook 01 or writing into `results/eda_reduction_clustering/` (read-only, contract rule).

## Approach

Read the cleaned **training** partition through `nids.data.load_clean_partitions().train`, subsample it
with `nids.sampling.stratified_subsample`, fit `nids.preprocessing.build_preprocessor_pair()` on that
subsample's `feature_columns(include_ttl)` slice, and cluster the resulting dense matrix directly. No
dimensionality reduction precedes any `fit`. `attack_cat` and `label` are held in separate arrays used
only for stratification, colouring, and post-hoc ARI/NMI/contingency.

## Decisions

- **D1** `notebook_id` is `clustering_reduction`; `02_...` is rejected by `^[a-z][a-z0-9_]{0,63}$`.
- **D2** K-range is `2..12`, this notebook's own choice, stated in the notebook: it brackets the 10
  `attack_cat` classes from both sides and matches notebook 01's grid so the comparison shares an axis.
- **D3** Everything clusters on the subsample; the subsample **is** the analysis population, not an estimator of it.
- **D4** Spectral clustering is capped at 5,000 rows, a nested stratified subsample of the same 10,000.
- **D5** Ward is computed once per variant as a `scipy` linkage matrix and cut per `k`, not refitted per `k`.
- **D6** DBSCAN `eps` comes from a deterministic normalised-chord knee on the sorted k-distance curve.
- **D7** Stability = 5 seeded 80% resamples, pairwise ARI on shared rows; 10 pairs reported.
- **D8** Notebook 01 comparison reads its `manifest.json` if present, otherwise degrades with a note.
- **D9** Non-finite metrics are registered as the string `"inf"`; `json.dumps` cannot emit `Infinity`.
- **D10** Single delivery slice, one commit; a notebook is reviewed section by section.

## Risks

| Risk | Mitigation |
|---|---|
| 15-minute budget | Explicit caps (§ design Compute Budget); Ward cut from one linkage; embeddings computed once and reused for both colourings. |
| Ward memory on 10,000 rows (~0.4 GB condensed) | Guarded: on `MemoryError` fall back to a 5,000-row nested subsample and register a note. |
| Notebook 01 unfinished | Comparison section is read-only, optional, and note-backed; it never raises. |
| DBSCAN degenerating to all-noise in ~55 dimensions | Reported as a finding with the noise fraction, not hidden; `eps` rule and its fallbacks are stated. |
