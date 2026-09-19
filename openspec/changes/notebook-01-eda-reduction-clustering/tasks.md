# Tasks: Notebook 01 — EDA, Dimensionality Reduction and Clustering

> Unattended execution. There is no user review checkpoint between this task list and `sdd-apply`.
> Delivery is a single slice, one commit, accepted `size:exception` (~1,900 authored lines against the
> 400-line budget) — already settled by the user; do not re-ask. Strict TDD is disabled for this
> project; `openspec/config.yaml` `rules.tasks` states notebooks are exploratory and require no pytest
> tests. `src/nids/**` and `openspec/changes/project-foundation/**` are owned by another concurrently
> running agent — every task below either produces files under `notebooks/` and
> `results/eda_reduction_clustering/`, or reads `src/nids/**` / `openspec/changes/project-foundation/**`
> read-only.

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 1,600 – 2,200 authored lines; plan for ~1,900 (proposal Effort Estimate) |
| 400-line budget risk | High |
| Chained PRs recommended | No |
| Suggested split | Single PR / single commit |
| Delivery strategy | ask-on-risk (session default) — already resolved by the user to `size:exception` for this change |
| Chain strategy | size-exception |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: size-exception
400-line budget risk: High

The user has already accepted a single-slice, one-commit delivery with `size:exception` for this
change (session context, delivery strategy). A notebook is reviewed section by section rather than
line by line, which is what makes the single slice tolerable (proposal, Effort Estimate). No chain
split is proposed because the 73 cells are one coherent, cross-referencing artifact (shared
`VariantSpace`, shared `writer`, shared helper functions defined once in `s0_09_helpers`) that cannot
be partially committed without leaving `results/eda_reduction_clustering/manifest.json` incomplete —
the very failure mode `ResultsWriter.close()` is designed to prevent (Decision 3).

### Suggested Work Units

| Unit | Goal | Likely PR | Focused test command | Runtime harness | Rollback boundary |
|------|------|-----------|----------------------|-----------------|-------------------|
| 1 | Author all ~73 cells of `notebooks/01_eda_reduction_clustering.ipynb`, execute it twice (Decision 13), and commit it together with the generated `results/eda_reduction_clustering/` folder | Single commit (size:exception) | `uv run pytest` — must stay green; this change adds no new tests | `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/01_eda_reduction_clustering.ipynb` — must exit 0 with no `error` output cells | `git revert <sha>` removes `notebooks/01_eda_reduction_clustering.ipynb` and the entire `results/eda_reduction_clustering/` folder only; no source module, dependency, test, or other producer's `results/` folder is affected (proposal, Rollback Plan) |

---

## Requirement legend

Every task below cites one or more of these 13 requirements from
`openspec/changes/notebook-01-eda-reduction-clustering/specs/eda-reduction-clustering/spec.md`:

| ID | Requirement (title) |
|----|----------------------|
| Req1 | The notebook consumes the `nids` package boundary and never re-implements cleaning or preprocessing |
| Req2 | Only the cleaned training partition is used; the test partition is never read or referenced |
| Req3 | `attack_cat` and `label` never enter any model fit; `attack_cat` MAY be used post-hoc only |
| Req4 | The notebook respects the AGENTS.md data rules it consumes but does not implement |
| Req5 | The results folder identity is `eda_reduction_clustering` |
| Req6 | Descriptive statistics are reported for the cleaned training partition |
| Req7 | Correlation analysis and multicollinearity (VIF) are reported |
| Req8 | Principal Component Analysis quantifies redundancy in the feature space |
| Req9 | K-Means and DBSCAN clustering are reported on the 90%-variance PCA space |
| Req10 | Cluster profiling reports original-scale feature means and class composition, excluding ARI/NMI |
| Req11 | Dual with/without-TTL reporting applies wherever the toggle can change a conclusion |
| Req12 | All exported artifacts follow the results-output-contract and are reproducible |
| Req13 | The notebook executes end to end in a single, verifiable pass |

**Note on Q1.** The orchestrator has resolved Q1 under option A: the committed
`NOTEBOOK_ID_PATTERN = ^[a-z][a-z0-9_]{0,63}$` (`src/nids/paths.py:19`, read-only) rejects a leading
digit, verified by an actual `ValueError`. This is final, not provisional. `NOTEBOOK_ID =
"eda_reduction_clustering"`, results at `results/eda_reduction_clustering/`, notebook file stays
`notebooks/01_eda_reduction_clustering.ipynb`.

## Global constraints (apply to every authoring task in Phases 1–7)

These are not repeated on every checkbox; they bind the whole notebook and are checked once in
Phase 8.

- Author `notebooks/01_eda_reduction_clustering.ipynb` directly as hand-written nbformat v4.5 JSON
  (`"nbformat_minor": 5`, load-bearing so `id` fields survive re-execution). No jupytext conversion,
  no committed generator script (Decision 1). A throwaway assembly script MAY live in the session
  scratchpad only, never in the repository.
- Cell id convention `s<section>_<two-digit ordinal>_<slug>`, exactly as listed in each phase below.
  Ids are written once and never regenerated.
- Every code cell starts with `"execution_count": null` and `"outputs": []` before the first
  execution.
- No new dependency. Imports are limited to `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
  (`PCA`, `KMeans`, `DBSCAN`, `NearestNeighbors`, `LinearRegression`, `silhouette_score`,
  `calinski_harabasz_score`), `dataclasses`, and `nids`. `statsmodels` and `jupytext` are never added.
- Seed `42` is the only seed/`random_state` literal anywhere in the notebook (Req4).
- Forbidden patterns anywhere in a code cell (Req1): `read_csv`, `.duplicated(`, `StandardScaler(`,
  `OneHotEncoder(`, `drop(columns=` **on a partition frame**, `np.log1p` on a partition column, any
  `"-"` → `"none"` mapping, any rare-category threshold, `to_csv(`, `savefig(` outside
  `ResultsWriter`. The VIF loop's local `others = design[[c for c in usable if c != column]]` is
  positional selection, never `.drop(columns=`, so the `drop(columns=` grep stays literal with no
  footnote (Decision 5 / R4).
- Every figure-producing cell ends with `plt.close(fig)` as its last statement (never a bare `fig`).
- Every `ResultsWriter.add_table` call supplies `sort_by` with an ascending key, on a
  `reset_index()`ed frame.
- Every figure title, axis label, legend entry, table column name and markdown cell is in English
  (AGENTS.md, Req12).
- `attack_cat` and `label` never reach a `.fit(X, y)` target and never enter a feature matrix passed
  to `.fit(...)` (Req3).

---

## Phase 0: Pre-flight verification (before authoring any cell)

> **Gate.** Phase 8, task 8.1 ("Re-verify the five consumed `nids` signatures... before writing the
> first cell") is listed at the end of this document only because it is one of the session's mandatory
> closing tasks. `sdd-apply` MUST execute task 8.1 first, before task 1.1, because
> `project-foundation` slices 3–5 land between planning and apply and drift is possible. This phase 0
> task is the one additional pre-flight check that is not part of that closing-task list.

- [x] 0.1 Verify the `ResultsWriter` artifact-name uniqueness assumption against committed
  `src/nids/results.py` (read-only). The design assumes the name registry is **global** across tables
  and figures (not per-kind), and on that assumption renamed the figure
  `pca_top_loadings_pc1_pc2` → `pca_top_loadings_pc1_pc2_bars` (R9) to avoid colliding with the
  identically named table. Read `src/nids/results.py`'s `add_table` / `add_figure` implementation and
  its name-registration data structure.
  - If the registry is confirmed global (one shared set/dict of registered names across kinds): proceed
    with the design's names exactly as given in Phase 4 (the figure stays
    `pca_top_loadings_pc1_pc2_bars`); no further action.
  - If the registry turns out to be per-kind (tables and figures tracked separately): the rename is
    harmless but unnecessary. Keep `pca_top_loadings_pc1_pc2_bars` anyway for consistency with this
    task list and the design's figure-specification tables — do not revert to the colliding name.
  - Spec: Req12. Outcome: one line in the apply report stating which behavior was observed
    (global vs per-kind) and confirming the figure name used in Phase 4, task 4.7.

---

## Phase 1: §0 — Setup and guards (14 cells)

All cells target `notebooks/01_eda_reduction_clustering.ipynb`.

- [x] 1.1 `s0_00_title` (md): Title, the three questions the notebook answers (data shape, feature
  redundancy, whether unsupervised structure exists), how to re-run it
  (`uv run jupyter nbconvert --to notebook --execute --inplace
  --ExecutePreprocessor.timeout=-1 notebooks/01_eda_reduction_clustering.ipynb`), and the
  `results/eda_reduction_clustering/` folder name. Spec: Req5, Req13. Outcome: markdown cell renders
  with no `outputs`/`execution_count` fields.
- [x] 1.2 `s0_01_imports` (code): import `dataclasses`, `numpy`, `pandas`,
  `matplotlib.pyplot`, `seaborn`, the five sklearn classes, and
  `nids.columns.{feature_columns, TTL_SHORTCUT_COLUMNS, CATEGORICAL_COLUMNS, SKEWED_COLUMNS,
  numeric_feature_columns, skewed_feature_columns}`, `nids.data.load_clean_partitions`,
  `nids.preprocessing.{build_preprocessor, build_preprocessor_pair}`,
  `nids.sampling.stratified_subsample`, `nids.results.ResultsWriter`,
  `nids.validation.validate_raw_data`. Spec: Req1. Outcome: cell executes without error; every
  data-touching name originates from `nids.columns`, `nids.data`, `nids.preprocessing`,
  `nids.sampling`, `nids.results`, or `nids.validation`.
- [x] 1.3 `s0_02_style` (code): `%matplotlib inline`; `sns.set_theme(style="whitegrid",
  context="notebook")`; the `plt.rcParams.update({...})` block from Decision 10 (dpi 100, opaque white
  figure/axes/savefig facecolor, `DejaVu Sans`, title/label sizes, `legend.frameon=True`);
  `CLASS_ORDER = sorted(train["attack_cat"].unique().tolist())` **cannot run here** — move that one
  line into `s0_03_constants` after `train` exists, or defer `CLASS_ORDER`/`CLASS_COLORS` construction
  to immediately after `s0_06_load`. Also set `pd.set_option` display-limit defaults (max_rows,
  max_columns) once. Spec: Req12. Outcome: no visible output; rcParams pinned.
- [x] 1.4 `s0_03_constants` (code): `NOTEBOOK_ID = "eda_reduction_clustering"`, `SEED = 42`,
  `VARIANTS = {"with_ttl": True, "without_ttl": False}`, `K_RANGE = range(2, 13)`,
  `VARIANCE_THRESHOLDS = (0.90, 0.95)`, `KEY_FEATURES = ("dur", "sbytes", "dbytes", "rate", "sload",
  "dload")` (frozen, all in `SKEWED_COLUMNS` — R13), `MAX_ROWS = 10_000`, `FLOOR = 50`. Spec: Req4,
  Req5, Req11. Outcome: constants defined; no output.
- [x] 1.5 `s0_04_data_guard_md` (md): why raw validation runs before anything else — a truncated or
  replaced CSV must fail at this cell, not halfway through section 3. Spec: Req13. Outcome: markdown
  renders.
- [x] 1.6 `s0_05_validate` (code): `report = validate_raw_data()`; `print(report.render())`; `assert
  report.ok`. Spec: Req1, Req13. Outcome: assertion passes; report prints; cell has no `error` output.
- [x] 1.7 `s0_06_load` (code): `cleaned = load_clean_partitions()` — called once, no arguments
  overriding `comparison_key` or `raw_dir`; `train = cleaned.train`; `del cleaned`; `assert "cleaned"
  not in globals()`. Build `CLASS_ORDER = sorted(train["attack_cat"].unique().tolist())` and
  `CLASS_COLORS = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(CLASS_ORDER)}` here (moved from
  1.3 — needs `train`); `assert len(CLASS_ORDER) <= 10` else fall back to `tab20`. Spec: Req2, Req4.
  Outcome: `train` is the only name bound from `load_clean_partitions`'s return value; `cleaned` is
  unreachable after this cell; `CLASS_ORDER`/`CLASS_COLORS` fixed for every later figure.
- [x] 1.8 `s0_07_boundary_md` (md): the `attack_cat` boundary — the four legitimate uses
  (descriptive stats, `stratify_by`, post-hoc PCA scatter colour, post-hoc cluster composition) and
  the three structural guarantees (allow-list, `ColumnTransformer(remainder="drop")`, no `.fit(X, y)`
  anywhere). Spec: Req3. Outcome: markdown renders.
- [x] 1.9 `s0_08_assert_labels` (code) — **runtime assertion cell 1**:
  ```python
  for include_ttl in (True, False):
      selected = feature_columns(include_ttl)
      assert "attack_cat" not in selected and "label" not in selected
      assert "row_position" not in selected
  assert len(feature_columns(True)) == 39 and len(feature_columns(False)) == 37
  assert "cleaned" not in globals()
  print("Label boundary and test-partition guards passed.")
  ```
  Spec: Req3 (scenario: label-boundary assertions present and pass), Req2. Outcome: cell executes
  without raising; saved output contains the printed confirmation line.
- [x] 1.10 `s0_09_helpers` (code): define, in this order, `VariantSpace` (frozen dataclass: `key`,
  `include_ttl`, `columns`, `pipeline`, `names`, `design`, `modelled`, `blocks`), `build_variant_space`,
  `variance_inflation_factors` (the three-guard version from Decision 5: zero-variance columns dropped
  and reported first with `status="zero_variance"`; `r2 >= 1.0` or non-finite → `vif=inf`,
  `status="perfect_collinearity"`; else `status="ok"`), `fix_component_signs` (largest-magnitude
  loading forced positive per component, ties broken by `np.argmax`'s first-index behavior, zero
  pivots treated as positive), `select_eps` (returns `(eps, kdist, rule)` where `rule` is one of
  `chord`, `flat_curve_median`, `smallest_positive`), `record_metric` and `record_note` (raise
  `ValueError` on a duplicate name, then call `writer.add_metric`/`writer.add_note`), `new_figure`
  (`(nrows, ncols, *, figsize) -> (Figure, Axes)`). Spec: Req1, Req7, Req8, Req9, Req12. Outcome: all
  eight names defined; no execution error; no data touched yet.
- [x] 1.11 `s0_10_spaces` (code): `pipelines = build_preprocessor_pair()`; for each `key, include_ttl
  in VARIANTS.items()`, `pipe = pipelines[key].fit(train[feature_columns(include_ttl)])` (the frame is
  named at the call site) and build `spaces[key] = build_variant_space(key, include_ttl)` populating
  `columns`, `pipeline`, `names` (`pipeline.get_feature_names_out().tolist()`), `design`
  (`pipe.transform(...)`), `blocks` (`numeric`/`skewed`/`onehot` → names, from
  `numeric_feature_columns`/`skewed_feature_columns`/one-hot output names), and `modelled` (the
  numeric+skewed columns of `design`, by name, as a `DataFrame`). Spec: Req1, Req4, Req11. Outcome:
  `spaces = {"with_ttl": VariantSpace(...), "without_ttl": VariantSpace(...)}`; `spaces["with_ttl"
  ].design.shape[1] == ~52` (36 modelled + one-hot), `spaces["with_ttl"].modelled.shape[1] == 36`,
  `spaces["without_ttl"].modelled.shape[1] == 34`.
- [x] 1.12 `s0_11_assert_matrix` (code) — **runtime assertion cell 2**:
  ```python
  for key, space in spaces.items():
      assert list(space.pipeline.feature_names_in_) == feature_columns(space.include_ttl)
      assert not ({"attack_cat", "label"} & set(space.names))
      assert space.design.shape[0] == len(train)
  print("Fitted design matrices carry no target column, in either TTL variant.")
  ```
  Spec: Req3 (scenario: assert `{"attack_cat", "label"}` do not intersect
  `pipeline.get_feature_names_out()`), Req4 (scenario: no `id`/`attack_cat`/`label` reaches a fitted
  pipeline's input frame). Outcome: cell executes without raising; saved output contains the printed
  confirmation line.
- [x] 1.13 `s0_12_subsamples` (code) — the Decision 4 workaround: build
  `index_frame = pd.DataFrame({"attack_cat": train["attack_cat"].to_numpy(), "row_position":
  np.arange(len(train), dtype=np.int64)})`; call `sub_default = stratified_subsample(index_frame,
  max_rows=10_000, floor=50, seed=42)` and `sub_proportional = stratified_subsample(index_frame,
  max_rows=10_000, floor=0, seed=42)`; `positions = sub_default.data["row_position"].to_numpy()`. Add
  the guard `assert "row_position" not in feature_columns(True)`. This is the only supported way to
  recover which rows of the full partition were drawn, because `SubsampleResult.data` is
  `.reset_index(drop=True)`d and exposes no `selected_positions` accessor (foundation finding, recorded
  in design Decision 4 — do not attempt to patch `src/nids/sampling.py`). Spec: Req4, Req9 (feeds
  Phase 5). Outcome: `positions` is an `int64` array of length ≤10,000; `sub_default.data` and
  `sub_proportional.data` each carry `attack_cat` and `row_position` only.
- [x] 1.14 `s0_13_writer` (code): `writer = ResultsWriter(NOTEBOOK_ID)`; print `writer.directory`;
  register via `record_metric`: `train_rows` (`len(train)`), `attack_cat_classes`
  (`train["attack_cat"].nunique()`), `feature_columns_with_ttl` (`len(feature_columns(True))`),
  `feature_columns_without_ttl` (`len(feature_columns(False))`), `silhouette_sample_rows`
  (`len(sub_default.data)`), `silhouette_sensitivity_sample_rows` (`len(sub_proportional.data)`).
  Spec: Req5 (scenario: `ResultsWriter` constructed with `notebook_id="eda_reduction_clustering"`),
  Req12. Outcome: `writer.directory` prints a path ending in `results/eda_reduction_clustering`; six
  metrics registered, none starting with a digit.

---

## Phase 2: §1 — Descriptive statistics (13 cells)

- [x] 2.1 `s1_00_header` (md): what section 1 establishes; why it deliberately overlaps
  `results/data_cleaning/` content (each producer folder must be interpretable in isolation — the
  cleaning report shows before-vs-after, this notebook describes the cleaned train partition only).
  Spec: Req6. Outcome: markdown renders.
- [x] 2.2 `s1_01_shape` (code): column inventory — `position, column, dtype, non_null, role` — role
  ∈ {`feature_numeric`, `feature_skewed`, `feature_categorical`, `target`}, computed at run time from
  `train`, no hardcoded row count. `add_table(..., name="partition_shape_and_dtypes",
  sort_by=["position"])`. Spec: Req6 (scenario: descriptive tables registered; scenario: no count
  hardcoded). Outcome: table registered; row count equals `train.shape[1]`.
- [x] 2.3 `s1_02_numeric_summary` (code): extended `describe` per numeric feature — `feature, count,
  mean, median, std, min, q25, q75, max, skew, mean_median_ratio` (raw units; `mean_median_ratio` null
  where median is 0). `add_table(..., name="numeric_summary_mean_vs_median", sort_by=["feature"])`.
  Spec: Req6. Outcome: table registered, one row per numeric feature.
- [x] 2.4 `s1_03_categorical` (code): long-format `value_counts` for `proto`, `service`, `state` —
  `column, category, rows, share`, computed at run time (the `"-"` category in `service` appears
  as-is, never remapped). `add_table(..., name="categorical_value_counts", sort_by=["column",
  "category"])`. Spec: Req6, Req4 (`"-"` not remapped by this notebook). Outcome: table registered;
  `service` category list includes `"-"`.
- [x] 2.5 `s1_04_attack_cat` (code): `attack_cat, rows, share`, computed at run time.
  `add_table(..., name="attack_cat_distribution", sort_by=["attack_cat"])`. Spec: Req6 (scenario: no
  count hardcoded), Req3 (descriptive use). Outcome: table registered; row count equals
  `train["attack_cat"].nunique()`.
- [x] 2.6 `s1_05_imbalance_md` (md): reading the class imbalance; why the bar chart uses a log
  x-axis (largest class outnumbers smallest by ~2 orders of magnitude). Spec: Req6. Outcome: markdown
  renders.
- [x] 2.7 `s1_06_fig_attack_bars` (code): sorted horizontal bars (rows descending), log x-axis, one
  bar per class coloured from `CLASS_COLORS`. `add_figure(..., name="attack_cat_distribution_bars",
  title="Attack category distribution in the cleaned training partition", ...)`. Spec: Req6 (scenario:
  descriptive figures registered), Req12 (English labels). Outcome: figure registered at dpi 150;
  `plt.close(fig)` is the cell's last statement.
- [x] 2.8 `s1_07_skew_md` (md): right skew and the `log1p` rationale; states plainly that the ECDF
  panel that follows shows the *same* curves under two axis parameterisations, because an ECDF is
  invariant under any strictly increasing transform — this is R8, a stated finding, not a bug to
  avoid. Spec: Req6. Outcome: markdown renders.
- [x] 2.9 `s1_08_fig_hist_raw` (code): 2×3 histogram grid over `KEY_FEATURES`, `bins=50`, raw
  units, log y-axis. `add_figure(..., name="key_feature_histograms_raw", ...)`. Spec: Req6. Outcome:
  figure registered.
- [x] 2.10 `s1_09_fig_hist_log1p` (code): same 2×3 grid after `log1p` **applied to a notebook-local
  numpy array** (`np.log1p(train[feature].to_numpy())`), never assigned back onto `train` — no
  `np.log1p` call may target a partition column in place.
  `add_figure(..., name="key_feature_histograms_log1p", ...)`. Spec: Req6, Req1 (no re-implementation
  of the pipeline's `log1p` on a partition column). Outcome: figure registered; `train` unmodified.
- [x] 2.11 `s1_10_fig_ecdf` (code) — **R8, do not "fix" this back into two different curves**: 1×2
  panels, six step curves per panel, same six ECDFs in both panels. Left panel x-axis: raw value,
  `symlog` scale with `linthresh=1`, label "Value (symlog scale, linear below 1)". Right panel x-axis:
  `log1p(value)`, linear scale, label "log1p(value)". Y-axis both panels: "Cumulative proportion of
  rows". `add_figure(..., name="key_feature_ecdf_raw_vs_log1p",
  description="...The curves are identical by construction: log1p is strictly increasing...")`. Spec:
  Req6, Req12 (English labels). Outcome: figure registered with two panels showing visually identical
  curve shapes (re-parameterised x-axis only), not a duplicate-curve overlay.
- [x] 2.12 `s1_11_fig_boxplots` (code): 3×2 horizontal boxplot grid over `KEY_FEATURES`, grouped by
  `attack_cat` (`CLASS_ORDER`), `showfliers=False`, `whis=(1, 99)`, symlog x-axis (`linthresh=1`),
  bars coloured from `CLASS_COLORS`. `add_figure(..., name="key_feature_boxplots_by_attack_cat", ...)`.
  Spec: Req6, Req3 (descriptive grouping use). Outcome: figure registered; caption/markdown states
  whiskers are 1st/99th percentiles, not Tukey.
- [x] 2.13 `s1_12_findings` (md): what section 1 establishes, in prose. Spec: Req6, Req11 (states
  dual-TTL reporting is not applicable to section 1 — descriptive stats cover all columns). Outcome:
  markdown renders.

---

## Phase 3: §2 — Correlation and collinearity (9 cells)

- [x] 3.1 `s2_00_header` (md): scope of section 2; why the correlation matrices are single-variant
  while VIF is dual-variant (Pearson/Spearman are pairwise — removing two columns deletes two rows and
  columns without changing any remaining cell; VIF is multivariate — removing two columns changes
  every other feature's value). Spec: Req7, Req11. Outcome: markdown renders.
- [x] 3.2 `s2_01_matrices` (code): `pearson_correlation_matrix` and `spearman_correlation_matrix`,
  each 36×36, computed on `spaces["with_ttl"].modelled` only. `add_table(...,
  name="pearson_correlation_matrix", sort_by=["feature"])` and
  `add_table(..., name="spearman_correlation_matrix", sort_by=["feature"])`. Spec: Req7 (scenario: VIF
  dual, correlation single — the matrices appear exactly once with no `_with_ttl`/`_without_ttl`
  counterpart). Outcome: two 36×36 tables registered, no `variant` column.
- [x] 3.3 `s2_02_fig_pearson` (code): diverging heatmap, `cmap="RdBu_r"`, `vmin=-1`, `vmax=1`,
  `center=0`, unannotated, 11×9.5 in, 7-pt tick labels rotated 90° on x. Colorbar label "Pearson
  correlation coefficient". `add_figure(..., name="pearson_correlation_heatmap", ...)`. Spec: Req7
  (scenario: correlation heatmaps use a diverging palette centred at zero, `vmin=-1`, `vmax=1`).
  Outcome: figure registered.
- [x] 3.4 `s2_03_fig_spearman` (code): same palette/limits/geometry as 3.3. Colorbar label "Spearman
  rank correlation coefficient". `add_figure(..., name="spearman_correlation_heatmap", ...)`. Spec:
  Req7. Outcome: figure registered.
- [x] 3.5 `s2_04_pairs` (code): upper-triangle pairs with `|r| > 0.9` under Pearson or Spearman —
  `feature_a, feature_b, pearson, spearman, method_flagged` (`method_flagged` ∈ `pearson`, `spearman`,
  `both`). `add_table(..., name="high_correlation_pairs", sort_by=["feature_a", "feature_b"])`;
  `record_metric("correlated_pairs_above_0_9", len(pairs), ...)`. Spec: Req7 (`high_correlation_pairs`
  requirement). Outcome: table and metric registered.
- [x] 3.6 `s2_05_vif_md` (md): VIF definition (`1/(1-R²)`), the three guards, why one-hot columns are
  excluded, why the modelled (post-`log1p`, pre-scaling-irrelevant) form is used. Spec: Req7. Outcome:
  markdown renders.
- [x] 3.7 `s2_06_vif` (code): call `variance_inflation_factors(space.modelled)` for both variants,
  concatenate with a `variant` column, `feature, r_squared, vif, status`. `add_table(...,
  name="variance_inflation_factors", sort_by=["variant", "feature"])`.
  **Design correction — mandatory**: register `max_vif_with_ttl` / `max_vif_without_ttl` as the
  **string** `"inf"` when any feature in that variant has `vif == float("inf")`, and as the maximum
  finite float otherwise. Never register a Python float `inf` — `json.dumps(float("inf"))` emits the
  bare token `Infinity`, which is not valid JSON and would make `manifest.json` unparseable by a strict
  reader. The `variance_inflation_factors` table itself still carries the numeric `inf` in its `vif`
  column (CSV renders it as text `inf`, unambiguous). Spec: Req7 (scenario: perfectly collinear feature
  reports `inf`, no `ZeroDivisionError`; scenario: zero-variance column dropped and flagged; scenario:
  VIF reported for both variants), Req11. Outcome: table registered with `status` ∈ {`ok`,
  `zero_variance`, `perfect_collinearity`}; both metrics registered, each either a float or the literal
  string `"inf"`.
- [x] 3.8 `s2_07_fig_vif` (code): grouped vertical bars, two bars per feature (variant pair colours
  `#1f77b4` / `#d62728`), log y-axis. Infinite VIF bars: replace with `10 × max(finite VIF)`, hatch
  `///`, separate legend entry "VIF is infinite (perfect collinearity)" — the clipped height is a
  drawing convention only, never registered as a number. `add_figure(...,
  name="variance_inflation_factors_bars", ...)`. Spec: Req7, Req11. Outcome: figure registered; no
  infinite value is written to the figure's underlying data beyond the clipped drawing height.
- [x] 3.9 `s2_08_findings` (md): which feature families are redundant (e.g. `swin`/`dwin`, the
  `ct_*` family); how the TTL toggle moved every other feature's VIF. Spec: Req7, Req11. Outcome:
  markdown renders.

---

## Phase 4: §3 — Dimensionality reduction (12 cells)

- [x] 4.1 `s3_00_header` (md): what PCA is being asked; the design-matrix width per variant (~52
  columns, 36/34 modelled). Spec: Req8. Outcome: markdown renders.
- [x] 4.2 `s3_01_fit` (code): for each variant, `pca = PCA(n_components=None, svd_solver="full",
  random_state=42).fit(space.design)`; `scores = pca.transform(space.design)`; `components, scores =
  fix_component_signs(pca.components_, scores)`; `cumulative = np.cumsum(pca.explained_variance_ratio_)`;
  `n90 = int(np.searchsorted(cumulative, 0.90, side="left") + 1)`; `n95` likewise at 0.95. Store per
  variant: `pca_by_variant`, `components_by_variant`, `scores` (sign-fixed, never re-transformed later),
  `n90`, `n95`. `attack_cat` must not appear anywhere in `space.design`. Spec: Req8 (`svd_solver="full"`,
  `random_state=42`; scenario: PCA loadings stable across reruns), Req3 (fit precedes any
  `attack_cat` reference). Outcome: `pca_by_variant`, `scores`, `n90`/`n95` populated for both variants;
  no error.
- [x] 4.3 `s3_02_explained` (code): `variant, component, explained, cumulative`, both variants.
  `add_table(..., name="pca_explained_variance", sort_by=["variant", "component"])`. Spec: Req8,
  Req11. Outcome: table registered.
- [x] 4.4 `s3_03_thresholds` (code): `variant, threshold, n_components` for thresholds `(0.90, 0.95)`.
  `add_table(..., name="pca_components_for_variance", sort_by=["variant", "threshold"])`;
  `record_metric` for `pca_components_90_with_ttl`, `pca_components_90_without_ttl`,
  `pca_components_95_with_ttl`, `pca_components_95_without_ttl`. Spec: Req8 (scenario: components
  reaching 90%/95% reported for both variants, each a positive integer ≤ input column count). Outcome:
  table and four metrics registered.
- [x] 4.5 `s3_04_fig_scree` (code): explained variance ratio per component, both variants overlaid,
  variant-pair colours and linestyles (`-`/`--`). `add_figure(..., name="pca_scree_plot", ...)`. Spec:
  Req8. Outcome: figure registered.
- [x] 4.6 `s3_05_fig_cumulative` (code): cumulative curve, both variants, plus grey dashed horizontal
  reference lines at 0.90 and 0.95. `add_figure(..., name="pca_cumulative_variance", ...)`. Spec: Req8.
  Outcome: figure registered.
- [x] 4.7 `s3_06_loadings` (code): top 15 loadings by `|value|` for PC1 and PC2, per variant —
  `variant, component, rank, feature, loading, abs_loading`. `add_table(...,
  name="pca_top_loadings_pc1_pc2", sort_by=["variant", "component", "rank"])`. Spec: Req8. Outcome:
  table registered.
- [x] 4.8 `s3_07_fig_loadings` (code): 2×2 panel grid (variant × component {PC1, PC2}), horizontal
  loading bars, coloured by feature block (`numeric`/`skewed`/`onehot`) via `space.blocks`. Registered
  under the **renamed** figure name `pca_top_loadings_pc1_pc2_bars` (per Phase 0, task 0.1 and design
  R9 — the table above is named `pca_top_loadings_pc1_pc2` without `_bars`; they must not collide).
  `add_figure(..., name="pca_top_loadings_pc1_pc2_bars", ...)`. Spec: Req8, Req12 (name uniqueness).
  Outcome: figure registered under a name distinct from the table in 4.7.
- [x] 4.9 `s3_08_onehot_md` (md): F6 — the one-hot variance asymmetry; why it is measured, not fixed
  (`build_preprocessor`'s categorical branch has no scaler, out of scope to change). Spec: Req8.
  Outcome: markdown renders.
- [x] 4.10 `s3_09_blocks` (code): the F6 measurement exactly as specified in design
  ([F6 measurement](../notebook-01-eda-reduction-clustering/design.md) — read-only reference, this
  notebook implements it locally): `column_variance = frame.var(axis=0, ddof=1)`; `total =
  column_variance.sum()`; self-check assertion `np.isclose(total,
  pca_by_variant[key].explained_variance_.sum(), rtol=1e-9)`; per block (`numeric`, `skewed`,
  `onehot`): `n_columns`, `share_of_total_variance`, `share_of_pc1_loading_sq`,
  `share_of_pc2_loading_sq` (squared unit-norm loadings summed per block, each column also sums to
  1.0 per variant). `add_table(..., name="pca_variance_by_feature_block", sort_by=["variant",
  "block"])`; `record_metric` for `onehot_share_of_total_variance_with_ttl` /
  `..._without_ttl`; `record_note("onehot_variance_asymmetry", ...)` stating the measured share, the
  structural cause, and that a numeric-heavy PC1 is expected and not evidence of uninformative
  categorical features. Spec: Req8, Req11, Req12 (note). Outcome: table (6 rows: 2 variants × 3
  blocks) and two metrics registered; the trace-preservation assertion passes; the note is present.
- [x] 4.11 `s3_10_fig_scatter` (code): 1×2 panels (one per variant), PC1 vs PC2 scores on
  `scores[positions]`, `s=5, alpha=0.4, linewidths=0`, coloured post hoc by `attack_cat` via
  `CLASS_COLORS`, classes drawn in descending size (ties by name ascending), legend outside axes with
  `markerscale=4`. The `PCA().fit(...)` call this cell references (`s3_01_fit`) must precede any
  reference to `attack_cat` in the cell that builds this figure. `add_figure(...,
  name="pca_scatter_pc1_pc2_by_attack_cat", ...)`. Spec: Req8 (scenario: PCA scatter coloring is
  post-hoc), Req3. Outcome: figure registered; `attack_cat` is not a column of the matrix passed to
  `.fit(...)`.
- [x] 4.12 `s3_11_findings` (md): how many real degrees of freedom exist; how the TTL toggle changed
  the component count. Spec: Req8, Req11. Outcome: markdown renders.

---

## Phase 5: §4 — Clustering on the reduced space (13 cells)

- [x] 5.1 `s4_00_header` (md): states the fit-on-full / evaluate-on-sample split before any number
  appears — K-Means is fitted on the full cleaned partition; silhouette is evaluated on the `floor=50`
  subsample using labels from that full-partition fit. Spec: Req9. Outcome: markdown renders.
- [x] 5.2 `s4_01_sweep` (code): for each variant and `k in K_RANGE` (2..12), `kmeans_fits[(variant,
  k)] = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scores[:, :n90])` — the frame passed to
  `.fit` has row count equal to the full cleaned training partition. Record, per `(variant, k)`:
  `inertia` (`kmeans.inertia_`, free), `calinski_harabasz_score` on the full partition, and
  `silhouette_score` on `scores[positions, :n90]` using `kmeans.labels_[positions]` — guarded: if
  `len(np.unique(labels[positions])) < 2`, record `NaN` and continue rather than raise. Spec: Req9
  (scenario: K-Means fitted on full partition), Req4 (seed 42), Req3 (`attack_cat`/`label` never
  passed as `.fit` target). Outcome: `kmeans_fits` holds 22 fitted estimators (11 k-values × 2
  variants); a sweep results structure holds inertia/silhouette/CH per `(variant, k)`.
- [x] 5.3 `s4_02_sweep_table` (code): `variant, k, inertia, silhouette, calinski_harabasz`, long
  format. `add_table(..., name="kmeans_k_sweep_metrics", sort_by=["variant", "k"])`;
  `record_note("silhouette_is_subsample_estimate", ...)` — states sample size, stratification key
  (`attack_cat`), floor (50), seed (42), and that the value is a class-rebalanced estimate. Spec: Req9
  (scenario: notes array contains `silhouette_is_subsample_estimate`), Req11. Outcome: table (22 rows)
  and note registered.
- [x] 5.4 `s4_03_fig_elbow` (code): inertia vs `k`, both variants overlaid, variant-pair colours.
  `add_figure(..., name="kmeans_elbow_inertia", ...)`. Spec: Req9. Outcome: figure registered.
- [x] 5.5 `s4_04_fig_silhouette` (code): silhouette vs `k`, both variants overlaid, selected `k`
  ringed (marker emphasised) once `selected_k` is available from task 5.6 — if this cell must execute
  before 5.6 in top-to-bottom order, compute the argmax selection locally in this cell too (same rule
  as 5.6) so the figure and the metric never disagree. `add_figure(...,
  name="kmeans_silhouette_by_k", ...)`. Spec: Req9. Outcome: figure registered.
- [x] 5.6 `s4_05_fig_ch` (code): Calinski-Harabasz vs `k`, both variants overlaid.
  `add_figure(..., name="kmeans_calinski_harabasz_by_k", ...)`. Spec: Req9. Outcome: figure registered.
- [x] 5.7 `s4_06_select_k` (code): `selected_k[variant] = min(K_RANGE, key=lambda k: (-sweep[(variant,
  k)].silhouette, k))` — argmax silhouette, ties broken toward the smaller `k`. At the selected `k`
  only, recompute silhouette on `scores[sub_proportional.data["row_position"].to_numpy(), :n90]` using
  `kmeans_fits[(variant, selected_k[variant])].labels_[those positions]` (the `floor=0`
  population-representative sensitivity check). `record_metric` for `kmeans_selected_k_{v}`,
  `kmeans_silhouette_at_selected_k_{v}`, `kmeans_silhouette_floor0_sensitivity_{v}`. Spec: Req9
  (scenario: `floor=0` sensitivity silhouette reported alongside default-floor silhouette). Outcome:
  six metrics registered (3 stems × 2 variants).
- [x] 5.8 `s4_07_dbscan_md` (md): why DBSCAN is sampled (ball-tree neighbour search degrades near ten
  dimensions; memory is driven by neighbourhood size, not row count); the `eps` rule written out in
  words; `min_samples = 2 × n_components_90[variant]`. Spec: Req9. Outcome: markdown renders.
- [x] 5.9 `s4_08_eps` (code): for each variant, `sample = scores[positions, :n90[variant]]`; `k =
  min_samples = 2 * n90[variant]`; compute the k-th nearest-neighbour distance for every point via
  `NearestNeighbors(n_neighbors=k).fit(sample).kneighbors(sample)`; sort ascending as `kdist`; call
  `select_eps(sample, min_samples)` implementing the normalised-chord rule with all three guards
  (`span == 0` → median fallback + note; selected `eps == 0` → smallest positive k-distance; no
  positive k-distance → DBSCAN skipped for that variant, metrics `NaN`). `record_metric` for
  `dbscan_eps_{v}` and `dbscan_min_samples_{v}`. Spec: Req9 (scenario: `eps` selected deterministically
  and marked on the k-distance plot — the plot itself is task 5.10). Outcome: two metrics per variant
  registered; `dbscan_eps_{v}` equals the value later drawn in `s4_09_fig_kdist` and later passed to
  `DBSCAN(eps=...)` in `s4_10_dbscan`.
- [x] 5.10 `s4_09_fig_kdist` (code): sorted k-distance curves, both variants, each with a horizontal
  dashed reference line at its variant's selected `eps`, in the matching variant colour.
  `add_figure(..., name="dbscan_k_distance_plot", ...)`. Spec: Req9 (scenario: `eps` used to fit
  `DBSCAN` equals the value annotated on this figure). Outcome: figure registered; the drawn reference
  line value is read from the same `dbscan_eps_{v}` metric computed in 5.9, not recomputed.
- [x] 5.11 `s4_10_dbscan` (code): for each variant, `DBSCAN(eps=dbscan_eps[v],
  min_samples=min_samples[v]).fit(scores[positions, :n90[v]])`; cluster sizes including the `-1` noise
  label — `variant, cluster, rows, share`. `add_table(..., name="dbscan_cluster_summary",
  sort_by=["variant", "cluster"])`; `record_metric` for `dbscan_cluster_count_{v}` (clusters excluding
  noise) and `dbscan_noise_fraction_{v}`; `record_note("dbscan_is_subsample_property", ...)` and
  `record_note("dbscan_eps_not_transferable", ...)` (states whether the chord rule degenerated to a
  fallback, per variant). Spec: Req9 (scenario: DBSCAN noise fraction disclosed as subsample property —
  both note ids present). Outcome: table and four metrics registered; both notes present.
- [x] 5.12 `s4_11_allocation` (code): `sub_default.to_frame()` and `sub_proportional.to_frame()`
  concatenated with a `sampler` column (`default_floor50` / `proportional_floor0`), the sampler's
  `label` column renamed `attack_cat` — `sampler, attack_cat, available, allocated, proportional,
  floor_applied, population_share, sample_share`. `add_table(..., name="clustering_sample_allocation",
  sort_by=["sampler", "attack_cat"])`. Spec: Req9 (allocation reporting requirement). Outcome: table
  registered, evidencing how far the `floor=50` sample over-represents minority classes relative to
  `floor=0`.
- [x] 5.13 `s4_12_findings` (md): whether unsupervised structure exists; whether the two variants
  agree on the selected `k`. Spec: Req9, Req11. Outcome: markdown renders.

---

## Phase 6: §5 — Cluster profiling (5 cells)

- [x] 6.1 `s5_00_header` (md): profiling is post hoc; no metric is optimised against `attack_cat`;
  ARI/NMI are explicitly out of scope (notebook 2's job). Spec: Req10. Outcome: markdown renders.
- [x] 6.2 `s5_01_labels` (code): `final_labels[v] = kmeans_fits[(v, selected_k[v])].labels_` for both
  variants — reuse the sweep's already-fitted estimator, **no refit** (Decision 8; a refit could
  silently disagree with the sweep if any parameter drifted between call sites). Spec: Req10 (basis
  for the two profiling tables below). Outcome: `final_labels` populated for both variants without a
  new `KMeans(...).fit(...)` call in this cell.
- [x] 6.3 `s5_02_profile` (code): per cluster, per **original, unscaled** numeric + skewed feature —
  `cluster_mean`, `global_mean` (both in original units, from `train`, not from `space.design`),
  `ratio`, `std_deviations` (gap expressed in global standard deviations; null where global mean or
  std is 0) — `variant, cluster, feature, cluster_mean, global_mean, ratio, std_deviations`.
  `add_table(..., name="cluster_profile_original_units", sort_by=["variant", "cluster", "feature"])`.
  Spec: Req10 (scenario: cluster profiling uses unscaled original-unit features — means MUST match the
  original training-partition feature scale, not the standardized scale). Outcome: table registered;
  values traced back to `train[feature]`, not to the standardized `space.design` column.
- [x] 6.4 `s5_03_composition` (code): `attack_cat` composition per cluster, including zero cells —
  `variant, cluster, attack_cat, rows, share_of_cluster, share_of_class`. `add_table(...,
  name="cluster_attack_cat_composition", sort_by=["variant", "cluster", "attack_cat"])`;
  `record_note("attack_cat_is_post_hoc_only", ...)` — states `attack_cat` is used for stratification,
  colouring and post-hoc profiling only, never a feature or fitting target, and that no ARI/NMI is
  computed here. Spec: Req10 (scenario: no external cluster-validation metric present — searched
  patterns `adjusted_rand_score`, `normalized_mutual_info_score`, `ARI`, `NMI` must not appear
  anywhere), Req3. Outcome: table and note registered; the cell contains no call to
  `adjusted_rand_score` or `normalized_mutual_info_score`.
- [x] 6.5 `s5_04_findings` (md): what the clusters appear to represent, in original units. Spec:
  Req10, Req11. Outcome: markdown renders.

---

## Phase 7: §6 — Conclusions and close (7 cells)

> Decision 13's two-pass authoring loop applies to this phase only: cells `s6_02_structure`,
> `s6_03_clusters`, `s6_04_limitations` are first authored with the *questions and rules of
> interpretation* (no numbers), the notebook is executed once (Phase 8, task 8.2), the printed findings
> from `s6_01_summary` are read, these three markdown cells are then rewritten with the observed
> numbers and conclusions, and the notebook is executed a **second** time — that second execution is
> the committed state (Phase 8, task 8.2 is run twice; see its note).

- [x] 7.1 `s6_00_header` (md): how to read section 6. Spec: Req11, Req13. Outcome: markdown renders.
- [x] 7.2 `s6_01_summary` (code): assemble a compact decisive-metrics frame from `METRICS` (the
  ordered dict from `record_metric`) and **print it only** — this cell registers nothing through
  `add_table`/`add_figure`. `record_note("ttl_shortcut_effect", ...)` — states, with the measured
  numbers, whether removing `sttl`/`ct_state_ttl` changed the maximum VIF, the 90% component count, the
  selected `k`, and the cluster composition. Spec: Req11 (scenario: section 6 states where TTL removal
  changes conclusions), Req12 (note). Outcome: printed frame in the saved output; note registered.
- [x] 7.3 `s6_02_structure` (md, authored in the second pass): what structure exists, quoting the
  observed numbers from `s6_01_summary`'s printed output. Spec: Req9, Req11. Outcome: markdown renders
  with concrete numbers, not placeholders.
- [x] 7.4 `s6_03_clusters` (md, authored in the second pass): what the clusters appear to represent,
  cross-referencing `cluster_profile_original_units` and `cluster_attack_cat_composition`. Spec:
  Req10, Req11. Outcome: markdown renders with concrete numbers.
- [x] 7.5 `s6_04_limitations` (md, authored in the second pass): subsample caveats (silhouette,
  DBSCAN), the one-hot variance asymmetry, the TTL effect, and an explicit statement of what this
  notebook does **not** claim (no ARI/NMI, no supervised result — that is notebook 2). Must
  explicitly state, for VIF, PCA, and clustering results, whether removing the TTL shortcut pair
  changed the reported conclusion. Spec: Req11 (scenario: section 6 states in prose whether removing
  TTL changed the conclusion for VIF, PCA, and clustering). Outcome: markdown renders; all three
  results (VIF, PCA, clustering) are explicitly addressed.
- [x] 7.6 `s6_05_close` (code): `writer.write_json({"metrics": {k: v for k, (v, _) in
  METRICS.items()}, "notes": NOTES}, name="counts")`; `manifest = writer.close()`; re-read
  `manifest.json` from disk and assert every declared `tables[].path` and `figures[].path` resolves to
  an existing file; assert the manifest declares exactly 17 tables, 16 figures, 29 metrics, 6 notes.
  Spec: Req12 (scenario: manifest is complete and self-describing — parses, every path resolves, all
  six notes present), Req13 (scenario: manifest self-check cell passes during execution). Outcome:
  `results/eda_reduction_clustering/counts.json` and `manifest.json` written; the self-check assertion
  passes and prints confirmation; no cell after this one performs further writes.
- [x] 7.7 `s6_06_rerun` (md): the exact re-run command
  (`uv run jupyter nbconvert --to notebook --execute --inplace
  --ExecutePreprocessor.timeout=-1 notebooks/01_eda_reduction_clustering.ipynb`) and what a clean rerun
  is expected to change — one line of `manifest.json`'s `generated_at` field, nothing else in
  `tables/*.csv`, `figures/*.png`, or `counts.json`; the `.ipynb` itself is expected to differ in its
  inline PNG base64 blobs (documented limit, not a failure). Spec: Req12 (byte-identical tables
  scenario), Req13. Outcome: markdown renders with the literal command.

---

## Phase 8: Execution and verification (mandatory closing tasks, in this order)

> These five tasks are listed last because they are the session's mandatory closing tasks. Task 8.1's
> action — re-verifying signatures against committed source — MUST be performed first in wall-clock
> terms, before Phase 1 task 1.1 is started (see the Phase 0 header note). Tasks 8.2–8.5 run after all
> 73 cells in Phases 1–7 are authored.

- [x] 8.1 Re-verify the five consumed `nids` signatures against **committed** source before writing
  the first cell of Phase 1 — `project-foundation` slices 4 and 5 land between planning and apply, so
  drift is possible. Read (read-only): `src/nids/data.py` (`load_clean_partitions`),
  `src/nids/preprocessing.py` (`build_preprocessor`, `build_preprocessor_pair`),
  `src/nids/sampling.py` (`stratified_subsample`), `src/nids/results.py` (`ResultsWriter` and its
  methods), and `src/nids/columns.py` (`feature_columns`, `numeric_feature_columns`,
  `skewed_feature_columns`, `TTL_SHORTCUT_COLUMNS`, `CATEGORICAL_COLUMNS`, `SKEWED_COLUMNS`). Compare
  each against the signatures recorded in
  `openspec/changes/notebook-01-eda-reduction-clustering/design.md` (read-only, section "The `nids`
  API surface consumed") and against
  `openspec/changes/project-foundation/design.md` (read-only). If any signature has drifted, update
  the affected Phase 1–7 tasks above to match the committed signature before authoring the
  corresponding cell — do not author against a stale signature. Spec: Req1. Outcome: one line per
  consumed call in the apply report confirming "matches design" or naming the drift and the task(s)
  updated.
- [x] 8.2 Execute the notebook with the exact command:
  `uv run jupyter nbconvert --to notebook --execute --inplace
  --ExecutePreprocessor.timeout=-1 notebooks/01_eda_reduction_clustering.ipynb`. Run it once after all
  73 cells are authored (Phases 1–7 complete, section 6 markdown still carries questions/rules only,
  no numbers). Read the printed findings from `s6_01_summary`'s output, rewrite `s6_02_structure`,
  `s6_03_clusters`, `s6_04_limitations` with the observed numbers (Phase 7 note), then run this exact
  same command a **second** time — that second execution's output is the committed state (Decision
  13). Spec: Req13 (scenario: `nbconvert --execute` completes without error — process exits 0, no cell
  carries an `error` output). Outcome: two recorded exit codes, both `0`; the second run's notebook
  file is what gets committed.
- [x] 8.3 Verify `results/eda_reduction_clustering/manifest.json` exists, is valid JSON parseable by
  `json.load`, and lists every exported file with each declared `tables[].path` and `figures[].path`
  present on disk under `results/eda_reduction_clustering/`. Cross-check the inventory totals: 17
  tables, 16 figures, 29 metrics, 6 notes, and that the `notes` array contains all six ids:
  `silhouette_is_subsample_estimate`, `dbscan_is_subsample_property`, `dbscan_eps_not_transferable`,
  `attack_cat_is_post_hoc_only`, `onehot_variance_asymmetry`, `ttl_shortcut_effect`. Spec: Req12
  (scenario: the manifest is complete and self-describing). Outcome: `json.load` succeeds; every
  listed path resolves; counts match exactly; all six note ids present.
- [x] 8.4 Verify the two `attack_cat` boundary assertion cells (`s0_08_assert_labels`,
  `s0_11_assert_matrix`) executed without error in the output notebook — inspect their saved
  `outputs` for an `error` output type (there must be none) and confirm each cell's printed
  confirmation string is present in a `stream`/`execute_result` output. Spec: Req3 (scenario: the
  label-boundary assertions are present and pass — both assertions MUST have executed without raising
  an error in the notebook's saved outputs). Outcome: both cells show clean, non-error outputs with the
  expected printed confirmation text.
- [x] 8.5 Run the full `uv run pytest` suite. It must stay green — this change adds no new tests
  (notebooks are exploratory per `openspec/config.yaml` `rules.tasks`), but must not break the
  existing package under `src/nids/**` (read-only) or `tests/**`. Spec: none directly (regression
  guard for the project's existing test suite, per AGENTS.md "Functions in src/ have ... pytest tests
  in tests/"). Outcome: `uv run pytest` exits 0; no test newly fails as a result of this change.

Also run the two mechanical static checks named in the proposal's Success Criteria, as part of task
8.3's verification pass:
- `rg -n 'load_raw_test|\.test\b|test_df|X_test' notebooks/01_eda_reduction_clustering.ipynb` returns
  no match in a code cell.
- `rg -n 'read_csv|duplicated\(|StandardScaler\(|OneHotEncoder\(|drop\(columns=|to_csv\(|savefig\('
  notebooks/01_eda_reduction_clustering.ipynb` returns no match (the VIF cell uses positional selection
  per the Global constraints section, so this grep stays literal with no scoping exception needed).

Report the measured `du -sh notebooks/01_eda_reduction_clustering.ipynb` and
`du -sh results/eda_reduction_clustering/` in the apply report — proposal F4 estimates ~10 MB combined;
if either measurement exceeds twice that estimate, report it as a finding, not a silent pass.

---

## Apply execution notes (unattended run)

- **Task 0.1 outcome**: `src/nids/results.py`'s `ResultsWriter` registers table and figure names into
  one shared `self._registered_artifact_names: set[str]`, checked in `_register_artifact_name`, which
  both `add_table` and `add_figure` call. The registry is confirmed **global** across kinds. Proceeded
  with the design's names exactly as given; the figure stays `pca_top_loadings_pc1_pc2_bars`.
- **Task 8.1 outcome**: re-verified all five consumed signatures against committed source before
  authoring cell 1 — `load_clean_partitions` (`src/nids/data.py`), `build_preprocessor` /
  `build_preprocessor_pair` (`src/nids/preprocessing.py`), `stratified_subsample`
  (`src/nids/sampling.py`), `ResultsWriter` and its methods (`src/nids/results.py`), and
  `feature_columns` / `numeric_feature_columns` / `skewed_feature_columns` / `TTL_SHORTCUT_COLUMNS` /
  `CATEGORICAL_COLUMNS` / `SKEWED_COLUMNS` (`src/nids/columns.py`). All five match the design
  verbatim — no drift, no task updates required.
- **Compliance fix found during verification**: the first authored `attack_cat_is_post_hoc_only` note
  text contained the literal substrings `ARI` and `NMI` inside a code-cell string, which the spec's
  "No external cluster-validation metric is present" scenario's `rg` search matches literally in any
  code cell (not only literal function calls). Reworded to "adjusted Rand index, normalised mutual
  information" (no abbreviation) and re-executed. Guard now clean:
  `rg -n '\bARI\b|\bNMI\b|adjusted_rand_score|normalized_mutual_info_score'` returns no match.
- **Execution**: two authoring-loop passes (Decision 13) plus one corrective third pass after the note
  fix above — all three exited `0` with zero `error` outputs; the third pass is the committed state.
  Wall-clock: pass 1 ≈ 1m58s, pass 2 ≈ 43s, pass 3 (committed) ≈ 43s. Code-cell execution counts are
  sequential `1..50` in the committed notebook.
- **Manifest verification**: `results/eda_reduction_clustering/manifest.json` parses; 17 tables, 16
  figures, 29 metrics, 6 notes (all six ids present); every declared `tables[].path` /
  `figures[].path` resolves to an existing file on disk.
- **Static guards**: both proposal `rg` guards and the spec's ARI/NMI guard return no match in the
  committed notebook. No `random_state=`/`seed=` literal other than `42` appears.
- **Zero-variance VIF branch**: unreachable in this run — no feature in either TTL variant had exactly
  zero variance on this partition (`is_sm_ips_ports` is near-constant but not exactly constant), so
  every VIF row has `status` `ok` or `perfect_collinearity`, never `zero_variance`. The guard code path
  exists and is exercised by unit-level reasoning, not by this specific dataset run.
- **`uv run pytest`**: 193 passed, 4 deselected (`slow`), 0 failed — unchanged from before this change.
- **`data/raw/`**: file timestamps and MD5 checksums unchanged across all three execution passes.
- **Measured size**: `notebooks/01_eda_reduction_clustering.ipynb` = 192K;
  `results/eda_reduction_clustering/` = 2.2M. Combined ≈2.4 MB, well under the proposal's ~10 MB
  estimate — not a finding.
- **Key measured numbers** (with_ttl / without_ttl): `train_rows`=107,740; PCA 90%-variance components
  = 12 / 12; 95%-variance components = 17 / 16; `max_vif` = inf / inf (`ackdat`, `synack`, `tcprtt`
  perfectly collinear in both variants); `correlated_pairs_above_0_9` = 16; K-Means selected `k` = 10 /
  10 (agree); silhouette at selected k = 0.472 / 0.442; floor=0 sensitivity silhouette = 0.468 / 0.437;
  DBSCAN `eps` = 4.873 / 4.914 (chord rule, not degenerate in either variant); DBSCAN clusters
  (excluding noise) = 3 / 3; DBSCAN noise fraction = 0.27% / 0.27%; one-hot share of total variance =
  4.05% / 4.27%.
- **Nothing was committed or staged** — no git mutations were performed. `git status --porcelain`
  confirms only `notebooks/01_eda_reduction_clustering.ipynb` (new) and
  `results/eda_reduction_clustering/**` (new) under this agent's scope; sibling agents'
  concurrently-produced files (`notebooks/02_clustering_reduction.ipynb`,
  `results/clustering_reduction/`, `skills/**`, `tests/test_skills.py`,
  `openspec/changes/notebook-02-clustering-reduction/tasks.md`) were not touched by this agent.
