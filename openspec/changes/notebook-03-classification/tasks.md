# Tasks: Notebook 03 — Supervised Classification

> Unattended execution. There is no user review checkpoint between this task list and `sdd-apply`.
> Single delivery slice, one commit. Strict TDD is disabled for this project
> (`openspec/config.yaml` `strict_tdd: false`); `rules.tasks` states notebooks are exploratory and
> require no pytest tests. `src/nids/**` is owned by a concurrently running agent — every task below
> writes only under `notebooks/` and `results/classification/`, and reads `src/nids/**` read-only.
> `src/nids/results.py` (`ResultsWriter`) lands in a concurrent commit; if it is absent when apply
> starts, stop and report rather than writing a local substitute.

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 1,300 – 1,700 authored lines; plan for ~1,500 |
| 400-line budget risk | High |
| Chained PRs recommended | No |
| Suggested split | Single PR / single commit |
| Delivery strategy | ask-on-risk (session default), resolved to `size:exception` for this change |
| Chain strategy | size-exception |

Decision needed before apply: No. A 47-cell notebook is reviewed section by section, and the
manifest is only complete once every cell has run — a partial commit would leave
`results/classification/manifest.json` declaring files that do not exist, the exact failure
`ResultsWriter.close()` exists to prevent.

### Suggested Work Units

| Unit | Goal | PR | Check | Rollback boundary |
|---|---|---|---|---|
| 1 | Author all 47 cells, execute once, commit the notebook together with `results/classification/` | single commit (`size:exception`) | `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/03_classification.ipynb` exits 0 with no `error` output cell; `uv run pytest` stays green | `git revert <sha>` removes only `notebooks/03_classification.ipynb` and `results/classification/` |

## Phase 1 — Scaffold and guards

- [x] 1.1 Create `notebooks/03_classification.ipynb` as nbformat v4.5 JSON with fixed cell ids `c00`–`c46`.
- [x] 1.2 Cells `c00`–`c02`: title markdown, imports, the constants block from design (`SEED=42`, `CV_FOLDS=3`, `N_ITER_MAX=15`, `SEARCH_ROWS=30_000`, `GP_MAX_ROWS=2_000`, `SVC_MAX_ROWS=5_000`, `KNN_REF_ROWS=20_000`, `COST_FN=20.0`, `COST_FP=1.0`, `SUBSAMPLE_FLOOR=50`, `VARIANT_COLORS`), and the compute-budget markdown table.
- [x] 1.3 Cells `c03`–`c04`: `ResultsWriter("classification")`, `perf_counter()` start, `load_clean_partitions()`, printed shapes and class counts.
- [x] 1.4 Cell `c05`: assertion that `attack_cat` and `label` are absent from `feature_columns(True)` and `feature_columns(False)`, and that the lengths are 39 and 37.
- [x] 1.5 Cell `c06`: `TEST_FIT_COUNT`, the `fit_train_only()` wrapper, `TRAIN_FRAME_ID`, and the test-frame fingerprint.
- [x] 1.6 Cells `c07`–`c08`: the four `stratified_subsample` draws (seed 42, floor 50), `subsample_allocations` and `dataset_shapes` tables, and `load_label_noise_floor()` reading `results/data_cleaning/manifest.json` with a graceful `"unavailable"` path and a stated message.

## Phase 2 — Variants and model inventory

- [x] 2.1 Cells `c09`–`c11`: `VARIANTS`, `build_preprocessor_pair()`, the four target vectors, `CLASSES`.
- [x] 2.2 Cells `c12`–`c14`: `MODEL_SPECS` for all seven models with the exact search spaces, `n_iter` values, search frames, training allocations and `search_n_jobs` from the design's model matrix; register `model_inventory` carrying `training_rows` and `subsampled` on every row.

## Phase 3 — Train-only hyperparameter search

- [x] 3.1 Cells `c15`–`c16`: `run_search()` building `Pipeline([("features", build_preprocessor(include_ttl)), ("model", est)])` inside `RandomizedSearchCV(scoring="f1_macro", cv=StratifiedKFold(3, shuffle=True, random_state=42), refit=False, random_state=42)`; assert `n_iter <= 15` at construction.
- [x] 3.2 Cells `c17`–`c18`: run every search for `with_ttl` then `without_ttl`, timing each with `perf_counter()`. Gaussian Process is skipped (D2, fixed kernel).
- [x] 3.3 Cell `c19`: register `cv_search_results_with_ttl` and `cv_search_results_without_ttl`.

## Phase 4 — Final fits and the test-fit guard

- [x] 4.1 Cells `c20`–`c21`: refit each winner on its full training allocation through `fit_train_only`, `RandomForestClassifier` flipping to `n_jobs=-1`; record per-model wall clock.
- [x] 4.2 Cell `c22`: assert `TEST_FIT_COUNT == 0` and that the test fingerprint is unchanged.

## Phase 5 — The single test evaluation

- [x] 5.1 Cells `c23`–`c24`: one prediction pass per model per variant over `cleaned.test[feature_columns(include_ttl)]`; set `test_evaluations_count = 1`.
- [x] 5.2 Cell `c25`: `test_metrics_with_ttl` and `test_metrics_without_ttl` with accuracy, balanced accuracy, macro F1, weighted F1, `training_rows` and `subsampled` on every row.
- [x] 5.3 Cell `c26`: `model_comparison` with both variants, `macro_f1_delta`, `training_rows` and `subsampled` — the table a reader uses to see why a 2,000-row Gaussian Process is not a 107,740-row Random Forest.
- [x] 5.4 Cells `c27`–`c28`: per-class metrics tables for both variants and long-format confusion matrices for each variant's best model.
- [x] 5.5 Cells `c29`–`c30`: the six multiclass figures, English labels, variant palette, exactly as specified in the design's figure table.

## Phase 6 — Binary view and the cost threshold

- [x] 6.1 Cell `c32`: rebuild each variant's best estimator with its tuned params against the `label` target and refit on full train.
- [x] 6.2 Cell `c33`: `cross_val_predict(method="predict_proba", cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=-1)` on the 30,000-row train search subsample — out-of-fold TRAIN probabilities only.
- [x] 6.3 Cell `c34`: sweep `t` over `linspace(0.01, 0.99, 197)`, minimise `20*FN(t) + 1*FP(t)`, break ties toward the smallest `t`; register `binary_threshold_sweep` and `binary_operating_points` covering 0.5 and `t_star` for both variants.
- [x] 6.4 Cells `c35`–`c37`: a single test `predict_proba` pass, ROC/AUC and PR/AP, `binary_curve_points`, the three binary figures, and the binary metrics — any infinite value registered as the string `"inf"`.

## Phase 7 — Floor, budget, notes, conclusions

- [x] 7.1 Cells `c38`–`c39`: `label_noise_floor` table plus `label_noise_error_floor_rows` and `label_noise_error_floor_share`, sourced from the cleaning report, reported beside the accuracy numbers as a floor no model can beat.
- [x] 7.2 Cell `c40`: `runtime_budget` table (per model, per variant, wall-clock seconds) and `total_runtime_seconds`.
- [x] 7.3 Cell `c41`: register all seven notes — `single_test_evaluation`, `subsample_trained_models`, `label_noise_error_floor`, `ttl_shortcut_comparison`, `cost_threshold_assumption` (state FN:FP = 20:1 and its one-line justification), `benchmark_non_comparability`, `soc_conclusions`.
- [x] 7.4 Cells `c42`–`c43`: plain-language SOC conclusions in markdown — what a SOC would actually deploy, what the TTL delta means about the testbed, what the error floor means for expected alert quality, and what the subsample caps do and do not let us claim.

## Phase 8 — Close and verify

- [x] 8.1 Cells `c44`–`c46`: `writer.close()`, the manifest round-trip assertion (every declared path exists, every file under `tables/` and `figures/` is declared), and the reproducibility/limits markdown.
- [x] 8.2 Execute `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/03_classification.ipynb`; require exit 0, no `error` output cell, and a total wall clock under ~15 minutes.
- [x] 8.3 Read `results/classification/manifest.json` back from disk and confirm 17 tables, 9 figures, ~28 metrics and 7 notes, every artifact name matching `^[a-z][a-z0-9_]{0,63}$`.
- [x] 8.4 Run `uv run pytest` and confirm it is still green (this change adds no test).
- [ ] 8.5 Commit the notebook and `results/classification/` together, Conventional Commit, on a feature branch.

## Acceptance criteria

- [x] A1 `test_evaluations_count` is 1 and no cell fits, fit_transforms or resamples the testing partition.
- [x] A2 Neither `attack_cat` nor `label` appears in any fitted feature matrix; both assertion cells pass.
- [x] A3 Every model-ranking table carries accuracy, balanced accuracy, macro F1, weighted F1, `training_rows` and `subsampled`.
- [x] A4 Every result exists for both `with_ttl` and `without_ttl`, and `model_comparison` carries the signed delta.
- [x] A5 The Gaussian Process trained on at most 2,000 rows, `SVC` on at most 5,000, kNN on at most a 20,000-row reference set, every subsample drawn with seed 42 and floor 50.
- [x] A6 The cost threshold was selected from out-of-fold training probabilities, FN:FP = 20:1 is stated in the notebook and in a note, and both operating points are exported.
- [x] A7 The label-noise floor is reported beside the results, or the degradation message is present if `results/data_cleaning/` was absent.
- [x] A8 `nbconvert` exits 0 in one pass, under ~15 minutes, with no hardcoded absolute path and no new dependency.


## Apply results (verification evidence)

- 8.2 result: `nbconvert` exit 0, no `error` output cell, wall clock **5m26s** (real), well under the ~15 minute budget.
- 8.3 result: `results/classification/manifest.json` — schema_version=1.0, notebook_id=classification,
  **tables=17, figures=9, metrics=30 (~28 target), notes=7**. All 26 declared table/figure paths verified to
  exist on disk; zero undeclared files under `tables/`/`figures/`.
- 8.4 result: `uv run pytest -q` — **181 passed, 1 deselected** (unchanged from the pre-apply baseline; this
  change adds no test).
- Leak guards: Assertion 1 passed (39/37 features, targets disjoint from feature allow-list). Assertion 2
  passed (`TEST_FIT_COUNT=0`, test fingerprint unchanged). `test_evaluations_count=1`. Assertion 3 passed
  (manifest round-trip, 26/26 files present and declared).
- `data/raw/` unchanged (2 files, original sizes/timestamps, confirmed via `ls -la`).
- 8.5 (commit) intentionally **not performed** — explicit apply-time scope restriction: "DO NOT COMMIT OR
  STAGE. No git mutations. The parent commits." `git status --short` confirms only untracked files
  (`notebooks/03_classification.ipynb`, `results/classification/`, plus sibling agents' own untracked
  `notebooks/01_*`, `notebooks/02_*`, `results/eda_reduction_clustering/`, `results/clustering_reduction/`,
  none of which this apply touched); nothing staged, no commits created.

## Deviations recorded during apply

1. **Naming collision fix (design bug, not a scope change).** The design's Table list and Figure table both
   use the name `confusion_matrix_best_with_ttl` / `_without_ttl` — one for a CSV table, one for a PNG
   figure. `ResultsWriter` shares one name registry across `add_table` and `add_figure`
   (`_registered_artifact_names`), so registering both under the identical name raises `ValueError` at
   runtime on the second call. Fixed by keeping the CSV table names as designed (`confusion_matrix_best_with_ttl`
   / `_without_ttl`, the D14 machine-readable source) and renaming the heatmap PNGs to
   `confusion_matrix_heatmap_with_ttl` / `_without_ttl`. All 26 final table+figure names verified unique
   before execution; per the apply brief, the committed `results.py` signature takes precedence over the
   design snippet.
2. **GP/Dummy search step literally skipped, not CV-scored.** Design cell `c17` states "Gaussian Process is
   skipped (D2, fixed kernel)" — implemented literally: `run_search()` returns immediately with
   `best_params={}`, `mean_cv_macro_f1="not_searched"`, `elapsed_seconds=0.0` for any model with
   `param_distributions=None` (dummy and gaussian_process), instead of running an extra `cross_val_score`
   pass. This avoided roughly 30 extra one-vs-rest Gaussian Process sub-fits during the search phase (only
   the single final-fit GP training remains per variant), meaningfully reducing runtime risk with no loss of
   required output — `dummy_macro_f1` is sourced from the TEST metrics table instead, which is the more
   meaningful baseline comparison anyway.
3. **SVC `probability=True` used only for the one binary-view refit and its out-of-fold CV pass** (per the
   apply brief's explicit guidance), not for the multiclass search/final fit, which stays
   `probability=False` for speed since only `.predict` is needed there.

## Key results (see `results/classification/manifest.json` and `tables/` for full detail)

- Best model both variants: **HistGradientBoostingClassifier** — with_ttl macro F1 = 0.5116 (balanced
  accuracy 0.5768, accuracy 0.7430); without_ttl macro F1 = 0.5029 (balanced accuracy 0.5529, accuracy
  0.7415). `macro_f1_ttl_delta` for the with_ttl-best model = **+0.0087** (small TTL shortcut contribution
  for the winning model; larger, up to +0.025, for `knn`).
- Dummy baseline macro F1 = 0.0893 — every real model clears the trivial baseline by a wide margin.
- Worst per-class F1 for the best model (with_ttl): `Backdoor` 0.072, `Analysis` 0.107, `DoS` 0.224 — not
  `Worms` (0.591 despite only 44 test rows, thanks to the floor=50 stratified subsampling keeping it
  represented during search).
- Binary view: ROC-AUC ~0.983 and average precision ~0.987 for both variants. Cost-optimal threshold
  (FN:FP=20:1, chosen from out-of-fold train probabilities): t*=0.075 (with_ttl), t*=0.085 (without_ttl);
  expected-cost reduction vs the default 0.5 threshold: 82.1% (with_ttl), 82.5% (without_ttl).
- Label-noise floor: 4,294 rows / 5.499% of the retained test set (from `results/data_cleaning/manifest.json`).
  The best model's test error (~25.7% for with_ttl) is well above this floor — the floor explains only a
  small fraction of total error, most of the gap is genuine model imperfection on the harder classes.
- Runtime: total_runtime_seconds=321.5s (~5.4 min) inside the notebook's own timer; `nbconvert` wall clock
  5m26s end to end. Slowest single steps: GP final fits (~79s and ~73s per variant); everything else under
  ~14s per step.
