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

- [ ] 1.1 Create `notebooks/03_classification.ipynb` as nbformat v4.5 JSON with fixed cell ids `c00`–`c46`.
- [ ] 1.2 Cells `c00`–`c02`: title markdown, imports, the constants block from design (`SEED=42`, `CV_FOLDS=3`, `N_ITER_MAX=15`, `SEARCH_ROWS=30_000`, `GP_MAX_ROWS=2_000`, `SVC_MAX_ROWS=5_000`, `KNN_REF_ROWS=20_000`, `COST_FN=20.0`, `COST_FP=1.0`, `SUBSAMPLE_FLOOR=50`, `VARIANT_COLORS`), and the compute-budget markdown table.
- [ ] 1.3 Cells `c03`–`c04`: `ResultsWriter("classification")`, `perf_counter()` start, `load_clean_partitions()`, printed shapes and class counts.
- [ ] 1.4 Cell `c05`: assertion that `attack_cat` and `label` are absent from `feature_columns(True)` and `feature_columns(False)`, and that the lengths are 39 and 37.
- [ ] 1.5 Cell `c06`: `TEST_FIT_COUNT`, the `fit_train_only()` wrapper, `TRAIN_FRAME_ID`, and the test-frame fingerprint.
- [ ] 1.6 Cells `c07`–`c08`: the four `stratified_subsample` draws (seed 42, floor 50), `subsample_allocations` and `dataset_shapes` tables, and `load_label_noise_floor()` reading `results/data_cleaning/manifest.json` with a graceful `"unavailable"` path and a stated message.

## Phase 2 — Variants and model inventory

- [ ] 2.1 Cells `c09`–`c11`: `VARIANTS`, `build_preprocessor_pair()`, the four target vectors, `CLASSES`.
- [ ] 2.2 Cells `c12`–`c14`: `MODEL_SPECS` for all seven models with the exact search spaces, `n_iter` values, search frames, training allocations and `search_n_jobs` from the design's model matrix; register `model_inventory` carrying `training_rows` and `subsampled` on every row.

## Phase 3 — Train-only hyperparameter search

- [ ] 3.1 Cells `c15`–`c16`: `run_search()` building `Pipeline([("features", build_preprocessor(include_ttl)), ("model", est)])` inside `RandomizedSearchCV(scoring="f1_macro", cv=StratifiedKFold(3, shuffle=True, random_state=42), refit=False, random_state=42)`; assert `n_iter <= 15` at construction.
- [ ] 3.2 Cells `c17`–`c18`: run every search for `with_ttl` then `without_ttl`, timing each with `perf_counter()`. Gaussian Process is skipped (D2, fixed kernel).
- [ ] 3.3 Cell `c19`: register `cv_search_results_with_ttl` and `cv_search_results_without_ttl`.

## Phase 4 — Final fits and the test-fit guard

- [ ] 4.1 Cells `c20`–`c21`: refit each winner on its full training allocation through `fit_train_only`, `RandomForestClassifier` flipping to `n_jobs=-1`; record per-model wall clock.
- [ ] 4.2 Cell `c22`: assert `TEST_FIT_COUNT == 0` and that the test fingerprint is unchanged.

## Phase 5 — The single test evaluation

- [ ] 5.1 Cells `c23`–`c24`: one prediction pass per model per variant over `cleaned.test[feature_columns(include_ttl)]`; set `test_evaluations_count = 1`.
- [ ] 5.2 Cell `c25`: `test_metrics_with_ttl` and `test_metrics_without_ttl` with accuracy, balanced accuracy, macro F1, weighted F1, `training_rows` and `subsampled` on every row.
- [ ] 5.3 Cell `c26`: `model_comparison` with both variants, `macro_f1_delta`, `training_rows` and `subsampled` — the table a reader uses to see why a 2,000-row Gaussian Process is not a 107,740-row Random Forest.
- [ ] 5.4 Cells `c27`–`c28`: per-class metrics tables for both variants and long-format confusion matrices for each variant's best model.
- [ ] 5.5 Cells `c29`–`c30`: the six multiclass figures, English labels, variant palette, exactly as specified in the design's figure table.

## Phase 6 — Binary view and the cost threshold

- [ ] 6.1 Cell `c32`: rebuild each variant's best estimator with its tuned params against the `label` target and refit on full train.
- [ ] 6.2 Cell `c33`: `cross_val_predict(method="predict_proba", cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=-1)` on the 30,000-row train search subsample — out-of-fold TRAIN probabilities only.
- [ ] 6.3 Cell `c34`: sweep `t` over `linspace(0.01, 0.99, 197)`, minimise `20*FN(t) + 1*FP(t)`, break ties toward the smallest `t`; register `binary_threshold_sweep` and `binary_operating_points` covering 0.5 and `t_star` for both variants.
- [ ] 6.4 Cells `c35`–`c37`: a single test `predict_proba` pass, ROC/AUC and PR/AP, `binary_curve_points`, the three binary figures, and the binary metrics — any infinite value registered as the string `"inf"`.

## Phase 7 — Floor, budget, notes, conclusions

- [ ] 7.1 Cells `c38`–`c39`: `label_noise_floor` table plus `label_noise_error_floor_rows` and `label_noise_error_floor_share`, sourced from the cleaning report, reported beside the accuracy numbers as a floor no model can beat.
- [ ] 7.2 Cell `c40`: `runtime_budget` table (per model, per variant, wall-clock seconds) and `total_runtime_seconds`.
- [ ] 7.3 Cell `c41`: register all seven notes — `single_test_evaluation`, `subsample_trained_models`, `label_noise_error_floor`, `ttl_shortcut_comparison`, `cost_threshold_assumption` (state FN:FP = 20:1 and its one-line justification), `benchmark_non_comparability`, `soc_conclusions`.
- [ ] 7.4 Cells `c42`–`c43`: plain-language SOC conclusions in markdown — what a SOC would actually deploy, what the TTL delta means about the testbed, what the error floor means for expected alert quality, and what the subsample caps do and do not let us claim.

## Phase 8 — Close and verify

- [ ] 8.1 Cells `c44`–`c46`: `writer.close()`, the manifest round-trip assertion (every declared path exists, every file under `tables/` and `figures/` is declared), and the reproducibility/limits markdown.
- [ ] 8.2 Execute `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/03_classification.ipynb`; require exit 0, no `error` output cell, and a total wall clock under ~15 minutes.
- [ ] 8.3 Read `results/classification/manifest.json` back from disk and confirm 17 tables, 9 figures, ~28 metrics and 7 notes, every artifact name matching `^[a-z][a-z0-9_]{0,63}$`.
- [ ] 8.4 Run `uv run pytest` and confirm it is still green (this change adds no test).
- [ ] 8.5 Commit the notebook and `results/classification/` together, Conventional Commit, on a feature branch.

## Acceptance criteria

- [ ] A1 `test_evaluations_count` is 1 and no cell fits, fit_transforms or resamples the testing partition.
- [ ] A2 Neither `attack_cat` nor `label` appears in any fitted feature matrix; both assertion cells pass.
- [ ] A3 Every model-ranking table carries accuracy, balanced accuracy, macro F1, weighted F1, `training_rows` and `subsampled`.
- [ ] A4 Every result exists for both `with_ttl` and `without_ttl`, and `model_comparison` carries the signed delta.
- [ ] A5 The Gaussian Process trained on at most 2,000 rows, `SVC` on at most 5,000, kNN on at most a 20,000-row reference set, every subsample drawn with seed 42 and floor 50.
- [ ] A6 The cost threshold was selected from out-of-fold training probabilities, FN:FP = 20:1 is stated in the notebook and in a note, and both operating points are exported.
- [ ] A7 The label-noise floor is reported beside the results, or the degradation message is present if `results/data_cleaning/` was absent.
- [ ] A8 `nbconvert` exits 0 in one pass, under ~15 minutes, with no hardcoded absolute path and no new dependency.
