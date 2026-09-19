# Design: Notebook 03 — Supervised Classification

One notebook, `notebooks/03_classification.ipynb`, 47 cells, `notebook_id = "classification"`.
Seed 42 everywhere. Executes in under ~15 minutes on 16 threads.

## Decisions

| # | Decision |
|---|---|
| D1 | Search runs on a 30,000-row stratified train subsample; the winner is refit on the model's full training allocation — the only way 14 searches fit the budget, and still train-only. |
| D2 | `GaussianProcessClassifier` gets no search at all (fixed `1.0 * RBF(1.0)`, `n_restarts_optimizer=0`) because O(n^3) x 10 one-vs-rest fits cannot absorb a search. |
| D3 | `KNeighborsClassifier` is capped at a 20,000-row reference set: brute-force kNN is O(n_train x n_test) and 107,740 x 78,073 x ~52 dims alone blows the budget. Third cap beyond the brief's two, disclosed in every table. |
| D4 | Outer `RandomizedSearchCV(n_jobs=-1)`, inner estimator `n_jobs=1` for `RandomForestClassifier`, to avoid thread oversubscription; the final refit flips to `n_jobs=-1`. |
| D5 | `HistGradientBoostingClassifier` search uses outer `n_jobs=4` because its OpenMP inner parallelism already saturates cores. |
| D6 | Ranking metric is macro F1 — the class distribution spans `Normal` ~51,890 against `Worms` ~127. |
| D7 | The binary view reuses each variant's best multiclass estimator with its tuned params, refit on `label`. No second search. |
| D8 | FN:FP = 20:1. One line: a missed intrusion carries incident-response and breach cost roughly an order of magnitude above one analyst triaging a false alarm; 20:1 is a deliberately conservative SOC figure, stated as an assumption, not a measurement. |
| D9 | The cost threshold is chosen from `cross_val_predict` out-of-fold probabilities on the 30,000-row train search subsample, then applied once to test. Choosing it on test would be selection on test. |
| D10 | Two TTL variants are one dict iterated once, never a copy-pasted cell: `build_preprocessor_pair()` returns both. |
| D11 | The test partition is guarded by a counter: a `fit_train_only()` wrapper increments a module-level counter if the frame it receives is not the train frame; the final assertion requires zero. |
| D12 | The label-noise floor is read from `results/data_cleaning/manifest.json` metrics, never hardcoded; missing folder degrades to the string `"unavailable"` plus a note. |
| D13 | Figures use an explicit two-colour variant palette (`with_ttl` `#2E5A88`, `without_ttl` `#C0603A`) so the TTL comparison reads identically in every chart. |
| D14 | Confusion matrices are exported both as a long-format CSV (true, pred, count, share) and as a row-normalised heatmap; the CSV is the machine-readable source. |
| D15 | Every per-model wall clock is measured with `time.perf_counter()` and exported, so an overrun is visible rather than guessed. |
| D16 | `set_output(transform="pandas")` is not used; the pipelines stay on numpy arrays, and feature names come from `get_feature_names_out()` only for the assertion cell. |

## Constants block (cell `c01`)

```python
SEED = 42
CV_FOLDS = 3
N_ITER_MAX = 15
SEARCH_ROWS = 30_000     # stratified TRAIN subsample used by every search
GP_MAX_ROWS = 2_000      # O(n^3) time, O(n^2) memory, one fit per class
SVC_MAX_ROWS = 5_000     # O(n^2..n^3)
KNN_REF_ROWS = 20_000    # O(n_train x n_test) at predict time
COST_FN, COST_FP = 20.0, 1.0
SUBSAMPLE_FLOOR = 50
VARIANT_COLORS = {"with_ttl": "#2E5A88", "without_ttl": "#C0603A"}
```

## `nids` calls used

`nids.data.load_clean_partitions()` · `nids.columns.feature_columns(include_ttl)` ·
`nids.columns.TARGET_COLUMNS` · `nids.preprocessing.build_preprocessor(include_ttl_features=...)` ·
`nids.preprocessing.build_preprocessor_pair()` · `nids.sampling.stratified_subsample(df,
stratify_by="attack_cat", max_rows=..., floor=50, seed=42)` and `SubsampleResult.to_frame()` ·
`nids.results.ResultsWriter("classification")` · `nids.paths.results_root()`.
Nothing else. No cleaning or preprocessing logic is written in the notebook.

## Cell-by-cell outline

| Cell | Kind | Content |
|---|---|---|
| c00 | md | Title, question, the "test partition is spent once" statement. |
| c01 | code | Imports (`nids`, sklearn, pandas, numpy, matplotlib, seaborn, time, json), constants block above. |
| c02 | md | The hard compute budget, stated as a table: the four caps, `n_jobs=-1`, `n_iter<=15`, 3 folds. |
| c03 | code | `writer = ResultsWriter("classification")`; `t_notebook = perf_counter()`. |
| c04 | code | `cleaned = load_clean_partitions()`; print train/test shapes and class counts. |
| c05 | code | **Assertion 1**: `set(TARGET_COLUMNS).isdisjoint(feature_columns(True))` and `(False)`; lengths are 39 / 37. |
| c06 | code | `TEST_FIT_COUNT = 0`; `fit_train_only(estimator, X, y)` wrapper + `TRAIN_FRAME_ID = id(cleaned.train)`; `TEST_FINGERPRINT = (cleaned.test.shape, tuple(cleaned.test.columns))`. |
| c07 | code | Subsamples: `search_sub` (30k), `gp_sub` (2k), `svc_sub` (5k), `knn_sub` (20k); register `subsample_allocations`. |
| c08 | code | `load_label_noise_floor()` reading `results/data_cleaning/manifest.json`; graceful `"unavailable"`. Register `dataset_shapes`. |
| c09 | md | The two-variant structure. |
| c10 | code | `VARIANTS = {"with_ttl": True, "without_ttl": False}`; `PREPROCESSORS = build_preprocessor_pair()` (rebuilt per fit, never reused fitted). |
| c11 | code | `y_train = cleaned.train["attack_cat"]`, `y_test = cleaned.test["attack_cat"]`, `y_train_bin = cleaned.train["label"]`, `y_test_bin = cleaned.test["label"]`; `CLASSES = sorted(y_train.unique())`. |
| c12 | md | Model inventory and why each cap exists. |
| c13 | code | `MODEL_SPECS` (table below): estimator factory, param distributions, `n_iter`, training allocation, `subsampled` flag. |
| c14 | code | Register `model_inventory`. |
| c15 | md | Search protocol: train only, `StratifiedKFold(3, shuffle=True, random_state=42)`, `scoring="f1_macro"`. |
| c16 | code | `run_search(spec, variant)` → `Pipeline([("features", build_preprocessor(include_ttl)), ("model", est)])` wrapped in `RandomizedSearchCV(..., refit=False, n_jobs=spec.search_n_jobs, random_state=42)`, fitted on `spec.search_frame`. |
| c17 | code | Run all searches, `with_ttl`; collect best params, mean CV macro F1, elapsed. |
| c18 | code | Run all searches, `without_ttl`. |
| c19 | code | Register `cv_search_results_with_ttl`, `cv_search_results_without_ttl`. |
| c20 | md | Final fits. |
| c21 | code | Refit each winner on its full training allocation via `fit_train_only`; store `FITTED[variant][model]`; record elapsed. |
| c22 | code | **Assertion 2**: `TEST_FIT_COUNT == 0` and the test fingerprint is unchanged. |
| c23 | md | **The single test evaluation.** |
| c24 | code | One pass: `PREDICTIONS[variant][model] = pipe.predict(cleaned.test[feature_columns(include_ttl)])`; `test_evaluations_count = 1`. |
| c25 | code | `test_metrics_with_ttl`, `test_metrics_without_ttl`: accuracy, balanced accuracy, macro F1, weighted F1, `training_rows`, `subsampled`. |
| c26 | code | `model_comparison`: both variants side by side plus `macro_f1_delta`, `training_rows`, `subsampled`. |
| c27 | code | `per_class_metrics_with_ttl` / `_without_ttl` from `classification_report(output_dict=True)`. |
| c28 | code | `confusion_matrix_best_with_ttl` / `_without_ttl`, long format (`true`, `pred`, `count`, `row_share`). |
| c29 | code | Figures `model_comparison_macro_f1`, `model_comparison_balanced_accuracy`. |
| c30 | code | Figures `confusion_matrix_best_with_ttl`, `confusion_matrix_best_without_ttl`, `per_class_f1_best_model`, `training_rows_vs_macro_f1`. |
| c31 | md | The binary attack-vs-normal view and its cost model. |
| c32 | code | Per variant, rebuild the best multiclass estimator with its tuned params, target `label`, refit on full train. |
| c33 | code | `cross_val_predict(..., cv=StratifiedKFold(3,...), method="predict_proba", n_jobs=-1)` on `search_sub` → out-of-fold TRAIN probabilities. |
| c34 | code | Cost sweep + `t_star`; register `binary_threshold_sweep`, `binary_operating_points`. |
| c35 | code | Single test `predict_proba`; ROC/AUC and PR/AP per variant; register `binary_curve_points`. |
| c36 | code | Figures `roc_curves_binary`, `precision_recall_curves_binary`, `cost_vs_threshold`. |
| c37 | code | Register binary metrics. |
| c38 | md | The label-noise error floor, beside the results. |
| c39 | code | Register `label_noise_floor` table, floor metrics, floor note. |
| c40 | code | Register `runtime_budget` and `total_runtime_seconds`. |
| c41 | code | Register all seven notes. |
| c42 | md | Plain-language SOC conclusions. |
| c43 | code | Print the final summary table. |
| c44 | code | `writer.close()`; print manifest path and pruned files. |
| c45 | code | **Assertion 3**: re-read `manifest.json`, every declared path exists, every file under `tables/`+`figures/` is declared. |
| c46 | md | Reproducibility, limits, what this notebook does not claim. |

## Model matrix, search spaces and runtime estimate

Search space sizes are deliberately small; `n_iter` never exceeds 15 and folds are always 3.

| Model | Search space (`n_iter`) | Search frame | Final training rows | Est. s / variant |
|---|---|---|---|---|
| `DummyClassifier(strategy="stratified", random_state=42)` | none | — | 107,740 | 1 |
| `LogisticRegression(class_weight="balanced", max_iter=300, tol=1e-3)` | `C` loguniform(1e-2, 1e2) (5) | `search_sub` | 107,740 | 25 |
| `KNeighborsClassifier(n_jobs=-1)` | `n_neighbors` {5,11,21,31} x `weights` {uniform, distance} (6) | `knn_sub` | 20,000 (**subsampled**) | 30 |
| `RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42)` | `max_depth` {None,16,24} x `max_features` {sqrt,0.3} x `min_samples_leaf` {1,2,5} (5) | `search_sub` | 107,740 | 45 |
| `HistGradientBoostingClassifier(max_iter=150, early_stopping=True, random_state=42)` | `learning_rate` {0.05,0.1,0.2} x `max_leaf_nodes` {31,63} x `l2_regularization` {0,1} (5) | `search_sub` | 107,740 | 80 |
| `SVC(kernel="rbf", class_weight="balanced", probability=False, cache_size=1000, random_state=42)` | `C` {1,10,100} x `gamma` {scale,0.01,0.1} (5) | `svc_sub` | 5,000 (**subsampled**) | 30 |
| `GaussianProcessClassifier(kernel=1.0*RBF(1.0), n_restarts_optimizer=0, max_iter_predict=20, multi_class="one_vs_rest", n_jobs=-1, random_state=42)` | none (**D2**) | — | 2,000 (**subsampled**) | 45 |

Per variant ≈ 256 s; two variants ≈ 512 s. Data load and cleaning ≈ 45 s, binary view ≈ 120 s,
figures and writes ≈ 40 s. **Total ≈ 12 minutes**, inside the ~15-minute budget with ~20 % headroom.
If `runtime_budget.csv` shows any model over 150 s per variant, apply should report it rather than
silently exceeding the budget.

## Two-variant TTL structure

One loop, never a duplicated cell. For `variant, include_ttl in VARIANTS.items()`: a fresh
`build_preprocessor(include_ttl_features=include_ttl)` is constructed per fit (never a reused fitted
object), the feature frame is `frame[feature_columns(include_ttl)]` (39 vs 37 names), and every
registered artifact name carries the `_with_ttl` / `_without_ttl` suffix or a `variant` column.
`leakage_key_columns()` is never passed the toggle — dataset identity always includes both TTL
columns. `model_comparison.csv` is the single table where both variants meet, carrying
`macro_f1_with_ttl`, `macro_f1_without_ttl` and `macro_f1_delta`.

## Cost-based threshold arithmetic

Binary target is `label` (1 = attack). For a threshold `t` applied to `P(attack)`:

```
FN(t) = #{y = 1 and p <  t}          FP(t) = #{y = 0 and p >= t}
expected_cost(t) = COST_FN * FN(t) + COST_FP * FP(t)        # COST_FN = 20, COST_FP = 1
t_star = argmin over t in linspace(0.01, 0.99, 197) of expected_cost(t)   # ties -> smallest t
```

The argmin runs on **out-of-fold training probabilities** (D9). `t_star` is then applied once to the
single test `predict_proba` pass. Reported per variant: `t_star`, expected cost at `t_star`,
expected cost at 0.5, the percentage cost reduction, and the full confusion counts at both points.
A cost ratio that would divide by zero registers as the string `"inf"`.

## Figure specifications

All labels, titles, legends and tick labels in English. `dpi=150` via `ResultsWriter`.

| Figure | Type | Axes | Colour |
|---|---|---|---|
| `model_comparison_macro_f1` | grouped horizontal bars | y: model (ranked); x: `Macro F1 on the test partition` (0–1) | variant palette |
| `model_comparison_balanced_accuracy` | grouped horizontal bars | y: model; x: `Balanced accuracy on the test partition` | variant palette |
| `confusion_matrix_best_with_ttl` | annotated heatmap, row-normalised | x: `Predicted attack category`; y: `True attack category` | `Blues` |
| `confusion_matrix_best_without_ttl` | annotated heatmap, row-normalised | same | `Oranges` |
| `per_class_f1_best_model` | grouped vertical bars | x: `Attack category` (support desc); y: `F1 score` | variant palette |
| `roc_curves_binary` | line | x: `False positive rate`; y: `True positive rate`; chance diagonal grey dashed | variant palette |
| `precision_recall_curves_binary` | line | x: `Recall`; y: `Precision`; prevalence baseline grey dashed | variant palette |
| `cost_vs_threshold` | line + markers | x: `Decision threshold on P(attack)`; y: `Expected cost (FN:FP = 20:1)`; dashed vertical at `t_star`, dot at 0.5 | variant palette |
| `training_rows_vs_macro_f1` | annotated scatter, log x | x: `Training rows (log scale)`; y: `Macro F1`; every point labelled with its model | variant palette |

## ResultsWriter registration inventory

**Tables (16):** `dataset_shapes`, `subsample_allocations`, `model_inventory`,
`cv_search_results_with_ttl`, `cv_search_results_without_ttl`, `test_metrics_with_ttl`,
`test_metrics_without_ttl`, `model_comparison`, `per_class_metrics_with_ttl`,
`per_class_metrics_without_ttl`, `confusion_matrix_best_with_ttl`,
`confusion_matrix_best_without_ttl`, `binary_curve_points`, `binary_threshold_sweep`,
`binary_operating_points`, `label_noise_floor`, plus `runtime_budget` (17 with it).

**Figures (9):** as specified above.

**Metrics (~28):** `n_train_rows`, `n_test_rows`, `n_classes`, `n_features_with_ttl`,
`n_features_without_ttl`, `random_seed`, `cv_folds`, `search_n_iter_max`, `search_subsample_rows`,
`gp_training_rows`, `svc_training_rows`, `knn_reference_rows`, `test_evaluations_count`,
`dummy_macro_f1`, `best_model_with_ttl`, `best_macro_f1_with_ttl`,
`best_balanced_accuracy_with_ttl`, `best_model_without_ttl`, `best_macro_f1_without_ttl`,
`best_balanced_accuracy_without_ttl`, `macro_f1_ttl_delta`, `binary_roc_auc_with_ttl`,
`binary_average_precision_with_ttl`, `cost_ratio_fn_to_fp`, `cost_optimal_threshold_with_ttl`,
`cost_optimal_threshold_without_ttl`, `expected_cost_reduction_with_ttl`,
`label_noise_error_floor_rows`, `label_noise_error_floor_share`, `total_runtime_seconds`.

**Notes (7):** `single_test_evaluation`, `subsample_trained_models`, `label_noise_error_floor`,
`ttl_shortcut_comparison`, `cost_threshold_assumption`, `benchmark_non_comparability`,
`soc_conclusions`.

Every name matches `^[a-z][a-z0-9_]{0,63}$` and none starts with a digit.

## Structural leak prevention in this notebook

| Guard | Where |
|---|---|
| Targets are an allow-list, never a set difference | `feature_columns()`, asserted in `c05` |
| `remainder="drop"` in the preprocessor | inherited from `build_preprocessor` |
| Every fit goes through `fit_train_only` | `c06`, counter asserted zero in `c22` |
| Test frame fingerprint unchanged | `c06` → `c22` |
| Threshold chosen on out-of-fold train probabilities | `c33` → `c34` |
| Exactly one `predict` pass over test | `c24`, metric `test_evaluations_count = 1` |
