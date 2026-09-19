# Proposal: Notebook 03 — Supervised Classification

> One executed notebook, one `results/` producer folder. No Python source, no dependency change.
> This is the **only** notebook permitted to touch the cleaned testing partition, and only once,
> after all model selection is complete.

## Intent

`project-foundation` delivers the cleaned partitions, the fitted-on-train preprocessing pipeline,
the seeded stratified subsampler and the results writer. `notebook-01` established what the feature
space looks like. Nothing yet answers the project's actual question: **can a model tell attack
families apart on held-out data, and how much of any apparent skill is the `sttl` / `ct_state_ttl`
testbed shortcut?**

This notebook fits seven classifier families on the cleaned training partition, selects every
hyperparameter by cross-validation on training data alone, evaluates once on the cleaned testing
partition, and reports every number twice — with and without the TTL shortcut pair.

## Scope

### In scope

| # | Deliverable |
|---|---|
| 1 | `notebooks/03_classification.ipynb`, committed with outputs, executed in a single `nbconvert` pass. |
| 2 | Seven models inside `build_preprocessor`-based Pipelines: `DummyClassifier`, `LogisticRegression`, `KNeighborsClassifier`, `RandomForestClassifier`, `HistGradientBoostingClassifier`, `SVC(RBF)`, `GaussianProcessClassifier`. |
| 3 | Multiclass target `attack_cat`; a binary attack-vs-normal view (`label`) for ROC, precision-recall and a cost-based threshold. |
| 4 | `RandomizedSearchCV` on train only, `StratifiedKFold(3)`, `n_iter <= 15`, `class_weight="balanced"` where supported. |
| 5 | A single test evaluation reporting accuracy, balanced accuracy, macro F1, weighted F1, per-class precision/recall/F1, confusion matrices and one comparison table across all models and both TTL variants. |
| 6 | `results/classification/` with 17 tables, 9 figures, ~28 metrics and 7 disclosure notes, all registered through `ResultsWriter` (`notebook_id = "classification"`). |
| 7 | The label-noise error floor from `results/data_cleaning/` reported beside every accuracy number, degrading gracefully with a stated message when that folder is absent. |
| 8 | Runtime assertion cells proving the target columns are absent from the feature matrix and that no `fit`/`fit_transform` ever saw the testing partition. |
| 9 | Plain-language SOC conclusions. |

### Out of scope

Deep learning, feature engineering beyond `build_preprocessor`, new dependencies, changes to
`src/nids/**`, any second evaluation pass on the testing partition, calibration studies, and
per-class cost matrices beyond the single binary FN:FP ratio.

## Why now

The foundation's `build_preprocessor_pair()` exists precisely so a supervised notebook cannot forget
AGENTS.md's with-and-without-TTL requirement; enforcement of "both numbers are reported" belongs
here. The testing partition has been sitting unused and leakage-cleaned since `project-foundation`;
this change spends that budget once and records the result permanently under the output contract.

## Hard compute budget

The notebook MUST execute end to end in under ~15 minutes on 16 threads. That budget is met by four
declared limits, not by luck: a 30,000-row stratified **search** subsample used by every
`RandomizedSearchCV` (train-only, so no leakage); `GaussianProcessClassifier` capped at 2,000 rows;
`SVC(RBF)` capped at 5,000 rows; `KNeighborsClassifier` capped at a 20,000-row reference set. Every
subsample-trained model is labelled as such in every table, and the comparison table carries the
training row count on each row so a 2,000-row Gaussian Process is never silently compared against a
107,740-row Random Forest.

## Risks and rollback

| Risk | Mitigation |
|---|---|
| Budget overrun on `GaussianProcessClassifier` or `HistGradientBoostingClassifier`. | Fixed kernel and no search for the Gaussian Process; measured per-model wall clock exported as `tables/runtime_budget.csv` so an overrun is visible, not guessed. |
| The threshold is tuned on test. | The cost threshold is chosen from out-of-fold **training** probabilities and applied once to test; a runtime assertion counts test-partition fits and requires zero. |
| `results/data_cleaning/` not yet produced. | The floor loader degrades to a stated message and a `"unavailable"` string metric; nothing else in the notebook depends on it. |

Rollback: `git revert <sha>` removes `notebooks/03_classification.ipynb` and
`results/classification/` only. No source module, dependency, test or other producer folder is
touched.

## Decisions taken here

D1 KNN gets a third subsample cap (20,000 reference rows) beyond the two the brief names, because
brute-force kNN cost is O(n_train x n_test) and 107,740 x 78,073 x 52 dims alone exceeds the budget.
D2 Hyperparameter search runs on a 30,000-row stratified train subsample; the winning configuration
is then refit on the model's full training allocation.
D3 The binary cost ratio is fixed at FN:FP = 20:1.
D4 The binary view reuses the best multiclass model per variant rather than searching again.
