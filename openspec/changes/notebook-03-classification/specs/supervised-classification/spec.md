# Supervised Classification Specification

## Purpose

Define the requirements for `notebooks/03_classification.ipynb` and its `results/classification/`
producer folder: how the seven classifier families are selected on training data, how the cleaned
testing partition is spent exactly once, which metrics an imbalanced ten-class problem MUST report,
and how the TTL shortcut comparison and the label-noise error floor are disclosed.

## ADDED Requirements

### Requirement: Inherited data rules

This notebook MUST obey every AGENTS.md data rule it consumes, without re-implementing any of them.
The `id` column MUST already be dropped by `nids.cleaning`. `attack_cat` and `label` MUST NEVER be
input features; they are targets only. Every imputer, encoder and scaler MUST be fitted inside a
scikit-learn `Pipeline` on training data only. The `"-"` value in `service` MUST already be mapped
to its own `"none"` category, never imputed with the mode. `proto`, `service` and `state` values
under 1 percent of rows MUST already be grouped into `"other"` before one-hot encoding. Random seed
42 MUST be used for every split, subsample, search and model. Gaussian Process work MUST run on a
stratified subsample of at most 10,000 rows. All of the above MUST be obtained by importing
`nids.columns`, `nids.data`, `nids.preprocessing` and `nids.sampling`; the notebook MUST NOT
duplicate cleaning or preprocessing logic.

#### Scenario: Preprocessing is imported, not re-implemented

- GIVEN the executed notebook
- WHEN its code cells are inspected for cleaning or preprocessing logic
- THEN every transformation MUST come from a `nids` import
- AND no cell MAY define a second imputer, encoder, scaler or rare-category grouper

### Requirement: The target columns never enter a fit as features

`attack_cat` and `label` MUST NEVER appear in any fitted feature matrix. The feature set MUST be
obtained from `nids.columns.feature_columns(include_ttl)`, which is an allow-list, and never from a
set difference. The notebook MUST contain a runtime assertion cell that fails execution when either
target name is present in the selected feature columns or in the fitted preprocessor's
`get_feature_names_out()`.

#### Scenario: The assertion fails a notebook that leaked a target

- GIVEN a fitted pipeline whose `get_feature_names_out()` contains `attack_cat` or `label`
- WHEN the assertion cell runs
- THEN the cell MUST raise
- AND the `nbconvert` execution MUST exit non-zero

### Requirement: No fit or fit_transform ever touches the testing partition

Every estimator, transformer and pipeline MUST be fitted on the cleaned training partition or on a
subsample of it. The cleaned testing partition MAY only be passed to `transform`, `predict`,
`predict_proba` and `decision_function`. The notebook MUST contain a runtime assertion cell proving
that the count of fits performed against the testing partition is zero.

#### Scenario: The testing partition is only transformed

- GIVEN the executed notebook
- WHEN every call passing the testing frame is inspected
- THEN none MAY be `fit`, `fit_transform`, `fit_predict` or `fit_resample`
- AND the test-fit counter asserted at the end of the notebook MUST be zero

#### Scenario: The testing frame is unmodified end to end

- GIVEN the cleaned testing partition as returned by `nids.data.load_clean_partitions`
- WHEN its shape and column order are compared before the first prediction and after the last
- THEN they MUST be identical

### Requirement: Model selection happens entirely on training data, evaluated once on test

Every hyperparameter MUST be selected by `RandomizedSearchCV` over a `StratifiedKFold` split of
training data only, with `n_iter` at most 15 and exactly 3 folds. The cleaned testing partition MUST
be evaluated exactly once, after every model has been finally refit. The notebook MUST NOT contain a
second evaluation pass, a test-informed model choice, or a test-informed threshold choice.

#### Scenario: Search configuration stays inside the declared bounds

- GIVEN any `RandomizedSearchCV` instance constructed by the notebook
- WHEN its `n_iter` and `cv` are read
- THEN `n_iter` MUST be less than or equal to 15
- AND `cv` MUST be a `StratifiedKFold` with `n_splits = 3` and `random_state = 42`

#### Scenario: The test partition is scored exactly once per model and variant

- GIVEN the executed notebook and its manifest
- WHEN the metric `test_evaluations_count` is read
- THEN it MUST equal 1
- AND every test-partition metric in `results/classification/` MUST derive from that single pass

### Requirement: Imbalanced classification reports macro F1 and balanced accuracy

Every multiclass result MUST report accuracy, balanced accuracy, macro F1 and weighted F1 together;
accuracy alone MUST NEVER be reported as a model's headline score. Per-class precision, recall, F1
and support MUST be exported for every model and both TTL variants, and a confusion matrix MUST be
exported for the best model of each variant. Model ranking MUST use macro F1.

#### Scenario: A metrics table carrying accuracy also carries the imbalanced metrics

- GIVEN any table under `results/classification/tables/` reporting test accuracy
- WHEN its columns are read
- THEN `balanced_accuracy`, `macro_f1` and `weighted_f1` MUST also be present on the same row

#### Scenario: Rare classes are visible per class

- GIVEN the per-class metrics tables
- WHEN they are filtered to a rare class such as `Worms`
- THEN a precision, recall, F1 and support value MUST be present for every model and both variants

### Requirement: Every result is reported with and without the TTL shortcut pair

Each of the seven models MUST be searched, refit and evaluated twice: once with `sttl` and
`ct_state_ttl` in the feature set and once without, using
`nids.preprocessing.build_preprocessor(include_ttl_features=...)`. The comparison table MUST carry
both variants and the macro F1 delta between them. The TTL toggle MUST NOT be applied to
`nids.columns.leakage_key_columns()`, which is a dataset-identity key rather than a modelling
exclusion.

#### Scenario: The comparison table exposes the shortcut's contribution

- GIVEN `tables/model_comparison.csv`
- WHEN a model's row is read
- THEN it MUST carry a macro F1 for `with_ttl`, a macro F1 for `without_ttl`, and their signed delta

### Requirement: Compute budget with declared subsample caps and visible training-set sizes

The notebook MUST execute end to end in under approximately 15 minutes on 16 threads and MUST state
its limits explicitly. `GaussianProcessClassifier` MUST be trained on at most 2,000 rows,
`SVC(RBF)` on at most 5,000 rows, and `KNeighborsClassifier` on at most a 20,000-row reference set.
Every subsample MUST be produced by `nids.sampling.stratified_subsample` with `seed = 42` and
`floor = 50` so rare classes survive. `n_jobs = -1` MUST be used wherever supported. Every model
trained on a subsample MUST be flagged as such in every results table, and every table comparing
models MUST carry each model's training row count on the same row.

#### Scenario: A reader can never compare unequal training sizes unknowingly

- GIVEN any table under `results/classification/tables/` that ranks two or more models
- WHEN a row is read
- THEN it MUST carry a `training_rows` value and a boolean `subsampled` flag for that model

#### Scenario: Rare classes survive every subsample

- GIVEN any subsample drawn by the notebook
- WHEN its `attack_cat` value counts are read
- THEN every class present in the source frame MUST be present in the subsample, subject to the
  declared per-class floor

### Requirement: Cost-based binary threshold with explicitly stated costs

The binary attack-vs-normal view MUST export ROC and precision-recall curves and MUST select an
operating threshold by minimising an explicitly stated expected cost. The false-negative to
false-positive cost ratio MUST be stated numerically in the notebook, in the design and in a
manifest note, with a one-line justification. The threshold MUST be selected from out-of-fold
probabilities computed on training data and applied to the testing partition exactly once; it MUST
NOT be selected from test-partition probabilities.

#### Scenario: The threshold's provenance is training data

- GIVEN the cost-optimal threshold reported in the manifest
- WHEN the cell that computed it is inspected
- THEN its probabilities MUST come from a cross-validated fit over training data
- AND no test-partition probability MAY participate in the argmin

#### Scenario: Both operating points are reported

- GIVEN `tables/binary_operating_points.csv`
- WHEN it is read
- THEN it MUST contain the default 0.5 threshold row and the cost-optimal row for each variant
- AND each row MUST carry true positives, false positives, false negatives, true negatives and
  expected cost

### Requirement: The label-noise error floor is reported beside the results

The notebook MUST read the retained feature-identical, label-contradictory row count and its share
of the retained testing set from `results/data_cleaning/manifest.json` and MUST report them beside
the accuracy results as a near-certain minimum error rate no model can eliminate. When
`results/data_cleaning/` does not exist, the notebook MUST continue to completion, MUST register the
affected metric as the string `"unavailable"`, and MUST state the degradation in a note rather than
substituting a hardcoded number.

#### Scenario: The floor is present when the cleaning report exists

- GIVEN `results/data_cleaning/manifest.json` is present
- WHEN `results/classification/manifest.json` is read
- THEN `label_noise_error_floor_rows` and `label_noise_error_floor_share` MUST carry values sourced
  from the cleaning report, not from a literal in the notebook

#### Scenario: The notebook degrades gracefully without the cleaning report

- GIVEN `results/data_cleaning/` does not exist
- WHEN the notebook is executed
- THEN it MUST complete successfully
- AND the floor metrics MUST be the string `"unavailable"`
- AND a note MUST state that the floor could not be read and why

### Requirement: Output contract compliance under notebook_id classification

Every table and figure MUST be registered through `ResultsWriter` under `notebook_id =
"classification"`, writing to `results/classification/` with `tables/`, `figures/` and a
`manifest.json` that declares every exported file. Every artifact name MUST match
`^[a-z][a-z0-9_]{0,63}$` and MUST NOT begin with a digit. Any metric whose computed value is
infinite MUST be registered as the string `"inf"`, never as a float infinity. The notebook MUST NOT
write outside `results/classification/` and MUST NOT modify anything under `data/raw/`.

#### Scenario: The manifest declares every file the folder holds

- GIVEN a completed notebook run
- WHEN `results/classification/manifest.json` is parsed
- THEN every `path` under `tables` and `figures` MUST resolve to an existing file in that folder
- AND every file under `tables/` and `figures/` MUST be declared by the manifest

#### Scenario: An infinite metric is stored as a string

- GIVEN a computed metric that evaluates to infinity
- WHEN it is registered
- THEN its manifest `value` MUST be the string `"inf"`

### Requirement: Deterministic single-pass execution

The notebook MUST execute cleanly from a cold kernel via
`uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1
notebooks/03_classification.ipynb`, exiting zero with no error output cell. It MUST be nbformat v4.5
JSON with fixed, stable cell ids. It MUST use no hardcoded absolute path and MUST add no dependency
beyond scikit-learn, scipy, pandas, numpy, matplotlib, seaborn and umap-learn.

#### Scenario: Two runs agree on every reported number

- GIVEN two consecutive executions over identical cleaned inputs
- WHEN their `tables/` files are compared
- THEN every table file MUST be byte-identical between the two runs

#### Scenario: Execution stays within the declared budget

- GIVEN a completed run on a 16-thread machine
- WHEN `tables/runtime_budget.csv` is read
- THEN the sum of per-model wall-clock seconds plus overhead SHOULD be under approximately 900
  seconds
- AND any overrun MUST be visible in that table rather than undisclosed
