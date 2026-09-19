# Preprocessing Pipeline Specification

## Purpose

Define the fitted boundary of this project: every parameter learned from row statistics MUST be fitted inside a scikit-learn `Pipeline`/`ColumnTransformer` on the training partition only, and a testing partition MUST only ever be transformed, never fitted. This capability restates the `AGENTS.md` data rules that govern it directly: every imputer, encoder, and scaler is fitted inside a Pipeline on training data only (split-first fitting); rare `proto` values (under 1 percent of rows) are grouped into `"other"` before one-hot encoding; the `id` column is never a feature and `attack_cat`/`label` are never used as input features; and the seed for every fit and every downstream model remains 42.

This capability's fitted operations consume the training partition **after** it has passed through the data-cleaning capability. They MUST NEVER fit against the raw, pre-cleaning training partition, because deduplication materially reshapes the categorical distributions those fits depend on.

## Requirements

### Requirement: Rare-category grouping is fitted state, never a hardcoded list

The system MUST provide a fitted transformer that learns, at `fit()` time, the set of `state` and `proto` category values whose share of the fitted training partition's rows is at or above a 1 percent threshold. At `transform()` time, any value not in that learned set — whether it is rare in the training partition or entirely unseen there — MUST be mapped to `"other"`. The learned category set MUST be stored as fitted state on the transformer instance; it MUST NEVER be hardcoded as a literal list anywhere in the source.

#### Scenario: The frequent-category set is learned from training row counts

- GIVEN a training partition where a `proto` value occupies 2 percent of rows and another `proto` value occupies 0.5 percent of rows
- WHEN the rare-category transformer is fitted on that training partition
- THEN the 2 percent value MUST be included in the learned frequent-category set
- AND the 0.5 percent value MUST NOT be included in the learned frequent-category set

#### Scenario: Both rare-in-train and unseen-in-train values map to "other" at transform time

- GIVEN a fitted rare-category transformer whose learned frequent-category set excludes a specific `state` value present in training, and a testing frame containing that value plus a second `state` value that never appeared in training
- WHEN `transform()` is called on the testing frame
- THEN both the rare-in-train value and the entirely-unseen value MUST be mapped to `"other"`
- AND every value present in the learned frequent-category set MUST pass through unchanged

#### Scenario: Fitting on different frames produces different learned category sets

- GIVEN two training frames with materially different `state`/`proto` distributions
- WHEN the rare-category transformer is fitted separately on each frame
- THEN the two fitted transformers MUST report different learned frequent-category sets
- AND this difference MUST be observable from the transformer's fitted state, confirming the set is learned, not hardcoded

### Requirement: The rare-category threshold is fitted on the cleaned training partition only

The rare-category transformer MUST be fitted on the training partition produced by the data-cleaning capability's full cleaning sequence, never on the raw, pre-cleaning training partition. Because deduplication changes the category-frequency distribution materially, fitting on the wrong input silently changes which categories are treated as frequent.

#### Scenario: Fitting against raw, pre-cleaning input is rejected as an integration error

- GIVEN a raw training partition that has not yet passed through the data-cleaning capability's cleaning sequence
- WHEN a caller attempts to fit the rare-category transformer, or any other fitted component of this capability, directly on that raw partition
- THEN the resulting fitted state MUST NOT be treated as valid for this project's Pipeline, and the integration MUST instead fit only on cleaned training output

### Requirement: One-hot encoding is fitted on the training partition only

Category encoding for the pipeline's categorical inputs MUST be fitted on the training partition only, with unseen-category handling configured so that a category value absent from training never raises an error at transform time.

#### Scenario: An unseen category at transform time does not crash the pipeline

- GIVEN a fitted pipeline whose categorical encoder was fit on a training partition that never contained a specific category value
- WHEN a testing frame containing that unseen category value is transformed
- THEN the transform MUST complete without raising an error
- AND the unseen value MUST be handled according to the encoder's configured unseen-category behavior

### Requirement: Skewed volumetric features are log-transformed then scaled, scaler fitted on train only

The pipeline MUST apply a stateless `log1p` transformation to the designated skewed volumetric features, followed by a `StandardScaler` whose mean and standard deviation are fitted on the training partition only. Both steps MUST be composed inside the same `Pipeline` object so that a single `fit(X_train)` followed by `transform(X_test)` performs both steps consistently, with no separate manual step a caller could omit.

#### Scenario: Scaler statistics come from train only

- GIVEN a pipeline fit on a training partition
- WHEN the fitted scaler's mean and standard deviation are inspected
- THEN they MUST reflect only the training partition's log-transformed values
- AND they MUST NOT reflect any row from the testing partition

#### Scenario: log1p and scaling apply consistently to both partitions via the same fitted pipeline

- GIVEN a pipeline fit on the training partition
- WHEN the same fitted pipeline transforms the training partition and, separately, the testing partition
- THEN both transforms MUST apply the same `log1p` function followed by the same fitted scaling parameters
- AND no separate, uncomposed `log1p` step MAY exist outside the pipeline

### Requirement: The TTL shortcut inclusion toggle

The pipeline factory MUST accept a boolean parameter controlling whether `sttl` and `ct_state_ttl` — the known testbed shortcut features — are included in the feature set. This MUST be a structural toggle over which columns enter the transformer; it MUST NOT change any already-fitted value.

#### Scenario: Excluding the TTL shortcut removes both columns from the output

- GIVEN the pipeline factory is called with the TTL-inclusion toggle set to exclude the shortcut features
- WHEN the resulting pipeline transforms a partition
- THEN neither `sttl` nor `ct_state_ttl` MAY be present in the output feature matrix

#### Scenario: Including the TTL shortcut keeps both columns in the output

- GIVEN the pipeline factory is called with the TTL-inclusion toggle set to include the shortcut features
- WHEN the resulting pipeline transforms a partition
- THEN both `sttl` and `ct_state_ttl` MUST be present in the output feature matrix

#### Scenario: Both toggle states are producible from the same factory for paired reporting

- GIVEN the requirement to report supervised results both with and without the TTL shortcut features
- WHEN the pipeline factory is called twice, once with each toggle state, on the same underlying data
- THEN the two resulting pipelines MUST differ only in the presence of `sttl` and `ct_state_ttl`
- AND every other fitted parameter derived from row statistics MUST be computed independently and correctly for each toggle state

### Requirement: The feature set is an explicit allow-list

The set of columns entering the feature matrix MUST be produced by an explicit allow-list function, never by subtracting known target columns from the full column set. `attack_cat` and `label` MUST NEVER appear in the feature set, under either TTL-inclusion toggle state.

#### Scenario: Neither target column appears in the feature matrix regardless of the TTL toggle

- GIVEN the feature-column allow-list function is called once with the TTL toggle enabled and once with it disabled
- WHEN the resulting column lists are inspected
- THEN `attack_cat` MUST NOT appear in either resulting list
- AND `label` MUST NOT appear in either resulting list

#### Scenario: A skipped upstream drop step does not reintroduce a target column

- GIVEN a hypothetical caller error where a target column was not removed by an earlier step
- WHEN the feature-column allow-list function is used to select features
- THEN the allow-list MUST still exclude `attack_cat` and `label`, because it enumerates permitted columns rather than excluding forbidden ones

### Requirement: No fitted component of this capability accepts a test partition for fitting

No public function exposed by this capability's preprocessing module MAY accept a testing partition as an argument for the purpose of fitting. Every fit operation MUST take only a training partition; every function that accepts a testing partition MUST only ever call `transform()` on already-fitted state.

#### Scenario: The preprocessing surface exposes no test-fitting path

- GIVEN the complete public surface of this capability's preprocessing module
- WHEN that surface is inspected for functions that both fit a component and accept a testing partition
- THEN no such function MAY exist
