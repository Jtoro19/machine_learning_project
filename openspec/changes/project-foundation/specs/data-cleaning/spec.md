# Data Cleaning Specification

## Purpose

Define the dataset-level, unfitted cleaning operations that every notebook depends on, in one mandated order, so no two notebooks disagree on the training and testing row counts they start from. This capability restates and enforces the project's data rules from `AGENTS.md`: the `id` column is dropped and never enters the feature set; `attack_cat` and `label` are never used as input features; every fitted transformation happens after this capability's output, split-first, on the training partition only; `"-"` in `service` is treated as its own category, never imputed with the mode; rare `proto` values are grouped only after this capability's cleaned training output is available; and the seed for every downstream split, subsample, or model remains 42.

Cleaned-partition row counts are **outputs** of this capability, never **inputs** to it. No requirement, and no implementation of this capability, may assert a literal cleaned row count — every such value is a run-time output of the cleaning report (see the results-output-contract capability). Only the raw, unmodified input shapes (175,341 rows by 45 columns for training; 82,332 rows by 45 columns for testing) describe fixed, pre-existing properties of the input, and MAY be asserted directly.

## Requirements

### Requirement: Mandated cleaning step order

The system MUST execute the following steps, in exactly this order, for every cleaning run. The order MUST be encoded so that no caller can invoke the steps out of sequence or skip a step silently.

| Step | Operation | Partition(s) | Mutates rows? |
|---|---|---|---|
| 1 | Drop the fixed columns `id`, `stcpb`, `dtcpb`, `is_ftp_login` | Both | No |
| 2 | Map `service` value `"-"` to `"none"` | Both | No |
| 3 | Drop exact duplicate rows, comparing the full post-drop row | Train only | Yes |
| 4 | Count duplicate rows within the test partition (diagnostic) | Test | No — rows are kept |
| 5 | Drop test rows matching a train row on the configured comparison key | Test only | Yes |
| 6 | Count train feature combinations carrying more than one `attack_cat` (diagnostic) | Train | No |

#### Scenario: Column drop and service mapping run before any row comparison

- GIVEN a fixture whose duplicate-row status under steps 3 and 5 would differ depending on whether the dropped columns are still present
- WHEN the full cleaning sequence runs on that fixture
- THEN steps 1 and 2 MUST have completed before step 3 or step 5 evaluates any row for duplication or leakage
- AND the row-comparison outcome MUST reflect the post-drop, post-mapping columns only

#### Scenario: The step order is not reassignable by a caller

- GIVEN the public cleaning entry point that runs steps 1 through 6
- WHEN it is invoked on a train and test frame
- THEN it MUST perform all six steps in the mandated order as a single operation
- AND no separate public function MAY allow steps 3–6 to run before steps 1–2 have completed on the same frame

### Requirement: Fixed column drop and service normalization apply to both partitions unfitted

Dropping `id`, `stcpb`, `dtcpb`, and `is_ftp_login`, and mapping `service` value `"-"` to `"none"`, MUST be applied identically to both the training and testing partitions. Neither operation MAY depend on any row-frequency statistic, learned parameter, or fitted state — they are fixed, constant operations. The `"-"` value in `service` MUST be treated as an explicit category to rename, never imputed with the training partition's mode or any other frequency-derived value.

#### Scenario: Exactly the four fixed columns are dropped from both partitions

- GIVEN the raw training and testing partitions before cleaning
- WHEN the column-drop step runs on each partition
- THEN `id`, `stcpb`, `dtcpb`, and `is_ftp_login` MUST be absent from both resulting frames
- AND every other original column MUST remain present in both resulting frames

#### Scenario: Service `"-"` values map to `"none"` without using the mode

- GIVEN a partition containing one or more rows where `service` equals `"-"`
- WHEN the service-mapping step runs
- THEN every `"-"` value MUST become `"none"`
- AND the count of resulting `"none"` values MUST equal the count of `"-"` values before mapping
- AND no most-frequent (mode) `service` value MAY be computed or substituted in place of `"-"`

### Requirement: Train-only exact-duplicate removal

The system MUST drop exact duplicate rows from the training partition only, comparing the full post-drop, post-mapping row. This step MUST NOT be applied to the testing partition.

#### Scenario: Train row count strictly decreases when duplicates exist

- GIVEN a training partition containing at least one exact duplicate row after steps 1–2
- WHEN the train-deduplication step runs
- THEN the resulting training row count MUST be strictly less than the row count before the step
- AND every remaining training row MUST be unique on the full post-drop row

#### Scenario: Test partition is untouched by train deduplication

- GIVEN a training and a testing partition, each having completed steps 1–2
- WHEN the train-deduplication step runs
- THEN the testing partition's row count MUST be unchanged
- AND no row of the testing partition MAY be inspected, reordered, or dropped by this step

### Requirement: Test-internal duplicate diagnostic never removes rows

The system MUST count duplicate rows within the testing partition as a read-only diagnostic. This count MUST be reported. Test-internal duplicates MUST NOT be dropped by this or any other step in this capability.

#### Scenario: Test-internal duplicates are counted but retained

- GIVEN a testing partition containing at least one row that duplicates another row within the same partition
- WHEN the test-internal-duplicate diagnostic runs
- THEN it MUST return a positive count
- AND the testing partition's row count MUST be unchanged before and after the diagnostic runs

### Requirement: Parameterized cross-partition leakage removal, test-only and direction-safe

The system MUST provide a leakage-removal operation that drops testing rows matching a training row on a configured comparison key. The comparison key MUST be an explicit parameter with exactly two supported values: `"full_row"` (the default — features plus `attack_cat` plus `label`) and `"features_only"` (features alone). This operation MUST only ever remove rows from the testing partition; it MUST NEVER remove, reorder, or otherwise mutate any row of the training partition, regardless of which comparison key is configured.

#### Scenario: Leakage removal reduces test rows and leaves train untouched under the default key

- GIVEN a training partition and a testing partition where at least one testing row exactly matches a training row on the full post-drop row (including `attack_cat` and `label`)
- WHEN leakage removal runs with the default `"full_row"` comparison key
- THEN the resulting testing row count MUST be strictly less than the row count before the step
- AND the resulting training row count MUST be exactly equal to the row count before the step

#### Scenario: Leakage removal never mutates train regardless of comparison key

- GIVEN a training partition and a testing partition
- WHEN leakage removal runs once with `"full_row"` and, independently, once with `"features_only"` on the same original inputs
- THEN the training partition returned by each run MUST be identical to the training partition passed in
- AND only the testing partition MAY differ in row count between the two runs

#### Scenario: Feature-identical, label-contradictory test rows are retained and counted under the default key

- GIVEN a training partition and a testing partition where at least one testing row matches a training row on features alone but disagrees on `attack_cat` or `label`
- WHEN leakage removal runs with the default `"full_row"` comparison key
- THEN that testing row MUST NOT be removed
- AND the operation MUST report a `retained_contradictory` diagnostic value that counts it

#### Scenario: The features-only key never retains a contradictory row

- GIVEN a training partition and a testing partition containing feature-identical, label-contradictory row pairs
- WHEN leakage removal runs with the `"features_only"` comparison key
- THEN the `retained_contradictory` diagnostic value MUST be zero
- AND every feature-identical testing row, regardless of label agreement, MUST be removed

#### Scenario: Removed and retained counts satisfy the cross-key arithmetic identity

- GIVEN the same training and testing partitions, both already having completed steps 1–4
- WHEN leakage removal runs once with `"full_row"` and once with `"features_only"`
- THEN the sum of `removed` and `retained_contradictory` under `"full_row"` MUST equal `removed` under `"features_only"`

### Requirement: Label-noise diagnostic on the training partition

The system MUST count the number of distinct feature combinations in the training partition that carry more than one `attack_cat` value, as a read-only diagnostic. This step MUST NOT mutate the training partition.

#### Scenario: Conflicting feature combinations are counted without mutation

- GIVEN a training partition containing at least one feature combination that appears with two different `attack_cat` values across separate rows
- WHEN the label-noise diagnostic runs
- THEN it MUST return a count of at least one
- AND the training partition's row count MUST be unchanged before and after the diagnostic runs

### Requirement: No fit or fit_transform anywhere in this capability

No function in this capability MAY expose a `fit` or `fit_transform` interface, and no function in this capability MAY accept a testing partition for the purpose of learning any parameter from it. Every operation in this capability is either a fixed transformation or a read-only diagnostic.

#### Scenario: The cleaning surface exposes no fit interface

- GIVEN the complete public surface of this capability's cleaning operations
- WHEN that surface is inspected
- THEN no function MAY be named or behave as `fit` or `fit_transform`
- AND no function MAY take a testing partition as an argument for any purpose other than being the target of a read-only diagnostic or the target of row removal in leakage removal

### Requirement: Cleaning output precedes any fitted preprocessing step

Because deduplication substantially reshapes the training partition's categorical distributions, any fitted parameter that depends on training-row category frequencies (including, but not limited to, the rare-category threshold used by the preprocessing-pipeline capability) MUST be computed from this capability's cleaned training output, and MUST NEVER be computed from the raw, pre-cleaning training partition.

#### Scenario: A category-frequency-dependent fit consumes cleaned output, not raw input

- GIVEN a raw training partition and its cleaned counterpart produced by this capability, where a categorical column's value distribution differs materially between the two
- WHEN any downstream component fits a parameter derived from that column's row-frequency distribution
- THEN it MUST fit against the cleaned training partition
- AND it MUST NOT fit against the raw, pre-cleaning training partition

### Requirement: No hardcoded cleaned-partition row counts

No cleaned-partition row count, produced by any step in this capability, MAY appear as a literal value in this capability's implementation or in its tests. Every assertion about cleaning correctness MUST be expressed as an invariant or relationship that holds independent of the specific measured counts.

#### Scenario: Two runs over identical inputs reproduce identical counts

- GIVEN the same raw training and testing partitions
- WHEN the full cleaning sequence runs twice, independently, over those same inputs
- THEN every count produced by the two runs (duplicates dropped, test-internal duplicates, leakage removed, retained-contradictory, label-noise combinations) MUST be identical between the two runs

#### Scenario: No literal cleaned count exists in the implementation or its tests

- GIVEN the complete source of this capability's implementation and its test suite
- WHEN that source is inspected for integer literals representing a cleaned-partition row count
- THEN no such literal MAY be present
- AND every correctness assertion MUST instead reference a computed value or a relationship between computed values
