# Data Loading Specification

## Purpose

Give every consumer of the UNSW-NB15 partitions one deterministic, read-only entry point, and give a contributor with an empty `data/raw/` directory a clear, actionable failure instead of a cryptic file-not-found error. This capability never downloads data automatically and never mutates `data/raw/`.

## Requirements

### Requirement: Read-only loading of both raw partitions

The system MUST provide functions that load the UNSW-NB15 training and testing partitions from `data/raw/` with deterministic column dtypes. Loading MUST NOT write to, rename, or otherwise modify any file under `data/raw/`.

#### Scenario: Both raw partitions load with stable dtypes

- GIVEN both UNSW-NB15 CSV files are present under `data/raw/`
- WHEN the training-partition loader and the testing-partition loader are each called
- THEN each MUST return a data frame with deterministic, explicitly-declared dtypes for its string-typed columns
- AND repeated calls in the same environment MUST return frames with identical dtypes

#### Scenario: Raw files are never modified by loading

- GIVEN both UNSW-NB15 CSV files are present under `data/raw/`
- WHEN the training-partition loader and the testing-partition loader are each called any number of times
- THEN the raw files on disk MUST be byte-identical before and after every call

### Requirement: Raw data validation with manual-download guidance

The system MUST provide a validation entry point, runnable as a standalone module, that verifies `data/raw/` holds both expected UNSW-NB15 CSV files with the expected raw row and column counts. When a required file is missing or its shape does not match the expected raw shape, the validation entry point MUST print manual UNSW-NB15 download instructions and MUST exit with a non-zero status. The system MUST NOT implement, invoke, or reference any automated dataset download.

#### Scenario: Both raw files present with the expected shape

- GIVEN `data/raw/` contains both UNSW-NB15 CSV files matching the documented raw shapes (175,341 rows by 45 columns for training; 82,332 rows by 45 columns for testing)
- WHEN the validation entry point runs
- THEN it MUST report both files as present and correctly shaped
- AND it MUST exit with a zero status

#### Scenario: A required raw file is missing

- GIVEN `data/raw/` is missing at least one of the two expected UNSW-NB15 CSV files
- WHEN the validation entry point runs
- THEN it MUST print instructions for manually downloading and placing the missing file
- AND it MUST exit with a non-zero status
- AND it MUST NOT attempt to download the file itself

#### Scenario: A present raw file does not match the expected raw shape

- GIVEN a UNSW-NB15 CSV file is present under `data/raw/` but its row count or column count does not match the documented expected raw shape for that partition
- WHEN the validation entry point runs
- THEN it MUST report the shape mismatch
- AND it MUST exit with a non-zero status

### Requirement: No automated dataset acquisition anywhere in the package

No module in `src/nids/` MAY perform a network request, invoke a download tool, or otherwise acquire the dataset automatically. Dataset acquisition MUST remain a manual step performed by the user outside this package.

#### Scenario: Validation failure never triggers acquisition

- GIVEN the validation entry point has detected a missing or malformed raw file
- WHEN it reports that failure
- THEN its only side effects MUST be printing instructions and exiting non-zero
- AND it MUST NOT initiate any download, network call, or file-fetch operation
