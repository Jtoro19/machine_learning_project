# Notebook Skills Specification

## Purpose

Package each of the three analysis notebooks (EDA+reduction+clustering, clustering+reduction,
classification) as a standalone, parametrized Claude Code skill so the same analysis can run
against any tabular dataset from the command line, following the results-output-contract.

## Requirements

### Requirement: One skill directory per notebook with a documented trigger

The system MUST provide `skills/eda-reduction-clustering/`, `skills/clustering-reduction/`, and
`skills/classification/`, each with a `SKILL.md` carrying YAML frontmatter (`name`,
`description`) whose `description` states when the skill should be invoked, plus a Python script
usable as a CLI entry point.

#### Scenario: A skill's frontmatter is parseable and states its trigger

- GIVEN one of the three `SKILL.md` files
- WHEN its YAML frontmatter is parsed
- THEN it MUST contain non-empty `name` and `description` fields
- AND `description` MUST state the condition under which the skill applies

### Requirement: Every script exposes the same four parameters

Each script MUST accept `--dataset` (path to a CSV), `--target` (target column name),
`--exclude` (zero or more columns to exclude from features, repeatable or comma-separated), and
`--output` (output folder), via `argparse`, and MUST print usage and exit zero on `--help`.

#### Scenario: --help succeeds without touching any dataset

- GIVEN any of the three scripts
- WHEN invoked with only `--help`
- THEN it MUST print usage text and exit with status 0
- AND it MUST NOT require `--dataset` to be present

### Requirement: The target and excluded columns never reach the feature matrix

Each script MUST derive its feature set as every dataset column minus `--target` minus every
`--exclude` entry, and MUST assert, before fitting or transforming, that neither the target nor
any excluded column is present among the features passed to the preprocessor.

#### Scenario: An assertion blocks a leaking feature set

- GIVEN a script invoked with `--target label --exclude attack_cat`
- WHEN the feature matrix is constructed
- THEN it MUST NOT contain a `label` or `attack_cat` column
- AND the script MUST raise before fitting if either column were present

### Requirement: Scripts are dataset-agnostic, with UNSW-NB15 as a detected special case

Each script MUST derive its feature set and preprocessing routing from the supplied dataset and
CLI parameters, not from hardcoded UNSW-NB15 column names. `nids.columns` constants and
`nids.preprocessing.build_preprocessor` MAY be used only when the supplied dataset's columns are
detected to match the UNSW-NB15 raw schema; otherwise a generic preprocessing pipeline (built
from the dataset's own detected numeric/categorical dtypes) MUST be used.

#### Scenario: A synthetic dataset runs end-to-end

- GIVEN a small synthetic tabular dataset with a mix of numeric and categorical columns and a
  multiclass target, none of it matching the UNSW-NB15 schema
- WHEN a skill script is run against it
- THEN it MUST complete without raising
- AND it MUST NOT import or reference any UNSW-specific column name as a hardcoded literal in
  the routing decision

#### Scenario: A UNSW-NB15-shaped dataset uses the frozen pipeline

- GIVEN a dataset whose columns match `nids.columns.RAW_COLUMNS`
- WHEN a skill script is run against it with no `--exclude` beyond the TTL shortcut pair
- THEN it MUST route through `nids.preprocessing.build_preprocessor`

### Requirement: Outputs follow the results-output-contract

Each script MUST write its outputs under `--output` using `nids.results.ResultsWriter`: a
`manifest.json` with `schema_version`, `notebook_id`, `generated_at`, `tables`, `figures`, and
`metrics`, with every declared path resolving to an existing file.

#### Scenario: A run produces a self-describing, valid manifest

- GIVEN a completed script run
- WHEN `<output>/manifest.json` is parsed
- THEN parsing MUST succeed
- AND every table and figure path it declares MUST exist on disk relative to that folder

### Requirement: Deterministic seeding

Every random operation (subsampling, splitting, clustering initialization, model fitting) MUST
use seed 42.

#### Scenario: Two runs over identical input produce identical metrics

- GIVEN the same script run twice over the same input dataset and parameters
- WHEN the `metrics` array of each run's manifest is compared
- THEN every numeric metric value MUST be identical between the two runs
