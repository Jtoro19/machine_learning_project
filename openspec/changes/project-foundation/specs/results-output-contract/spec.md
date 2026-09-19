# Results Output Contract Specification

## Purpose

Let a future dashboard and future skills discover every notebook's and script's outputs by scanning `results/`, without knowing any producer's internals and without any producer coordinating writes with another. This capability defines the folder layout, file-naming convention, and machine-readable manifest schema that every producer under `results/` MUST follow, and defines the cleaning report as this contract's first producer.

## Requirements

### Requirement: Self-describing per-producer output folder layout

Every producer writing to `results/` MUST write its outputs under `results/<notebook_id>/`, containing a `manifest.json` at the folder root, table files under a `tables/` subfolder, and figure files under a `figures/` subfolder. Each producer's folder MUST be fully self-describing: no producer's output folder MAY depend on another producer's folder to be interpretable.

#### Scenario: A producer's folder is interpretable in isolation

- GIVEN a single producer's output folder under `results/<notebook_id>/`
- WHEN that folder is inspected without reference to any other folder under `results/`
- THEN its `manifest.json` MUST describe every table and figure that folder contains
- AND every path referenced by that manifest MUST resolve to a file within that same folder

#### Scenario: No producer writes outside its own folder

- GIVEN two producers writing to `results/<notebook_id_a>/` and `results/<notebook_id_b>/` respectively
- WHEN both producers run, in either order
- THEN neither producer MAY write, modify, or delete any file under the other producer's folder

### Requirement: One manifest.json per folder with the required schema

Each producer's `manifest.json` MUST contain at minimum: `schema_version`, `notebook_id`, `generated_at`, a `tables` array, a `figures` array, and a `metrics` array. Each entry in `tables` and `figures` MUST include a `path` relative to the manifest's own folder, a `title`, a `type`, and a `description`. Each entry in `metrics` MUST include a `name`, a `value`, and a `description`, and `value` MAY be a number or a string. There MUST be exactly one `manifest.json` per producer folder; the system MUST NEVER require or maintain a single global index file referencing every producer.

#### Scenario: A manifest round-trips through JSON parsing with all declared paths present

- GIVEN a producer has finished writing its output folder
- WHEN its `manifest.json` is parsed as JSON
- THEN parsing MUST succeed
- AND every `path` value declared under `tables` and `figures` MUST resolve to an existing file, relative to the manifest's own folder

#### Scenario: A metrics entry tolerates both numeric and string values

- GIVEN a producer's manifest containing one metrics entry whose `value` is numeric and another whose `value` is a string
- WHEN a consumer reads the `metrics` array
- THEN it MUST be able to read both entries without a type error, treating `value` as either a number or a string

#### Scenario: Consumers discover outputs without a global index

- GIVEN multiple producer folders exist under `results/`
- WHEN a consumer needs to enumerate every available output
- THEN it MUST be able to do so by scanning for every `results/*/manifest.json` file
- AND no producer MAY be required to register itself in any shared or global index file

### Requirement: Deterministic, timestamp-free file naming

Every file written under a producer's `tables/` and `figures/` subfolders MUST use a descriptive, `snake_case` filename with no timestamp embedded in the filename. Any generation timestamp MUST live only inside `manifest.json`'s `generated_at` field.

#### Scenario: File names carry no timestamp

- GIVEN a producer's output folder
- WHEN the filenames under `tables/` and `figures/` are inspected
- THEN none MAY contain a timestamp component
- AND the only location for the generation timestamp MUST be the `generated_at` field of `manifest.json`

### Requirement: Reruns over identical inputs are reproducible at the byte level for table files

Two runs of the same producer over the same inputs MUST produce byte-identical table files, aside from the `generated_at` field inside `manifest.json`. Table files MUST have stable row ordering and MUST NOT embed timestamps.

#### Scenario: Two consecutive runs produce identical table bytes

- GIVEN a producer run twice, consecutively, over the same input data
- WHEN each run's table files under `tables/` are compared byte-for-byte between the two runs
- THEN every table file MUST be byte-identical across the two runs

### Requirement: The cleaning report is this contract's first producer

The system MUST provide a cleaning report producer, runnable as a standalone module, that writes to `results/data_cleaning/` following this capability's folder layout, manifest schema, and reproducibility requirements. The cleaning report MUST compute every count it reports at run time from the data-cleaning capability's output; it MUST NEVER hardcode a cleaned-partition count. The report MUST include, at minimum: a table of row counts per partition after each ordered cleaning step, a table of `attack_cat` distribution before and after cleaning for both partitions, a table of `state` distribution before and after deduplication, a table of every rare `state`/`proto` category grouped into `"other"` with its row count, and a figure showing class balance before and after cleaning. The report MUST state the leakage comparison key used, the count of rows removed, and the count of retained feature-identical label-contradictory rows together with that count's share of the retained testing set. The report MUST include a written note that removing leaking testing rows makes this project's test metrics non-comparable to published UNSW-NB15 benchmarks on the full, unmodified testing set, and a written note describing the retained feature-identical label-contradictory rows as a near-certain minimum error rate that no model can eliminate.

#### Scenario: The cleaning report folder is complete and self-describing

- GIVEN the cleaning report producer has run to completion against present raw data
- WHEN `results/data_cleaning/` is inspected
- THEN it MUST contain `manifest.json` with `notebook_id` set to `"data_cleaning"`
- AND it MUST contain the four required tables and the one required figure
- AND every path in the manifest MUST resolve to an existing file within that folder

#### Scenario: Every reported count is computed, not hardcoded

- GIVEN two independent runs of the cleaning report producer over the same raw input data
- WHEN the `metrics` array of each run's `manifest.json` is compared
- THEN every metric value MUST be identical between the two runs
- AND no metric value MAY match a literal hardcoded in the producer's source rather than a value computed from the data-cleaning capability's output for that run

#### Scenario: The leakage comparison key and both leakage counts are reported together

- GIVEN the cleaning report producer has run using a specific leakage comparison key
- WHEN its manifest's `metrics` array is inspected
- THEN it MUST include the comparison key used
- AND it MUST include the count of rows removed by leakage removal
- AND it MUST include the count of retained feature-identical label-contradictory rows and that count's share of the retained testing set

#### Scenario: Both mandatory disclosure notes are present

- GIVEN the cleaning report producer has run to completion
- WHEN its output is inspected
- THEN it MUST include the benchmark non-comparability note
- AND it MUST include the retained-error-floor note

### Requirement: results/ is committed while data/raw/ stays untouched by this contract

The `results/` directory and every producer's output under it MUST be tracked by version control. This capability MUST NOT modify, require modification of, or otherwise touch any file under `data/raw/`.

#### Scenario: A producer's output is available without rerunning it

- GIVEN a producer has already written its output folder and that folder has been committed
- WHEN a consumer (for example a future dashboard) needs that producer's outputs
- THEN it MUST be able to read them directly from the committed `results/` tree
- AND it MUST NOT need to rerun the producer to obtain them

#### Scenario: Raw data remains untouched by every producer

- GIVEN any producer following this contract has run, including the cleaning report
- WHEN `data/raw/` is inspected before and after that run
- THEN it MUST be byte-identical
