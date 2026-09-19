# Network Intrusion Detection on UNSW-NB15

This project analyzes and models the UNSW-NB15 network intrusion dataset: a reusable
`nids` package under `src/` handles data loading, cleaning, and preprocessing, and
notebooks under `notebooks/` build the exploratory and modelling work on top of it.

## Setup

```bash
uv sync
```

This installs the `nids` package editable into a local virtual environment (`.venv`),
along with every declared dependency.

## Data acquisition

This project uses the official UNSW-NB15 training and testing partitions. The
dataset is **not** bundled with the repository and is **never downloaded
automatically** — no module under `src/nids/` performs a network request.

1. Download the UNSW-NB15 dataset from UNSW Canberra Cyber's official source.
2. Save the training partition as `UNSW_NB15_training-set.csv`.
3. Save the testing partition as `UNSW_NB15_testing-set.csv`.
4. Place both files under `data/raw/` at the repository root. `data/raw/` is
   read-only to this package: nothing here writes to, renames, or otherwise
   modifies files under it.

Run `uv run python -m nids.validation` to confirm both files are present and
correctly shaped. If a file is missing or malformed, the command prints these
same instructions and exits with a non-zero status instead of guessing.

## Running notebooks

_Placeholder — filled in during the results slice._

## Validation and reports

_Placeholder — filled in during the results slice._

## Known limitations

_Placeholder — filled in during the results slice._
