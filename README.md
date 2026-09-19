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

```bash
uv run jupyter lab
```

Run this from the repository root, not from inside `notebooks/`. `uv run` activates
the project's `.venv` (where `nids` is installed editable via `uv sync`) before
launching Jupyter, so every notebook cell's `import nids` resolves against this
project's package and every path `nids.paths.repo_root()` resolves stays anchored
here.

A separately registered Jupyter kernel (for example a global `ipykernel install`
run outside this project, or a kernel pointed at a different Python interpreter)
would **not** see the package: `nids` is installed only inside this project's
`.venv`, so a kernel backed by a different interpreter has no `nids` on its
`sys.path` and `import nids` fails. Always launch Jupyter through `uv run` from
this repository so the notebook process and the `.venv` it imports from are the
same interpreter.

## Validation and reports

```bash
uv run python -m nids.validation
uv run python -m nids.cleaning_report
```

`nids.validation` checks that both raw partition files are present under
`data/raw/`, match the documented shape and schema, parsed with the expected
dtypes, and contain no null cells; on failure it prints the same manual
acquisition instructions as the "Data acquisition" section above and exits
non-zero.

`nids.cleaning_report` runs `nids.validation` first (so it never half-runs
against a missing or malformed file) and then writes the committed
`results/data_cleaning/` folder: `manifest.json`, a human-readable `counts.json`
mirror, four tables (`row_counts_before_after.csv`, `attack_cat_distribution.csv`,
`state_distribution.csv`, `rare_categories.csv`), and one figure
(`class_balance_before_after.png`). `results/` is committed and self-describing —
a consumer can read `results/data_cleaning/manifest.json` directly without
rerunning anything.

## Known limitations

- **Test-set metrics are not comparable to published UNSW-NB15 benchmarks.**
  This project drops testing rows that leak into the training partition (rows
  matching a training row on the leakage comparison key). Any metric computed
  on this project's cleaned testing set is therefore computed on a different,
  smaller testing set than the one published benchmarks report against.
- **A near-certain minimum error rate remains in the testing set.** After
  leakage removal, some testing rows still match a training row on every
  feature but carry a different label (`test_contradictory_rows_retained` in
  `results/data_cleaning/counts.json`, about 5.5% of the retained testing set
  on the current raw data). No model can classify both members of such a
  feature-identical, label-contradictory pair correctly, so this count is a
  floor no classifier trained on these features can eliminate.
- **The 10,000-row stratified subsample distorts minority-class share.**
  `nids.sampling.stratified_subsample`'s `floor` guarantees every class at
  least `floor` rows (50 by default) even when its proportional share would
  round to fewer. On the current cleaned training partition this over-samples
  the smallest class, `Worms`: 127 available rows (about 0.12% of the training
  partition) become 57 allocated rows in the 10,000-row subsample (0.57% of
  the subsample) — roughly 4.8x its true population share. This is deliberate
  (AGENTS.md requires Gaussian Process, spectral clustering, and t-SNE to run
  on a subsample of at most 10,000 rows, and a class with too few rows is
  otherwise invisible to these methods), but any notebook reporting class
  proportions from the subsample must disclose this distortion rather than
  present the subsample's class balance as representative of the full,
  cleaned training partition.
