# Network Intrusion Detection on UNSW-NB15

This project analyzes and models the UNSW-NB15 network intrusion dataset in three
stages: data cleaning, unsupervised structure analysis (two independent notebooks),
and supervised classification. A reusable `nids` package under `src/` handles data
loading, cleaning, and preprocessing; notebooks under `notebooks/` build the
exploratory and modelling work on top of it; `skills/` exposes the same analyses as
standalone, dataset-agnostic command-line tools; and `dashboard/` renders every
notebook's committed results.

## Setup

```bash
uv sync
```

Requires Python 3.14. This installs the `nids` package editable into a local virtual
environment (`.venv`), along with every declared dependency.

## Data acquisition

The UNSW-NB15 training and testing partitions are **not** bundled with this
repository and are **never downloaded automatically** — no module under `src/nids/`
performs a network request.

1. Download the UNSW-NB15 dataset from UNSW Canberra Cyber's official source.
2. Save the training partition as `UNSW_NB15_training-set.csv`.
3. Save the testing partition as `UNSW_NB15_testing-set.csv`.
4. Place both files under `data/raw/` at the repository root. `data/raw/` is
   gitignored and read-only to this package: nothing here writes to, renames, or
   otherwise modifies files under it.

```bash
uv run python -m nids.validation
```

Confirms both files are present and correctly shaped (schema, dtypes, no null
cells). If a file is missing or malformed, this command prints the manual
acquisition instructions above and exits non-zero instead of guessing.

## Running the notebooks

```bash
uv run jupyter lab
```

Run this from the repository root, not from inside `notebooks/`, so `import nids`
resolves against this project's `.venv` with no separate `ipykernel install` step.

To execute a notebook headlessly and refresh its committed outputs and results:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/<name>.ipynb
```

| Notebook | Produces |
|---|---|
| `01_eda_reduction_clustering.ipynb` | Summary statistics, correlation/VIF analysis, PCA, and a K-Means sweep on the full cleaned training partition → `results/eda_reduction_clustering/` |
| `02_clustering_reduction.ipynb` | K-Means, Ward, DBSCAN, and spectral clustering compared on a 10,000-row stratified subsample, with external validation (ARI, NMI) against `attack_cat` → `results/clustering_reduction/` |
| `03_classification.ipynb` | Supervised model comparison (macro F1, balanced accuracy), binary threshold cost analysis, and per-class diagnostics on the cleaned test partition → `results/classification/` |

`nids.cleaning_report` (`uv run python -m nids.cleaning_report`) runs
`nids.validation` first, then writes the committed `results/data_cleaning/` folder.
`results/` is committed and self-describing — a consumer can read any
`results/<name>/manifest.json` directly without rerunning anything.

## Running the skills

`skills/` packages the notebook analyses as standalone, parametrized, dataset-agnostic
Claude Code skills — runnable against any tabular CSV with a target column, not just
UNSW-NB15:

| Skill | Script | Does |
|---|---|---|
| `eda-reduction-clustering` | `skills/eda-reduction-clustering/eda_reduction_clustering.py` | Summary statistics, PCA(2D), and a KMeans clustering pass |
| `clustering-reduction` | `skills/clustering-reduction/clustering_reduction.py` | Stratified subsample (capped at 10,000 rows) + KMeans vs Agglomerative clustering comparison on a PCA embedding |
| `classification` | `skills/classification/classification.py` | RandomForest classification report (macro F1, balanced accuracy, confusion matrix), with an automatic with/without-TTL variant when `sttl`/`ct_state_ttl` are both present |

All three share the same CLI contract:

```bash
uv run python skills/<skill-name>/<script>.py \
  --dataset path/to/data.csv \
  --target target_column_name \
  --exclude id \
  --output results/my_run
```

| Flag | Required | Notes |
|---|---|---|
| `--dataset` | yes | Path to a CSV file |
| `--target` | yes | Target/label column name |
| `--exclude` | no | Repeatable and/or comma-separated; extra columns dropped from the feature set (e.g. an id column) |
| `--output` | yes | Must resolve inside `results/`; its final path component becomes the manifest's `notebook_id` and must match `^[a-z][a-z0-9_]{0,63}$` (lowercase `snake_case`, e.g. `results/my_run`, not `results/My-Run`) |

`--seed` (default `42`) is also available on all three. Run any script with `--help`
for the full flag list. Only the UNSW-NB15 schema is auto-detected and cleaned; for
any other dataset, identifier-like columns are not dropped automatically — name them
via `--exclude`. Every script follows the
[results-output-contract](openspec/changes/project-foundation/specs/results-output-contract/spec.md)
(`manifest.json` + `tables/` + `figures/`) under `--output` — see each
`skills/<skill-name>/SKILL.md` for the full per-skill contract.

## Running the dashboard

```bash
uv run streamlit run dashboard/app.py
```

Read-only: it discovers every `results/*/manifest.json` dynamically and renders each
section's tables, figures, metrics, and notes in manifest order. It reads only from
`results/` and never re-executes a notebook or touches `data/raw/`.

## Key findings

All figures below come from the committed `results/*/manifest.json` files.

### Cleaning (`results/data_cleaning/`)

| Metric | Value |
|---|---|
| Training rows retained | 107,740 |
| Testing rows retained | 78,084 |
| Duplicate training rows dropped | 67,601 |
| Leaking testing rows removed (matched a training row on the full row) | 4,248 |
| Duplicate rows retained inside the testing partition | 26,387 |
| Label-noise feature combinations in training | 1,772 |

**The error floor.** 4,294 retained testing rows are feature-identical to a training
row but carry a different label — about 5.5% of the retained testing set. No
classifier trained on these features can get both members of such a pair right, so
this is a minimum error rate no model can beat.

### Unsupervised structure (`results/eda_reduction_clustering/`, `results/clustering_reduction/`)

- K-Means selects k=10 in both TTL variants on the full training partition
  (silhouette 0.472 with TTL, 0.442 without).
- 12 principal components reach 90% of design-matrix variance, in both TTL variants.
- `ackdat`, `synack`, and `tcprtt` are perfectly collinear (VIF = infinite).

**The TTL result, stated carefully.** Dropping `sttl`/`ct_state_ttl` barely moves
supervised performance (best model's macro F1 changes by +0.0087) but collapses
unsupervised alignment with `attack_cat` (best ARI 0.163 → -0.005, NMI 0.289 →
0.005). The clusters still exist without the shortcut — they simply stop
corresponding to attack categories.

### Supervised classification (`results/classification/`)

| Model | Macro F1 | Balanced accuracy | Accuracy |
|---|---|---|---|
| HistGradientBoosting (best, with TTL) | 0.512 | 0.577 | 0.743 |
| RandomForest (best balanced accuracy) | 0.503 | 0.638 | 0.692 |

- Binary attack-vs-normal view: ROC-AUC 0.983, PR-AUC 0.987.
- At an FN:FP cost ratio of 20:1, the cost-optimal decision threshold is 0.075,
  cutting expected cost about 82% versus the default 0.5.
- Hardest classes are **not** the rarest: Backdoor (F1 0.072) and Analysis (F1
  0.107) are the two hardest, while Worms — only 44 testing rows — reaches F1 0.591.

## Limitations

- **Not comparable to published UNSW-NB15 benchmarks.** Removing testing rows that
  leak into the training partition makes every metric here computed on a different,
  smaller testing set than published benchmarks report against (78,084 rows here
  versus the full 82,332-row testing set).
- **The ~5.5% label-noise error floor** (above) is a near-certain minimum error rate
  inherited from the raw data, not a modelling shortfall.
- **Some models train on capped subsamples, not the full 107,740 training rows**:
  GaussianProcessClassifier (2,000 rows), SVC (5,000 rows), and
  KNeighborsClassifier (20,000 reference rows). These are not directly comparable to
  models trained on the full partition; every results table carries its
  `training_rows` count for that reason.
- **The 10,000-row stratified subsample distorts minority-class share.** Its
  per-class floor of 50 rows over-represents the smallest class: Worms becomes about
  0.57% of the subsample against a true population share of about 0.12%.
- **Compute budgets were capped** to keep each notebook under roughly 15 minutes of
  execution; hyperparameter search and clustering sweeps are bounded, and results
  would differ under exhaustive search.
