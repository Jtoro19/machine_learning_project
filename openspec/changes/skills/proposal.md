# Proposal: skills

## Why

`project-foundation` shipped `src/nids/` (preprocessing, sampling, results-output-contract)
and sibling agents are writing the three analysis notebooks against it. Nothing yet packages
that analysis as a reusable, parametrized command a user or agent can invoke outside a notebook
cell. This milestone adds one Claude Code skill per notebook so the same EDA/reduction/
clustering/classification logic can run against any tabular dataset, not just the committed
UNSW-NB15 CSVs.

## What Changes

- Add three Claude Code skills under `skills/`: `eda-reduction-clustering`,
  `clustering-reduction`, `classification`. Each ships a `SKILL.md` (frontmatter + usage) and a
  parametrized `argparse` Python script (`--dataset`, `--target`, `--exclude`, `--output`).
- Add a small shared helper module (`skills/_shared/common.py`) so the three scripts reuse one
  schema-detection routine and one generic (non-UNSW) preprocessing fallback instead of each
  reimplementing it.
- Scripts reuse `nids.preprocessing.build_preprocessor`, `nids.sampling.stratified_subsample`,
  `nids.results.ResultsWriter`, and `nids.columns` constants **when, and only when, the supplied
  dataset is detected to match the UNSW-NB15 raw schema**; otherwise they fall back to a generic
  sklearn pipeline built from the dataset's own detected numeric/categorical dtypes.
- Add `tests/test_skills.py`: each script is exercised in-process (`main(argv)`, no subprocess)
  against a small synthetic dataset generated in `tmp_path`, plus an opt-in `@pytest.mark.slow`
  real-UNSW-NB15 run guarded by a `data/raw/` existence skip.
- Append an "Skills" section to `README.md` documenting install/invocation. No existing README
  section is rewritten.

## Non-Goals

- No change to `src/nids/`, `notebooks/`, `results/`, or `dashboard/` — those are owned by
  sibling work in flight.
- No new modelling capability beyond what each notebook already exercises (PCA/KMeans-style
  reduction+clustering, and one classifier with macro F1 / balanced accuracy).
- No git commit — the parent orchestrator commits.

## Impact

- Affected capability: `notebook-skills` (new).
- Affected code: `skills/**` (new), `tests/test_skills.py` (new), `README.md` (append-only).
- Risk: dataset-agnosticism for the UNSW-specific pieces (`build_preprocessor`,
  `SKEWED_COLUMNS` log1p routing) is bounded — see design.md "Schema detection" — by falling
  back to a generic pipeline whenever the supplied dataset or `--exclude` set does not exactly
  match what the frozen `nids` pipeline supports.
