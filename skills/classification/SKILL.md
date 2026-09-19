---
name: classification
description: >-
  Use when the user wants a supervised classification report (RandomForest, macro F1,
  balanced accuracy, confusion matrix) over a tabular CSV dataset with a target column
  (e.g. "train a classifier on this dataset and report macro F1", "how well can this
  data predict the target"). Dataset-agnostic; not specific to the UNSW-NB15 project
  data, but automatically reports a with/without-TTL-shortcut variant when `sttl` and
  `ct_state_ttl` are both present in the feature set.
---

# Classification

Trains a `RandomForestClassifier` on a 70/30 stratified split, reports macro F1 and
balanced accuracy (never plain accuracy alone), and writes a per-class classification
report plus a confusion matrix figure, following the
[results-output-contract](../../openspec/changes/project-foundation/specs/results-output-contract/spec.md)
(`manifest.json` + `tables/` + `figures/`).

## When to use

Invoke this skill when the user wants a quick, honest (macro-averaged, imbalance-aware)
classification baseline over a CSV dataset.

## Usage

```bash
uv run python skills/classification/classification.py \
  --dataset path/to/data.csv \
  --target target_column_name \
  --exclude id \
  --output results/my_classification_run
```

Same `--dataset`/`--target`/`--exclude`/`--output`/`--seed` contract as the other two
skills. Run `--help` for the full flag list without needing a dataset.

**Only the UNSW-NB15 schema is detected and cleaned automatically.** For any other
dataset, identifier-like columns are NOT dropped automatically — the caller is
responsible for naming them via `--exclude` (as in the `--exclude id` example above).

## What it does

1. When the input matches the UNSW-NB15 raw schema, cleans it first through the
   project's own `nids.cleaning` steps (drop the fixed unused columns, normalize
   `service`) — the same column-level cleaning the canonical `nids` pipeline requires
   before fitting `RareCategoryGrouper`.
2. Splits the (possibly cleaned) input 70/30, stratified by `--target` when every class
   has at least 2 rows, otherwise unstratified, THEN drops exact-duplicate rows from the
   resulting TRAIN split only — never before the split, and never from the held-out
   split (`nids.cleaning.drop_train_duplicates`'s own contract is "training partition
   only"; see `eda-reduction-clustering/SKILL.md`'s identical note for the full
   rationale).
3. Fits preprocessing on the training split only (UNSW-detected pipeline or generic
   fallback, same routing as `eda-reduction-clustering`).
4. Fits `RandomForestClassifier(n_estimators=200, random_state=42)` and predicts on the
   held-out test split.
5. Reports macro F1 and balanced accuracy (AGENTS.md: "Imbalanced classification must
   report macro F1 and balanced accuracy, not only accuracy").
6. **When both `sttl` and `ct_state_ttl` are present in the feature set**, repeats steps
   2-5 with those two columns additionally excluded, and reports both variants side by
   side (AGENTS.md: "Report supervised results with and without sttl and
   ct_state_ttl").
7. Writes: one `classification_report_<variant>` table and one
   `confusion_matrix_<variant>` figure per variant; `macro_f1`, `balanced_accuracy`
   metrics (plus `*_without_ttl` metrics when the TTL variant ran).

Neither the target column nor any `--exclude` column ever reaches the feature matrix —
asserted twice per variant, exactly as in `eda-reduction-clustering`.
