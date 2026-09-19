# Project Rules

## Language
- All documentation is in English: file and folder names, specs, READMEs, commit messages, code comments, variable names, notebook markdown, chart titles and axis labels.

## Stack
- Python 3.14 in a local virtual environment (.venv) managed with uv.
- Dataset: UNSW-NB15 official partitions (training and testing CSV) stored in data/raw/. Raw data is never modified.

## Data rules
- Drop the id column. Never use attack_cat or label as input features.
- Split first. Every imputer, encoder and scaler is fitted inside a scikit-learn Pipeline on training data only.
- Treat "-" in service as its own category, never impute it with the mode.
- Group rare proto values (under 1 percent of rows) into "other" before one-hot encoding.
- Report supervised results with and without sttl and ct_state_ttl (known testbed shortcut).
- Use random seed 42 for every split, subsample and model.
- Gaussian Process, spectral clustering and t-SNE run on a stratified subsample of at most 10000 rows.

## Code
- Shared preprocessing lives in src/. Notebooks import it instead of duplicating logic.
- Every notebook exports tables (CSV or JSON) and figures (PNG) to results/<notebook_name>/ following the output contract in the project spec.
- Functions in src/ have type hints, docstrings and pytest tests in tests/.
- No hardcoded absolute paths.

## Review checklist
- Any fit or fit_transform on test data is a blocking issue.
- Any feature derived from the target is a blocking issue.
- Imbalanced classification must report macro F1 and balanced accuracy, not only accuracy.
