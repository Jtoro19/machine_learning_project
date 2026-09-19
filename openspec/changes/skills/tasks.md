# Tasks: skills

TDD: disabled for this change (per dispatch instructions); functional tests still required.

- [x] 1. `skills/_shared/common.py` + `__init__.py`: argparse builder, `is_unsw_raw_schema`,
      `build_generic_preprocessor`, leakage assertion helper, output-path -> ResultsWriter wiring.
- [x] 2. `skills/eda-reduction-clustering/SKILL.md` + `eda_reduction_clustering.py`
      (PCA + KMeans EDA report, see design Decision 5).
- [x] 3. `skills/clustering-reduction/SKILL.md` + `clustering_reduction.py`
      (stratified subsample + KMeans vs Agglomerative comparison, see design Decision 5).
- [x] 4. `skills/classification/SKILL.md` + `classification.py`
      (RandomForest, macro F1 + balanced accuracy, TTL with/without variant, see design
      Decision 5).
- [x] 5. `tests/test_skills.py`: synthetic-dataset fixture, one in-process `main(argv)` test per
      skill (manifest validity, no target/excluded leakage), `@pytest.mark.slow` UNSW-real-data
      variant per skill guarded by `data/raw/` skipif.
- [x] 6. Append a "Skills" section to `README.md` (install/invocation only, append-only).
- [x] 7. Verification: `uv run pytest -q` green (193 passed, 4 deselected); `--help` for all
      three scripts confirmed exit 0; one end-to-end run reported (manual `manual_eda_run`
      manifest, and pytest integration runs); `data/raw/` unchanged (byte-identical file sizes
      before/after); `git status --short` reported. All 3 `@pytest.mark.slow` real-UNSW-NB15
      tests also run and pass (`-m slow`), since `data/raw/` is present in this environment.

## Review findings fixed

- [x] Blocking finding 1 (target-leakage): `skills/_shared/common.py::build_preprocessor_for`
      routed the UNSW-vs-generic pipeline choice on the shape of `--exclude` (only stripping
      `DROPPED_COLUMNS`/`TARGET_COLUMNS` when `exclude` was empty or TTL-shaped), so
      `--target attack_cat --exclude id` let `label` (a perfect `attack_cat != "Normal"`
      predictor) reach the feature matrix. Fixed: routing is now keyed on
      `is_unsw_raw_schema(df)` alone; `DROPPED_COLUMNS`/`TARGET_COLUMNS` are always
      subtracted first, and `--exclude` can only remove more columns, never fewer.
- [x] Blocking finding 2 (`id` not dropped): same root cause as finding 1; closed by the same
      fix. `tests/test_skills.py`'s three `test_runs_end_to_end_on_unsw_nb15` tests now call
      `_assert_unsw_run_excludes_dropped_columns` (new helper) after each run, which reruns
      `compute_features`/`build_preprocessor_for` directly and asserts `id` (and every other
      `DROPPED_COLUMNS`/`TARGET_COLUMNS` member) is absent from the fitted
      `get_feature_names_out()` — proving the fix, not encoding around the bug.
- [x] Guard widened: `assert_no_leakage` gained an optional `df` kwarg; when supplied and the
      UNSW schema is detected, it now also asserts no `TARGET_COLUMNS`/`DROPPED_COLUMNS`
      member is present, independent of `target`/`exclude`. All three skill scripts now pass
      `df=` at both call sites. New regression tests in `tests/test_skills.py::
      TestCommonHelpers` prove both the old leak scenario and the widened assertion.
- [x] Major finding 3 (preprocessing outside `src/`): `build_generic_preprocessor` moved from
      `skills/_shared/common.py` into `src/nids/preprocessing.py`, renamed
      `_build_generic_preprocessor` (kept underscore-private — an existing hard AST guard,
      `tests/test_leak_safety.py::test_preprocessing_public_functions_take_no_dataframe_parameter`,
      requires every public parameter in this module to be `bool`-typed or absent, per design
      Decision 5's frozen no-data public surface; a public `list[str]`-parameter function would
      have broken that invariant). `common.py` re-exports it under its original name at its one
      call site. `build_preprocessor`'s signature is untouched. New tests in
      `tests/test_preprocessing.py`.
- [x] Major finding 4 (inconsistent split methodology): removed the train/test split from
      `eda_reduction_clustering.py`, matching `clustering_reduction.py` (neither skill fits a
      supervised model, so AGENTS.md's "split first" rule doesn't apply to either — there is no
      held-out set to protect). Rationale documented in both scripts' module docstrings and
      both `SKILL.md` files, each noting `classification` (supervised) DOES split.
- [x] Minor finding 5: `resolve_results_writer` — `generated_at` and the return type annotated.
- [x] Minor finding 6: `classification.py::_run_variant` — `df`, `writer`, and the return type
      annotated.
- [x] Minor finding 7: `eda_reduction_clustering.py` — unused `import argparse` removed.
- [x] Minor finding 8: `eda_reduction_clustering.py` — the mid-function `import numpy as np` is
      gone entirely; removing the split (finding 4) also removed its only use (`np.vstack`).
- [x] Minor finding 9: with the split removed (finding 4), `df[features].describe()` and every
      other reported table in `eda_reduction_clustering.py` now run over the same single
      whole-dataset partition; confirmed no table mixes partitions.
- [x] Minor finding 10: the confusion-matrix figure description rewritten from
      "Row-normalized-free confusion matrix" to "Confusion matrix (raw counts, not
      row-normalized) on the held-out test split."

## Review findings fixed (round 2)

- [x] Blocking finding 1 (unsplit `fit_transform` in both unsupervised skills, plus the
      test-hole that could point `--dataset` at the testing CSV): **the round-1 "no-split
      exemption" for `eda-reduction-clustering.py` and `clustering-reduction.py` is
      explicitly WITHDRAWN — the parent orchestrator's earlier call that AGENTS.md's
      split-first rule doesn't apply to unsupervised fitting was wrong, because the rule
      governs the fit/fit_transform OPERATION, not the model that follows it.** Restored
      `train_test_split` (seed 42, stratified by the target when every class has at least
      2 rows via the new `common.resolve_stratify` helper, otherwise unstratified) in both
      scripts; the pipeline now `.fit()`s on the TRAIN split only and `.transform()`s
      whichever partition is then reduced/clustered. Removed the exemption paragraphs from
      both module docstrings and both `SKILL.md` files, replaced with an accurate
      "fitted on the training split only, per AGENTS.md's split-first rule, even though no
      supervised model is fitted" statement. `tests/test_skills.py`'s slow tests no longer
      fall back to `sorted(raw_data_dir().glob("*.csv"))` (which could select the testing
      CSV); a new `_require_training_csv()` helper requires
      `nids.loading.RAW_TRAIN_FILENAME` explicitly and `pytest.skip`s if absent.
- [x] Blocking finding 2 (frozen pipeline fitted on RAW data): `skills/_shared/common.py`
      gained `clean_dataset_if_unsw(df)`, which detects the UNSW-NB15 raw schema on the
      loaded frame and, when it matches, routes it through the project's own
      `nids.cleaning` step functions (`drop_unused_columns`, `normalize_service`,
      `drop_train_duplicates`, reused directly, never re-implemented) plus
      `nids.loading`'s `STRING_DTYPE`/`STRING_COLUMNS` declarations, before any
      preprocessing pipeline is fit. All three skill scripts now call
      `common.load_dataset` -> `common.clean_dataset_if_unsw` before `compute_features`.
      Because cleaning removes `DROPPED_COLUMNS`, a fresh `is_unsw_raw_schema` check on the
      cleaned frame would always be `False`; `build_preprocessor_for` and
      `assert_no_leakage` now take an explicit `is_unsw_schema` parameter (computed once,
      before cleaning) instead of re-detecting, so the UNSW routing decision survives
      cleaning correctly. Verified: the skill-path cleaned row count and
      `RareCategoryGrouper.frequent_categories_` kept-category sets for `proto`/`service`/
      `state` are byte-identical to `nids.cleaning.clean_partitions().train` fitted through
      `nids.preprocessing.build_preprocessor()` on the real training CSV (107,740 rows
      both ways). Documented in all three `SKILL.md` files.
- [x] Blocking finding 3 (`id` reaching `summary_statistics.csv`): fixed the one-line bug
      in `eda_reduction_clustering.py` — the summary table now uses `feature_columns_to_use`
      (the UNSW-routing-safe list) instead of the schema-blind `features`. Added a
      slow-test assertion (`TestEdaReductionClustering::test_runs_end_to_end_on_unsw_nb15`)
      that `summary_statistics.csv` contains no `DROPPED_COLUMNS` member; verified manually
      on the real training CSV with `--target attack_cat --exclude label` (output written
      to a scratch folder, not `results/`) — no `id`/`stcpb`/`dtcpb`/`is_ftp_login` present.
- [x] Lower-severity finding 4 (bare `assert` leakage guards vanish under `python -O`):
      every leakage guard in `skills/_shared/common.py` (`compute_features`,
      `assert_no_leakage`) now `raise`s `ValueError` instead of `assert`ing. Updated the
      three existing tests that expected `AssertionError` to expect `ValueError`.
- [x] Lower-severity finding 5 (`leaked_excluded` in `compute_features` can never fire):
      deleted the dead check and replaced it with a real, previously-missing guard — every
      `--exclude` entry is now validated against the dataset's actual columns before
      `compute_features` filters them out, raising `ValueError` on a typo'd/non-existent
      `--exclude` name instead of silently no-op'ing it. Added two new regression tests.

## Review findings fixed (round 3)

- [x] Blocking finding 1 (`--exclude id` crashed on real UNSW input): all three scripts
      call `clean_dataset_if_unsw()` before `compute_features()`, and cleaning drops
      `DROPPED_COLUMNS` (including `id`) before `compute_features` validated `--exclude`
      names — so the documented `--exclude id` usage example raised `ValueError` on
      every real UNSW CSV. Fixed: `compute_features` gained an `original_columns`
      parameter; every script now captures `original_columns = list(raw_df.columns)`
      immediately after loading, before cleaning, and threads it through, so `--exclude`
      is validated against the PRE-cleaning column set (a genuine typo still raises; an
      already-dropped canonical column does not). Added
      `tests/test_skills.py::TestEdaReductionClustering::
      test_runs_end_to_end_on_unsw_nb15_with_exclude_id` (a `@pytest.mark.slow` test
      using `--exclude id` against the real UNSW training CSV — every prior slow test
      only ever used `--exclude label`) plus direct unit coverage in
      `tests/test_tabular.py::TestComputeFeatures`. Verified manually: real training CSV,
      `--exclude id`, output to a scratch folder, exit 0.
- [x] Blocking finding 2 (deduplication ran before the split): `common.py`'s
      `clean_dataset_if_unsw` (now `nids.tabular.clean_dataset_if_unsw`) ran
      `nids.cleaning.drop_train_duplicates` over the ENTIRE frame before any split, so
      held-out rows were deduplicated too, despite that function's own docstring saying
      "training partition only". Fixed: `clean_dataset_if_unsw` now performs only the two
      COLUMN-level steps (`drop_unused_columns`, `normalize_service`); a new
      `nids.tabular.split_and_dedupe(df, target, test_size=, seed=, is_unsw_schema=)`
      splits the FULL frame first, then applies `dedupe_train_split` (UNSW-schema only)
      to the resulting TRAIN partition alone. All three scripts now call
      `split_and_dedupe` instead of a raw `train_test_split`. Ordering documented in
      each script's module docstring and each `SKILL.md`. Verified manually on the real
      training CSV: held-out (test) row count matches a plain 30% split of the
      pre-dedup, column-cleaned frame (52,603 vs. an expected 52,602 — within sklearn's
      rounding), train row count dropped from 122,739 to 78,507 (44,232 duplicates
      removed, TRAIN ONLY), `train_df.duplicated().any()` is `False`, and
      `test_df.duplicated().any()` is `True` (proving the held-out split was never
      touched). Direct regression tests in
      `tests/test_tabular.py::TestSplitAndDedupe`/`TestDedupeTrainSplit`/
      `TestCleanDatasetIfUnsw`.
- [x] Standards finding 3 (shared preprocessing outside `src/`): `is_unsw_raw_schema`,
      `clean_dataset_if_unsw`, `build_preprocessor_for`, `numeric_and_categorical_columns`,
      `compute_features`, `resolve_stratify`, and the new `split_and_dedupe`/
      `dedupe_train_split` moved from `skills/_shared/common.py` into the new
      `src/nids/tabular.py` (AGENTS.md: "Shared preprocessing lives in `src/`").
      `common.py` now re-exports them under the same names and keeps only argparse,
      CSV loading, `ResultsWriter` wiring, and the (clustering-only, not a preprocessing
      concern) `choose_n_clusters` helper as its own code. New direct tests in
      `tests/test_tabular.py`.
- [x] Standards finding 4 (private alias defeated its own guard): `_build_generic_preprocessor`
      renamed PUBLIC (`nids.preprocessing.build_generic_preprocessor`) — the previous
      underscore-hack-plus-re-export existed only to dodge a blunt AST guard, not to
      comply with its actual intent. The guard
      (`tests/test_leak_safety.py::test_preprocessing_public_functions_reject_data_shaped_parameters`,
      renamed from `..._take_no_dataframe_parameter`) is now precise: it rejects a
      parameter annotated as DataFrame/Series/ndarray/array-like BY ANNOTATION TEXT
      (`DataFrame`, `Series`, `ndarray`, `ArrayLike` substrings, or an `npt.` prefix),
      while explicitly permitting `bool` and `list[str]`. Added
      `test_refined_guard_still_catches_a_dataframe_annotated_parameter`, proving the
      widened guard still flags `pd.DataFrame`/`pd.Series`/`np.ndarray`/`npt.ArrayLike`/
      `npt.NDArray[...]`-annotated parameters while passing `bool`/`list[str]` ones.
      Module docstring updated to state the refined rule and why it changed.
- [x] Standards finding 5 (stale module docstring): `src/nids/preprocessing.py`'s module
      docstring rewritten to state accurately that no function accepts a DataFrame or
      array-like data parameter, that a `list[str]` of column names is permitted, and
      that the frozen `build_preprocessor`/`build_preprocessor_pair` surface (`bool` or
      no arguments) is unchanged.
- [x] Standards finding 6 (hardcoded threshold): `RareCategoryGrouper(threshold=0.01)`
      in `src/nids/preprocessing.py`'s `build_generic_preprocessor` now reads
      `nids.columns.RARE_CATEGORY_THRESHOLD` instead of a hardcoded `0.01` literal.
- [x] Worth-fixing finding 7 (supervision leaking into unsupervised analysis): both
      clustering skills previously set `n_clusters = max(2, min(y_train.nunique(), 10))`
      — derived from the target — then scored that clustering against that same target.
      Fixed: both scripts accept an explicit `--n-clusters INT` CLI flag; when omitted,
      the new `skills._shared.common.choose_n_clusters(X, seed, k_min=2, k_max=10)`
      chooses `k` via a silhouette-score sweep computed purely from the (already
      preprocessed/PCA-transformed) feature matrix, never the target. Both the chosen
      `n_clusters` and `n_clusters_selection_method` (`user_specified`/
      `silhouette_sweep`/`fixed_minimum`) are recorded as metrics and documented in both
      `SKILL.md` files. New tests:
      `tests/test_skills.py::TestCommonHelpers::test_choose_n_clusters_never_looks_at_the_target`/
      `test_choose_n_clusters_falls_back_to_fixed_minimum_for_a_tiny_dataset`, and an
      end-to-end `--n-clusters` override test per clustering skill.
- [x] Worth-fixing finding 8 (results contract not enforced): `common.resolve_results_writer`
      now validates the final `--output` path component against
      `nids.paths.NOTEBOOK_ID_PATTERN` directly and raises a clear, pattern-naming
      `ValueError` BEFORE any filesystem side effect (previously relied solely on
      `ResultsWriter.__init__`'s own validation, which runs after `root.mkdir()`).
      New tests: `test_resolve_results_writer_rejects_invalid_output_name`/
      `test_resolve_results_writer_accepts_valid_output_name`.
- [x] Worth-fixing finding 9 (generic-path gap undocumented): all three `SKILL.md` files
      now state plainly that only the UNSW-NB15 schema is detected and cleaned
      automatically; for any other dataset the caller is responsible for excluding
      identifier columns via `--exclude`.

## Review findings fixed (round 4)

- [x] Blocking finding 1 (artifacts could escape `results/`): `resolve_results_writer`
      validated `notebook_id` against `NOTEBOOK_ID_PATTERN` but never validated its
      LOCATION, using `ResultsWriter`'s `root=` escape hatch to write a full manifest
      anywhere on disk (e.g. `--output /tmp/anything`), bypassing the containment
      `nids.paths.results_dir()` enforces. Fixed: by default, `output_path.parent.resolve()`
      is now required to be `results_root()` itself or a descendant of it, raising
      `ValueError` naming the violation before `root.mkdir(...)` runs. Added an explicit,
      documented `allow_external: bool = False` escape hatch for tests only (the
      production CLI path never passes it). `tests/test_skills.py::TestCommonHelpers::
      test_resolve_results_writer_accepts_valid_output_name` (which asserted the
      bypassing behaviour against `tmp_path`) was renamed
      `..._with_allow_external` and now passes the flag explicitly; new
      `test_resolve_results_writer_rejects_out_of_tree_output_by_default` proves the
      default rejection, and `test_resolve_results_writer_accepts_output_inside_results_root`
      proves the default acceptance path (via a monkeypatched `results_root()`, never
      the real repo `results/` tree). All three end-to-end skill test classes
      (`TestEdaReductionClustering`/`TestClusteringReduction`/`TestClassification`)
      gained a class-scoped `_stub_results_root` autouse fixture pointing
      `nids.paths.results_root()` at each test's own `tmp_path`, since their existing
      `main(argv)` calls pass a `tmp_path`-based `--output` standing in for the real
      results tree.
- [x] Blocking finding 2 (unbounded silhouette sweep): `eda-reduction-clustering.py`
      called `choose_n_clusters` over the FULL train split (~122k rows on real
      UNSW-NB15), unlike `clustering-reduction.py` which subsamples to 10,000 rows
      first — `silhouette_score` is O(n²), evaluated up to 9 times. Fixed: every
      `silhouette_score` call inside `choose_n_clusters` now passes
      `sample_size=10_000, random_state=seed`, capping the cost regardless of caller.
      Documented in the function docstring and `skills/eda-reduction-clustering/SKILL.md`.
      `uv run pytest -q -m slow tests/test_skills.py` now completes in ~31s (4 tests).
- [x] Standards finding 3 (`assert_no_leakage` outside `src/`): moved from
      `skills/_shared/common.py` into `src/nids/tabular.py` (same category as the
      round-3 move — it enforces the project's hardest data rule using
      `nids.columns.TARGET_COLUMNS`/`DROPPED_COLUMNS`). `common.py` re-exports it
      under the same name for its call sites; module docstring updated. New direct
      tests in `tests/test_tabular.py::TestAssertNoLeakage` (5 cases: pass-through,
      target-present, excluded-present, UNSW-schema-via-`df`, UNSW-schema-via-explicit-flag,
      non-UNSW-schema-skips-UNSW-check) — `tests/test_skills.py::TestCommonHelpers`
      keeps its existing indirect coverage through the re-export.
- [x] Minor finding 4 (variable shadowing): `tests/test_leak_safety.py::
      test_cleaning_report_never_fits_on_the_test_partition` — the AST walker
      variable at the former lines 397-400 (reassigning `root`, already bound to
      `repo_root()` at line 365) renamed to `node_root`.

## Review findings fixed (round 5)

- [x] Finding 3 (silent empty subsample for a non-string stratify column):
      `src/nids/sampling.py::stratified_subsample` stringified `stratify_by`'s
      unique values for `labels_sorted`, then compared that string against the
      original (e.g. int64) column for the row mask -- for `--target label` on
      UNSW-NB15, every count came back 0, the largest-remainder allocation
      silently allocated nothing, and the function returned an EMPTY frame with
      no exception. Fixed: `label_to_value` keeps the ACTUAL unique value
      (original dtype) for both the `available` count and the row-position
      mask; only the diagnostics label text is stringified, preserving the
      existing sort order. Also added a hard invariant check in
      `_largest_remainder_allocation` that raises `ValueError` if the
      allocation ever sums to less than `max_rows`, instead of silently
      returning a short/empty allocation again. New tests in
      `tests/test_sampling.py`:
      `test_int64_stratify_column_returns_non_empty_correctly_allocated_subsample`
      (proves a non-empty, correctly-proportioned subsample on an int64 column)
      and `test_largest_remainder_allocation_raises_instead_of_silently_returning_empty`.
- [x] Finding 1 (column names re-typed outside `columns.py`): `src/nids/
      tabular.py::build_preprocessor_for` re-typed `{"sttl", "ct_state_ttl"}`
      locally instead of importing `nids.columns.TTL_SHORTCUT_COLUMNS`; fixed.
      `skills/classification/classification.py` defined its own module-level
      `TTL_SHORTCUT_COLUMNS = ("sttl", "ct_state_ttl")`, shadowing the
      canonical one; deleted, now imports `nids.columns.TTL_SHORTCUT_COLUMNS`
      directly. Verified: `rg -n '"sttl"' src/ skills/` shows the literal only
      in `src/nids/columns.py`.
- [x] Finding 4 (TTL paired reporting silently skippable): `classification.py`
      previously ran only the `"full"` variant with no manifest record of
      whether the TTL shortcut pair was present. Added
      `_ttl_shortcut_state(features)` (`"both"`/`"none"`/`"partial"`),
      recorded unconditionally as the new `ttl_shortcut_columns_present`
      metric, plus a `writer.add_note("ttl_shortcut_comparison", ...)` call
      (previously unused) explaining what was and wasn't compared in every
      case, including the previously-silent partial-exclusion case. Verified
      manually: a `both`-state run (synthetic `sttl`/`ct_state_ttl` columns)
      and a `none`-state run both produce the metric and note in
      `manifest.json`.
- [x] Finding 2 (`ensure_repo_root_on_syspath` didn't run before the first
      `nids` import it enables): `skills/_shared/common.py` defined the guard
      but only called it from inside each script's `main()`, AFTER this
      module's own `from nids.tabular import (...)` at module scope had
      already executed and already required `nids` to be importable -- so the
      docstring's "from any working directory" promise did not hold without a
      separate `uv sync` install. Fixed by making it genuinely work: the guard
      function moved above the `nids.tabular` import and is now called once,
      at module scope, immediately before that import.
- [x] Finding 5 (bare-expression asserts): `tests/test_tabular.py`'s
      `test_passes_when_target_and_exclude_are_absent` and
      `test_non_unsw_schema_does_not_check_unsw_columns` were missing `assert`
      (ruff B015) -- both fixed.
- [x] Finding 6 (tests fitting on un-split frames): fixed all four sites --
      `tests/test_skills.py::_assert_unsw_run_excludes_dropped_columns`,
      `tests/test_skills.py::TestCommonHelpers::
      test_build_preprocessor_for_generic_never_leaks_target_or_excluded_names`,
      `tests/test_skills.py::TestCommonHelpers::
      test_build_preprocessor_for_unsw_schema_drops_target_and_dropped_columns_regardless_of_exclude`,
      and `tests/test_tabular.py::TestBuildPreprocessorFor::
      test_generic_path_for_non_unsw_dataframe` -- each now calls
      `split_and_dedupe`/`common.split_and_dedupe` first and fits only on the
      resulting train partition, matching `tests/test_preprocessing.py`'s
      existing convention and AGENTS.md's "fit inside a Pipeline on training
      data only" rule with no test-code carve-out.
- [x] Finding 7 (smaller items): added direct `resolve_exclude` coverage in
      `tests/test_skills.py::TestCommonHelpers` (repeated-flag form,
      comma-separated form, mixed form, whitespace/empty-piece handling,
      de-duplication, empty input); `tests/test_tabular.py`'s
      `test_non_unsw_frame_train_is_never_deduplicated` now seeds
      `np.random.default_rng(42)` instead of `0`, matching every other
      fixture; `src/nids/preprocessing.py`'s docstring reference to the old
      test name `test_preprocessing_public_functions_take_no_dataframe_parameter`
      updated to the current `..._reject_data_shaped_parameters`; and
      `tests/test_leak_safety.py`'s `_DATAFRAME_ANNOTATION_NAMES` docstring no
      longer claims a `frozenset` has "source order".

## Known deviations / drops

- `is_unsw_raw_schema` combined with a partial TTL exclusion (exactly one of
  `sttl`/`ct_state_ttl` excluded, not both) falls back to the generic preprocessing path
  instead of attempting to bend `nids.preprocessing.build_preprocessor`'s fixed boolean-toggle
  surface to a partial column subset it was never designed to accept (see design Decision 2).
  This is a deliberate scope narrowing, not a dropped requirement — every dataset shape still
  runs successfully, just through the generic pipeline in that one edge case.
- No `[project.scripts]` CLI entry-point installer was added; scripts are invoked directly via
  `uv run python skills/<name>/<script>.py`, matching how the sibling notebooks already invoke
  `nids`. Explicitly out of scope per design.md.
