"""Direct tests for `nids.tabular` -- the dataset-agnostic column/dtype-shaping helpers
moved here from `skills/_shared/common.py` (round-3 review standards finding 3, AGENTS.md:
"Shared preprocessing lives in `src/`").

`skills/_shared/common.py` re-exports these under the same names, and
`tests/test_skills.py::TestCommonHelpers` exercises them indirectly through that
re-export; this module tests `nids.tabular`'s functions directly, per AGENTS.md's
"Functions in `src/` have type hints, docstrings and pytest tests in `tests/`."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nids.columns import RAW_COLUMNS
from nids.tabular import (
    assert_no_leakage,
    build_preprocessor_for,
    clean_dataset_if_unsw,
    compute_features,
    dedupe_train_split,
    is_unsw_raw_schema,
    numeric_and_categorical_columns,
    resolve_stratify,
    split_and_dedupe,
)


def _make_unsw_shaped_frame(n_rows: int = 80, *, duplicate_first_n: int = 0) -> pd.DataFrame:
    """A synthetic frame carrying every `RAW_COLUMNS` name (so `is_unsw_raw_schema`
    is true), with `label` derived from `attack_cat`. When `duplicate_first_n` > 0,
    the first `duplicate_first_n` rows are exact-duplicated once (appended), so
    `drop_train_duplicates` has something real to remove."""
    rng = np.random.default_rng(42)
    data: dict[str, object] = {}
    for column in RAW_COLUMNS:
        if column in ("proto", "service", "state"):
            data[column] = rng.choice(["tcp", "udp", "-"], size=n_rows)
        elif column == "attack_cat":
            data[column] = rng.choice(["Normal", "Exploits", "DoS"], size=n_rows)
        elif column == "label":
            continue
        else:
            data[column] = rng.uniform(0.0, 100.0, size=n_rows)
    data["label"] = (np.asarray(data["attack_cat"]) != "Normal").astype(int)
    frame = pd.DataFrame(data)
    if duplicate_first_n:
        frame = pd.concat([frame, frame.iloc[:duplicate_first_n]], ignore_index=True)
    return frame


class TestIsUnswRawSchema:
    def test_true_for_full_raw_column_set(self) -> None:
        df = _make_unsw_shaped_frame(10)
        assert is_unsw_raw_schema(df) is True

    def test_false_for_unrelated_columns(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert is_unsw_raw_schema(df) is False


class TestResolveStratify:
    def test_returns_y_when_every_class_has_two_rows(self) -> None:
        y = pd.Series(["a", "a", "b", "b"])
        assert resolve_stratify(y) is y

    def test_returns_none_when_a_class_has_one_row(self) -> None:
        y = pd.Series(["a", "a", "b"])
        assert resolve_stratify(y) is None

    def test_returns_none_for_empty_series(self) -> None:
        assert resolve_stratify(pd.Series([], dtype=object)) is None


class TestComputeFeatures:
    def test_excludes_target_and_excluded_columns(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "a": [1.0, 2.0], "target": ["x", "y"]})
        features = compute_features(df, target="target", exclude=["id"])
        assert features == ["a"]

    def test_raises_on_missing_target(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not found in dataset columns"):
            compute_features(df, target="missing", exclude=[])

    def test_raises_on_exclude_typo_absent_from_df_columns(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "target": ["x", "y"]})
        with pytest.raises(ValueError, match="not present in dataset columns"):
            compute_features(df, target="target", exclude=["typo_column"])

    def test_tolerates_exclude_already_dropped_by_cleaning_via_original_columns(self) -> None:
        """Round-3 blocking finding 1: `clean_dataset_if_unsw` drops `id` (a
        `DROPPED_COLUMNS` member) before `compute_features` runs, so validating
        `--exclude id` against the CLEANED frame's columns crashed on every real
        UNSW run. `original_columns` (captured before cleaning) fixes that: a
        pre-cleaning-valid, post-cleaning-absent name is tolerated, while a name
        absent from BOTH sets still raises."""
        cleaned_df = pd.DataFrame({"a": [1.0, 2.0], "target": ["x", "y"]})
        original_columns = ["id", "a", "target"]

        features = compute_features(
            cleaned_df, target="target", exclude=["id"], original_columns=original_columns
        )
        assert features == ["a"]

        with pytest.raises(ValueError, match="not present in dataset columns"):
            compute_features(
                cleaned_df,
                target="target",
                exclude=["typo_column"],
                original_columns=original_columns,
            )

    def test_without_original_columns_a_dropped_exclude_name_still_raises(self) -> None:
        """Belt-and-braces: the tolerance is opt-in via `original_columns`. A
        caller that omits it (passing an already-cleaned frame directly, matching
        the previous behavior) still gets the old strict validation."""
        cleaned_df = pd.DataFrame({"a": [1.0, 2.0], "target": ["x", "y"]})
        with pytest.raises(ValueError, match="not present in dataset columns"):
            compute_features(cleaned_df, target="target", exclude=["id"])


class TestNumericAndCategoricalColumns:
    def test_splits_by_detected_dtype(self) -> None:
        df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"], "unused": [1, 2]})
        numeric, categorical = numeric_and_categorical_columns(df, ["num", "cat"])
        assert numeric == ["num"]
        assert categorical == ["cat"]


class TestCleanDatasetIfUnsw:
    def test_non_unsw_frame_is_returned_unchanged(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        cleaned, is_unsw = clean_dataset_if_unsw(df)
        assert is_unsw is False
        pd.testing.assert_frame_equal(cleaned, df)

    def test_unsw_frame_drops_dropped_columns_and_normalizes_service(self) -> None:
        from nids.columns import DROPPED_COLUMNS

        df = _make_unsw_shaped_frame(20)
        cleaned, is_unsw = clean_dataset_if_unsw(df)
        assert is_unsw is True
        for column in DROPPED_COLUMNS:
            assert column not in cleaned.columns
        assert "-" not in set(cleaned["service"])

    def test_unsw_frame_is_not_deduplicated_here(self) -> None:
        """Round-3 blocking finding 2: `clean_dataset_if_unsw` must NOT run
        `drop_train_duplicates` -- that step only belongs after a split, applied
        to the resulting train partition alone (`split_and_dedupe`/
        `dedupe_train_split`), never on the full pre-split frame."""
        df = _make_unsw_shaped_frame(20, duplicate_first_n=5)
        cleaned, is_unsw = clean_dataset_if_unsw(df)
        assert is_unsw is True
        assert len(cleaned) == len(df)  # no rows removed here


class TestDedupeTrainSplit:
    def test_non_unsw_train_df_is_returned_unchanged(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2]})
        result = dedupe_train_split(df, is_unsw_schema=False)
        pd.testing.assert_frame_equal(result, df)

    def test_unsw_train_df_has_exact_duplicates_removed(self) -> None:
        raw = _make_unsw_shaped_frame(20, duplicate_first_n=5)
        cleaned, _ = clean_dataset_if_unsw(raw)
        assert len(cleaned) == 25

        deduped = dedupe_train_split(cleaned, is_unsw_schema=True)
        assert len(deduped) == 20


class TestSplitAndDedupe:
    def test_split_runs_before_dedupe_train_only_held_out_untouched(self) -> None:
        """Round-3 blocking finding 2's core proof: split the FULL frame first,
        then deduplicate the TRAIN split only -- the held-out/test split must
        keep every row `train_test_split` gave it, byte-identical, even when it
        happens to contain rows that are exact duplicates of each other or of
        train rows."""
        raw = _make_unsw_shaped_frame(200, duplicate_first_n=40)
        cleaned, is_unsw = clean_dataset_if_unsw(raw)
        assert is_unsw is True
        rows_before_split = len(cleaned)

        train_df, test_df = split_and_dedupe(
            cleaned, "attack_cat", test_size=0.3, seed=42, is_unsw_schema=is_unsw
        )

        # The held-out split's row COUNT matches what train_test_split alone
        # would produce over the PRE-dedup frame (30% of 240 = exactly 72). If
        # dedup ran BEFORE the split (the round-3 bug), the pool would already
        # be shrunk by the ~40 exact duplicates before the 30% cut, producing a
        # smaller test split instead -- so this exact count is the regression
        # proof that split runs first.
        expected_test_rows = round(rows_before_split * 0.3)
        assert len(test_df) == expected_test_rows

        # And the train split itself was actually deduplicated (scoped to train
        # only) -- the second half of the ordering proof.
        assert not train_df.duplicated().any()

    def test_non_unsw_frame_train_is_never_deduplicated(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "a": rng.normal(size=100),
                "target": rng.choice(["x", "y"], size=100),
            }
        )
        train_df, test_df = split_and_dedupe(
            df, "target", test_size=0.3, seed=42, is_unsw_schema=False
        )
        assert len(train_df) + len(test_df) == len(df)


class TestBuildPreprocessorFor:
    def test_generic_path_for_non_unsw_dataframe(self) -> None:
        rng = np.random.default_rng(42)
        n_rows = 40
        df = pd.DataFrame(
            {
                "num1": rng.normal(size=n_rows),
                "cat1": rng.choice(["a", "b"], size=n_rows),
                "target": rng.choice(["x", "y"], size=n_rows),
            }
        )
        features = compute_features(df, target="target", exclude=[])
        pipeline, feature_columns_to_use = build_preprocessor_for(df, features, [])
        assert feature_columns_to_use == features

        # Round-5 review finding 6: this test previously `fit_transform`ed the
        # whole, unsplit frame -- fit on the TRAIN split only, per AGENTS.md's
        # "Every imputer, encoder and scaler is fitted inside a scikit-learn
        # Pipeline on training data only."
        train_df, test_df = split_and_dedupe(
            df, "target", test_size=0.3, seed=42, is_unsw_schema=False
        )
        assert len(train_df) + len(test_df) == n_rows
        transformed = pipeline.fit_transform(train_df[feature_columns_to_use])
        assert transformed.shape[0] == len(train_df)

    def test_unsw_path_drops_target_and_dropped_columns_regardless_of_exclude(self) -> None:
        from nids.columns import DROPPED_COLUMNS, TARGET_COLUMNS

        df = _make_unsw_shaped_frame(30)
        features = compute_features(df, target="attack_cat", exclude=["id"])
        assert "label" in features  # compute_features alone doesn't know the schema

        pipeline, feature_columns_to_use = build_preprocessor_for(df, features, ["id"])
        assert "label" not in feature_columns_to_use
        assert "id" not in feature_columns_to_use
        for column in DROPPED_COLUMNS:
            assert column not in feature_columns_to_use
        for column in TARGET_COLUMNS:
            assert column not in feature_columns_to_use


class TestAssertNoLeakage:
    """Direct tests for `assert_no_leakage` (round-4 review standards finding 3: moved
    here from `skills/_shared/common.py`, which re-exports it for its call sites --
    `tests/test_skills.py::TestCommonHelpers` still exercises it indirectly through
    that re-export)."""

    def test_passes_when_target_and_exclude_are_absent(self) -> None:
        assert assert_no_leakage(["a", "b"], target="target", exclude=["id"]) is None

    def test_raises_on_target_present(self) -> None:
        with pytest.raises(ValueError, match="target"):
            assert_no_leakage(["a", "target"], target="target", exclude=[])

    def test_raises_on_excluded_column_present(self) -> None:
        with pytest.raises(ValueError, match="excluded"):
            assert_no_leakage(["a", "id"], target="target", exclude=["id"])

    def test_unsw_schema_via_df_kwarg_catches_target_sibling(self) -> None:
        """Before this guard, only `target`/`exclude` membership was checked, so
        `label` (a perfect `attack_cat != "Normal"` predictor) could leak in
        alongside a `target="attack_cat"` call. Detected here via the `df` kwarg."""
        df = _make_unsw_shaped_frame()
        assert is_unsw_raw_schema(df)

        with pytest.raises(ValueError, match="UNSW-NB15"):
            assert_no_leakage(["dur", "label"], target="attack_cat", exclude=["id"], df=df)

        with pytest.raises(ValueError, match="UNSW-NB15"):
            assert_no_leakage(["dur", "id"], target="attack_cat", exclude=[], df=df)

    def test_unsw_schema_via_explicit_flag_catches_dropped_column(self) -> None:
        """`is_unsw_schema` lets a caller pass the pre-cleaning routing decision
        directly (e.g. because `df` was already cleaned, and `is_unsw_raw_schema`
        on a cleaned frame always returns False)."""
        with pytest.raises(ValueError, match="UNSW-NB15"):
            assert_no_leakage(
                ["dur", "stcpb"], target="attack_cat", exclude=[], is_unsw_schema=True
            )

    def test_non_unsw_schema_does_not_check_unsw_columns(self) -> None:
        """A non-UNSW frame (or `is_unsw_schema=False`) never raises on `label`/`id`
        unless they are `target` or a member of `exclude` -- the UNSW-specific check
        only applies when the schema is actually detected."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert is_unsw_raw_schema(df) is False
        assert assert_no_leakage(["dur", "label"], target="attack_cat", exclude=[], df=df) is None
