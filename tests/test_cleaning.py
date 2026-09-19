"""Tests for `src/nids/cleaning.py`.

Uses the engineered `raw_train`/`raw_test`/`cleaned` fixtures from
`tests/conftest.py`. Every count asserted here that is not one of the two
frozen raw dataset shapes is a computed value, never a hardcoded
cleaned-partition row count (data-cleaning spec — "No hardcoded
cleaned-partition row counts").
"""

import pandas as pd
import pytest

from nids.cleaning import (
    CleaningResult,
    ClassDistributionDiagnostics,
    DeduplicationDiagnostics,
    InternalDuplicateDiagnostics,
    LabelConflictDiagnostics,
    LeakageDiagnostics,
    ServiceDiagnostics,
    clean_partitions,
    count_label_conflicts,
    count_test_internal_duplicates,
    drop_test_leakage,
    drop_train_duplicates,
    drop_unused_columns,
    normalize_service,
)
from nids.columns import DROPPED_COLUMNS, KEPT_COLUMNS


def test_step_order_drop_before_duplicate_check(raw_test: pd.DataFrame) -> None:
    """Duplicate status differs depending on whether `stcpb` was dropped first.

    Comparing on the kept columns plus the (not-yet-dropped) `stcpb` column
    only finds the duplicate pair that agrees on `stcpb` too. Comparing on
    the post-drop columns (the mandated order) finds both engineered pairs.
    """
    naive_subset = [*KEPT_COLUMNS, "stcpb"]
    naive_duplicates = int(raw_test.duplicated(subset=naive_subset, keep="first").sum())

    dropped, _ = drop_unused_columns(raw_test)
    diagnostics = count_test_internal_duplicates(dropped)

    assert naive_duplicates == 1
    assert diagnostics.duplicate_rows == 2


def test_drop_unused_columns_removes_exactly_the_four_columns(
    raw_train: pd.DataFrame,
) -> None:
    result, diagnostics = drop_unused_columns(raw_train)
    for column in DROPPED_COLUMNS:
        assert column not in result.columns
    for column in KEPT_COLUMNS:
        assert column in result.columns
    assert list(result.columns) == list(KEPT_COLUMNS)
    assert diagnostics.dropped == DROPPED_COLUMNS
    assert diagnostics.columns_after == len(KEPT_COLUMNS)


def test_drop_unused_columns_raises_keyerror_on_missing_column(
    raw_train: pd.DataFrame,
) -> None:
    incomplete = raw_train.drop(columns=["stcpb"])
    with pytest.raises(KeyError):
        drop_unused_columns(incomplete)


def test_normalize_service_maps_dash_to_none_without_a_mode(
    raw_train: pd.DataFrame,
) -> None:
    dropped, _ = drop_unused_columns(raw_train)
    mapped, diagnostics = normalize_service(dropped, partition="train")
    assert "-" not in set(mapped["service"])
    assert diagnostics.dash_rows_before > 0
    assert diagnostics.none_rows_after == diagnostics.dash_rows_before
    assert diagnostics.replacement_label == "none"


def test_clean_partitions_output_has_forty_one_columns(cleaned: CleaningResult) -> None:
    assert list(cleaned.train.columns) == list(KEPT_COLUMNS)
    assert list(cleaned.test.columns) == list(KEPT_COLUMNS)
    for column in DROPPED_COLUMNS:
        assert column not in cleaned.train.columns
        assert column not in cleaned.test.columns


def test_service_dash_mapping_invariant_holds_for_both_partitions(
    cleaned: CleaningResult,
) -> None:
    assert isinstance(cleaned.service_train, ServiceDiagnostics)
    assert cleaned.service_train.none_rows_after == cleaned.service_train.dash_rows_before
    assert cleaned.service_test.none_rows_after == cleaned.service_test.dash_rows_before
    # The engineered fixture carries 46 udp rows with service == "-".
    assert cleaned.service_train.dash_rows_before == 46


def test_train_dedup_strictly_decreases_train_rows_only(
    cleaned: CleaningResult, raw_test: pd.DataFrame
) -> None:
    assert isinstance(cleaned.deduplication, DeduplicationDiagnostics)
    assert cleaned.deduplication.rows_after < cleaned.deduplication.rows_before
    assert cleaned.deduplication.duplicate_rows_dropped == 6
    # Deduplication never touches the test partition.
    assert cleaned.test_internal_duplicates.rows == len(raw_test)


def test_internal_duplicates_are_counted_and_retained(cleaned: CleaningResult) -> None:
    assert isinstance(cleaned.test_internal_duplicates, InternalDuplicateDiagnostics)
    assert cleaned.test_internal_duplicates.duplicate_rows == 2
    assert cleaned.test_internal_duplicates.duplicate_rows > 0


def test_drop_test_leakage_direction_and_arithmetic_identity(
    cleaned: CleaningResult, raw_test: pd.DataFrame
) -> None:
    test1, _ = drop_unused_columns(raw_test)
    test2, _ = normalize_service(test1, partition="test")
    train3 = cleaned.train

    test_full, leakage_full = drop_test_leakage(train3, test2, comparison_key="full_row")
    test_features, leakage_features = drop_test_leakage(
        train3, test2, comparison_key="features_only"
    )

    assert isinstance(leakage_full, LeakageDiagnostics)
    assert leakage_full.removed == 3
    assert leakage_full.retained_contradictory == 4
    assert leakage_features.removed == 7
    assert leakage_features.retained_contradictory == 0

    # The cross-key arithmetic identity from the spec.
    assert leakage_full.removed + leakage_full.retained_contradictory == leakage_features.removed

    # Test row counts strictly decrease; train row count is untouched by
    # either run (this test's own function checks direction more directly).
    assert len(test_full) < len(test2)
    assert len(test_features) < len(test2)
    assert len(train3) == cleaned.deduplication.rows_after


def test_leakage_never_mutates_train_across_both_keys(
    cleaned: CleaningResult, raw_test: pd.DataFrame
) -> None:
    test1, _ = drop_unused_columns(raw_test)
    test2, _ = normalize_service(test1, partition="test")
    train_before = cleaned.train.copy()

    drop_test_leakage(cleaned.train, test2, comparison_key="full_row")
    drop_test_leakage(cleaned.train, test2, comparison_key="features_only")

    pd.testing.assert_frame_equal(cleaned.train, train_before)


def test_count_label_conflicts_on_engineered_fixture(cleaned: CleaningResult) -> None:
    assert isinstance(cleaned.label_conflicts, LabelConflictDiagnostics)
    assert cleaned.label_conflicts.conflicting_feature_combinations == 2
    assert cleaned.label_conflicts.rows_in_conflicting_combinations == 4


def test_count_label_conflicts_does_not_mutate_train(cleaned: CleaningResult) -> None:
    rows_before = len(cleaned.train)
    count_label_conflicts(cleaned.train)
    assert len(cleaned.train) == rows_before


def test_row_count_timeline_reads_only_diagnostics(cleaned: CleaningResult) -> None:
    """Built with empty frames, `row_count_timeline()` must still report the
    real diagnostic counts -- proving it never touches `.train`/`.test`."""
    empty_result = CleaningResult(
        train=pd.DataFrame(),
        test=pd.DataFrame(),
        drop_columns=cleaned.drop_columns,
        service_train=cleaned.service_train,
        service_test=cleaned.service_test,
        deduplication=cleaned.deduplication,
        test_internal_duplicates=cleaned.test_internal_duplicates,
        leakage=cleaned.leakage,
        label_conflicts=cleaned.label_conflicts,
        class_distribution=cleaned.class_distribution,
    )

    timeline = empty_result.row_count_timeline()
    real_timeline = cleaned.row_count_timeline()

    assert [entry.rows_before for entry in timeline] == [
        entry.rows_before for entry in real_timeline
    ]
    assert [entry.rows_after for entry in timeline] == [
        entry.rows_after for entry in real_timeline
    ]

    dedup_entry = next(entry for entry in timeline if entry.operation == "drop_train_duplicates")
    assert dedup_entry.rows_before == cleaned.deduplication.rows_before
    assert dedup_entry.rows_after == cleaned.deduplication.rows_after
    assert dedup_entry.rows_removed == cleaned.deduplication.duplicate_rows_dropped


def test_clean_output_index_is_a_clean_range_index(cleaned: CleaningResult) -> None:
    assert isinstance(cleaned.train.index, pd.RangeIndex)
    assert isinstance(cleaned.test.index, pd.RangeIndex)
    assert cleaned.train.index[0] == 0
    assert cleaned.test.index[0] == 0


def test_two_independent_runs_reproduce_identical_counts(
    raw_train: pd.DataFrame, raw_test: pd.DataFrame
) -> None:
    first = clean_partitions(raw_train, raw_test)
    second = clean_partitions(raw_train, raw_test)

    assert first.deduplication == second.deduplication
    assert first.test_internal_duplicates == second.test_internal_duplicates
    assert first.leakage == second.leakage
    assert first.label_conflicts == second.label_conflicts
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)


def test_class_distribution_captured_before_and_after(cleaned: CleaningResult) -> None:
    assert isinstance(cleaned.class_distribution, ClassDistributionDiagnostics)
    assert sum(cleaned.class_distribution.train_before.values()) == 206
    assert sum(cleaned.class_distribution.train_after.values()) == cleaned.deduplication.rows_after
