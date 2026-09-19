"""Tests for `src/nids/data.py`'s `load_clean_partitions` facade."""

from pathlib import Path

import pandas as pd

from nids.cleaning import clean_partitions
from nids.data import load_clean_partitions
from nids.loading import RAW_TEST_FILENAME, RAW_TRAIN_FILENAME, load_raw_test, load_raw_train


def _write_synthetic_raw(
    tmp_path: Path, raw_train: pd.DataFrame, raw_test: pd.DataFrame
) -> Path:
    raw_train.to_csv(tmp_path / RAW_TRAIN_FILENAME, index=False)
    raw_test.to_csv(tmp_path / RAW_TEST_FILENAME, index=False)
    return tmp_path


def test_load_clean_partitions_matches_manual_composition(
    tmp_path: Path, raw_train: pd.DataFrame, raw_test: pd.DataFrame
) -> None:
    """`load_clean_partitions` composes loading and cleaning identically to a
    manual `load_raw_train` + `load_raw_test` + `clean_partitions` chain over
    the same synthetic CSVs."""
    raw_dir = _write_synthetic_raw(tmp_path, raw_train, raw_test)

    result = load_clean_partitions(raw_dir=raw_dir)
    expected = clean_partitions(
        load_raw_train(raw_dir=raw_dir), load_raw_test(raw_dir=raw_dir)
    )

    pd.testing.assert_frame_equal(result.train, expected.train)
    pd.testing.assert_frame_equal(result.test, expected.test)
    assert result.deduplication == expected.deduplication
    assert result.leakage == expected.leakage
    assert result.label_conflicts == expected.label_conflicts


def test_load_clean_partitions_forwards_comparison_key(
    tmp_path: Path, raw_train: pd.DataFrame, raw_test: pd.DataFrame
) -> None:
    """`comparison_key` reaches `clean_partitions` unchanged."""
    raw_dir = _write_synthetic_raw(tmp_path, raw_train, raw_test)

    full_row = load_clean_partitions(comparison_key="full_row", raw_dir=raw_dir)
    features_only = load_clean_partitions(comparison_key="features_only", raw_dir=raw_dir)

    assert full_row.leakage.comparison_key == "full_row"
    assert features_only.leakage.comparison_key == "features_only"
    assert features_only.leakage.retained_contradictory == 0
    assert (
        full_row.leakage.removed + full_row.leakage.retained_contradictory
        == features_only.leakage.removed
    )
