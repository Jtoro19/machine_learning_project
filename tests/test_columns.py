"""Tests for the frozen column allow-lists and their selector functions."""

import pandas as pd
import pytest

from nids import columns


def test_raw_columns_has_forty_five_entries() -> None:
    assert len(columns.RAW_COLUMNS) == 45


def test_kept_columns_has_forty_one_entries() -> None:
    assert len(columns.KEPT_COLUMNS) == 41


@pytest.mark.parametrize(
    ("include_ttl", "expected_count"),
    [(True, 39), (False, 37)],
)
def test_feature_columns_count_per_toggle(
    include_ttl: bool, expected_count: int
) -> None:
    assert len(columns.feature_columns(include_ttl)) == expected_count


@pytest.mark.parametrize("include_ttl", [True, False])
def test_feature_columns_never_include_targets(include_ttl: bool) -> None:
    features = columns.feature_columns(include_ttl)
    assert "attack_cat" not in features
    assert "label" not in features


@pytest.mark.parametrize("key", ["full_row", "features_only"])
def test_leakage_key_columns_always_include_ttl_shortcut(key: str) -> None:
    key_columns = set(columns.leakage_key_columns(key))
    assert set(columns.TTL_SHORTCUT_COLUMNS) <= key_columns


def test_leakage_key_columns_takes_no_ttl_parameter() -> None:
    import inspect

    signature = inspect.signature(columns.leakage_key_columns)
    assert "include_ttl" not in signature.parameters


def test_routing_groups_are_pairwise_disjoint_and_exhaustive() -> None:
    categorical = set(columns.CATEGORICAL_COLUMNS)
    skewed = set(columns.SKEWED_COLUMNS)
    numeric = set(columns.numeric_feature_columns(include_ttl=True))

    assert categorical.isdisjoint(skewed)
    assert categorical.isdisjoint(numeric)
    assert skewed.isdisjoint(numeric)
    assert categorical | skewed | numeric == set(columns.FEATURE_COLUMNS)


def test_skewed_columns_are_non_negative_on_a_synthetic_fixture() -> None:
    fixture = pd.DataFrame(
        {column: [0.0, 1.5, 42.0] for column in columns.SKEWED_COLUMNS}
    )

    for column in columns.SKEWED_COLUMNS:
        assert (fixture[column] >= 0).all()
