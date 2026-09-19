"""Integration tests for `nids.preprocessing.build_preprocessor`.

Fits and transforms the fitted pipeline against the engineered `cleaned`
fixture (see `tests/conftest.py`), which is the only way to exercise the
`ColumnTransformer` end to end. Covers the preprocessing-pipeline spec's
"One-hot encoding fitted on training only", "Skewed features" (both
scenarios), "TTL toggle" (all 3 scenarios), "feature set explicit allow-list"
(integration confirmation), and "No fitted component accepts a test
partition for fitting" scenarios.
"""

import numpy as np
import pandas as pd
import pytest

import nids.preprocessing as preprocessing_module
from nids.cleaning import CleaningResult
from nids.columns import TTL_SHORTCUT_COLUMNS, feature_columns, numeric_feature_columns
from nids.preprocessing import build_preprocessor, build_preprocessor_pair


def test_fit_train_transform_test_shapes_agree(cleaned: CleaningResult) -> None:
    pipe = build_preprocessor()
    pipe.fit(cleaned.train[feature_columns()])
    train_out = pipe.transform(cleaned.train[feature_columns()])
    test_out = pipe.transform(cleaned.test[feature_columns()])
    assert train_out.shape[0] == len(cleaned.train)
    assert test_out.shape[0] == len(cleaned.test)
    assert train_out.shape[1] == test_out.shape[1]


def test_feature_names_out_excludes_both_targets(cleaned: CleaningResult) -> None:
    """Belt-and-braces: fitting on the FULL 41-column cleaned frame (targets
    included) still excludes both targets from the output, via
    `remainder="drop"`."""
    pipe = build_preprocessor()
    pipe.fit(cleaned.train)
    names = pipe.get_feature_names_out()
    assert "attack_cat" not in names
    assert "label" not in names


def test_excluding_ttl_drops_exactly_two_columns(cleaned: CleaningResult) -> None:
    with_ttl = build_preprocessor(include_ttl_features=True)
    without_ttl = build_preprocessor(include_ttl_features=False)
    with_ttl.fit(cleaned.train[feature_columns(True)])
    without_ttl.fit(cleaned.train[feature_columns(False)])

    with_names = set(with_ttl.get_feature_names_out())
    without_names = set(without_ttl.get_feature_names_out())

    assert with_names - without_names == set(TTL_SHORTCUT_COLUMNS)
    assert len(with_names) - len(without_names) == len(TTL_SHORTCUT_COLUMNS)
    for ttl_column in TTL_SHORTCUT_COLUMNS:
        assert ttl_column not in without_names


def test_no_nan_in_output_proves_log1p_before_scale(cleaned: CleaningResult) -> None:
    """If `StandardScaler` ran before `log1p`, centred values below -1 would
    make `log1p` return NaN. A clean output is the ordering proof."""
    pipe = build_preprocessor()
    pipe.fit(cleaned.train[feature_columns()])
    transformed = pipe.transform(cleaned.test[feature_columns()])
    assert not np.isnan(transformed).any()


def test_set_output_pandas_yields_readable_column_names(cleaned: CleaningResult) -> None:
    pipe = build_preprocessor()
    pipe.set_output(transform="pandas")
    pipe.fit(cleaned.train[feature_columns()])
    transformed = pipe.transform(cleaned.test[feature_columns()])
    assert isinstance(transformed, pd.DataFrame)
    assert "sbytes" in transformed.columns
    assert "proto_tcp" in transformed.columns
    assert not any(column.startswith("categorical__") for column in transformed.columns)
    assert not any(column.startswith("numeric__") for column in transformed.columns)


def test_unseen_proto_produces_other_column_not_all_zero(cleaned: CleaningResult) -> None:
    """The `proto == "gre"` fixture row, unseen at fit time, MUST produce
    `proto_other == 1` -- proof `OneHotEncoder(handle_unknown="ignore")` is
    unreachable belt-and-braces, never actually exercised (design Decision 7)."""
    pipe = build_preprocessor()
    pipe.set_output(transform="pandas")
    pipe.fit(cleaned.train[feature_columns()])
    transformed = pipe.transform(cleaned.test[feature_columns()])

    gre_mask = cleaned.test["proto"] == "gre"
    assert gre_mask.sum() == 1, "expected exactly one engineered gre row in cleaned.test"
    gre_index = cleaned.test.index[gre_mask][0]

    assert "proto_other" in transformed.columns
    row = transformed.loc[gre_index]
    assert row["proto_other"] == 1.0
    onehot_proto_columns = [c for c in transformed.columns if c.startswith("proto_")]
    assert row[onehot_proto_columns].sum() == 1.0, (
        "expected exactly one active proto_* column (proto_other), not an "
        "all-zero one-hot block"
    )


def test_build_preprocessor_pair_differs_only_in_ttl(cleaned: CleaningResult) -> None:
    pair = build_preprocessor_pair()
    assert set(pair) == {"with_ttl", "without_ttl"}

    pair["with_ttl"].fit(cleaned.train[feature_columns(True)])
    pair["without_ttl"].fit(cleaned.train[feature_columns(False)])

    with_names = set(pair["with_ttl"].get_feature_names_out())
    without_names = set(pair["without_ttl"].get_feature_names_out())
    assert with_names - without_names == set(TTL_SHORTCUT_COLUMNS)

    with_out = pair["with_ttl"].transform(cleaned.test[feature_columns(True)])
    without_out = pair["without_ttl"].transform(cleaned.test[feature_columns(False)])
    assert not np.isnan(with_out).any()
    assert not np.isnan(without_out).any()


def test_scaler_statistics_reflect_only_training_rows(cleaned: CleaningResult) -> None:
    """The fitted `StandardScaler` inside the `numeric` branch MUST reflect
    only the training partition, never any row from testing."""
    pipe = build_preprocessor()
    pipe.fit(cleaned.train[feature_columns()])
    numeric_scaler = pipe.named_steps["features"].named_transformers_["numeric"]
    numeric_columns = numeric_feature_columns()
    expected_mean = cleaned.train[numeric_columns].mean().to_numpy()
    np.testing.assert_allclose(numeric_scaler.mean_, expected_mean, rtol=1e-8)


def test_build_preprocessor_raises_on_overlapping_routing_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`build_preprocessor`'s runtime invariant MUST raise before constructing
    any transformer when a column would land in more than one of the
    categorical/skewed/numeric routing groups (design's routing assertion,
    an until-now-untested raise path). Triggered by monkeypatching
    `RARE_GROUPED_COLUMNS` to include `dur`, which `SKEWED_COLUMNS` (and
    therefore the `skewed` branch) already claims."""
    monkeypatch.setattr(preprocessing_module, "RARE_GROUPED_COLUMNS", ("proto", "dur"))
    with pytest.raises(ValueError, match="routing groups overlap"):
        preprocessing_module.build_preprocessor()


def test_build_preprocessor_raises_when_routing_groups_do_not_cover_selected_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`build_preprocessor`'s coverage invariant MUST raise when the union of
    the three routing groups is narrower than the selected feature set
    (design's routing assertion, an until-now-untested raise path).
    Triggered by monkeypatching `numeric_feature_columns` to silently drop
    one column, simulating a bad edit to `columns.py`."""
    original_numeric_feature_columns = preprocessing_module.numeric_feature_columns

    def _missing_one_column(include_ttl: bool = True) -> list[str]:
        columns = original_numeric_feature_columns(include_ttl)
        return columns[1:]

    monkeypatch.setattr(
        preprocessing_module, "numeric_feature_columns", _missing_one_column
    )
    with pytest.raises(ValueError, match="do not exactly cover"):
        preprocessing_module.build_preprocessor()
