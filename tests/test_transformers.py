"""Unit tests for `nids.transformers.RareCategoryGrouper`.

Covers the preprocessing-pipeline spec's "Rare-category grouping is fitted
state, never a hardcoded list" (all 3 scenarios) and "The rare-category
threshold is fitted on the cleaned training partition only" scenario.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from nids.cleaning import CleaningResult
from nids.columns import OTHER_CATEGORY
from nids.transformers import RareCategoryGrouper


def test_boundary_exactly_at_threshold_is_kept(cleaned: CleaningResult) -> None:
    """`arp` sits at exactly 2/200 == 1.0% in the cleaned train fixture and MUST
    be kept as itself (the boundary is inclusive, `>=`)."""
    grouper = RareCategoryGrouper()
    grouper.fit(cleaned.train[["proto"]])
    assert "arp" in grouper.frequent_categories_["proto"]


def test_below_threshold_is_grouped(cleaned: CleaningResult) -> None:
    """`ospf` and `sctp` each sit at 1/200 == 0.5% in the cleaned train fixture
    and MUST be grouped -- strictly below 1% is never kept."""
    grouper = RareCategoryGrouper()
    grouper.fit(cleaned.train[["proto"]])
    frequent = grouper.frequent_categories_["proto"]
    assert "ospf" not in frequent
    assert "sctp" not in frequent


def test_frequent_categories_learned_not_hardcoded() -> None:
    """The 2%/0.5% split from the preprocessing-pipeline spec's first scenario,
    built as an isolated synthetic frame independent of the shared fixture."""
    frame_above = pd.DataFrame({"proto": ["tcp"] * 98 + ["rare"] * 2})  # rare == 2/100 == 2%
    grouper_above = RareCategoryGrouper()
    grouper_above.fit(frame_above)
    assert "rare" in grouper_above.frequent_categories_["proto"]

    frame_below = pd.DataFrame({"proto": ["tcp"] * 199 + ["rare"]})  # rare == 1/200 == 0.5%
    grouper_below = RareCategoryGrouper()
    grouper_below.fit(frame_below)
    assert "rare" not in grouper_below.frequent_categories_["proto"]


def test_rare_and_unseen_both_map_to_other() -> None:
    """A value rare in training and a value never seen in training MUST both
    map to `"other"` at transform time, via the identical code path."""
    train_below = pd.DataFrame({"state": ["FIN"] * 199 + ["INT"]})  # INT: 1/200 == 0.5% rare
    grouper = RareCategoryGrouper()
    grouper.fit(train_below[["state"]])

    test_frame = pd.DataFrame({"state": ["INT", "NEVER_SEEN", "FIN"]})
    transformed = grouper.transform(test_frame)
    assert transformed["state"].tolist() == [OTHER_CATEGORY, OTHER_CATEGORY, "FIN"]


def test_unseen_in_fit_maps_to_other_on_real_fixture(cleaned: CleaningResult) -> None:
    """The `proto == "gre"` value, present only in the engineered test fixture,
    is unseen at fit time and MUST map to `"other"`, not raise."""
    grouper = RareCategoryGrouper()
    grouper.fit(cleaned.train[["proto"]])
    gre_rows = cleaned.test.loc[cleaned.test["proto"] == "gre", ["proto"]]
    assert len(gre_rows) == 1, "expected the engineered gre row to survive cleaning"
    transformed = grouper.transform(gre_rows)
    assert transformed["proto"].tolist() == [OTHER_CATEGORY]


def test_fitted_on_frame_a_applied_to_frame_b_keeps_a_category_set() -> None:
    """A grouper fitted on frame A and applied to frame B MUST use A's
    frequent-category set, not recompute one from B."""
    frame_a = pd.DataFrame({"proto": ["tcp"] * 198 + ["boundary"] * 2})  # boundary: 1.0% kept in A
    frame_b = pd.DataFrame({"proto": ["boundary"] * 1 + ["tcp"] * 199})  # boundary: 0.5% in B alone

    grouper = RareCategoryGrouper()
    grouper.fit(frame_a)
    assert "boundary" in grouper.frequent_categories_["proto"]

    transformed_b = grouper.transform(frame_b)
    # "boundary" survives in B's transform because A's fitted state says so,
    # even though B alone would have grouped it at 0.5%.
    assert (transformed_b["proto"] == "boundary").sum() == 1
    assert "boundary" in transformed_b["proto"].tolist()


def test_grouper_fit_differs_between_raw_and_cleaned(
    raw_train: pd.DataFrame, cleaned: CleaningResult
) -> None:
    """Fitting on the raw train fixture vs the cleaned train fixture MUST
    produce different learned category sets (preprocessing-pipeline spec --
    "Fitting on different frames produces different learned category sets").

    The engineered fixture makes this observable through `service`: the raw
    frame still carries the literal `"-"` dash value (46/206 rows), while the
    cleaned frame carries `"none"` in its place (46/200 rows) after
    `normalize_service` runs. The two fitted `frequent_categories_["service"]`
    sets therefore differ by construction -- proof that fitting on the wrong
    input (raw, pre-cleaning) learns the wrong category set, which is exactly
    the failure design Decision 3 exists to prevent.
    """
    raw_grouper = RareCategoryGrouper()
    raw_grouper.fit(raw_train[["proto", "service", "state"]])

    cleaned_grouper = RareCategoryGrouper()
    cleaned_grouper.fit(cleaned.train[["proto", "service", "state"]])

    assert raw_grouper.frequent_categories_ != cleaned_grouper.frequent_categories_
    assert "-" in raw_grouper.frequent_categories_["service"]
    assert "-" not in cleaned_grouper.frequent_categories_["service"]
    assert "none" in cleaned_grouper.frequent_categories_["service"]
    assert "none" not in raw_grouper.frequent_categories_["service"]


def test_clone_get_params_set_params_round_trip() -> None:
    grouper = RareCategoryGrouper(threshold=0.02, other_label="rare_bucket")
    cloned = clone(grouper)
    assert cloned.get_params() == grouper.get_params()
    cloned.set_params(threshold=0.05)
    assert cloned.threshold == 0.05
    assert grouper.threshold == 0.02  # unaffected by the clone's mutation


def test_fit_transform_equals_fit_then_transform() -> None:
    frame = pd.DataFrame({"proto": ["tcp"] * 90 + ["rare"] * 10})
    via_chain = RareCategoryGrouper().fit(frame).transform(frame)
    via_fit_transform = RareCategoryGrouper().fit_transform(frame)
    pd.testing.assert_frame_equal(via_chain, via_fit_transform)


def test_other_label_collision_raises_value_error() -> None:
    frame = pd.DataFrame({"proto": ["tcp"] * 50 + [OTHER_CATEGORY] * 50})
    with pytest.raises(ValueError, match="other_label"):
        RareCategoryGrouper(other_label=OTHER_CATEGORY).fit(frame)


def test_non_dataframe_input_raises_type_error_naming_the_type() -> None:
    array: np.ndarray[Any, Any] = np.array([["tcp"], ["udp"]])
    with pytest.raises(TypeError, match="ndarray"):
        RareCategoryGrouper().fit(array)


def test_threshold_out_of_range_raises_value_error() -> None:
    frame = pd.DataFrame({"proto": ["tcp", "udp"]})
    with pytest.raises(ValueError, match="threshold"):
        RareCategoryGrouper(threshold=1.0).fit(frame)
    with pytest.raises(ValueError, match="threshold"):
        RareCategoryGrouper(threshold=-0.1).fit(frame)


def test_empty_frame_raises_value_error() -> None:
    frame = pd.DataFrame({"proto": pd.Series([], dtype="str")})
    with pytest.raises(ValueError, match="empty"):
        RareCategoryGrouper().fit(frame)


def test_transform_before_fit_raises_not_fitted() -> None:
    from sklearn.exceptions import NotFittedError

    frame = pd.DataFrame({"proto": ["tcp"]})
    with pytest.raises(NotFittedError):
        RareCategoryGrouper().transform(frame)


def test_transform_rejects_mismatched_columns() -> None:
    fit_frame = pd.DataFrame({"proto": ["tcp", "udp"]})
    grouper = RareCategoryGrouper().fit(fit_frame)
    with pytest.raises(ValueError, match="columns"):
        grouper.transform(pd.DataFrame({"state": ["FIN", "INT"]}))


def test_transform_never_mutates_input() -> None:
    frame = pd.DataFrame({"proto": ["tcp"] * 90 + ["rare"] * 10})
    original = frame.copy()
    RareCategoryGrouper().fit(frame).transform(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_get_feature_names_out_one_to_one() -> None:
    frame = pd.DataFrame({"proto": ["tcp", "udp"], "state": ["FIN", "INT"]})
    grouper = RareCategoryGrouper().fit(frame)
    names = grouper.get_feature_names_out()
    assert list(names) == ["proto", "state"]
    echoed = grouper.get_feature_names_out(["proto", "state"])
    assert list(echoed) == ["proto", "state"]
    with pytest.raises(ValueError, match="input_features"):
        grouper.get_feature_names_out(["proto"])
