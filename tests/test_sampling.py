"""Unit tests for `nids.sampling.stratified_subsample`.

Covers every stratified-subsampling spec requirement's scenarios:
deterministic bounded subsample, minimum-per-class floor, class-smaller-than-
floor, input-at-or-below-cap, and deterministic-ordering precondition.
"""

import numpy as np
import pandas as pd
import pytest

from nids.sampling import stratified_subsample


@pytest.fixture
def wide_frame() -> pd.DataFrame:
    """A 1,000-row, 10-class frame with a 3-row class and a 700-row class.

    Class composition (label ascending, `attack_cat` column):
        class_00:   3 rows  (smaller than any floor used in these tests)
        class_01: 700 rows  (dominant majority class)
        class_02..class_08 (7 classes): 37 rows each == 259 rows
        class_09:  38 rows

    Total: 3 + 700 + 259 + 38 == 1000. Each row carries a unique `row_id` so
    tests can assert on exactly *which* rows were selected, not just counts.
    Row order is `attack_cat`-grouped, then explicitly reset -- the
    deterministic order `stratified_subsample` requires as a precondition.
    """
    counts = {"class_00": 3, "class_01": 700}
    for i in range(2, 9):
        counts[f"class_{i:02d}"] = 37
    counts["class_09"] = 38
    assert sum(counts.values()) == 1000

    rows: list[dict[str, object]] = []
    row_id = 0
    for label in sorted(counts):
        for _ in range(counts[label]):
            rows.append({"row_id": row_id, "attack_cat": label})
            row_id += 1
    frame = pd.DataFrame(rows)
    assert len(frame) == 1000
    return frame


def test_repeated_calls_with_same_seed_return_identical_rows(wide_frame: pd.DataFrame) -> None:
    first = stratified_subsample(wide_frame, max_rows=100, floor=5, seed=42)
    second = stratified_subsample(wide_frame, max_rows=100, floor=5, seed=42)
    assert sorted(first.data["row_id"].tolist()) == sorted(second.data["row_id"].tolist())


def test_output_never_exceeds_the_row_cap(wide_frame: pd.DataFrame) -> None:
    result = stratified_subsample(wide_frame, max_rows=100, floor=5, seed=42)
    assert len(result.data) <= 100
    assert result.total_rows == 100


def test_class_smaller_than_floor_yields_all_its_rows(wide_frame: pd.DataFrame) -> None:
    result = stratified_subsample(wide_frame, max_rows=100, floor=5, seed=42)
    by_label = {allocation.label: allocation for allocation in result.allocations}
    small_class = by_label["class_00"]
    assert small_class.available == 3
    assert small_class.allocated == 3
    assert small_class.floor_applied is True
    assert (result.data["attack_cat"] == "class_00").sum() == 3


def test_allocation_sums_exactly_to_max_rows(wide_frame: pd.DataFrame) -> None:
    result = stratified_subsample(wide_frame, max_rows=100, floor=5, seed=42)
    assert sum(allocation.allocated for allocation in result.allocations) == 100
    assert len(result.data) == 100


def test_floor_zero_matches_independently_computed_proportional_allocation(
    wide_frame: pd.DataFrame,
) -> None:
    """Hand-computed largest-remainder reference for `max_rows=100, floor=0` on
    `wide_frame`. Exact shares (`n_c / 1000 * 100 == n_c / 10`):

        class_00: 0.3 -> floor 0, frac 0.3
        class_01: 70.0 -> floor 70, frac 0.0
        class_02..08 (7x 37 rows): 3.7 each -> floor 3, frac 0.7
        class_09 (38 rows): 3.8 -> floor 3, frac 0.8

    Sum of floors: 0 + 70 + 7*3 + 3 = 94, leaving a residual of 6. Ranked by
    fractional remainder descending, ties broken by label ascending:
    class_09 (0.8), then class_02..class_06 (0.7, first five of the seven
    tied classes in label order) -- exactly 6 classes receive one extra row.
    class_07 and class_08 (also at 0.7) do not, since only 6 slots remain.
    """
    expected = {
        "class_00": 0,
        "class_01": 70,
        "class_02": 4,
        "class_03": 4,
        "class_04": 4,
        "class_05": 4,
        "class_06": 4,
        "class_07": 3,
        "class_08": 3,
        "class_09": 4,
    }
    assert sum(expected.values()) == 100

    result = stratified_subsample(wide_frame, max_rows=100, floor=0, seed=42)
    realized = {allocation.label: allocation.allocated for allocation in result.allocations}
    assert realized == expected
    assert all(not allocation.floor_applied for allocation in result.allocations)
    assert len(result.data) == 100


def test_input_at_or_below_cap_returned_unchanged(wide_frame: pd.DataFrame) -> None:
    small = wide_frame.iloc[:50].reset_index(drop=True)
    result = stratified_subsample(small, max_rows=100, floor=5, seed=42)
    assert len(result.data) == len(small)
    pd.testing.assert_frame_equal(
        result.data.sort_values("row_id").reset_index(drop=True),
        small.sort_values("row_id").reset_index(drop=True),
    )
    assert all(not allocation.floor_applied for allocation in result.allocations)


def test_input_at_or_below_cap_consumes_no_rng(
    wide_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = wide_frame.iloc[:20].reset_index(drop=True)

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("default_rng must not be called when N <= max_rows")

    monkeypatch.setattr(np.random, "default_rng", _fail)
    stratified_subsample(small, max_rows=1000, floor=5, seed=42)


def test_infeasible_floor_raises_naming_floor_and_max_rows(wide_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="floor=200") as excinfo:
        stratified_subsample(wide_frame, max_rows=100, floor=200, seed=42)
    assert "max_rows=100" in str(excinfo.value)


def test_deterministic_ordering_precondition(wide_frame: pd.DataFrame) -> None:
    """Two frames with the same row content but different construction order,
    each reset to the same canonical deterministic order before the call,
    MUST select the same underlying rows."""
    frame_a = wide_frame.sort_values("row_id").reset_index(drop=True)
    shuffled = wide_frame.sample(frac=1.0, random_state=42)
    frame_b = shuffled.sort_values("row_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(frame_a, frame_b)

    result_a = stratified_subsample(frame_a, max_rows=100, floor=5, seed=42)
    result_b = stratified_subsample(frame_b, max_rows=100, floor=5, seed=42)
    assert sorted(result_a.data["row_id"].tolist()) == sorted(result_b.data["row_id"].tolist())


def test_max_rows_must_be_positive(wide_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="max_rows"):
        stratified_subsample(wide_frame, max_rows=0)


def test_floor_must_be_non_negative(wide_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="floor"):
        stratified_subsample(wide_frame, floor=-1)


def test_empty_dataframe_raises_value_error() -> None:
    empty = pd.DataFrame({"attack_cat": pd.Series([], dtype="str")})
    with pytest.raises(ValueError, match="empty"):
        stratified_subsample(empty)


def test_missing_stratify_column_raises_key_error(wide_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        stratified_subsample(wide_frame, stratify_by="does_not_exist")


def test_all_ten_classes_present_at_default_floor_on_wide_frame(wide_frame: pd.DataFrame) -> None:
    # base_total at floor=50 is min(3,50)+min(700,50)+7*min(37,50)+min(38,50)
    # == 3+50+259+38 == 350, so max_rows must clear that floor before the
    # allocation is feasible; 400 leaves the largest-remainder pass exercised.
    result = stratified_subsample(wide_frame, max_rows=400, floor=50, seed=42)
    labels = {allocation.label for allocation in result.allocations}
    assert labels == {f"class_{i:02d}" for i in range(10)}
    by_label = {allocation.label for allocation in result.allocations if allocation.allocated > 0}
    assert by_label == labels


def test_to_frame_has_one_row_per_class(wide_frame: pd.DataFrame) -> None:
    result = stratified_subsample(wide_frame, max_rows=100, floor=5, seed=42)
    table = result.to_frame()
    assert len(table) == 10
    assert set(table["label"]) == {f"class_{i:02d}" for i in range(10)}


def test_floored_classes_lists_only_floor_affected_labels(wide_frame: pd.DataFrame) -> None:
    result = stratified_subsample(wide_frame, max_rows=100, floor=5, seed=42)
    assert "class_00" in result.floored_classes
    assert "class_01" not in result.floored_classes
