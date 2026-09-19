"""Six ordered, unfitted cleaning operations and their diagnostics.

Every mutating step returns `(new_frame, diagnostics)`; the two diagnostic-only
steps (`count_test_internal_duplicates`, `count_label_conflicts`) return only a
dataclass, no frame. `clean_partitions()` is the single composition point that
runs all six steps in the mandated order (see design Decision 2 and the
"Mandated cleaning step order" requirement in the data-cleaning spec).

This module is pure: no class, no `fit`, no `fit_transform`, no file I/O, and
no function mutates its argument (pandas 3's Copy-on-Write makes chained
assignment a silent no-op, so every step builds and returns a new frame).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from nids.columns import (
    DROPPED_COLUMNS,
    FEATURE_COLUMNS,
    KEPT_COLUMNS,
    SERVICE_DASH,
    SERVICE_NONE,
    LeakageKey,
    leakage_key_columns,
)


@dataclass(frozen=True, slots=True)
class DropColumnsDiagnostics:
    """Diagnostics for step 1, the fixed column drop."""

    dropped: tuple[str, ...]
    columns_before: int
    columns_after: int


@dataclass(frozen=True, slots=True)
class ServiceDiagnostics:
    """Diagnostics for step 2, the `service` `"-"` -> `"none"` mapping."""

    partition: str
    dash_rows_before: int
    none_rows_after: int
    replacement_label: str


@dataclass(frozen=True, slots=True)
class DeduplicationDiagnostics:
    """Diagnostics for step 3, train-only exact-duplicate removal."""

    rows_before: int
    rows_after: int
    duplicate_rows_dropped: int
    state_counts_before: Mapping[str, int]
    state_counts_after: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class InternalDuplicateDiagnostics:
    """Diagnostics for step 4, the test-internal duplicate count."""

    rows: int
    duplicate_rows: int
    duplicate_share: float


@dataclass(frozen=True, slots=True)
class LeakageDiagnostics:
    """Diagnostics for step 5, cross-partition leakage removal."""

    comparison_key: LeakageKey
    rows_before: int
    rows_after: int
    removed: int
    retained_contradictory: int
    retained_contradictory_share: float


@dataclass(frozen=True, slots=True)
class LabelConflictDiagnostics:
    """Diagnostics for step 6, the training label-noise diagnostic."""

    conflicting_feature_combinations: int
    rows_in_conflicting_combinations: int


@dataclass(frozen=True, slots=True)
class ClassDistributionDiagnostics:
    """`attack_cat` value counts captured before step 1 and after step 6."""

    train_before: Mapping[str, int]
    train_after: Mapping[str, int]
    test_before: Mapping[str, int]
    test_after: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class StepRowCounts:
    """One row-count observation for `CleaningResult.row_count_timeline()`."""

    step: int
    operation: str
    partition: str
    rows_before: int
    rows_after: int
    rows_removed: int


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """The cleaned partitions plus every diagnostic the six steps produced."""

    train: pd.DataFrame
    test: pd.DataFrame
    drop_columns: DropColumnsDiagnostics
    service_train: ServiceDiagnostics
    service_test: ServiceDiagnostics
    deduplication: DeduplicationDiagnostics
    test_internal_duplicates: InternalDuplicateDiagnostics
    leakage: LeakageDiagnostics
    label_conflicts: LabelConflictDiagnostics
    class_distribution: ClassDistributionDiagnostics

    def row_count_timeline(self) -> list[StepRowCounts]:
        """Project the six ordered steps as table rows.

        Reads only diagnostics fields. Never references `self.train` or
        `self.test` — the empty-frame unit test in `tests/test_cleaning.py`
        proves this by building a `CleaningResult` with empty frames and real
        diagnostics and checking the returned counts are still correct.

        Returns:
            One `StepRowCounts` per (step, partition) pair that step touches,
            in execution order.
        """
        train_raw_rows = sum(self.class_distribution.train_before.values())
        test_raw_rows = sum(self.class_distribution.test_before.values())
        return [
            StepRowCounts(1, "drop_unused_columns", "train", train_raw_rows, train_raw_rows, 0),
            StepRowCounts(1, "drop_unused_columns", "test", test_raw_rows, test_raw_rows, 0),
            StepRowCounts(2, "normalize_service", "train", train_raw_rows, train_raw_rows, 0),
            StepRowCounts(2, "normalize_service", "test", test_raw_rows, test_raw_rows, 0),
            StepRowCounts(
                3,
                "drop_train_duplicates",
                "train",
                self.deduplication.rows_before,
                self.deduplication.rows_after,
                self.deduplication.duplicate_rows_dropped,
            ),
            StepRowCounts(
                4,
                "count_test_internal_duplicates",
                "test",
                self.test_internal_duplicates.rows,
                self.test_internal_duplicates.rows,
                0,
            ),
            StepRowCounts(
                5,
                "drop_test_leakage",
                "test",
                self.leakage.rows_before,
                self.leakage.rows_after,
                self.leakage.removed,
            ),
            StepRowCounts(
                6,
                "count_label_conflicts",
                "train",
                self.deduplication.rows_after,
                self.deduplication.rows_after,
                0,
            ),
        ]


def _row_key(df: pd.DataFrame, columns: Sequence[str]) -> pd.MultiIndex:
    """Single construction point for every row-comparison key.

    Args:
        df: The frame to key.
        columns: The columns making up the key, in order.

    Returns:
        A `MultiIndex` built from `df[columns]`, used for `.isin(...)`
        row-membership tests and for `pd.factorize` grouping.
    """
    return pd.MultiIndex.from_frame(df[list(columns)])


def drop_unused_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, DropColumnsDiagnostics]:
    """Step 1: drop the four fixed columns and reindex to canonical order.

    Args:
        df: A raw or otherwise unprocessed frame carrying `DROPPED_COLUMNS`
            and every `KEPT_COLUMNS` name.

    Returns:
        `(new_frame, diagnostics)`. `new_frame` has exactly `KEPT_COLUMNS`, in
        that order, regardless of `df`'s original column order.

    Raises:
        KeyError: `df` is missing a column in `DROPPED_COLUMNS` or in
            `KEPT_COLUMNS`, naming the missing column(s).
    """
    missing_to_drop = [column for column in DROPPED_COLUMNS if column not in df.columns]
    if missing_to_drop:
        raise KeyError(f"Cannot drop missing column(s): {missing_to_drop}")

    columns_before = len(df.columns)
    dropped = df.drop(columns=list(DROPPED_COLUMNS))

    missing_kept = [column for column in KEPT_COLUMNS if column not in dropped.columns]
    if missing_kept:
        raise KeyError(f"Missing expected column(s) after drop: {missing_kept}")

    result = dropped.reindex(columns=list(KEPT_COLUMNS))
    diagnostics = DropColumnsDiagnostics(
        dropped=DROPPED_COLUMNS,
        columns_before=columns_before,
        columns_after=len(result.columns),
    )
    return result, diagnostics


def normalize_service(
    df: pd.DataFrame, *, partition: str
) -> tuple[pd.DataFrame, ServiceDiagnostics]:
    """Step 2: map `service` value `"-"` to `"none"`, never via the mode.

    Args:
        df: A frame carrying a `service` column.
        partition: `"train"` or `"test"`, recorded in the diagnostics.

    Returns:
        `(new_frame, diagnostics)`. `new_frame` never contains `"-"` in
        `service`.
    """
    dash_before = int((df["service"] == SERVICE_DASH).sum())
    mapped = df.assign(
        service=df["service"].where(df["service"] != SERVICE_DASH, SERVICE_NONE)
    )
    none_after = int((mapped["service"] == SERVICE_NONE).sum())
    diagnostics = ServiceDiagnostics(
        partition=partition,
        dash_rows_before=dash_before,
        none_rows_after=none_after,
        replacement_label=SERVICE_NONE,
    )
    return mapped, diagnostics


def drop_train_duplicates(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, DeduplicationDiagnostics]:
    """Step 3: drop exact duplicate rows from the training partition only.

    Comparison is on the full post-drop, post-mapping row (`KEPT_COLUMNS`).
    `keep="first"` makes the result deterministic given the input's row order.

    Args:
        train: The post-step-2 training frame.

    Returns:
        `(deduplicated_train, diagnostics)`.
    """
    rows_before = len(train)
    state_before = {
        str(key): int(value) for key, value in train["state"].value_counts().items()
    }
    mask = train.duplicated(subset=list(KEPT_COLUMNS), keep="first")
    deduped = train.loc[~mask].reset_index(drop=True)
    rows_after = len(deduped)
    state_after = {
        str(key): int(value) for key, value in deduped["state"].value_counts().items()
    }
    diagnostics = DeduplicationDiagnostics(
        rows_before=rows_before,
        rows_after=rows_after,
        duplicate_rows_dropped=rows_before - rows_after,
        state_counts_before=state_before,
        state_counts_after=state_after,
    )
    return deduped, diagnostics


def count_test_internal_duplicates(test: pd.DataFrame) -> InternalDuplicateDiagnostics:
    """Step 4: count duplicate rows within the testing partition. Read-only.

    Args:
        test: The post-step-2 testing frame.

    Returns:
        Diagnostics only — no frame. The testing partition is never mutated
        by this step.
    """
    rows = len(test)
    duplicate_rows = int(test.duplicated(subset=list(KEPT_COLUMNS), keep="first").sum())
    share = duplicate_rows / rows if rows else 0.0
    return InternalDuplicateDiagnostics(
        rows=rows, duplicate_rows=duplicate_rows, duplicate_share=share
    )


def drop_test_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    comparison_key: LeakageKey = "full_row",
) -> tuple[pd.DataFrame, LeakageDiagnostics]:
    """Step 5: drop test rows matching a train row on the comparison key.

    Uses `MultiIndex.isin`, never `pd.merge` (design Decision 4) — under
    `"features_only"` the train feature keys are not unique, so a merge would
    silently multiply the returned test frame.

    Returns only the cleaned test frame, never a train frame, so this
    direction-critical step cannot mutate train even through a mistaken tuple
    unpack at the call site.

    Args:
        train: The post-step-3 training frame (already deduplicated).
        test: The post-step-2 testing frame.
        comparison_key: `"full_row"` (default) or `"features_only"`.

    Returns:
        `(test_clean, diagnostics)`. `train` is never touched.
    """
    key_columns = leakage_key_columns(comparison_key)
    rows_before = len(test)
    removed_mask = _row_key(test, key_columns).isin(_row_key(train, key_columns))
    test_clean = test.loc[~removed_mask].reset_index(drop=True)
    rows_after = len(test_clean)
    removed = rows_before - rows_after

    feature_mask = _row_key(test_clean, FEATURE_COLUMNS).isin(
        _row_key(train, FEATURE_COLUMNS)
    )
    retained_contradictory = int(feature_mask.sum())
    share = retained_contradictory / rows_after if rows_after else 0.0

    diagnostics = LeakageDiagnostics(
        comparison_key=comparison_key,
        rows_before=rows_before,
        rows_after=rows_after,
        removed=removed,
        retained_contradictory=retained_contradictory,
        retained_contradictory_share=share,
    )
    return test_clean, diagnostics


def count_label_conflicts(train: pd.DataFrame) -> LabelConflictDiagnostics:
    """Step 6: count train feature combinations carrying more than one `attack_cat`.

    Uses a single integer-groupby over `pd.factorize` codes, not a 39-key
    groupby, per design's implementation note.

    Args:
        train: The post-step-3 training frame.

    Returns:
        Diagnostics only — no frame. The training partition is never mutated
        by this step.
    """
    codes, _ = pd.factorize(_row_key(train, FEATURE_COLUMNS))
    codes_series = pd.Series(codes)
    pairs = pd.DataFrame(
        {"g": codes, "attack_cat": train["attack_cat"].to_numpy()}
    ).drop_duplicates()
    distinct_counts = pairs.groupby("g", sort=False).size()
    conflicting_mask = distinct_counts > 1
    conflicting_ids = distinct_counts.index[conflicting_mask]
    rows_in_conflicting = int(codes_series.isin(conflicting_ids).sum())
    return LabelConflictDiagnostics(
        conflicting_feature_combinations=int(conflicting_mask.sum()),
        rows_in_conflicting_combinations=rows_in_conflicting,
    )


def clean_partitions(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    *,
    comparison_key: LeakageKey = "full_row",
) -> CleaningResult:
    """Run the six mandated cleaning steps, in order, as a single operation.

    This is the only place the step order is written. No separate public
    function allows steps 3-6 to run before steps 1-2 have completed on the
    same frame (data-cleaning spec — "The step order is not reassignable by a
    caller").

    Args:
        train_raw: The raw training partition (45 columns).
        test_raw: The raw testing partition (45 columns).
        comparison_key: Forwarded to `drop_test_leakage`.

    Returns:
        A `CleaningResult` holding the cleaned partitions (with a clean
        `RangeIndex`) and every diagnostic the six steps produced.
    """
    train_before = {
        str(key): int(value)
        for key, value in train_raw["attack_cat"].value_counts().items()
    }
    test_before = {
        str(key): int(value)
        for key, value in test_raw["attack_cat"].value_counts().items()
    }

    train1, drop_columns_diag = drop_unused_columns(train_raw)
    test1, _ = drop_unused_columns(test_raw)

    train2, service_train_diag = normalize_service(train1, partition="train")
    test2, service_test_diag = normalize_service(test1, partition="test")

    train3, dedup_diag = drop_train_duplicates(train2)

    internal_dup_diag = count_test_internal_duplicates(test2)

    test3, leakage_diag = drop_test_leakage(train3, test2, comparison_key=comparison_key)

    label_conflict_diag = count_label_conflicts(train3)

    train_after = {
        str(key): int(value) for key, value in train3["attack_cat"].value_counts().items()
    }
    test_after = {
        str(key): int(value) for key, value in test3["attack_cat"].value_counts().items()
    }
    class_distribution = ClassDistributionDiagnostics(
        train_before=train_before,
        train_after=train_after,
        test_before=test_before,
        test_after=test_after,
    )

    return CleaningResult(
        train=train3.reset_index(drop=True),
        test=test3.reset_index(drop=True),
        drop_columns=drop_columns_diag,
        service_train=service_train_diag,
        service_test=service_test_diag,
        deduplication=dedup_diag,
        test_internal_duplicates=internal_dup_diag,
        leakage=leakage_diag,
        label_conflicts=label_conflict_diag,
        class_distribution=class_distribution,
    )
