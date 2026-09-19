"""Compose the data-cleaning capability's diagnostics into a committed report.

Runnable as `python -m nids.cleaning_report`. `main()` runs
`validation.validate_raw_data()` against the real raw directory first and
prints download instructions on failure, so the CLI entry point can never
half-run against a missing or malformed raw file -- `cleaning_report`
depends on `validation`, never the reverse.

**Pipeline rule compliance.** AGENTS.md requires every imputer, encoder
and scaler to be fitted inside a scikit-learn Pipeline on training data
only. The `RareCategoryGrouper` used to report learned categories is
therefore fitted inside a one-step `Pipeline` on the cleaned TRAINING
partition; `tests/test_leak_safety.py` enforces the fit target
structurally.

**Hard constraint (design Decision 2).** This module contains no `len()`,
`.shape`, `value_counts()`, `duplicated()`, or `groupby()` applied to a
partition frame anywhere. Every number reported below comes from
`nids.cleaning.CleaningResult`, `nids.validation.ValidationReport`, or a
`RareCategoryGrouper` fitted here on the cleaned training partition --
`tests/test_repo_hygiene.py` enforces this with an AST scan.
"""

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.pipeline import Pipeline

from nids import validation
from nids.cleaning import ClassDistributionDiagnostics, CleaningResult, DeduplicationDiagnostics
from nids.columns import LeakageKey, RARE_GROUPED_COLUMNS
from nids.data import load_clean_partitions
from nids.results import ResultsWriter
from nids.transformers import RareCategoryGrouper

NOTEBOOK_ID: Final[str] = "data_cleaning"

_BENCHMARK_NON_COMPARABILITY_NOTE_ID: Final[str] = "benchmark_non_comparability"
_BENCHMARK_NON_COMPARABILITY_TEXT: Final[str] = (
    "Removing testing rows that leak into the training partition (rows matching "
    "a training row on the comparison key) makes this project's test-set metrics "
    "non-comparable to published UNSW-NB15 benchmarks computed on the full, "
    "unmodified testing set."
)
_RETAINED_ERROR_FLOOR_NOTE_ID: Final[str] = "retained_error_floor"
_RETAINED_ERROR_FLOOR_TEXT: Final[str] = (
    "The retained feature-identical, label-contradictory testing rows are a "
    "near-certain minimum error rate that no model can eliminate: two rows with "
    "identical features but different labels cannot both be classified correctly "
    "by any function of the features alone."
)


def _raw_null_cells(raw_dir: Path | None) -> int:
    """Sum the null-cell count `validation.py` already computed per partition.

    Reads only `PartitionCheck.null_cells` fields -- never a `.isna()`/
    `.shape` call on a partition frame in this module (Decision 2).

    Args:
        raw_dir: Directory to validate. Defaults to `paths.raw_data_dir()`.

    Returns:
        The combined null-cell count across both raw partitions. A partition
        that could not be read at all (missing file) contributes `0`.
    """
    report = validation.validate_raw_data(raw_dir=raw_dir)
    return sum(check.null_cells or 0 for check in report.checks)


def _row_counts_table(result: CleaningResult) -> pd.DataFrame:
    """Project `CleaningResult.row_count_timeline()` as a table, unsorted.

    Args:
        result: The cleaning result to report on.

    Returns:
        One row per `(step, partition)` pair, in execution order --
        deliberately not re-sorted (design's `results.py` module section).
    """
    timeline = result.row_count_timeline()
    return pd.DataFrame(
        {
            "step": [row.step for row in timeline],
            "operation": [row.operation for row in timeline],
            "partition": [row.partition for row in timeline],
            "rows_before": [row.rows_before for row in timeline],
            "rows_after": [row.rows_after for row in timeline],
            "rows_removed": [row.rows_removed for row in timeline],
        }
    )


def _attack_cat_distribution_table(dist: ClassDistributionDiagnostics) -> pd.DataFrame:
    """Build the `attack_cat` before/after table for both partitions.

    Args:
        dist: `CleaningResult.class_distribution`.

    Returns:
        One row per `attack_cat`, sorted by `train_rows_after` descending
        then `attack_cat` ascending.
    """
    categories = sorted(
        set(dist.train_before)
        | set(dist.train_after)
        | set(dist.test_before)
        | set(dist.test_after)
    )
    frame = pd.DataFrame(
        {
            "attack_cat": categories,
            "train_rows_before": [dist.train_before.get(c, 0) for c in categories],
            "train_rows_after": [dist.train_after.get(c, 0) for c in categories],
            "test_rows_before": [dist.test_before.get(c, 0) for c in categories],
            "test_rows_after": [dist.test_after.get(c, 0) for c in categories],
        }
    )
    return frame.sort_values(
        by=["train_rows_after", "attack_cat"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def _state_distribution_table(dedup: DeduplicationDiagnostics) -> pd.DataFrame:
    """Build the `state` before/after-deduplication table for the training partition.

    Args:
        dedup: `CleaningResult.deduplication`.

    Returns:
        One row per `state`, sorted by `rows_before` descending then
        `state` ascending.
    """
    states = sorted(set(dedup.state_counts_before) | set(dedup.state_counts_after))
    frame = pd.DataFrame(
        {
            "state": states,
            "rows_before": [dedup.state_counts_before.get(s, 0) for s in states],
            "rows_after": [dedup.state_counts_after.get(s, 0) for s in states],
        }
    )
    return frame.sort_values(
        by=["rows_before", "state"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def _rare_categories_table(grouper: RareCategoryGrouper) -> pd.DataFrame:
    """Build the table of every category grouped into `"other"`, with its row count.

    Args:
        grouper: A `RareCategoryGrouper` already fitted on the cleaned
            training partition's `RARE_GROUPED_COLUMNS`.

    Returns:
        One row per `(column, category)` pair that was NOT kept as itself
        (that is, every category folded into `other_label`), sorted by
        `column` ascending, then `rows` descending, then `category`
        ascending.
    """
    records: list[dict[str, object]] = []
    for column, counts in grouper.category_counts_.items():
        frequent = grouper.frequent_categories_[column]
        for category, rows in counts.items():
            if category in frequent:
                continue
            records.append({"column": column, "category": category, "rows": rows})
    frame = pd.DataFrame(records, columns=["column", "category", "rows"])
    return frame.sort_values(
        by=["column", "rows", "category"], ascending=[True, False, True], kind="stable"
    ).reset_index(drop=True)


def _class_balance_figure(dist: ClassDistributionDiagnostics) -> plt.Figure:
    """Build the grouped-bar, log-scale class-balance figure for both partitions.

    Args:
        dist: `CleaningResult.class_distribution`.

    Returns:
        A matplotlib `Figure` with one subplot per partition, English title
        and axis labels, log y-scale (Worms ~127 against Normal ~51,890 is
        unreadable linearly).
    """
    categories = sorted(
        set(dist.train_before)
        | set(dist.train_after)
        | set(dist.test_before)
        | set(dist.test_after)
    )
    positions = list(range(len(categories)))
    width = 0.35

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 6))
    for axis, before, after, title in (
        (ax_train, dist.train_before, dist.train_after, "Training partition"),
        (ax_test, dist.test_before, dist.test_after, "Testing partition"),
    ):
        before_counts = [before.get(c, 0) for c in categories]
        after_counts = [after.get(c, 0) for c in categories]
        axis.bar(
            [p - width / 2 for p in positions], before_counts, width, label="Before cleaning"
        )
        axis.bar(
            [p + width / 2 for p in positions], after_counts, width, label="After cleaning"
        )
        axis.set_yscale("log")
        axis.set_xticks(positions)
        axis.set_xticklabels(categories, rotation=45, ha="right")
        axis.set_title(f"{title} class balance")
        axis.set_xlabel("attack_cat")
        axis.set_ylabel("Row count (log scale)")
        axis.legend()

    fig.suptitle("Class balance before and after cleaning")
    fig.tight_layout()
    return fig


def _metric_entries(
    result: CleaningResult, *, raw_null_cells: int
) -> list[tuple[str, int | float | str, str]]:
    """Build the twelve `(name, value, description)` metric tuples this report emits.

    Args:
        result: The cleaning result to report on.
        raw_null_cells: The combined raw null-cell count from `_raw_null_cells`.

    Returns:
        The metric tuples, in the fixed order named in the design's
        "cleaning_report.py" module section.
    """
    dedup = result.deduplication
    leakage = result.leakage
    label_conflicts = result.label_conflicts
    service_dash_total = (
        result.service_train.dash_rows_before + result.service_test.dash_rows_before
    )
    return [
        (
            "train_duplicate_rows_dropped",
            dedup.duplicate_rows_dropped,
            "Exact-duplicate rows dropped from the training partition (step 3).",
        ),
        (
            "test_internal_duplicate_rows",
            result.test_internal_duplicates.duplicate_rows,
            "Duplicate rows found within the testing partition itself (step 4, "
            "diagnostic only -- never dropped).",
        ),
        (
            "test_leakage_rows_dropped",
            leakage.removed,
            "Testing rows removed because they match a training row on the "
            "leakage comparison key (step 5).",
        ),
        (
            "test_leakage_comparison_key",
            leakage.comparison_key,
            "The leakage comparison key used for step 5: 'full_row' or "
            "'features_only'.",
        ),
        (
            "test_contradictory_rows_retained",
            leakage.retained_contradictory,
            "Retained testing rows that match a training row on features alone "
            "but carry a different label -- feature-identical, label-"
            "contradictory rows.",
        ),
        (
            "test_contradictory_share_of_retained",
            leakage.retained_contradictory_share,
            "test_contradictory_rows_retained as a share of the retained "
            "testing set (rows_after step 5).",
        ),
        (
            "label_noise_floor_combinations",
            label_conflicts.conflicting_feature_combinations,
            "Distinct training feature combinations carrying more than one "
            "attack_cat value (step 6, diagnostic only).",
        ),
        (
            "label_noise_floor_rows",
            label_conflicts.rows_in_conflicting_combinations,
            "Training rows that fall inside a conflicting feature combination.",
        ),
        (
            "service_dash_rows_mapped",
            service_dash_total,
            "Rows across both partitions where service == '-' was mapped to "
            "'none' (step 2).",
        ),
        (
            "raw_null_cells",
            raw_null_cells,
            "Null cells found across both raw partition files by "
            "validation.validate_raw_data().",
        ),
        (
            "train_rows_final",
            dedup.rows_after,
            "Training partition row count after all cleaning steps.",
        ),
        (
            "test_rows_final",
            leakage.rows_after,
            "Testing partition row count after all cleaning steps.",
        ),
    ]


def build_cleaning_report(
    *,
    comparison_key: LeakageKey = "full_row",
    raw_dir: Path | None = None,
    results_root: Path | None = None,
    generated_at: datetime | None = None,
) -> Path:
    """Run the cleaning steps and write `results/data_cleaning/`.

    Args:
        comparison_key: Leakage comparison strategy forwarded to
            `load_clean_partitions`.
        raw_dir: Directory to read the raw partitions from. Defaults to
            `paths.raw_data_dir()`.
        results_root: The `results/` root to write under. Defaults to
            `paths.results_root()`.
        generated_at: Explicit `manifest.json` timestamp override, forwarded
            to `ResultsWriter`.

    Returns:
        `results/data_cleaning/` (or `results_root/data_cleaning/` when
        `results_root` is given).
    """
    result = load_clean_partitions(comparison_key=comparison_key, raw_dir=raw_dir)
    raw_null_cells = _raw_null_cells(raw_dir)

    # AGENTS.md: "Every imputer, encoder and scaler is fitted inside a
    # scikit-learn Pipeline on training data only." The grouper is an encoder,
    # so it is fitted inside a Pipeline even though this is reporting output
    # rather than model input. Fitted on the cleaned TRAINING partition only.
    grouper_pipeline = Pipeline([("group_rare", RareCategoryGrouper())])
    grouper_pipeline.fit(result.train[list(RARE_GROUPED_COLUMNS)])
    grouper = grouper_pipeline.named_steps["group_rare"]

    with ResultsWriter(
        NOTEBOOK_ID, root=results_root, generated_at=generated_at
    ) as writer:
        writer.add_table(
            _row_counts_table(result),
            name="row_counts_before_after",
            title="Row counts before and after each cleaning step",
            description=(
                "Row counts per partition after each of the six ordered "
                "cleaning steps, in execution order."
            ),
        )
        writer.add_table(
            _attack_cat_distribution_table(result.class_distribution),
            name="attack_cat_distribution",
            title="attack_cat distribution before and after cleaning",
            description=(
                "attack_cat row counts per partition, before and after the "
                "ordered cleaning steps."
            ),
        )
        writer.add_table(
            _state_distribution_table(result.deduplication),
            name="state_distribution",
            title="state distribution before and after deduplication",
            description=(
                "state row counts in the training partition, before and "
                "after exact-duplicate removal (step 3)."
            ),
        )
        writer.add_table(
            _rare_categories_table(grouper),
            name="rare_categories",
            title="Rare categories grouped into 'other'",
            description=(
                "Every proto/service/state category below the 1% training-"
                "frequency threshold, grouped into 'other' by "
                "RareCategoryGrouper fitted on the cleaned training "
                "partition, with its row count."
            ),
        )
        writer.add_figure(
            _class_balance_figure(result.class_distribution),
            name="class_balance_before_after",
            title="Class balance before and after cleaning",
            description=(
                "Grouped bar chart (log y-scale) of attack_cat row counts "
                "before and after cleaning, per partition."
            ),
        )

        for name, value, description in _metric_entries(result, raw_null_cells=raw_null_cells):
            writer.add_metric(name, value, description)

        writer.add_note(_BENCHMARK_NON_COMPARABILITY_NOTE_ID, _BENCHMARK_NON_COMPARABILITY_TEXT)
        writer.add_note(_RETAINED_ERROR_FLOOR_NOTE_ID, _RETAINED_ERROR_FLOOR_TEXT)

        counts_payload: dict[str, object] = {
            entry.name: entry.value for entry in writer.metrics
        }
        counts_payload["notes"] = {note.id: note.text for note in writer.notes}
        writer.write_json(counts_payload, name="counts")

        directory = writer.directory

    plt.close("all")
    return directory


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: validate the real raw data, then build the report.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults
            to `sys.argv[1:]` via `argparse`.

    Returns:
        `0` on success, `1` if raw-data validation failed (download
        instructions are printed in that case, and the report is never
        run).
    """
    parser = argparse.ArgumentParser(
        prog="python -m nids.cleaning_report",
        description="Build the data-cleaning results report under results/data_cleaning/.",
    )
    parser.add_argument(
        "--comparison-key",
        choices=["full_row", "features_only"],
        default="full_row",
        help="Leakage comparison key forwarded to the cleaning step (default: full_row).",
    )
    args = parser.parse_args(argv)

    report = validation.validate_raw_data()
    if not report.ok:
        print(report.render())
        print()
        print(validation.download_instructions())
        return 1

    directory = build_cleaning_report(comparison_key=args.comparison_key)
    print(f"Wrote cleaning report to {directory}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
