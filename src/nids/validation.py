"""Raw dataset validation with manual-download guidance.

This module never downloads anything. `main()` prints a validation report and,
on failure, manual UNSW-NB15 acquisition instructions, then exits non-zero. No
module in this package performs a network request, invokes a download tool, or
otherwise acquires the dataset automatically.

`175_341`, `82_332`, and `45` are the only dataset row/column literals permitted
anywhere in `src/nids`. They are properties of the unmodified raw input, not of
any cleaning step, and they live in this module alone (see `RAW_EXPECTATIONS`).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from nids.loading import RAW_TEST_FILENAME, RAW_TRAIN_FILENAME, STRING_COLUMNS, STRING_DTYPE
from nids.columns import RAW_COLUMNS
from nids.paths import raw_data_dir


@dataclass(frozen=True, slots=True)
class PartitionExpectation:
    """The documented raw shape a partition file is expected to match."""

    filename: str
    rows: int
    columns: int


RAW_EXPECTATIONS: Final[tuple[PartitionExpectation, ...]] = (
    PartitionExpectation(RAW_TRAIN_FILENAME, 175_341, 45),
    PartitionExpectation(RAW_TEST_FILENAME, 82_332, 45),
)
"""The documented raw shape of each partition. The only place `175_341`, `82_332`,
and `45` may appear as literals anywhere in `src/nids`."""


@dataclass(frozen=True, slots=True)
class PartitionCheck:
    """The result of validating one raw partition file."""

    filename: str
    exists: bool
    rows: int | None
    columns: int | None
    expected_rows: int
    expected_columns: int
    columns_match: bool
    numeric_dtypes_ok: bool
    null_cells: int | None
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when the file exists, matches the expected shape and schema, every
        expected-numeric column parsed as numeric, and no null cell was found."""
        return (
            self.exists
            and self.rows == self.expected_rows
            and self.columns == self.expected_columns
            and self.columns_match
            and self.numeric_dtypes_ok
            and self.null_cells == 0
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The combined validation result for both raw partitions."""

    checks: tuple[PartitionCheck, ...]

    @property
    def ok(self) -> bool:
        """True when every partition check passed."""
        return all(check.ok for check in self.checks)

    def render(self) -> str:
        """Render a human-readable multi-line report of every partition check.

        Returns:
            A report naming, for each partition, its status, observed and expected
            shape, and any problems found. The final line is `OK` or `FAILED`.
        """
        lines: list[str] = []
        for check in self.checks:
            status = "OK" if check.ok else "FAIL"
            lines.append(f"[{status}] {check.filename}")
            if check.exists:
                lines.append(
                    f"    rows={check.rows} (expected {check.expected_rows}), "
                    f"columns={check.columns} (expected {check.expected_columns})"
                )
                lines.append(
                    f"    columns_match={check.columns_match}, "
                    f"numeric_dtypes_ok={check.numeric_dtypes_ok}, "
                    f"null_cells={check.null_cells}"
                )
            for problem in check.problems:
                lines.append(f"    - {problem}")
        lines.append("")
        lines.append("OK" if self.ok else "FAILED")
        return "\n".join(lines)


def _read_raw_frame(path: Path) -> pd.DataFrame:
    """Read a raw CSV file for validation, without enforcing the raw schema.

    Unlike `loading._read_partition`, this never raises on a schema mismatch, so a
    malformed file can still be reported with its actual shape. String dtypes are
    applied only to columns from STRING_COLUMNS that are actually present.

    Args:
        path: Path to a raw UNSW-NB15 CSV file.

    Returns:
        The parsed data frame, in whatever column order and count the file holds.
    """
    header = pd.read_csv(path, nrows=0).columns
    dtype = {column: STRING_DTYPE for column in STRING_COLUMNS if column in header}
    return pd.read_csv(path, dtype=dtype)


def _check_partition(directory: Path, expectation: PartitionExpectation) -> PartitionCheck:
    """Validate one raw partition file against its expectation.

    Args:
        directory: Directory the partition file is expected under.
        expectation: The documented shape this partition must match.

    Returns:
        The partition's validation result.
    """
    path = directory / expectation.filename
    if not path.is_file():
        return PartitionCheck(
            filename=expectation.filename,
            exists=False,
            rows=None,
            columns=None,
            expected_rows=expectation.rows,
            expected_columns=expectation.columns,
            columns_match=False,
            numeric_dtypes_ok=False,
            null_cells=None,
            problems=(f"{expectation.filename} is missing from {directory}",),
        )

    frame = _read_raw_frame(path)
    rows, columns = frame.shape
    columns_match = tuple(frame.columns) == RAW_COLUMNS

    problems: list[str] = []
    if rows != expectation.rows:
        problems.append(f"expected {expectation.rows} rows, found {rows}")
    if columns != expectation.columns:
        problems.append(f"expected {expectation.columns} columns, found {columns}")
    if not columns_match:
        problems.append("column names do not match the expected raw schema")

    expected_numeric_columns = [c for c in RAW_COLUMNS if c not in STRING_COLUMNS]
    present_numeric_columns = [c for c in expected_numeric_columns if c in frame.columns]
    non_numeric = [
        c for c in present_numeric_columns if not pd.api.types.is_numeric_dtype(frame[c])
    ]
    numeric_dtypes_ok = not non_numeric
    if non_numeric:
        problems.append(f"non-numeric dtype in expected-numeric column(s): {non_numeric}")

    null_cells = int(frame.isna().sum().sum())
    if null_cells:
        problems.append(f"{null_cells} null cell(s) found")

    return PartitionCheck(
        filename=expectation.filename,
        exists=True,
        rows=rows,
        columns=columns,
        expected_rows=expectation.rows,
        expected_columns=expectation.columns,
        columns_match=columns_match,
        numeric_dtypes_ok=numeric_dtypes_ok,
        null_cells=null_cells,
        problems=tuple(problems),
    )


def validate_raw_data(raw_dir: Path | None = None) -> ValidationReport:
    """Check presence, shape, column names, numeric dtypes and null count.

    Reads only from `raw_dir` (or `paths.raw_data_dir()` by default); never writes.

    Args:
        raw_dir: Directory to validate. Defaults to `paths.raw_data_dir()`.

    Returns:
        The combined validation result for both raw partitions.
    """
    directory = raw_dir if raw_dir is not None else raw_data_dir()
    checks = tuple(_check_partition(directory, expectation) for expectation in RAW_EXPECTATIONS)
    return ValidationReport(checks=checks)


def download_instructions(raw_dir: Path | None = None) -> str:
    """Manual UNSW-NB15 acquisition instructions, naming the two expected filenames.

    Args:
        raw_dir: Directory the files should be placed under. Defaults to
            `paths.raw_data_dir()`.

    Returns:
        A multi-line instruction string. Never triggers a download itself.
    """
    directory = raw_dir if raw_dir is not None else raw_data_dir()
    return (
        "Manual UNSW-NB15 dataset acquisition required. This package never "
        "downloads the dataset automatically.\n\n"
        "1. Download the official UNSW-NB15 training and testing partitions "
        "(UNSW Canberra Cyber, UNSW-NB15 dataset).\n"
        f"2. Save the training partition as {RAW_TRAIN_FILENAME!r}.\n"
        f"3. Save the testing partition as {RAW_TEST_FILENAME!r}.\n"
        f"4. Place both files under {directory}.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print the report; on failure also print download instructions.

    Never downloads anything.

    Args:
        argv: Unused. Accepted for a stable CLI entry-point signature.

    Returns:
        0 when both partitions validate successfully, 1 otherwise.
    """
    report = validate_raw_data()
    print(report.render())
    if report.ok:
        return 0
    print()
    print(download_instructions())
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
