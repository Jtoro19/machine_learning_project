"""Read-only loading of the raw UNSW-NB15 partitions.

Both partitions MUST go through :func:`_read_partition`, the single parse
implementation, so their dtypes are identical. Identical dtypes are the
precondition for the `MultiIndex` row-comparison used by the cleaning
capability's leakage-detection step (see design Decision 4). Nothing in this
module writes to, renames, or otherwise modifies any file under `data/raw/`.
"""

from pathlib import Path
from typing import Final

import pandas as pd

from nids.columns import RAW_COLUMNS
from nids.paths import raw_data_dir

RAW_TRAIN_FILENAME: Final[str] = "UNSW_NB15_training-set.csv"
"""Expected filename of the training partition under `data/raw/`."""

RAW_TEST_FILENAME: Final[str] = "UNSW_NB15_testing-set.csv"
"""Expected filename of the testing partition under `data/raw/`."""

STRING_DTYPE: Final[str] = "str"
"""Explicit pandas dtype applied to every string-typed raw column. Never inspect or
compare `.dtype` elsewhere in this package; compare values with Python `str`
operators instead (see design Decision 10)."""

STRING_COLUMNS: Final[tuple[str, ...]] = ("proto", "service", "state", "attack_cat")
"""Raw columns parsed with STRING_DTYPE rather than left to pandas' inference."""


class RawSchemaError(ValueError):
    """Raised when a raw CSV's column list does not match RAW_COLUMNS exactly."""


def raw_data_present(raw_dir: Path | None = None) -> bool:
    """Report whether both raw partition files exist on disk.

    Used by `pytest.mark.skipif` to guard tests that require the real dataset.

    Args:
        raw_dir: Directory to look in. Defaults to `paths.raw_data_dir()`.

    Returns:
        True when both `RAW_TRAIN_FILENAME` and `RAW_TEST_FILENAME` exist under
        `raw_dir`.
    """
    directory = raw_dir if raw_dir is not None else raw_data_dir()
    return (directory / RAW_TRAIN_FILENAME).is_file() and (
        directory / RAW_TEST_FILENAME
    ).is_file()


def load_raw_train(raw_dir: Path | None = None) -> pd.DataFrame:
    """Read the training partition read-only, with the declared string dtypes.

    Args:
        raw_dir: Directory to read from. Defaults to `paths.raw_data_dir()`.

    Returns:
        The training partition as a data frame with `RAW_COLUMNS` columns.

    Raises:
        RawSchemaError: The file's columns do not match `RAW_COLUMNS` exactly.
        FileNotFoundError: The file does not exist under `raw_dir`.
    """
    directory = raw_dir if raw_dir is not None else raw_data_dir()
    return _read_partition(directory / RAW_TRAIN_FILENAME)


def load_raw_test(raw_dir: Path | None = None) -> pd.DataFrame:
    """Read the testing partition read-only, with the declared string dtypes.

    Args:
        raw_dir: Directory to read from. Defaults to `paths.raw_data_dir()`.

    Returns:
        The testing partition as a data frame with `RAW_COLUMNS` columns.

    Raises:
        RawSchemaError: The file's columns do not match `RAW_COLUMNS` exactly.
        FileNotFoundError: The file does not exist under `raw_dir`.
    """
    directory = raw_dir if raw_dir is not None else raw_data_dir()
    return _read_partition(directory / RAW_TEST_FILENAME)


def _read_partition(path: Path) -> pd.DataFrame:
    """Single parse implementation. Both partitions MUST go through this function.

    `"-"` is deliberately not added to `na_values` and `keep_default_na` is left at
    its default, so `service == "-"` arrives as the literal two-character string
    the cleaning capability's normalization step expects.

    Args:
        path: Path to a raw UNSW-NB15 CSV file.

    Returns:
        The parsed data frame with `STRING_COLUMNS` cast to `STRING_DTYPE`.

    Raises:
        RawSchemaError: `path`'s columns do not match `RAW_COLUMNS` exactly, naming
            the file and the missing and unexpected columns.
        FileNotFoundError: `path` does not exist.
    """
    frame = pd.read_csv(path, dtype={column: STRING_DTYPE for column in STRING_COLUMNS})
    if tuple(frame.columns) != RAW_COLUMNS:
        actual = set(frame.columns)
        expected = set(RAW_COLUMNS)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RawSchemaError(
            f"{path}: columns do not match the expected raw schema. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )
    return frame
