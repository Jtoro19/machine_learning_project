"""Read-and-clean facade combining raw loading with the ordered cleaning steps.

This is the only module in the package that both reads files and cleans data,
which keeps `cleaning.py` pure (frames in, frames out) and therefore trivially
unit-testable with synthetic fixtures (see design Decision 11, deviation D3).
"""

from pathlib import Path

from nids.cleaning import CleaningResult, clean_partitions
from nids.columns import LeakageKey
from nids.loading import load_raw_test, load_raw_train


def load_clean_partitions(
    *,
    comparison_key: LeakageKey = "full_row",
    raw_dir: Path | None = None,
) -> CleaningResult:
    """Load both raw partitions and run the ordered cleaning steps.

    Args:
        comparison_key: Leakage comparison strategy forwarded to
            `clean_partitions`.
        raw_dir: Directory to read the raw partitions from. Defaults to
            `paths.raw_data_dir()` via the individual loaders.

    Returns:
        The `CleaningResult` produced by running `clean_partitions` over the
        freshly loaded training and testing partitions.

    Raises:
        RawSchemaError: A raw file's columns do not match `RAW_COLUMNS`.
        FileNotFoundError: A raw file does not exist under `raw_dir`.
    """
    train_raw = load_raw_train(raw_dir=raw_dir)
    test_raw = load_raw_test(raw_dir=raw_dir)
    return clean_partitions(train_raw, test_raw, comparison_key=comparison_key)
