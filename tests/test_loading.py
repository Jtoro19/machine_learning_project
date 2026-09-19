"""Tests for read-only loading of the raw UNSW-NB15 partitions."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nids import loading
from nids.columns import RAW_COLUMNS
from nids.loading import RAW_TEST_FILENAME, RAW_TRAIN_FILENAME, STRING_COLUMNS


def _write_partition(directory: Path, filename: str, frame: pd.DataFrame) -> Path:
    """Write `frame` as a raw-format CSV under `directory`, for test setup only."""
    path = directory / filename
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def two_row_frame(
    row_factory: Callable[..., dict[str, Any]],
    frame_factory: Callable[[list[Mapping[str, Any]]], pd.DataFrame],
) -> pd.DataFrame:
    """A minimal two-row raw-schema frame, one row carrying `service == "-"`."""
    rows = [
        row_factory(id=1, service="-"),
        row_factory(id=2, service="http"),
    ]
    return frame_factory(rows)


class TestLoadRawPartitions:
    def test_both_partitions_parse_to_identical_dtypes(
        self, tmp_path: Path, two_row_frame: pd.DataFrame
    ) -> None:
        _write_partition(tmp_path, RAW_TRAIN_FILENAME, two_row_frame)
        _write_partition(tmp_path, RAW_TEST_FILENAME, two_row_frame)

        train = loading.load_raw_train(raw_dir=tmp_path)
        test = loading.load_raw_test(raw_dir=tmp_path)

        assert list(train.dtypes) == list(test.dtypes)
        for column in STRING_COLUMNS:
            assert str(train.dtypes[column]) == "str"
            assert str(test.dtypes[column]) == "str"

    def test_dash_survives_as_literal_two_character_string(
        self, tmp_path: Path, two_row_frame: pd.DataFrame
    ) -> None:
        _write_partition(tmp_path, RAW_TRAIN_FILENAME, two_row_frame)

        train = loading.load_raw_train(raw_dir=tmp_path)

        assert (train["service"] == "-").sum() == 1
        assert train.loc[train["id"] == 1, "service"].iloc[0] == "-"

    def test_raw_schema_error_on_missing_column(
        self, tmp_path: Path, two_row_frame: pd.DataFrame
    ) -> None:
        broken = two_row_frame.drop(columns=["state"])
        path = _write_partition(tmp_path, RAW_TRAIN_FILENAME, broken)

        with pytest.raises(loading.RawSchemaError) as exc_info:
            loading.load_raw_train(raw_dir=tmp_path)

        message = str(exc_info.value)
        assert str(path) in message
        assert "state" in message

    def test_raw_schema_error_on_reordered_columns(
        self, tmp_path: Path, two_row_frame: pd.DataFrame
    ) -> None:
        reordered = two_row_frame[[*RAW_COLUMNS[1:], RAW_COLUMNS[0]]]
        path = _write_partition(tmp_path, RAW_TEST_FILENAME, reordered)

        with pytest.raises(loading.RawSchemaError) as exc_info:
            loading.load_raw_test(raw_dir=tmp_path)

        assert str(path) in str(exc_info.value)

    def test_raw_data_present_true_when_both_files_exist(
        self, tmp_path: Path, two_row_frame: pd.DataFrame
    ) -> None:
        _write_partition(tmp_path, RAW_TRAIN_FILENAME, two_row_frame)
        _write_partition(tmp_path, RAW_TEST_FILENAME, two_row_frame)

        assert loading.raw_data_present(raw_dir=tmp_path) is True

    @pytest.mark.parametrize("missing_filename", [RAW_TRAIN_FILENAME, RAW_TEST_FILENAME])
    def test_raw_data_present_false_when_one_file_missing(
        self, tmp_path: Path, two_row_frame: pd.DataFrame, missing_filename: str
    ) -> None:
        for filename in (RAW_TRAIN_FILENAME, RAW_TEST_FILENAME):
            if filename != missing_filename:
                _write_partition(tmp_path, filename, two_row_frame)

        assert loading.raw_data_present(raw_dir=tmp_path) is False

    def test_raw_data_present_false_on_empty_directory(self, tmp_path: Path) -> None:
        assert loading.raw_data_present(raw_dir=tmp_path) is False

    def test_loading_never_modifies_source_files(
        self, tmp_path: Path, two_row_frame: pd.DataFrame
    ) -> None:
        train_path = _write_partition(tmp_path, RAW_TRAIN_FILENAME, two_row_frame)
        test_path = _write_partition(tmp_path, RAW_TEST_FILENAME, two_row_frame)
        train_before = train_path.read_bytes()
        test_before = test_path.read_bytes()

        for _ in range(3):
            loading.load_raw_train(raw_dir=tmp_path)
            loading.load_raw_test(raw_dir=tmp_path)

        assert train_path.read_bytes() == train_before
        assert test_path.read_bytes() == test_before
