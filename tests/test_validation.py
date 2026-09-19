"""Tests for raw dataset validation and manual-download guidance.

Dataset row/column counts are never hardcoded here (see `test_repo_hygiene.py`'s
literal scan): tests compare against `PartitionCheck.expected_rows` /
`expected_columns`, which are sourced from `validation.RAW_EXPECTATIONS`, rather
than repeating the literal values.
"""

import ast
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nids import loading, validation
from nids.loading import RAW_TEST_FILENAME, RAW_TRAIN_FILENAME


def _write_partition(directory: Path, filename: str, frame: pd.DataFrame) -> Path:
    """Write `frame` as a raw-format CSV under `directory`, for test setup only."""
    path = directory / filename
    frame.to_csv(path, index=False)
    return path


class TestValidateRawDataMissingFiles:
    def test_missing_file_reports_not_ok(self, tmp_path: Path) -> None:
        report = validation.validate_raw_data(raw_dir=tmp_path)

        assert report.ok is False
        for check in report.checks:
            assert check.exists is False
            assert check.ok is False

    def test_main_prints_instructions_and_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(validation, "raw_data_dir", lambda: tmp_path)

        exit_code = validation.main()

        captured = capsys.readouterr()
        assert exit_code == 1
        assert RAW_TRAIN_FILENAME in captured.out
        assert RAW_TEST_FILENAME in captured.out
        assert "download" in captured.out.lower()


class TestValidateRawDataWrongShape:
    def test_shape_mismatch_reported_per_partition(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        row_factory: Callable[..., dict[str, Any]],
        frame_factory: Callable[[list[Mapping[str, Any]]], pd.DataFrame],
    ) -> None:
        frame = frame_factory([row_factory(id=index) for index in range(1, 3)])
        _write_partition(tmp_path, RAW_TRAIN_FILENAME, frame)
        _write_partition(tmp_path, RAW_TEST_FILENAME, frame)
        oversized_expectations = (
            validation.PartitionExpectation(RAW_TRAIN_FILENAME, 5, 45),
            validation.PartitionExpectation(RAW_TEST_FILENAME, 5, 45),
        )
        monkeypatch.setattr(validation, "RAW_EXPECTATIONS", oversized_expectations)

        report = validation.validate_raw_data(raw_dir=tmp_path)

        assert report.ok is False
        for check in report.checks:
            assert check.rows == 2
            assert check.expected_rows == 5
            assert check.ok is False
            assert any("rows" in problem for problem in check.problems)


class TestValidateRawDataWellFormed:
    def test_main_returns_zero_on_synthetic_well_formed_pair(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        row_factory: Callable[..., dict[str, Any]],
        frame_factory: Callable[[list[Mapping[str, Any]]], pd.DataFrame],
    ) -> None:
        frame = frame_factory([row_factory(id=index) for index in range(1, 4)])
        _write_partition(tmp_path, RAW_TRAIN_FILENAME, frame)
        _write_partition(tmp_path, RAW_TEST_FILENAME, frame)
        matching_expectations = (
            validation.PartitionExpectation(RAW_TRAIN_FILENAME, len(frame), 45),
            validation.PartitionExpectation(RAW_TEST_FILENAME, len(frame), 45),
        )
        monkeypatch.setattr(validation, "RAW_EXPECTATIONS", matching_expectations)
        monkeypatch.setattr(validation, "raw_data_dir", lambda: tmp_path)

        exit_code = validation.main()

        assert exit_code == 0


class TestValidationNeverAcquiresData:
    def test_module_source_has_no_network_or_subprocess_imports(self) -> None:
        source = Path(validation.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {"urllib", "requests", "subprocess", "socket", "http", "ftplib"}

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        assert imported.isdisjoint(forbidden)

    def test_failure_never_invokes_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("subprocess must never be invoked by validation")

        monkeypatch.setattr(subprocess, "run", _forbidden)
        monkeypatch.setattr(validation, "raw_data_dir", lambda: tmp_path)

        exit_code = validation.main()

        assert exit_code == 1


class TestRealRawData:
    """Guarded against the actual `data/raw/` (read-only). Skips cleanly without it."""

    @pytest.mark.skipif(not loading.raw_data_present(), reason="data/raw/ absent")
    def test_raw_partitions_match_expected_shape(self) -> None:
        report = validation.validate_raw_data()

        for check in report.checks:
            assert check.exists is True
            assert check.rows == check.expected_rows
            assert check.columns == check.expected_columns

    @pytest.mark.skipif(not loading.raw_data_present(), reason="data/raw/ absent")
    def test_raw_columns_match_allow_list(self) -> None:
        report = validation.validate_raw_data()

        for check in report.checks:
            assert check.columns_match is True
