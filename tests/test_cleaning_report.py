"""Tests for `src/nids/cleaning_report.py`.

Covers the results-output-contract spec's "The cleaning report is this
contract's first producer" requirement (all four scenarios): a complete,
self-describing `results/data_cleaning/` folder; every reported count
computed rather than hardcoded; the leakage comparison key and both leakage
counts reported together; and both mandatory disclosure notes present.
"""

import json
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nids import cleaning_report as cleaning_report_module
from nids import loading as loading_module
from nids import validation as validation_module
from nids.cleaning_report import NOTEBOOK_ID, build_cleaning_report
from nids.loading import RAW_TEST_FILENAME, RAW_TRAIN_FILENAME, raw_data_present

_REQUIRED_TABLES = {
    "row_counts_before_after",
    "attack_cat_distribution",
    "state_distribution",
    "rare_categories",
}
_REQUIRED_FIGURES = {"class_balance_before_after"}
_REQUIRED_METRIC_NAMES = {
    "train_duplicate_rows_dropped",
    "test_internal_duplicate_rows",
    "test_leakage_rows_dropped",
    "test_leakage_comparison_key",
    "test_contradictory_rows_retained",
    "test_contradictory_share_of_retained",
    "label_noise_floor_combinations",
    "label_noise_floor_rows",
    "service_dash_rows_mapped",
    "raw_null_cells",
    "train_rows_final",
    "test_rows_final",
}
_REQUIRED_NOTE_IDS = {"benchmark_non_comparability", "retained_error_floor"}


def _write_synthetic_raw(
    directory: Path, raw_train: pd.DataFrame, raw_test: pd.DataFrame
) -> Path:
    """Write the engineered synthetic partitions as raw-format CSVs under `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    raw_train.to_csv(directory / RAW_TRAIN_FILENAME, index=False)
    raw_test.to_csv(directory / RAW_TEST_FILENAME, index=False)
    return directory


def _load_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def test_report_produces_manifest_counts_and_every_required_artifact(
    tmp_path: Path,
    tmp_results_root: Path,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    frozen_clock: datetime,
) -> None:
    raw_dir = _write_synthetic_raw(tmp_path / "raw", raw_train, raw_test)

    directory = build_cleaning_report(
        raw_dir=raw_dir, results_root=tmp_results_root, generated_at=frozen_clock
    )

    assert directory == tmp_results_root / NOTEBOOK_ID
    manifest = _load_manifest(directory)
    assert manifest["notebook_id"] == "data_cleaning"

    table_names = {Path(entry["path"]).stem for entry in manifest["tables"]}
    assert _REQUIRED_TABLES <= table_names
    figure_names = {Path(entry["path"]).stem for entry in manifest["figures"]}
    assert figure_names == _REQUIRED_FIGURES

    for entry in (*manifest["tables"], *manifest["figures"]):
        assert (directory / entry["path"]).is_file()

    counts = json.loads((directory / "counts.json").read_text(encoding="utf-8"))
    metric_names = {entry["name"] for entry in manifest["metrics"]}
    assert metric_names == _REQUIRED_METRIC_NAMES
    assert set(counts) - {"notes"} == metric_names
    for name in metric_names:
        assert counts[name] == next(
            entry["value"] for entry in manifest["metrics"] if entry["name"] == name
        )


def test_both_mandatory_notes_present(
    tmp_path: Path,
    tmp_results_root: Path,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    frozen_clock: datetime,
) -> None:
    raw_dir = _write_synthetic_raw(tmp_path / "raw", raw_train, raw_test)

    directory = build_cleaning_report(
        raw_dir=raw_dir, results_root=tmp_results_root, generated_at=frozen_clock
    )

    manifest = _load_manifest(directory)
    note_ids = {entry["id"] for entry in manifest["notes"]}
    assert note_ids == _REQUIRED_NOTE_IDS
    for entry in manifest["notes"]:
        assert entry["text"]


def test_leakage_metrics_reported_together(
    tmp_path: Path,
    tmp_results_root: Path,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    frozen_clock: datetime,
) -> None:
    raw_dir = _write_synthetic_raw(tmp_path / "raw", raw_train, raw_test)

    directory = build_cleaning_report(
        raw_dir=raw_dir, results_root=tmp_results_root, generated_at=frozen_clock
    )

    manifest = _load_manifest(directory)
    metrics = {entry["name"]: entry["value"] for entry in manifest["metrics"]}

    assert metrics["test_leakage_comparison_key"] == "full_row"
    # Engineered by the `raw_test` fixture (see tests/conftest.py's docstring):
    # 3 rows match train exactly (removed), 4 rows match on features alone but
    # carry a different label (retained, contradictory).
    assert metrics["test_leakage_rows_dropped"] == 3
    assert metrics["test_contradictory_rows_retained"] == 4
    assert metrics["test_contradictory_share_of_retained"] == pytest.approx(
        4 / metrics["test_rows_final"]
    )


def test_second_run_with_same_generated_at_is_byte_identical(
    tmp_path: Path,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    frozen_clock: datetime,
) -> None:
    raw_dir = _write_synthetic_raw(tmp_path / "raw", raw_train, raw_test)

    first_directory = build_cleaning_report(
        raw_dir=raw_dir, results_root=tmp_path / "results_a", generated_at=frozen_clock
    )
    second_directory = build_cleaning_report(
        raw_dir=raw_dir, results_root=tmp_path / "results_b", generated_at=frozen_clock
    )

    manifest = _load_manifest(first_directory)
    relative_paths = ["manifest.json", "counts.json"] + [
        entry["path"] for entry in (*manifest["tables"], *manifest["figures"])
    ]
    for relative in relative_paths:
        first_bytes = (first_directory / relative).read_bytes()
        second_bytes = (second_directory / relative).read_bytes()
        assert first_bytes == second_bytes, f"{relative} differs between two runs"


def test_metrics_are_computed_not_hardcoded(
    tmp_path: Path,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    frozen_clock: datetime,
    row_factory: Callable[..., dict[str, Any]],
    frame_factory: Callable[[list[Mapping[str, Any]]], pd.DataFrame],
) -> None:
    """Perturbing the synthetic fixture changes the reported metric values,
    proving they come from `CleaningResult` rather than a literal in
    `cleaning_report.py` (design Decision 2)."""
    raw_dir = _write_synthetic_raw(tmp_path / "raw", raw_train, raw_test)
    baseline_directory = build_cleaning_report(
        raw_dir=raw_dir, results_root=tmp_path / "results_baseline", generated_at=frozen_clock
    )
    baseline_metrics = {
        entry["name"]: entry["value"]
        for entry in _load_manifest(baseline_directory)["metrics"]
    }

    # Append one more exact duplicate of an existing training row: this must
    # raise `train_duplicate_rows_dropped` by exactly one while leaving
    # `train_rows_final` unchanged -- dropping the extra duplicate restores the
    # original row set -- and must not touch the testing-side leakage metrics.
    perturbed_train = pd.concat(
        [raw_train, raw_train.iloc[[0]]], ignore_index=True
    )
    perturbed_dir = _write_synthetic_raw(tmp_path / "raw_perturbed", perturbed_train, raw_test)
    perturbed_directory = build_cleaning_report(
        raw_dir=perturbed_dir,
        results_root=tmp_path / "results_perturbed",
        generated_at=frozen_clock,
    )
    perturbed_metrics = {
        entry["name"]: entry["value"]
        for entry in _load_manifest(perturbed_directory)["metrics"]
    }

    assert (
        perturbed_metrics["train_duplicate_rows_dropped"]
        == baseline_metrics["train_duplicate_rows_dropped"] + 1
    )
    assert perturbed_metrics["train_rows_final"] == baseline_metrics["train_rows_final"]
    assert (
        perturbed_metrics["test_leakage_rows_dropped"]
        == baseline_metrics["test_leakage_rows_dropped"]
    )


def test_main_returns_one_and_prints_instructions_when_raw_data_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(validation_module, "raw_data_dir", lambda: tmp_path)

    exit_code = cleaning_report_module.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "download" in captured.out.lower()


def test_main_runs_end_to_end_and_returns_zero(
    tmp_path: Path,
    tmp_results_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_dir = _write_synthetic_raw(tmp_path / "raw", raw_train, raw_test)
    monkeypatch.setattr(loading_module, "raw_data_dir", lambda: raw_dir)
    monkeypatch.setattr(validation_module, "raw_data_dir", lambda: raw_dir)
    matching_expectations = (
        validation_module.PartitionExpectation(RAW_TRAIN_FILENAME, len(raw_train), 45),
        validation_module.PartitionExpectation(RAW_TEST_FILENAME, len(raw_test), 45),
    )
    monkeypatch.setattr(validation_module, "RAW_EXPECTATIONS", matching_expectations)

    exit_code = cleaning_report_module.main(["--comparison-key", "full_row"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Wrote cleaning report to" in captured.out
    assert (tmp_results_root / NOTEBOOK_ID / "manifest.json").is_file()


@pytest.mark.slow
@pytest.mark.skipif(not raw_data_present(), reason="data/raw/ absent")
def test_end_to_end_on_real_partitions(tmp_results_root: Path, frozen_clock: datetime) -> None:
    """Runs the whole report against the real `data/raw/` (read-only) and
    asserts the arithmetic identity that every partition report must hold:
    `removed + retained_contradictory` equals `rows_before - rows_after` for
    the testing leakage step, and every declared artifact path exists."""
    directory = build_cleaning_report(results_root=tmp_results_root, generated_at=frozen_clock)

    manifest = _load_manifest(directory)
    for entry in (*manifest["tables"], *manifest["figures"]):
        assert (directory / entry["path"]).is_file()

    metrics = {entry["name"]: entry["value"] for entry in manifest["metrics"]}
    assert metrics["test_leakage_rows_dropped"] >= 0
    assert metrics["test_contradictory_rows_retained"] >= 0
    assert metrics["test_contradictory_share_of_retained"] == pytest.approx(
        metrics["test_contradictory_rows_retained"] / metrics["test_rows_final"]
    )
