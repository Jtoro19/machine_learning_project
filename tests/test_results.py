"""Tests for `src/nids/results.py`'s `ResultsWriter`.

Covers the results-output-contract spec's "manifest round-trips through
JSON parsing with all declared paths present", "A metrics entry tolerates
both numeric and string values", "Consumers discover outputs without a
global index", "File names carry no timestamp", and "Two consecutive runs
produce identical table bytes" scenarios, plus every design-points row from
the "results.py" module section: path-traversal rejection, duplicate-name
rejection, manifest-written-last, deterministic PNG metadata, and
`prune_undeclared`.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from nids.results import ResultsWriter

_FIXED_TIMESTAMP = datetime(2024, 1, 1, tzinfo=UTC)


def _sample_frame() -> pd.DataFrame:
    """A tiny, deterministic table for artifact-writing tests."""
    return pd.DataFrame({"category": ["b", "a"], "count": [2, 1]})


def _sample_figure() -> "plt.Figure":
    """A tiny, deterministic matplotlib figure for artifact-writing tests."""
    fig, axis = plt.subplots()
    axis.bar(["a", "b"], [1, 2])
    return fig


def test_manifest_round_trips_and_every_declared_path_exists(tmp_path: Path) -> None:
    writer = ResultsWriter("sample_notebook", root=tmp_path, generated_at=_FIXED_TIMESTAMP)
    writer.add_table(_sample_frame(), name="counts", title="Counts", description="A table.")
    writer.add_figure(_sample_figure(), name="chart", title="Chart", description="A figure.")
    writer.add_metric("total", 3, "Total count.")
    writer.add_note("note_one", "A sample note.")
    directory = writer.close()

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "1.0"
    assert manifest["notebook_id"] == "sample_notebook"
    assert manifest["generated_at"] == "2024-01-01T00:00:00Z"
    assert len(manifest["tables"]) == 1
    assert len(manifest["figures"]) == 1
    assert len(manifest["metrics"]) == 1
    assert len(manifest["notes"]) == 1
    for entry in (*manifest["tables"], *manifest["figures"]):
        assert (directory / entry["path"]).is_file()


def test_metrics_tolerate_numeric_and_string_values(tmp_path: Path) -> None:
    writer = ResultsWriter("metric_types_check", root=tmp_path, generated_at=_FIXED_TIMESTAMP)
    writer.add_metric("numeric_metric", 42, "A numeric metric.")
    writer.add_metric("string_metric", "full_row", "A string metric.")
    directory = writer.close()

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    values = {entry["name"]: entry["value"] for entry in manifest["metrics"]}
    assert values["numeric_metric"] == 42
    assert values["string_metric"] == "full_row"


def test_no_global_index_required_across_producers(tmp_path: Path) -> None:
    for notebook_id in ("producer_a", "producer_b"):
        writer = ResultsWriter(notebook_id, root=tmp_path, generated_at=_FIXED_TIMESTAMP)
        writer.add_table(_sample_frame(), name="counts", title="Counts", description="A table.")
        writer.close()

    manifests = sorted(path.parent.name for path in tmp_path.glob("*/manifest.json"))
    assert manifests == ["producer_a", "producer_b"]
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "index.json").exists()


def test_file_names_carry_no_timestamp(tmp_path: Path) -> None:
    writer = ResultsWriter("timestamp_check", root=tmp_path, generated_at=_FIXED_TIMESTAMP)
    table_path = writer.add_table(
        _sample_frame(), name="counts", title="Counts", description="A table."
    )
    figure_path = writer.add_figure(
        _sample_figure(), name="chart", title="Chart", description="A figure."
    )
    writer.close()

    assert table_path.name == "counts.csv"
    assert figure_path.name == "chart.png"


def test_duplicate_artifact_name_raises(tmp_path: Path) -> None:
    writer = ResultsWriter("dup_artifact_check", root=tmp_path)
    writer.add_table(_sample_frame(), name="counts", title="Counts", description="A table.")
    with pytest.raises(ValueError, match="already registered"):
        writer.add_table(
            _sample_frame(), name="counts", title="Counts again", description="Different."
        )


def test_duplicate_metric_name_raises(tmp_path: Path) -> None:
    writer = ResultsWriter("dup_metric_check", root=tmp_path)
    writer.add_metric("total", 1, "First.")
    with pytest.raises(ValueError, match="already registered"):
        writer.add_metric("total", 2, "Second.")


def test_duplicate_note_id_raises(tmp_path: Path) -> None:
    writer = ResultsWriter("dup_note_check", root=tmp_path)
    writer.add_note("note_one", "First.")
    with pytest.raises(ValueError, match="already registered"):
        writer.add_note("note_one", "Second.")


@pytest.mark.parametrize(
    "invalid_name",
    ["..", "../escape", "/absolute", "counts.csv", "Bad-Id", "UPPER", ""],
)
def test_invalid_artifact_name_rejected_before_any_filesystem_call(
    tmp_path: Path, invalid_name: str
) -> None:
    writer = ResultsWriter("invalid_name_check", root=tmp_path)
    before = sorted(path.name for path in (writer.directory / "tables").iterdir())

    with pytest.raises(ValueError):
        writer.add_table(
            _sample_frame(), name=invalid_name, title="Bad", description="Bad."
        )

    after = sorted(path.name for path in (writer.directory / "tables").iterdir())
    assert before == after


@pytest.mark.parametrize(
    "invalid_notebook_id",
    ["..", "../escape", "/absolute", "has.ext", "Bad-Id", "UPPER", ""],
)
def test_invalid_notebook_id_rejected_before_any_filesystem_call(
    tmp_path: Path, invalid_notebook_id: str
) -> None:
    with pytest.raises(ValueError):
        ResultsWriter(invalid_notebook_id, root=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_manifest_written_last_not_on_exception(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="simulated mid-run failure"):
        with ResultsWriter(
            "crash_check", root=tmp_path, generated_at=_FIXED_TIMESTAMP
        ) as writer:
            writer.add_table(
                _sample_frame(), name="counts", title="Counts", description="A table."
            )
            writer.add_figure(
                _sample_figure(), name="chart", title="Chart", description="A figure."
            )
            raise RuntimeError("simulated mid-run failure")

    directory = tmp_path / "crash_check"
    assert not (directory / "manifest.json").exists()
    assert (directory / "tables" / "counts.csv").is_file()
    assert (directory / "figures" / "chart.png").is_file()


def test_two_runs_with_fixed_generated_at_are_byte_identical(
    tmp_path: Path, frozen_clock: datetime
) -> None:
    def _run(root: Path) -> Path:
        writer = ResultsWriter("determinism_check", root=root, generated_at=frozen_clock)
        writer.add_table(
            _sample_frame(),
            name="counts",
            title="Counts",
            description="A table.",
            sort_by=["category"],
        )
        writer.add_figure(
            _sample_figure(), name="chart", title="Chart", description="A figure."
        )
        writer.add_metric("total", 3, "Total.")
        writer.add_note("note_one", "A note.")
        return writer.close()

    first_directory = _run(tmp_path / "first")
    second_directory = _run(tmp_path / "second")

    for relative in ("manifest.json", "tables/counts.csv", "figures/chart.png"):
        first_bytes = (first_directory / relative).read_bytes()
        second_bytes = (second_directory / relative).read_bytes()
        assert first_bytes == second_bytes, f"{relative} differs between two runs"


def test_prune_undeclared_removes_only_orphaned_registered_extension_files(
    tmp_path: Path,
) -> None:
    writer = ResultsWriter("prune_check", root=tmp_path, generated_at=_FIXED_TIMESTAMP)
    writer.add_table(
        _sample_frame(), name="kept_table", title="Kept", description="Kept table."
    )
    writer.add_figure(
        _sample_figure(), name="kept_figure", title="Kept", description="Kept figure."
    )

    orphan_table = writer.directory / "tables" / "orphan.csv"
    orphan_table.write_text("orphan\n", encoding="utf-8")
    orphan_figure = writer.directory / "figures" / "orphan.png"
    orphan_figure.write_bytes(b"not a real png")
    non_registered_extension = writer.directory / "tables" / "notes.txt"
    non_registered_extension.write_text("keep me\n", encoding="utf-8")

    writer.close()

    assert not orphan_table.exists()
    assert not orphan_figure.exists()
    assert non_registered_extension.exists()
    assert (writer.directory / "tables" / "kept_table.csv").exists()
    assert (writer.directory / "figures" / "kept_figure.png").exists()


def test_prune_undeclared_reports_each_pruned_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    writer = ResultsWriter("prune_report_check", root=tmp_path, generated_at=_FIXED_TIMESTAMP)
    orphan = writer.directory / "tables" / "orphan.csv"
    orphan.write_text("orphan\n", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        writer.close()

    assert "tables/orphan.csv" in caplog.text


def test_prune_undeclared_false_leaves_orphans(tmp_path: Path) -> None:
    writer = ResultsWriter(
        "no_prune_check", root=tmp_path, generated_at=_FIXED_TIMESTAMP, prune_undeclared=False
    )
    orphan = writer.directory / "tables" / "orphan.csv"
    orphan.write_text("orphan\n", encoding="utf-8")
    writer.close()

    assert orphan.exists()


def test_png_bytes_carry_no_software_or_creation_time_metadata(tmp_path: Path) -> None:
    writer = ResultsWriter("png_metadata_check", root=tmp_path, generated_at=_FIXED_TIMESTAMP)
    figure_path = writer.add_figure(
        _sample_figure(), name="chart", title="Chart", description="A figure."
    )
    writer.close()

    raw = figure_path.read_bytes()
    assert b"Software" not in raw
    assert b"Creation Time" not in raw


def test_write_manifest_raises_if_declared_artifact_missing(tmp_path: Path) -> None:
    writer = ResultsWriter("missing_artifact_check", root=tmp_path)
    table_path = writer.add_table(
        _sample_frame(), name="counts", title="Counts", description="A table."
    )
    table_path.unlink()

    with pytest.raises(FileNotFoundError):
        writer.write_manifest()


def test_write_json_sidecar_is_unregistered_and_survives_close(tmp_path: Path) -> None:
    writer = ResultsWriter("sidecar_check", root=tmp_path, generated_at=_FIXED_TIMESTAMP)
    sidecar_path = writer.write_json({"total": 3}, name="counts")
    directory = writer.close()

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tables"] == []
    assert manifest["figures"] == []
    assert sidecar_path.is_file()
    assert sidecar_path == directory / "counts.json"
    assert json.loads(sidecar_path.read_text(encoding="utf-8")) == {"total": 3}


def test_add_json_table_registers_as_json_type(tmp_path: Path) -> None:
    writer = ResultsWriter("json_table_check", root=tmp_path, generated_at=_FIXED_TIMESTAMP)
    path = writer.add_json_table(
        {"a": 1}, name="payload", title="Payload", description="A JSON table."
    )
    directory = writer.close()

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tables"][0]["type"] == "json"
    assert manifest["tables"][0]["path"] == "tables/payload.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_generated_at_resolves_from_source_date_epoch_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    writer = ResultsWriter("epoch_check", root=tmp_path)
    directory = writer.close()

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated_at"] == "1970-01-01T00:00:00Z"


def test_registration_order_preserved_across_artifact_kinds(tmp_path: Path) -> None:
    writer = ResultsWriter("order_check", root=tmp_path)
    writer.add_table(_sample_frame(), name="second_table", title="Second", description="B.")
    writer.add_table(_sample_frame(), name="first_table", title="First", description="A.")
    writer.add_metric("z_metric", 1, "Z.")
    writer.add_metric("a_metric", 2, "A.")

    assert [entry.title for entry in writer.tables] == ["Second", "First"]
    assert [entry.name for entry in writer.metrics] == ["z_metric", "a_metric"]


def test_add_table_sort_by_applies_stable_ascending_sort(tmp_path: Path) -> None:
    writer = ResultsWriter("sort_check", root=tmp_path)
    frame = pd.DataFrame({"category": ["b", "a", "a"], "count": [2, 3, 1]})
    path = writer.add_table(
        frame,
        name="sorted_table",
        title="Sorted",
        description="Sorted table.",
        sort_by=["category", "count"],
    )

    written = pd.read_csv(path)
    assert written["category"].tolist() == ["a", "a", "b"]
    assert written["count"].tolist() == [1, 3, 2]


def test_context_manager_returns_self_and_writes_manifest_on_clean_exit(
    tmp_path: Path,
) -> None:
    with ResultsWriter("ctx_check", root=tmp_path, generated_at=_FIXED_TIMESTAMP) as writer:
        assert isinstance(writer, ResultsWriter)
        writer.add_metric("total", 1, "Total.")

    assert (writer.directory / "manifest.json").is_file()
