"""Smoke tests for the `dashboard` package: discovery, loading and formatting.

Every test drives `dashboard.loading` (and one import-only check of `dashboard.app`)
against a synthetic `results/` tree built in `tmp_path`. No test starts a Streamlit
server or calls `subprocess`.
"""

from __future__ import annotations

import ast
import json
import struct
import zlib
from pathlib import Path

import pytest

from dashboard.loading import (
    Note,
    Section,
    SectionStatus,
    disclosure_notes,
    discover_sections,
    find_model_comparison,
    format_metric_value,
    is_disclosure_note,
    load_section,
    load_table,
)

ERROR_FLOOR_TEXT = (
    "Conflicting feature combinations create an irreducible label-noise error floor."
)
BENCHMARK_TEXT = (
    "These results are not comparable to published UNSW-NB15 benchmarks."
)


def _write_manifest(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(directory: Path, relative_path: str) -> None:
    path = directory / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("model,f1\nrandom_forest,0.91\n", encoding="utf-8")


def _make_png_bytes() -> bytes:
    """Build a real, minimal 1x1 red PNG from raw bytes (no Pillow dependency)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw_scanline = b"\x00" + b"\xff\x00\x00"
    idat = zlib.compress(raw_scanline)
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _write_png(directory: Path, relative_path: str) -> None:
    path = directory / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_make_png_bytes())


@pytest.fixture
def synthetic_results(tmp_path: Path) -> Path:
    """Build a synthetic `results/` tree exercising every discovery outcome."""
    root = tmp_path / "results"

    data_cleaning = root / "data_cleaning"
    _write_csv(data_cleaning, "tables/summary.csv")
    _write_png(data_cleaning, "figures/plot.png")
    _write_manifest(
        data_cleaning,
        {
            "notebook_id": "data_cleaning",
            "schema_version": "1.0",
            "generated_at": "2024-01-01T00:00:00Z",
            "tables": [
                {
                    "path": "tables/summary.csv",
                    "title": "Summary",
                    "type": "csv",
                    "description": "Cleaning summary counts.",
                },
                {
                    "path": "../escape.csv",
                    "title": "Escape",
                    "type": "csv",
                    "description": "Attempts to escape the section folder.",
                },
            ],
            "figures": [
                {
                    "path": "figures/plot.png",
                    "title": "Plot",
                    "type": "png",
                    "description": "A diagnostic plot.",
                }
            ],
            "metrics": [
                {"name": "rows", "value": 1080, "description": "Row count."},
                {"name": "ratio", "value": "inf", "description": "A ratio metric."},
            ],
            "notes": [
                {"id": "error_floor", "text": ERROR_FLOOR_TEXT},
                {"id": "benchmark", "text": BENCHMARK_TEXT},
            ],
        },
    )

    classification = root / "classification"
    _write_csv(classification, "tables/model_comparison.csv")
    _write_manifest(
        classification,
        {
            "notebook_id": "classification",
            "schema_version": "1.0",
            "generated_at": "2024-01-02T00:00:00Z",
            "tables": [
                {
                    "path": "tables/model_comparison.csv",
                    "title": "Model comparison",
                    "type": "csv",
                    "description": "Macro F1 and balanced accuracy per model.",
                }
            ],
            "figures": [],
            "metrics": [],
            "notes": [],
        },
    )

    future_schema = root / "future_schema"
    _write_csv(future_schema, "tables/future.csv")
    _write_manifest(
        future_schema,
        {
            "notebook_id": "future_schema",
            "schema_version": "9.9",
            "generated_at": "2024-01-03T00:00:00Z",
            "tables": [
                {
                    "path": "tables/future.csv",
                    "title": "Future",
                    "type": "csv",
                    "description": "A table from a future schema version.",
                }
            ],
            "figures": [],
            "metrics": [],
            "notes": [],
        },
    )

    broken_json = root / "broken_json"
    broken_json.mkdir(parents=True, exist_ok=True)
    (broken_json / "manifest.json").write_text("{not json", encoding="utf-8")

    missing_file = root / "missing_file"
    _write_manifest(
        missing_file,
        {
            "notebook_id": "missing_file",
            "schema_version": "1.0",
            "generated_at": "2024-01-04T00:00:00Z",
            "tables": [],
            "figures": [
                {
                    "path": "figures/absent.png",
                    "title": "Absent",
                    "type": "png",
                    "description": "Declared but never written.",
                }
            ],
            "metrics": [],
            "notes": [],
        },
    )

    no_manifest = root / "no_manifest"
    no_manifest.mkdir(parents=True, exist_ok=True)

    return root


def _section_by_id(sections: tuple[Section, ...], notebook_id: str) -> Section:
    for section in sections:
        if section.notebook_id == notebook_id:
            return section
    raise AssertionError(f"no section named {notebook_id!r} in {[s.notebook_id for s in sections]}")


class TestDiscoverSections:
    def test_discovers_every_folder(self, synthetic_results: Path) -> None:
        sections = discover_sections(synthetic_results)
        ids = {section.notebook_id for section in sections}
        assert ids == {
            "data_cleaning",
            "classification",
            "future_schema",
            "broken_json",
            "missing_file",
            "no_manifest",
        }

    def test_absent_root_returns_empty_tuple(self, tmp_path: Path) -> None:
        assert discover_sections(tmp_path / "does_not_exist") == ()

    def test_no_argument_uses_results_root(self, tmp_results_root: Path) -> None:
        (tmp_results_root / "data_cleaning").mkdir(parents=True)
        (tmp_results_root / "data_cleaning" / "manifest.json").write_text(
            json.dumps({"notebook_id": "data_cleaning", "schema_version": "1.0"}),
            encoding="utf-8",
        )
        sections = discover_sections()
        assert [s.notebook_id for s in sections] == ["data_cleaning"]

    def test_valid_section_status_ok(self, synthetic_results: Path) -> None:
        section = _section_by_id(discover_sections(synthetic_results), "data_cleaning")
        assert section.status is SectionStatus.OK
        assert len(section.metrics) == 2
        assert len(section.notes) == 2
        assert len(section.figures) == 1

    def test_traversal_entry_is_skipped(self, synthetic_results: Path) -> None:
        section = _section_by_id(discover_sections(synthetic_results), "data_cleaning")
        assert [table.path for table in section.tables] == ["tables/summary.csv"]
        assert section.problem is not None
        assert "unsafe path" in section.problem

    def test_missing_manifest_status(self, synthetic_results: Path) -> None:
        section = _section_by_id(discover_sections(synthetic_results), "no_manifest")
        assert section.status is SectionStatus.MISSING_MANIFEST
        assert section.problem is not None

    def test_malformed_manifest_status(self, synthetic_results: Path) -> None:
        section = _section_by_id(discover_sections(synthetic_results), "broken_json")
        assert section.status is SectionStatus.MALFORMED
        assert section.problem

    def test_unsupported_schema_still_parses(self, synthetic_results: Path) -> None:
        section = _section_by_id(discover_sections(synthetic_results), "future_schema")
        assert section.status is SectionStatus.UNSUPPORTED_SCHEMA
        assert len(section.tables) == 1

    def test_missing_declared_file_marked_not_exists(self, synthetic_results: Path) -> None:
        section = _section_by_id(discover_sections(synthetic_results), "missing_file")
        assert section.status is SectionStatus.OK
        assert section.figures[0].exists is False


class TestLoadSection:
    def test_load_section_directly_on_missing_manifest(self, tmp_path: Path) -> None:
        directory = tmp_path / "empty_producer"
        directory.mkdir()
        section = load_section(directory)
        assert section.status is SectionStatus.MISSING_MANIFEST
        assert section.notebook_id == "empty_producer"


class TestLoadTable:
    def test_load_table_valid_csv(self, synthetic_results: Path) -> None:
        section = _section_by_id(discover_sections(synthetic_results), "data_cleaning")
        result = load_table(section.tables[0])
        assert result.problem is None
        assert result.frame is not None
        assert list(result.frame.columns) == ["model", "f1"]
        assert result.truncated is False

    def test_load_table_absent_file(self, synthetic_results: Path) -> None:
        section = _section_by_id(discover_sections(synthetic_results), "missing_file")
        result = load_table(section.figures[0])
        assert result.frame is None
        assert result.problem is not None

    def test_load_table_unparseable_file(self, tmp_path: Path) -> None:
        from dashboard.loading import Artifact

        bad_csv = tmp_path / "empty.csv"
        bad_csv.write_bytes(b"")
        artifact = Artifact(
            path="empty.csv",
            title="Empty",
            type="csv",
            description="",
            resolved=bad_csv,
            exists=True,
        )
        result = load_table(artifact)
        assert result.frame is None
        assert result.problem is not None


class TestFormatMetricValue:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("inf", "∞"),
            ("-inf", "-∞"),
            (float("inf"), "∞"),
            (float("nan"), "n/a"),
            (True, "true"),
            (1234, "1,234"),
            (1.5, "1.5000"),
            ("srcip_dstip", "srcip_dstip"),
        ],
    )
    def test_format_metric_value(self, value: object, expected: str) -> None:
        assert format_metric_value(value) == expected

    def test_bool_checked_before_int(self) -> None:
        assert format_metric_value(True) != format_metric_value(1)
        assert format_metric_value(False) != format_metric_value(0)


class TestDisclosureNotes:
    def test_is_disclosure_note_matches_keywords(self) -> None:
        assert is_disclosure_note(Note(id="a", text=ERROR_FLOOR_TEXT))
        assert is_disclosure_note(Note(id="b", text=BENCHMARK_TEXT))
        assert not is_disclosure_note(Note(id="c", text="An unrelated observation."))

    def test_disclosure_notes_finds_both_mandatory_notes(self, synthetic_results: Path) -> None:
        sections = discover_sections(synthetic_results)
        found = disclosure_notes(sections)
        texts = {note.text for _section, note in found}
        assert ERROR_FLOOR_TEXT in texts
        assert BENCHMARK_TEXT in texts


class TestFindModelComparison:
    def test_locates_classification_table(self, synthetic_results: Path) -> None:
        sections = discover_sections(synthetic_results)
        found = find_model_comparison(sections)
        assert found is not None
        section, artifact = found
        assert section.notebook_id == "classification"
        assert artifact.path == "tables/model_comparison.csv"

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        root = tmp_path / "results"
        _write_manifest(
            root / "data_cleaning",
            {"notebook_id": "data_cleaning", "schema_version": "1.0"},
        )
        sections = discover_sections(root)
        assert find_model_comparison(sections) is None


class TestAppImport:
    def test_import_dashboard_app_exposes_main(self) -> None:
        import dashboard.app as app_module

        assert callable(app_module.main)


class TestReadOnlyGuard:
    def test_no_write_mode_open_or_raw_data_references(self) -> None:
        dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
        write_modes = {"w", "wb", "a", "ab", "x", "xb", "w+", "a+", "r+"}

        for source_file in sorted(dashboard_dir.glob("*.py")):
            tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "raw_data_dir":
                    raise AssertionError(f"{source_file} references raw_data_dir")
                if isinstance(node, ast.Attribute) and node.attr == "raw_data_dir":
                    raise AssertionError(f"{source_file} references raw_data_dir")
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if "data/raw" in node.value:
                        raise AssertionError(f"{source_file} contains a 'data/raw' literal")
                if isinstance(node, ast.Call):
                    func = node.func
                    is_open_call = (isinstance(func, ast.Name) and func.id == "open") or (
                        isinstance(func, ast.Attribute) and func.attr == "open"
                    )
                    if not is_open_call:
                        continue
                    mode_value: str | None = None
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        mode_value = node.args[1].value
                    for keyword in node.keywords:
                        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                            mode_value = keyword.value.value
                    if isinstance(mode_value, str) and mode_value in write_modes:
                        raise AssertionError(
                            f"{source_file} opens a file in write mode {mode_value!r}"
                        )
