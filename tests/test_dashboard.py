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
    HEADLINE_SPECS,
    METRICS_PER_ROW,
    HeadlineSpec,
    Metric,
    Note,
    Section,
    SectionStatus,
    disclosure_notes,
    discover_sections,
    find_model_comparison,
    find_metric,
    format_metric_display,
    format_metric_value,
    headline_metrics,
    headline_rows,
    is_disclosure_note,
    is_numeric_metric,
    is_share_metric,
    load_section,
    load_table,
    metric_label,
    metric_rows,
    metrics_table,
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
            (1.5, "1.500"),
            (0.4052645710165401, "0.405"),
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


class _FakeColumn:
    """Stands in for one `st.columns` return value; usable as a context manager."""

    def __enter__(self) -> "_FakeColumn":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _FakeExpander:
    """Stands in for `st.expander`; records nothing beyond being entered."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> "_FakeExpander":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _StreamlitRecorder:
    """A drop-in stand-in for the `st` module that records what a render helper drew.

    Only the calls these tests assert on are implemented; every other Streamlit
    attribute resolves to a no-op through `__getattr__`, so a render helper can
    call `st.image`, `st.title` or anything else without the recorder needing to
    grow. That keeps the recorder honest about the one thing it exists to prove:
    how wide each `st.columns` row is.
    """

    def __init__(self) -> None:
        self.column_counts: list[int] = []
        self.metrics: list[dict[str, object]] = []
        self.markdowns: list[tuple[str, object]] = []
        self.captions: list[str] = []
        self.dataframes: list[object] = []
        self.expanders: list[str] = []
        self.warnings: list[str] = []
        self.order: list[str] = []

    def columns(self, spec: object, **_kwargs: object) -> list[_FakeColumn]:
        count = spec if isinstance(spec, int) else len(spec)  # type: ignore[arg-type]
        self.column_counts.append(count)
        self.order.append("columns")
        return [_FakeColumn() for _ in range(count)]

    def warning(self, body: str, **_kwargs: object) -> None:
        self.warnings.append(body)
        self.order.append("warning")

    def metric(
        self,
        label: str = "",
        value: object = "",
        **kwargs: object,
    ) -> None:
        self.metrics.append({"label": label, "value": value, **kwargs})

    def markdown(self, body: str, **kwargs: object) -> None:
        self.markdowns.append((body, kwargs.get("help")))

    def caption(self, body: str, **_kwargs: object) -> None:
        self.captions.append(body)

    def dataframe(self, data: object, **_kwargs: object) -> None:
        self.dataframes.append(data)
        self.order.append("dataframe")

    def expander(self, label: str, **_kwargs: object) -> _FakeExpander:
        self.expanders.append(label)
        return _FakeExpander(label)

    def __getattr__(self, _name: str):
        def _noop(*_args: object, **_kwargs: object) -> None:
            return None

        return _noop


def _metric_section(metrics: tuple[Metric, ...], notebook_id: str = "data_cleaning") -> Section:
    """Build a metrics-only `Section` for exercising the render helpers."""
    return Section(
        notebook_id=notebook_id,
        directory=Path("/nonexistent") / notebook_id,
        status=SectionStatus.OK,
        problem=None,
        schema_version="1.0",
        generated_at="2024-01-01T00:00:00Z",
        tables=(),
        figures=(),
        metrics=metrics,
        notes=(),
    )


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _StreamlitRecorder:
    """Swap `dashboard.app.st` for a recorder for the duration of one test."""
    import dashboard.app as app_module

    stub = _StreamlitRecorder()
    monkeypatch.setattr(app_module, "st", stub)
    return stub


class TestMetricLabel:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("train_duplicate_rows_dropped", "Train duplicate rows dropped"),
            ("test_leakage_comparison_key", "Test leakage comparison key"),
            ("best_macro_f1_with_ttl", "Best macro F1 with TTL"),
            ("pca_components_90_with_ttl", "PCA components 90 with TTL"),
            ("rows", "Rows"),
        ],
    )
    def test_humanises_metric_names(self, name: str, expected: str) -> None:
        assert metric_label(name) == expected

    def test_unsplittable_name_returned_unchanged(self) -> None:
        assert metric_label("_") == "_"


class TestMetricRows:
    def test_chunks_into_rows_of_at_most_four(self) -> None:
        metrics = tuple(Metric(name=f"m{i}", value=i, description="") for i in range(13))
        rows = metric_rows(metrics)
        assert [len(row) for row in rows] == [4, 4, 4, 1]

    def test_rows_preserve_manifest_order(self) -> None:
        metrics = tuple(Metric(name=f"m{i}", value=i, description="") for i in range(7))
        flattened = [metric for row in metric_rows(metrics) for metric in row]
        assert flattened == list(metrics)

    def test_empty_metrics_produce_no_rows(self) -> None:
        assert metric_rows(()) == ()

    def test_per_row_below_one_is_clamped(self) -> None:
        metrics = tuple(Metric(name=f"m{i}", value=i, description="") for i in range(3))
        assert [len(row) for row in metric_rows(metrics, 0)] == [1, 1, 1]

    def test_metrics_per_row_constant_is_four(self) -> None:
        assert METRICS_PER_ROW == 4


class TestMetricValueClassification:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1, True), (1.5, True), (True, False), ("full_row", False), ("inf", False)],
    )
    def test_is_numeric_metric(self, value: object, expected: bool) -> None:
        assert is_numeric_metric(Metric(name="m", value=value, description="")) is expected

    def test_share_detected_from_name(self) -> None:
        metric = Metric(name="dbscan_noise_fraction_with_ttl", value=0.0116, description="")
        assert is_share_metric(metric)
        assert format_metric_display(metric) == "1.16%"

    def test_share_detected_from_description(self) -> None:
        metric = Metric(
            name="test_contradictory_share_of_retained",
            value=0.05499205983300036,
            description="Retained rows as a share of the retained testing set.",
        )
        assert format_metric_display(metric) == "5.50%"

    def test_bare_ratio_above_one_is_not_a_percentage(self) -> None:
        metric = Metric(
            name="cost_ratio_fn_to_fp",
            value=20.0,
            description="The stated FN:FP cost ratio.",
        )
        assert not is_share_metric(metric)
        assert format_metric_display(metric) == "20"

    def test_already_percentage_value_is_not_multiplied_again(self) -> None:
        metric = Metric(
            name="expected_cost_reduction_with_ttl",
            value=82.10416153656735,
            description="Percentage reduction in expected cost.",
        )
        assert not is_share_metric(metric)
        assert format_metric_display(metric) == "82.104"

    def test_integer_gets_thousands_separator(self) -> None:
        metric = Metric(name="train_duplicate_rows_dropped", value=1_234, description="")
        assert format_metric_display(metric) == "1,234"

    def test_non_numeric_value_passes_through(self) -> None:
        metric = Metric(name="test_leakage_comparison_key", value="full_row", description="")
        assert format_metric_display(metric) == "full_row"


class TestMetricsTable:
    def test_lists_every_metric_with_name_value_and_description(self) -> None:
        metrics = (
            Metric(name="rows", value=1080, description="Row count."),
            Metric(name="key", value="full_row", description="Comparison key."),
        )
        frame = metrics_table(metrics)
        assert list(frame.columns) == ["name", "value", "description"]
        assert list(frame["name"]) == ["rows", "key"]
        assert list(frame["value"]) == ["1,080", "full_row"]
        assert list(frame["description"]) == ["Row count.", "Comparison key."]

    def test_empty_metrics_still_have_columns(self) -> None:
        assert list(metrics_table(()).columns) == ["name", "value", "description"]


class TestRenderMetricsGrid:
    """The regression guard: no section may ever render a row wider than four cards."""

    def test_no_row_exceeds_four_columns(self, recorder: _StreamlitRecorder) -> None:
        import dashboard.app as app_module

        metrics = tuple(
            Metric(name=f"metric_number_{i}", value=i * 1000, description=f"Metric {i}.")
            for i in range(13)
        )
        app_module.render_section(_metric_section(metrics))

        assert recorder.column_counts, "the metric grid rendered no columns at all"
        assert max(recorder.column_counts) <= METRICS_PER_ROW
        assert sum(recorder.column_counts) == len(metrics)

    def test_real_manifest_sections_never_exceed_four_columns(
        self, recorder: _StreamlitRecorder, synthetic_results: Path
    ) -> None:
        import dashboard.app as app_module

        for section in discover_sections(synthetic_results):
            recorder.column_counts.clear()
            app_module.render_section(section)
            assert all(count <= METRICS_PER_ROW for count in recorder.column_counts), (
                f"section {section.notebook_id} rendered a row wider than {METRICS_PER_ROW}"
            )

    def test_single_metric_renders_a_single_column(self, recorder: _StreamlitRecorder) -> None:
        import dashboard.app as app_module

        section = _metric_section((Metric(name="rows", value=1080, description=""),))
        app_module.render_section(section)
        assert recorder.column_counts == [1]

    def test_sectionless_metrics_render_no_columns(self, recorder: _StreamlitRecorder) -> None:
        import dashboard.app as app_module

        app_module.render_section(_metric_section(()))
        assert recorder.column_counts == []

    def test_numeric_metric_uses_human_label_with_name_in_help(
        self, recorder: _StreamlitRecorder
    ) -> None:
        import dashboard.app as app_module

        section = _metric_section(
            (Metric(name="train_duplicate_rows_dropped", value=1_234, description=""),)
        )
        app_module.render_section(section)

        assert len(recorder.metrics) == 1
        card = recorder.metrics[0]
        assert card["label"] == "Train duplicate rows dropped"
        assert card["value"] == "1,234"
        assert card["help"] == "train_duplicate_rows_dropped"

    def test_non_numeric_metric_is_a_badge_not_a_metric_card(
        self, recorder: _StreamlitRecorder
    ) -> None:
        import dashboard.app as app_module

        section = _metric_section(
            (Metric(name="test_leakage_comparison_key", value="full_row", description=""),)
        )
        app_module.render_section(section)

        assert recorder.metrics == []
        bodies = [body for body, _help in recorder.markdowns]
        assert any("full_row" in body and "Test leakage comparison key" in body for body in bodies)
        assert any(help_text == "test_leakage_comparison_key" for _body, help_text in recorder.markdowns)

    def test_descriptions_are_captioned_outside_the_columns(
        self, recorder: _StreamlitRecorder
    ) -> None:
        import dashboard.app as app_module

        description = "Exact-duplicate rows dropped from the training partition (step 3)."
        section = _metric_section(
            (Metric(name="train_duplicate_rows_dropped", value=1_234, description=description),)
        )
        app_module.render_section(section)

        assert any(description in caption for caption in recorder.captions)
        assert any("Train duplicate rows dropped" in caption for caption in recorder.captions)

    def test_all_metrics_expander_holds_every_metric(
        self, recorder: _StreamlitRecorder
    ) -> None:
        import dashboard.app as app_module

        metrics = tuple(
            Metric(name=f"metric_number_{i}", value=i, description=f"Metric {i}.")
            for i in range(9)
        )
        app_module.render_section(_metric_section(metrics))

        assert "All metrics" in recorder.expanders
        assert recorder.dataframes, "the All metrics expander rendered no table"
        frame = recorder.dataframes[-1]
        assert list(frame.columns) == ["name", "value", "description"]
        assert len(frame) == len(metrics)


class TestFindMetric:
    def test_finds_declared_metric(self, synthetic_results: Path) -> None:
        sections = discover_sections(synthetic_results)
        metric = find_metric(sections, "data_cleaning", "rows")
        assert metric is not None
        assert metric.value == 1080

    def test_returns_none_for_unknown_metric(self, synthetic_results: Path) -> None:
        sections = discover_sections(synthetic_results)
        assert find_metric(sections, "data_cleaning", "not_a_metric") is None

    def test_returns_none_for_unknown_section(self, synthetic_results: Path) -> None:
        sections = discover_sections(synthetic_results)
        assert find_metric(sections, "no_such_notebook", "rows") is None

    def test_returns_none_for_unparseable_section(self, synthetic_results: Path) -> None:
        sections = discover_sections(synthetic_results)
        assert find_metric(sections, "broken_json", "rows") is None


class TestHeadlineSpecs:
    def test_every_spec_declares_at_least_one_source(self) -> None:
        assert HEADLINE_SPECS
        for spec in HEADLINE_SPECS:
            assert spec.label
            assert spec.sources, f"{spec.label} declares no source"
            for notebook_id, name in spec.sources:
                assert notebook_id and name

    def test_specs_hold_no_hardcoded_values(self) -> None:
        """A headline slot may name a manifest and a metric — never a number."""
        for spec in HEADLINE_SPECS:
            for notebook_id, name in spec.sources:
                assert isinstance(notebook_id, str)
                assert isinstance(name, str)

    def test_labels_are_unique(self) -> None:
        labels = [spec.label for spec in HEADLINE_SPECS]
        assert len(labels) == len(set(labels))


class TestHeadlineMetrics:
    def test_resolves_from_declared_sources(self) -> None:
        section = _metric_section(
            (
                Metric(name="best_model_with_ttl", value="random_forest", description="Winner."),
                Metric(name="best_macro_f1_with_ttl", value=0.5116, description="Its macro F1."),
            ),
            notebook_id="classification",
        )
        specs = (
            HeadlineSpec(label="Best model", sources=(("classification", "best_model_with_ttl"),)),
            HeadlineSpec(
                label="Best macro F1",
                sources=(("classification", "best_macro_f1_with_ttl"),),
            ),
        )
        resolved = headline_metrics((section,), specs)
        assert [h.label for h in resolved] == ["Best model", "Best macro F1"]
        assert [format_metric_display(h.metric) for h in resolved] == ["random_forest", "0.512"]

    def test_falls_back_to_second_source(self) -> None:
        section = _metric_section(
            (Metric(name="n_train_rows", value=1_234, description=""),),
            notebook_id="classification",
        )
        specs = (
            HeadlineSpec(
                label="Cleaned train rows",
                sources=(
                    ("data_cleaning", "train_rows_final"),
                    ("classification", "n_train_rows"),
                ),
            ),
        )
        resolved = headline_metrics((section,), specs)
        assert len(resolved) == 1
        assert resolved[0].notebook_id == "classification"
        assert format_metric_display(resolved[0].metric) == "1,234"

    def test_prefers_the_first_available_source(self) -> None:
        owner = _metric_section(
            (Metric(name="train_rows_final", value=1_111, description=""),),
            notebook_id="data_cleaning",
        )
        fallback = _metric_section(
            (Metric(name="n_train_rows", value=2_222, description=""),),
            notebook_id="classification",
        )
        specs = (
            HeadlineSpec(
                label="Cleaned train rows",
                sources=(
                    ("data_cleaning", "train_rows_final"),
                    ("classification", "n_train_rows"),
                ),
            ),
        )
        resolved = headline_metrics((owner, fallback), specs)
        assert resolved[0].notebook_id == "data_cleaning"
        assert format_metric_display(resolved[0].metric) == "1,111"

    def test_unresolvable_slot_is_dropped(self) -> None:
        specs = (
            HeadlineSpec(label="Missing", sources=(("classification", "absent_metric"),)),
        )
        assert headline_metrics((), specs) == ()

    def test_no_sections_yields_no_headline(self) -> None:
        assert headline_metrics(()) == ()

    def test_source_names_the_producing_manifest(self) -> None:
        section = _metric_section(
            (Metric(name="best_model_with_ttl", value="random_forest", description=""),),
            notebook_id="classification",
        )
        specs = (
            HeadlineSpec(label="Best model", sources=(("classification", "best_model_with_ttl"),)),
        )
        assert headline_metrics((section,), specs)[0].source == (
            "classification / best_model_with_ttl"
        )

    def test_share_source_renders_as_a_percentage(self) -> None:
        section = _metric_section(
            (
                Metric(
                    name="label_noise_error_floor_share",
                    value=0.05499205983300036,
                    description="Error floor as a share of the retained testing set.",
                ),
            ),
            notebook_id="classification",
        )
        specs = (
            HeadlineSpec(
                label="Label-noise error floor",
                sources=(("classification", "label_noise_error_floor_share"),),
            ),
        )
        resolved = headline_metrics((section,), specs)
        assert format_metric_display(resolved[0].metric) == "5.50%"


class TestHeadlineRows:
    def test_chunks_at_four(self) -> None:
        section = _metric_section(
            tuple(Metric(name=f"m{i}", value=i, description="") for i in range(5)),
            notebook_id="classification",
        )
        specs = tuple(
            HeadlineSpec(label=f"Slot {i}", sources=(("classification", f"m{i}"),))
            for i in range(5)
        )
        rows = headline_rows(headline_metrics((section,), specs))
        assert [len(row) for row in rows] == [4, 1]

    def test_empty_headline_has_no_rows(self) -> None:
        assert headline_rows(()) == ()


class TestRenderOverviewHeadline:
    def _overview_sections(self) -> tuple[Section, ...]:
        cleaning = _metric_section(
            (
                Metric(name="train_rows_final", value=1_111, description="Train rows kept."),
                Metric(name="test_rows_final", value=2_222, description="Test rows kept."),
            ),
            notebook_id="data_cleaning",
        )
        classification = _metric_section(
            (
                Metric(name="best_model_with_ttl", value="random_forest", description="Winner."),
                Metric(name="best_macro_f1_with_ttl", value=0.5116, description="Its macro F1."),
                Metric(
                    name="label_noise_error_floor_share",
                    value=0.055,
                    description="Error floor as a share of the retained testing set.",
                ),
            ),
            notebook_id="classification",
        )
        return (cleaning, classification)

    def test_headline_never_exceeds_four_columns(self, recorder: _StreamlitRecorder) -> None:
        import dashboard.app as app_module

        app_module.render_overview(self._overview_sections())
        assert recorder.column_counts, "the headline rendered no columns at all"
        assert max(recorder.column_counts) <= METRICS_PER_ROW
        assert sum(recorder.column_counts) == len(HEADLINE_SPECS)

    def test_headline_values_come_from_the_manifests(
        self, recorder: _StreamlitRecorder
    ) -> None:
        import dashboard.app as app_module

        app_module.render_overview(self._overview_sections())
        cards = {card["label"]: card["value"] for card in recorder.metrics}
        assert cards["Best macro F1"] == "0.512"
        assert cards["Cleaned train rows"] == "1,111"
        assert cards["Cleaned test rows"] == "2,222"
        assert cards["Label-noise error floor"] == "5.50%"

    def test_best_model_is_a_badge_naming_its_manifest(
        self, recorder: _StreamlitRecorder
    ) -> None:
        import dashboard.app as app_module

        app_module.render_overview(self._overview_sections())
        assert "Best model" not in {card["label"] for card in recorder.metrics}
        badges = [(body, help_text) for body, help_text in recorder.markdowns if "`" in body]
        assert any("random_forest" in body and "Best model" in body for body, _ in badges)
        assert any(help_text == "classification / best_model_with_ttl" for _, help_text in badges)

    def test_headline_card_help_names_the_producing_manifest(
        self, recorder: _StreamlitRecorder
    ) -> None:
        import dashboard.app as app_module

        app_module.render_overview(self._overview_sections())
        helps = {card["label"]: card["help"] for card in recorder.metrics}
        assert helps["Cleaned train rows"] == "data_cleaning / train_rows_final"
        assert helps["Best macro F1"] == "classification / best_macro_f1_with_ttl"

    def test_section_table_is_still_rendered_below(
        self, recorder: _StreamlitRecorder
    ) -> None:
        import dashboard.app as app_module

        app_module.render_overview(self._overview_sections())
        assert recorder.dataframes, "the per-section table disappeared"
        frame = recorder.dataframes[-1]
        assert list(frame["section"]) == ["data_cleaning", "classification"]
        assert "metrics" in frame.columns

    def test_disclosures_stay_above_the_headline(
        self, recorder: _StreamlitRecorder
    ) -> None:
        """Mandatory warnings must not be demoted beneath the flattering number."""
        import dashboard.app as app_module

        sections = self._overview_sections()
        disclosed = Section(
            notebook_id="data_cleaning",
            directory=sections[0].directory,
            status=SectionStatus.OK,
            problem=None,
            schema_version="1.0",
            generated_at=None,
            tables=(),
            figures=(),
            metrics=sections[0].metrics,
            notes=(Note(id="error_floor", text=ERROR_FLOOR_TEXT),),
        )
        app_module.render_overview((disclosed, sections[1]))
        assert recorder.order.index("warning") < recorder.order.index("columns")

    def test_headline_absent_when_no_slot_resolves(
        self, recorder: _StreamlitRecorder
    ) -> None:
        import dashboard.app as app_module

        bare = _metric_section((Metric(name="unrelated", value=1, description=""),))
        app_module.render_overview((bare,))
        assert recorder.column_counts == []
        assert recorder.dataframes, "the per-section table should still render"

    def test_empty_results_tree_renders_no_headline(
        self, recorder: _StreamlitRecorder
    ) -> None:
        import dashboard.app as app_module

        app_module.render_overview(())
        assert recorder.column_counts == []

    def test_real_manifests_fill_every_headline_slot(
        self, recorder: _StreamlitRecorder
    ) -> None:
        """The repo's own results/ tree must satisfy all declared headline slots."""
        import dashboard.app as app_module

        app_module.render_overview(discover_sections())
        assert sum(recorder.column_counts) == len(HEADLINE_SPECS)
        assert max(recorder.column_counts) <= METRICS_PER_ROW


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
