"""Streamlit UI for the read-only results dashboard.

Run with `uv run streamlit run dashboard/app.py`. Module level holds only imports,
constants and the `sys.path` bootstrap below, so importing this module (as the test
suite does) renders nothing and starts no server. Every Streamlit call lives inside
`main()` or one of the render helpers it calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from dashboard.loading import (
    METRICS_PER_ROW,
    Artifact,
    HeadlineMetric,
    Metric,
    Section,
    SectionStatus,
    disclosure_notes,
    discover_sections,
    find_model_comparison,
    format_metric_display,
    headline_metrics,
    headline_rows,
    is_numeric_metric,
    load_table,
    metric_label,
    metric_rows,
    metrics_table,
    section_label,
)
from nids import paths

_STATUS_MESSAGES: dict[SectionStatus, str] = {
    SectionStatus.MISSING_MANIFEST: "This notebook has not run yet.",
    SectionStatus.UNREADABLE: "The manifest for this section could not be read.",
    SectionStatus.MALFORMED: "The manifest for this section is malformed.",
    SectionStatus.UNSUPPORTED_SCHEMA: (
        "This section uses an unsupported schema version; showing what could be parsed."
    ),
}


def _render_status_banner(section: Section) -> None:
    """Render the top-of-page banner for a section whose status is not OK."""
    message = _STATUS_MESSAGES.get(section.status)
    if message is None:
        return
    detail = f"{message} {section.problem}" if section.problem else message
    if section.status is SectionStatus.MISSING_MANIFEST:
        st.info(detail)
    elif section.status is SectionStatus.UNSUPPORTED_SCHEMA:
        st.warning(detail)
    else:
        st.error(detail)


def _render_metric_card(
    metric: Metric,
    *,
    label: str | None = None,
    help_text: str | None = None,
) -> None:
    """Render one metric inside its column: an `st.metric` card, or a text badge.

    Numeric metrics get a real `st.metric` card. Non-numeric ones (the leakage
    comparison key `"full_row"`, a winning model name, a `bool` flag) get a short
    badge line instead, because `st.metric` is built for measurements and renders
    a bare identifier as a cramped headline. Either way the human-readable label
    is what is shown and the technical manifest name lives in the help tooltip.

    Args:
        metric: The metric to draw.
        label: Overrides the label derived from `metric.name`. The Overview
            headline uses this to say "Best macro F1" rather than
            "Best macro F1 with TTL".
        help_text: Overrides the tooltip. The Overview headline uses it to name
            the producing manifest too, since a headline card is read far from
            the section it came from.
    """
    shown_label = label if label is not None else metric_label(metric.name)
    shown_help = help_text if help_text is not None else metric.name
    value = format_metric_display(metric)
    if is_numeric_metric(metric):
        st.metric(label=shown_label, value=value, help=shown_help, border=True)
        return
    st.markdown(f"**{shown_label}**  \n`{value}`", help=shown_help)


def _render_metric_row(row: tuple[Metric, ...]) -> None:
    """Render one grid row of metric cards, then their descriptions at full width.

    Descriptions are deliberately emitted *after* the `st.columns` block closes,
    so a 150-character sentence gets the whole page width instead of a quarter of
    it. `st.columns(len(row))` is always at most `METRICS_PER_ROW` wide because
    `metric_rows` chunked the metrics before we got here.
    """
    columns = st.columns(len(row))
    for column, metric in zip(columns, row, strict=True):
        with column:
            _render_metric_card(metric)
    for metric in row:
        if metric.description:
            st.caption(f"**{metric_label(metric.name)}** — {metric.description}")


def _render_metrics(section: Section) -> None:
    """Render every declared metric as a wrapping grid plus an "All metrics" table.

    The grid never puts more than `METRICS_PER_ROW` cards in one `st.columns`
    row, so no value is ever truncated to `"67…"`. The expander below it repeats
    every metric with its raw name and full description, so nothing the manifest
    declares can be hidden by the layout.
    """
    if not section.metrics:
        return
    st.subheader("Metrics")
    for row in metric_rows(section.metrics, METRICS_PER_ROW):
        _render_metric_row(row)
    with st.expander("All metrics"):
        st.dataframe(metrics_table(section.metrics))


def _render_notes(section: Section) -> None:
    """Render every note as an `st.warning`, in manifest order."""
    if not section.notes:
        return
    st.subheader("Notes")
    for note in section.notes:
        st.warning(note.text)


def _render_artifact_table(table: Artifact) -> None:
    """Render one declared table: title, description, and its data or a warning/error."""
    st.markdown(f"**{table.title or table.path}**")
    if table.description:
        st.caption(table.description)
    if not table.exists:
        st.warning(f"Declared file is missing: `{table.path}`")
        return
    result = load_table(table)
    if result.frame is None:
        st.error(result.problem or "Could not load this table.")
        return
    st.dataframe(result.frame)
    if result.truncated:
        st.caption("showing the first 5000 rows")


def _render_tables(section: Section) -> None:
    """Render every declared table, in manifest order."""
    if not section.tables:
        return
    st.subheader("Tables")
    for table in section.tables:
        _render_artifact_table(table)


def _render_figures(section: Section) -> None:
    """Render every declared figure, in manifest order."""
    if not section.figures:
        return
    st.subheader("Figures")
    for figure in section.figures:
        if figure.exists:
            st.image(str(figure.resolved), caption=figure.description or figure.title)
        else:
            st.warning(f"Declared file is missing: `{figure.path}`")


def render_section(section: Section) -> None:
    """Render one discovered section's page: banner, metrics, notes, tables, figures."""
    st.title(section_label(section))
    if section.problem and section.status is SectionStatus.OK:
        st.caption(section.problem)
    _render_status_banner(section)
    _render_metrics(section)
    _render_notes(section)
    _render_tables(section)
    _render_figures(section)


def _render_headline_row(row: tuple[HeadlineMetric, ...]) -> None:
    """Render one grid row of Overview headline cards, then their source captions.

    Reuses `_render_metric_card` so a headline card looks and behaves exactly like
    a section card, but overrides the label with the slot's curated wording and
    the tooltip with `<notebook_id> / <metric_name>`. On Overview the provenance
    matters: the reader is far from the section the number came from.
    """
    columns = st.columns(len(row))
    for column, headline in zip(columns, row, strict=True):
        with column:
            _render_metric_card(
                headline.metric,
                label=headline.label,
                help_text=headline.source,
            )
    for headline in row:
        if headline.metric.description:
            st.caption(f"**{headline.label}** — {headline.metric.description}")


def _render_headline(sections: tuple[Section, ...]) -> None:
    """Render the Overview headline grid, reading every value from the manifests.

    Renders nothing when no slot resolves, which is what happens before any
    notebook has run. The grid obeys `METRICS_PER_ROW` exactly like the section
    pages, so the five slots wrap as 4 + 1 rather than being squeezed into one row.
    """
    headlines = headline_metrics(sections)
    if not headlines:
        return
    st.subheader("Headline")
    for row in headline_rows(headlines, METRICS_PER_ROW):
        _render_headline_row(row)


def render_overview(sections: tuple[Section, ...]) -> None:
    """Render the Overview page: disclosures, headline grid, section listing, results root.

    The mandatory disclosures stay above the headline on purpose. The headline is
    the flattering number; the disclosures are the reasons it must not be read as
    a benchmark result. Demoting the warnings beneath the score would invert that.
    """
    st.title("UNSW-NB15 — Results Dashboard")

    for section, note in disclosure_notes(sections):
        st.warning(f"[{section.notebook_id}] {note.text}")

    if not sections:
        st.info("No results yet. Run a notebook, then reload this page.")
        st.caption(f"Results root: {paths.results_root()}")
        return

    _render_headline(sections)

    st.subheader("Sections")
    rows = [
        {
            "section": section.notebook_id,
            "status": section.status.value,
            "generated_at": section.generated_at or "",
            "tables": len(section.tables),
            "figures": len(section.figures),
            "metrics": len(section.metrics),
            "notes": len(section.notes),
        }
        for section in sections
    ]
    st.dataframe(pd.DataFrame(rows))
    st.caption(f"Results root: {paths.results_root()}")


def render_model_comparison(sections: tuple[Section, ...]) -> None:
    """Render the dedicated page for notebook 3's model comparison table."""
    found = find_model_comparison(sections)
    if found is None:
        st.info("Notebook 3 has not produced a model comparison table yet.")
        return

    section, table = found
    st.title("Model comparison")
    st.caption(f"Producer: {section.notebook_id}")
    if table.title:
        st.subheader(table.title)
    if table.description:
        st.caption(table.description)

    result = load_table(table)
    if result.frame is None:
        st.error(result.problem or "Could not load this table.")
        return
    st.dataframe(result.frame)
    if result.truncated:
        st.caption("showing the first 5000 rows")


def main() -> None:
    """Streamlit entry point: build the sidebar navigation and dispatch to a page."""
    st.set_page_config(page_title="UNSW-NB15 Results Dashboard", layout="wide")

    sections = discover_sections()
    labels = [section_label(section) for section in sections]
    pages = ["Overview", *labels, "Model comparison"]
    page = st.sidebar.radio("Page", pages)

    if page == "Overview":
        render_overview(sections)
    elif page == "Model comparison":
        render_model_comparison(sections)
    else:
        label_to_section = dict(zip(labels, sections, strict=True))
        render_section(label_to_section[page])


if __name__ == "__main__":
    main()
