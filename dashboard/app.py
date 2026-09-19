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
    Artifact,
    Section,
    SectionStatus,
    disclosure_notes,
    discover_sections,
    find_model_comparison,
    format_metric_value,
    load_table,
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


def _render_metrics(section: Section) -> None:
    """Render one `st.metric` per declared metric, in manifest order."""
    if not section.metrics:
        return
    st.subheader("Metrics")
    columns = st.columns(len(section.metrics))
    for column, metric in zip(columns, section.metrics, strict=True):
        with column:
            st.metric(label=metric.name, value=format_metric_value(metric.value))
            if metric.description:
                st.caption(metric.description)


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


def render_overview(sections: tuple[Section, ...]) -> None:
    """Render the Overview page: mandatory disclosures, section listing, results root."""
    st.title("UNSW-NB15 — Results Dashboard")

    for section, note in disclosure_notes(sections):
        st.warning(f"[{section.notebook_id}] {note.text}")

    if not sections:
        st.info("No results yet. Run a notebook, then reload this page.")
        st.caption(f"Results root: {paths.results_root()}")
        return

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
