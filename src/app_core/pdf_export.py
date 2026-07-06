"""PDF export helpers for AI insight reports."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from typing import Mapping
from xml.sax.saxutils import escape

import pandas as pd
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


@dataclass(frozen=True)
class InsightPdfSection:
    """One report section containing insight text and optional structured artifacts."""

    title: str
    insight_text: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    metrics: Mapping[str, object] = field(default_factory=dict)
    chart_png: bytes | None = None


@dataclass(frozen=True)
class InsightPdfDocument:
    """Top-level report container for one PDF output."""

    title: str
    generated_at: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    sections: list[InsightPdfSection] = field(default_factory=list)


def can_export_insight(insight_text: str) -> bool:
    """Return true when insight text is non-empty."""
    return bool(insight_text.strip())


def truncate_table_rows(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, bool]:
    """Clamp table rows for PDF rendering and return truncation flag."""
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1.")
    is_truncated = len(df) > max_rows
    return df.head(max_rows).copy(), is_truncated


def _kv_table_rows(mapping: Mapping[str, object]) -> list[list[str]]:
    rows: list[list[str]] = []
    for key, value in mapping.items():
        rows.append([str(key), str(value)])
    return rows


def _render_df_table(df: pd.DataFrame, max_rows: int) -> tuple[Table, bool]:
    trimmed_df, is_truncated = truncate_table_rows(df, max_rows=max_rows)
    rows = [list(trimmed_df.columns)] + trimmed_df.astype(str).values.tolist()
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table, is_truncated


def build_pdf_bytes(
    document: InsightPdfDocument,
    table_max_rows: int = 100,
    font_size: int = 10,
) -> bytes:
    """Render one full insight report to PDF bytes."""
    if font_size < 8:
        raise ValueError("font_size must be at least 8.")

    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=document.title,
    )

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "BodyTextDense",
        parent=styles["BodyText"],
        fontSize=font_size,
        leading=font_size + 3,
    )
    story: list[object] = [
        Paragraph(escape(document.title), styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"Generated at: {escape(document.generated_at)}", body_style),
        Spacer(1, 8),
    ]

    if document.metadata:
        story.append(Paragraph("Report Metadata", styles["Heading3"]))
        metadata_table = Table(_kv_table_rows(document.metadata), colWidths=[2.0 * inch, 4.8 * inch])
        metadata_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.extend([metadata_table, Spacer(1, 10)])

    for section in document.sections:
        story.append(Paragraph(escape(section.title), styles["Heading2"]))
        if section.metadata:
            section_meta_table = Table(
                _kv_table_rows(section.metadata),
                colWidths=[2.0 * inch, 4.8 * inch],
            )
            section_meta_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.extend([section_meta_table, Spacer(1, 8)])

        escaped_insight = escape(section.insight_text).replace("\n", "<br/>")
        story.extend(
            [
                Paragraph("AI Insight", styles["Heading3"]),
                Paragraph(escaped_insight, body_style),
                Spacer(1, 8),
            ]
        )

        if section.metrics:
            story.append(Paragraph("Metrics", styles["Heading3"]))
            metrics_table = Table(
                _kv_table_rows(section.metrics),
                colWidths=[2.0 * inch, 4.8 * inch],
            )
            metrics_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.extend([metrics_table, Spacer(1, 8)])

        for table_name, table_df in section.tables:
            story.append(Paragraph(escape(table_name), styles["Heading3"]))
            df_table, is_truncated = _render_df_table(table_df, max_rows=table_max_rows)
            story.append(df_table)
            if is_truncated:
                story.append(
                    Paragraph(
                        f"Table truncated to first {table_max_rows} rows for PDF export.",
                        body_style,
                    )
                )
            story.append(Spacer(1, 8))

        if section.chart_png:
            story.append(Paragraph("Chart", styles["Heading3"]))
            story.append(Image(io.BytesIO(section.chart_png), width=6.6 * inch, height=2.8 * inch))
            story.append(Spacer(1, 8))

        story.append(Spacer(1, 10))

    if not document.sections:
        story.append(Paragraph("No insight sections were available for this report.", body_style))

    pdf.build(story)
    return buffer.getvalue()


def build_timesfm_chart_png(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    history_value_column: str = "value",
    forecast_value_column: str = "prediction",
) -> bytes:
    """Build a PNG line chart from history and forecast data."""
    history_plot_df = history_df.copy()
    forecast_plot_df = forecast_df.copy()
    history_plot_df["timestamp"] = pd.to_datetime(history_plot_df["timestamp"], errors="coerce")
    forecast_plot_df["timestamp"] = pd.to_datetime(forecast_plot_df["timestamp"], errors="coerce")
    history_plot_df = history_plot_df.dropna(subset=["timestamp", history_value_column])
    forecast_plot_df = forecast_plot_df.dropna(subset=["timestamp", forecast_value_column])

    fig = Figure(figsize=(10, 4), dpi=140)
    axis = fig.subplots()
    has_history = not history_plot_df.empty
    has_forecast = not forecast_plot_df.empty
    if has_history:
        axis.plot(
            history_plot_df["timestamp"],
            history_plot_df[history_value_column],
            label="History",
            linewidth=1.8,
        )
    if has_forecast:
        axis.plot(
            forecast_plot_df["timestamp"],
            forecast_plot_df[forecast_value_column],
            label="Forecast",
            linewidth=1.8,
        )
    axis.set_title("TimesFM Forecast")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Value")
    axis.grid(alpha=0.2)
    if has_history or has_forecast:
        axis.legend(loc="best")
    fig.autofmt_xdate()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    fig.clear()
    return buffer.getvalue()


def build_batch_zip(per_file_pdfs: Mapping[str, bytes]) -> bytes:
    """Package many per-file PDFs into one ZIP archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in sorted(per_file_pdfs):
            archive.writestr(filename, per_file_pdfs[filename])
    return buffer.getvalue()
