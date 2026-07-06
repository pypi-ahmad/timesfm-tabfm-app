"""Tests for PDF export helpers."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from app_core.pdf_export import (
    InsightPdfDocument,
    InsightPdfSection,
    build_batch_zip,
    build_pdf_bytes,
    build_timesfm_chart_png,
    can_export_insight,
    truncate_table_rows,
)


def test_can_export_insight_requires_non_empty_text() -> None:
    assert can_export_insight("Trend is stable.")
    assert not can_export_insight("   ")


def test_truncate_table_rows_marks_when_truncated() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3]})

    trimmed, is_truncated = truncate_table_rows(frame, max_rows=2)

    assert is_truncated
    assert trimmed["x"].tolist() == [1, 2]


def test_truncate_table_rows_rejects_invalid_limit() -> None:
    frame = pd.DataFrame({"x": [1]})

    with pytest.raises(ValueError, match="at least 1"):
        truncate_table_rows(frame, max_rows=0)


def test_build_pdf_bytes_renders_full_report_payload() -> None:
    section = InsightPdfSection(
        title="TimesFM Forecast Insight",
        insight_text="Demand is expected to rise steadily over the horizon.",
        metadata={"Horizon": 3, "Run Type": "TimesFM Single"},
        tables=[("Forecast", pd.DataFrame({"timestamp": ["2024-01-01"], "prediction": [10.5]}))],
        metrics={"MAE": 0.14, "RMSE": 0.19},
    )
    document = InsightPdfDocument(
        title="TimesFM Insight Report",
        generated_at="2026-07-05 10:00:00 UTC",
        metadata={"Ollama model": "qwen3:4b"},
        sections=[section],
    )

    payload = build_pdf_bytes(document=document, table_max_rows=100, font_size=10)

    assert payload.startswith(b"%PDF")
    assert len(payload) > 800


def test_build_pdf_bytes_supports_multiple_sections() -> None:
    doc = InsightPdfDocument(
        title="Batch Insights",
        generated_at="2026-07-05 10:00:00 UTC",
        sections=[
            InsightPdfSection(title="file_a.csv", insight_text="A"),
            InsightPdfSection(title="file_b.csv", insight_text="B"),
        ],
    )

    payload = build_pdf_bytes(document=doc)

    assert payload.startswith(b"%PDF")


def test_build_timesfm_chart_png_returns_png_payload() -> None:
    history = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=4, freq="D"), "value": [1.0, 2.0, 3.0, 4.0]}
    )
    forecast = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-05", periods=2, freq="D"),
            "prediction": [4.5, 5.0],
        }
    )

    chart = build_timesfm_chart_png(history_df=history, forecast_df=forecast)

    assert chart.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(chart) > 200


def test_build_batch_zip_contains_expected_pdf_entries() -> None:
    zipped = build_batch_zip(
        {
            "a_report.pdf": b"%PDF-1.4\nA",
            "b_report.pdf": b"%PDF-1.4\nB",
        }
    )

    with zipfile.ZipFile(io.BytesIO(zipped), "r") as archive:
        names = archive.namelist()

    assert names == ["a_report.pdf", "b_report.pdf"]
