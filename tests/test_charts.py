"""Tests for Plotly chart helper functions."""

from __future__ import annotations

import pandas as pd
import pytest

from app_core.charts import (
    build_timesfm_backtest_figure,
    build_timesfm_forecast_figure,
    build_timesfm_residual_figure,
)


def test_build_timesfm_forecast_figure_has_history_and_forecast_traces() -> None:
    history = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=3, freq="D"), "value": [10.0, 11.0, 12.0]}
    )
    forecast = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-04", periods=2, freq="D"),
            "prediction": [12.5, 13.0],
        }
    )

    fig = build_timesfm_forecast_figure(history_df=history, forecast_df=forecast)

    assert len(fig.data) == 2
    names = [trace.name for trace in fig.data]
    assert names == ["History", "Forecast"]


def test_build_timesfm_forecast_figure_adds_quantile_band_trace() -> None:
    history = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=3, freq="D"), "value": [10.0, 11.0, 12.0]}
    )
    forecast = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-04", periods=2, freq="D"),
            "prediction": [12.5, 13.0],
            "p10": [12.1, 12.7],
            "p90": [12.9, 13.4],
        }
    )

    fig = build_timesfm_forecast_figure(
        history_df=history,
        forecast_df=forecast,
        show_quantile_band=True,
    )

    assert len(fig.data) == 4
    assert fig.data[1].name == "P90"
    assert fig.data[2].name == "P10-P90 Band"
    assert fig.data[2].fill == "tonexty"


def test_build_timesfm_backtest_figure_has_actual_and_prediction() -> None:
    comparison = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "actual": [10.0, 11.0, 12.0],
            "prediction": [9.5, 11.2, 12.4],
        }
    )

    fig = build_timesfm_backtest_figure(comparison_df=comparison)

    assert len(fig.data) == 2
    assert [trace.name for trace in fig.data] == ["Actual", "Prediction"]


def test_build_timesfm_residual_figure_has_residual_bar_values() -> None:
    comparison = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "actual": [10.0, 11.0, 12.0],
            "prediction": [9.5, 11.2, 12.4],
        }
    )

    fig = build_timesfm_residual_figure(comparison_df=comparison)

    assert len(fig.data) == 1
    assert fig.data[0].name == "Residual (Actual - Prediction)"
    assert list(fig.data[0].y) == pytest.approx([0.5, -0.2, -0.4])
