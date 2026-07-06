"""Interactive chart builders for TimesFM outputs."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp"])
    return normalized.sort_values("timestamp").reset_index(drop=True)


def build_timesfm_forecast_figure(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    show_quantile_band: bool = True,
) -> go.Figure:
    """Build interactive history + forecast figure with optional quantile ribbon."""
    if "timestamp" not in history_df.columns or "value" not in history_df.columns:
        raise ValueError("history_df must contain 'timestamp' and 'value' columns.")
    if "timestamp" not in forecast_df.columns or "prediction" not in forecast_df.columns:
        raise ValueError("forecast_df must contain 'timestamp' and 'prediction' columns.")

    history = _normalize_timestamp(history_df[["timestamp", "value"]])
    forecast_columns = ["timestamp", "prediction"]
    if "p10" in forecast_df.columns:
        forecast_columns.append("p10")
    if "p90" in forecast_df.columns:
        forecast_columns.append("p90")
    forecast = _normalize_timestamp(forecast_df[forecast_columns])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["timestamp"],
            y=history["value"],
            mode="lines",
            name="History",
            line={"width": 2},
        )
    )

    has_quantiles = show_quantile_band and {"p10", "p90"}.issubset(forecast.columns)
    if has_quantiles:
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"],
                y=forecast["p90"],
                mode="lines",
                name="P90",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"],
                y=forecast["p10"],
                mode="lines",
                name="P10-P90 Band",
                fill="tonexty",
                fillcolor="rgba(30, 136, 229, 0.20)",
                line={"width": 0},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["prediction"],
            mode="lines",
            name="Forecast",
            line={"width": 2},
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="TimesFM Forecast",
        xaxis_title="Timestamp",
        yaxis_title="Value",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def build_timesfm_backtest_figure(comparison_df: pd.DataFrame) -> go.Figure:
    """Build interactive actual vs prediction comparison figure."""
    required = {"timestamp", "actual", "prediction"}
    if not required.issubset(comparison_df.columns):
        raise ValueError("comparison_df must contain timestamp, actual, prediction columns.")

    comparison = _normalize_timestamp(comparison_df[["timestamp", "actual", "prediction"]])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comparison["timestamp"],
            y=comparison["actual"],
            mode="lines+markers",
            name="Actual",
            line={"width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comparison["timestamp"],
            y=comparison["prediction"],
            mode="lines+markers",
            name="Prediction",
            line={"width": 2},
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Backtest: Actual vs Prediction",
        xaxis_title="Timestamp",
        yaxis_title="Value",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def build_timesfm_residual_figure(comparison_df: pd.DataFrame) -> go.Figure:
    """Build interactive residual chart using actual-prediction deltas."""
    required = {"timestamp", "actual", "prediction"}
    if not required.issubset(comparison_df.columns):
        raise ValueError("comparison_df must contain timestamp, actual, prediction columns.")

    comparison = _normalize_timestamp(comparison_df[["timestamp", "actual", "prediction"]])
    residuals = comparison["actual"].astype(float) - comparison["prediction"].astype(float)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comparison["timestamp"],
            y=residuals,
            name="Residual (Actual - Prediction)",
            marker_color="rgba(244, 67, 54, 0.7)",
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="black")
    fig.update_layout(
        template="plotly_white",
        title="Backtest Residuals",
        xaxis_title="Timestamp",
        yaxis_title="Residual",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig
