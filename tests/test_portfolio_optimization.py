"""Tests for portfolio optimization utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from app_core.portfolio_optimization import (
    PortfolioOptimizationError,
    build_covariance_from_panel,
    optimize_mean_variance_long_only,
    validate_portfolio_forecast_inputs,
)


def test_optimize_mean_variance_long_only_respects_constraints() -> None:
    expected_returns = {"AAPL": 0.05, "MSFT": 0.03, "NVDA": 0.08}
    covariance = pd.DataFrame(
        {
            "AAPL": [0.04, 0.01, 0.015],
            "MSFT": [0.01, 0.03, 0.01],
            "NVDA": [0.015, 0.01, 0.06],
        },
        index=["AAPL", "MSFT", "NVDA"],
    )

    result = optimize_mean_variance_long_only(
        expected_returns=expected_returns,
        covariance_matrix=covariance,
        risk_aversion=1.0,
        max_weight=0.7,
    )

    assert pytest.approx(result.weights_df["weight"].sum(), rel=1e-6) == 1.0
    assert (result.weights_df["weight"] >= 0.0).all()
    assert (result.weights_df["weight"] <= 0.7 + 1e-8).all()


def test_build_covariance_from_panel_computes_non_empty_matrix() -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="D").tolist() * 2,
            "ticker": ["AAPL"] * 6 + ["MSFT"] * 6,
            "value": [100, 101, 102, 103, 104, 105, 200, 201, 202, 204, 206, 208],
        }
    )
    covariance = build_covariance_from_panel(panel)
    assert covariance.shape == (2, 2)
    assert set(covariance.columns) == {"AAPL", "MSFT"}


def test_optimize_mean_variance_long_only_rejects_nonfinite_expected_returns() -> None:
    covariance = pd.DataFrame(
        {
            "AAPL": [0.04, 0.01],
            "MSFT": [0.01, 0.03],
        },
        index=["AAPL", "MSFT"],
    )
    with pytest.raises(PortfolioOptimizationError, match="must be finite"):
        optimize_mean_variance_long_only(
            expected_returns={"AAPL": float("nan"), "MSFT": 0.02},
            covariance_matrix=covariance,
        )


def test_optimize_mean_variance_long_only_rejects_nonfinite_covariance() -> None:
    covariance = pd.DataFrame(
        {
            "AAPL": [0.04, float("nan")],
            "MSFT": [0.01, 0.03],
        },
        index=["AAPL", "MSFT"],
    )
    with pytest.raises(PortfolioOptimizationError, match="non-finite"):
        optimize_mean_variance_long_only(
            expected_returns={"AAPL": 0.05, "MSFT": 0.02},
            covariance_matrix=covariance,
        )


def test_optimize_mean_variance_long_only_aligns_tickers_deterministically() -> None:
    expected_returns = {"AAPL": 0.05, "MSFT": 0.03, "NVDA": 0.08}
    covariance = pd.DataFrame(
        {
            "AAPL": [0.04, 0.01, 0.02],
            "MSFT": [0.01, 0.03, 0.01],
            "GOOG": [0.02, 0.01, 0.05],
        },
        index=["AAPL", "MSFT", "GOOG"],
    )
    result = optimize_mean_variance_long_only(
        expected_returns=expected_returns,
        covariance_matrix=covariance,
        max_weight=0.8,
    )

    assert set(result.weights_df["ticker"]) == {"AAPL", "MSFT"}
    assert pytest.approx(result.weights_df["weight"].sum(), rel=1e-6) == 1.0


def test_optimize_mean_variance_long_only_raises_when_less_than_two_aligned_tickers() -> None:
    covariance = pd.DataFrame(
        {
            "AAPL": [0.04, 0.01],
            "MSFT": [0.01, 0.03],
        },
        index=["AAPL", "MSFT"],
    )
    with pytest.raises(PortfolioOptimizationError, match="at least two aligned tickers"):
        optimize_mean_variance_long_only(
            expected_returns={"AAPL": 0.05, "NVDA": 0.08},
            covariance_matrix=covariance,
        )


def test_build_covariance_from_panel_requires_two_valid_return_series() -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D").tolist() * 2,
            "ticker": ["AAPL"] * 4 + ["MSFT"] * 4,
            "value": [100, 101, 102, 103, 200, None, None, None],
        }
    )
    with pytest.raises(PortfolioOptimizationError, match="at least two tickers"):
        build_covariance_from_panel(panel)


def test_validate_portfolio_forecast_inputs_allows_multi_ticker_valid_data() -> None:
    panel_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D").tolist() * 2,
            "ticker": ["AAPL"] * 5 + ["MSFT"] * 5,
            "value": [100, 101, 103, 102, 104, 200, 201, 202, 203, 204],
        }
    )
    forecast_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-06", periods=2, freq="D").tolist() * 2,
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "prediction": [105.0, 106.0, 205.0, 206.0],
        }
    )

    validate_portfolio_forecast_inputs(panel_df=panel_df, forecast_df=forecast_df)


def test_validate_portfolio_forecast_inputs_rejects_single_forecastable_ticker() -> None:
    panel_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "ticker": ["AAPL"] * 5,
            "value": [100, 101, 102, 103, 104],
        }
    )
    forecast_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-06", periods=2, freq="D"),
            "ticker": ["AAPL", "AAPL"],
            "prediction": [105.0, 106.0],
        }
    )

    with pytest.raises(PortfolioOptimizationError, match="at least two forecastable tickers"):
        validate_portfolio_forecast_inputs(panel_df=panel_df, forecast_df=forecast_df)


def test_validate_portfolio_forecast_inputs_rejects_insufficient_return_history() -> None:
    panel_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D").tolist() * 2,
            "ticker": ["AAPL"] * 4 + ["MSFT"] * 4,
            "value": [100, 101, 102, 103, 200, None, None, None],
        }
    )
    forecast_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-05", periods=2, freq="D").tolist() * 2,
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "prediction": [104.0, 105.0, 201.0, 202.0],
        }
    )

    with pytest.raises(PortfolioOptimizationError, match="valid return history"):
        validate_portfolio_forecast_inputs(panel_df=panel_df, forecast_df=forecast_df)
