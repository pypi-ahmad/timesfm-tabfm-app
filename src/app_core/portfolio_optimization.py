"""Portfolio optimization utilities for multi-asset TimesFM forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class PortfolioOptimizationError(RuntimeError):
    """Raised when optimizer inputs are invalid or convergence fails."""


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    """Optimizer output for UI display and artifact export."""

    weights_df: pd.DataFrame
    expected_return: float
    expected_volatility: float


def _normalize_ticker(value: object) -> str:
    return str(value).strip().upper()


def _project_to_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    """Project a vector into the simplex with lower bound 0 and upper bound cap."""
    if cap <= 0.0 or cap > 1.0:
        raise PortfolioOptimizationError("Max weight cap must be in (0, 1].")
    n_assets = values.size
    if cap * n_assets < 1.0:
        raise PortfolioOptimizationError(
            "Max weight cap is infeasible for the number of assets."
        )

    weights = np.clip(values.astype(float), 0.0, cap)
    if np.isclose(weights.sum(), 1.0):
        return weights
    if weights.sum() == 0.0:
        weights = np.ones_like(weights) / n_assets

    active = np.ones(n_assets, dtype=bool)
    projected = np.zeros(n_assets, dtype=float)
    remaining = 1.0
    shifted = weights.copy()

    while np.any(active):
        active_values = shifted[active]
        theta = (active_values.sum() - remaining) / active_values.size
        candidate = shifted - theta

        high_mask = (candidate >= cap) & active
        low_mask = (candidate <= 0.0) & active
        mid_mask = active & ~high_mask & ~low_mask

        if not np.any(high_mask) and not np.any(low_mask):
            projected[mid_mask] = candidate[mid_mask]
            break

        if np.any(high_mask):
            projected[high_mask] = cap
            remaining -= cap * int(high_mask.sum())
            active[high_mask] = False
        if np.any(low_mask):
            projected[low_mask] = 0.0
            active[low_mask] = False

        if remaining < 0.0:
            remaining = 0.0
            break

    if projected.sum() == 0.0:
        projected = np.ones(n_assets, dtype=float) / n_assets
    else:
        projected = projected / projected.sum()
    return np.clip(projected, 0.0, cap)


def optimize_mean_variance_long_only(
    expected_returns: dict[str, float],
    covariance_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
    max_weight: float = 0.4,
    iterations: int = 500,
    learning_rate: float = 0.05,
) -> PortfolioOptimizationResult:
    """Optimize a long-only mean-variance portfolio with capped weights."""
    if not expected_returns:
        raise PortfolioOptimizationError("Expected returns cannot be empty.")
    if covariance_matrix.empty:
        raise PortfolioOptimizationError("Covariance matrix cannot be empty.")

    normalized_returns: dict[str, float] = {}
    for raw_ticker, raw_value in expected_returns.items():
        ticker = _normalize_ticker(raw_ticker)
        if not ticker:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise PortfolioOptimizationError(
                f"Expected return for ticker '{ticker}' is not numeric."
            ) from exc
        if not np.isfinite(value):
            raise PortfolioOptimizationError(
                f"Expected return for ticker '{ticker}' must be finite."
            )
        normalized_returns[ticker] = value

    if len(normalized_returns) < 2:
        raise PortfolioOptimizationError("Portfolio optimization requires at least two assets.")

    cov = covariance_matrix.copy()
    cov.index = cov.index.map(_normalize_ticker)
    cov.columns = cov.columns.map(_normalize_ticker)
    aligned_tickers = sorted(
        set(normalized_returns).intersection(set(cov.index)).intersection(set(cov.columns))
    )
    if len(aligned_tickers) < 2:
        raise PortfolioOptimizationError(
            "Portfolio optimization requires at least two aligned tickers between "
            "expected returns and covariance matrix."
        )

    means = np.array([normalized_returns[ticker] for ticker in aligned_tickers], dtype=float)
    cov = cov.reindex(index=aligned_tickers, columns=aligned_tickers)
    cov = cov.apply(pd.to_numeric, errors="coerce")
    cov_values = cov.to_numpy(dtype=float)
    if not np.isfinite(cov_values).all():
        raise PortfolioOptimizationError(
            "Covariance matrix contains non-finite values for aligned tickers."
        )
    cov_values = (cov_values + cov_values.T) / 2.0

    n_assets = means.size
    weights = np.ones(n_assets, dtype=float) / n_assets
    risk_penalty = float(risk_aversion)
    if not np.isfinite(risk_penalty) or risk_penalty <= 0.0:
        raise PortfolioOptimizationError("Risk aversion must be a positive finite value.")
    step = float(learning_rate)
    if not np.isfinite(step) or step <= 0.0:
        raise PortfolioOptimizationError("Learning rate must be a positive finite value.")
    max_iters = max(50, int(iterations))

    for _ in range(max_iters):
        gradient = means - (2.0 * risk_penalty * cov_values.dot(weights))
        candidate = weights + (step * gradient)
        weights = _project_to_capped_simplex(candidate, cap=float(max_weight))

    expected_return = float(means.dot(weights))
    expected_volatility = float(np.sqrt(max(weights.T.dot(cov_values).dot(weights), 0.0)))
    weights_df = pd.DataFrame(
        {
            "ticker": aligned_tickers,
            "weight": weights,
            "expected_return_contribution": weights * means,
        }
    ).sort_values("weight", ascending=False, ignore_index=True)
    return PortfolioOptimizationResult(
        weights_df=weights_df,
        expected_return=expected_return,
        expected_volatility=expected_volatility,
    )


def validate_portfolio_forecast_inputs(
    panel_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> None:
    """Validate panel and forecast outputs before portfolio optimization."""
    required_forecast = {"ticker", "prediction"}
    if not required_forecast.issubset(forecast_df.columns):
        raise PortfolioOptimizationError(
            "Forecast output is missing required columns for portfolio optimization."
        )

    forecastable_tickers = (
        forecast_df.dropna(subset=["ticker", "prediction"])["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )
    if len(forecastable_tickers) < 2:
        raise PortfolioOptimizationError(
            "Portfolio optimization requires at least two forecastable tickers."
        )

    normalized = panel_df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", "ticker", "value"])
    pivot = normalized.pivot_table(
        index="timestamp",
        columns="ticker",
        values="value",
        aggfunc="last",
    ).sort_index()
    returns = pivot.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    usable_returns = returns.dropna(axis=1, how="all")
    if usable_returns.shape[1] < 2:
        raise PortfolioOptimizationError(
            "Portfolio optimization requires at least two tickers with valid return history."
        )


def build_covariance_from_panel(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Build historical return covariance matrix from panel history data."""
    required = {"timestamp", "ticker", "value"}
    if not required.issubset(panel_df.columns):
        raise PortfolioOptimizationError(
            "Panel data must include timestamp, ticker, and value columns."
        )
    normalized = panel_df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", "ticker", "value"])
    pivot = normalized.pivot_table(
        index="timestamp",
        columns="ticker",
        values="value",
        aggfunc="last",
    ).sort_index()
    returns = pivot.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if returns.empty:
        raise PortfolioOptimizationError("Not enough history to compute covariance matrix.")
    usable_returns = returns.dropna(axis=1, how="all")
    if usable_returns.shape[1] < 2:
        raise PortfolioOptimizationError(
            "Need at least two tickers with valid return history to compute covariance matrix."
        )
    cov = usable_returns.cov().fillna(0.0)
    if cov.shape[0] < 2:
        raise PortfolioOptimizationError(
            "Need at least two tickers with valid return history to compute covariance matrix."
        )
    if cov.empty:
        raise PortfolioOptimizationError("Covariance matrix is empty.")
    return cov
