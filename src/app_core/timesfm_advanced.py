"""Advanced TimesFM orchestration for XReg, multi-asset runs, and backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app_core.batch_io import BatchInputItem
from app_core.metrics import compute_regression_metrics
from app_core.sentiment_service import apply_sentiment_bias
from app_core.timesfm_service import run_timesfm_forecast
from app_core.validators import ValidationError, validate_timesfm_input


@dataclass(frozen=True)
class MultiAssetForecastResult:
    """Forecast output for multiple assets in panel form."""

    forecast_df: pd.DataFrame


@dataclass(frozen=True)
class BacktestingFrameworkResult:
    """Fold-level and aggregate backtesting outputs."""

    fold_metrics_df: pd.DataFrame
    aggregate_metrics_df: pd.DataFrame


def _infer_seasonal_period(timestamps: pd.Series) -> int:
    """Infer seasonal period from timestamp frequency with safe fallback."""
    inferred = pd.infer_freq(pd.to_datetime(timestamps, errors="coerce").dropna())
    if not inferred:
        return 7

    normalized = inferred.upper()
    if normalized.startswith("W"):
        return 52
    if normalized.startswith("M"):
        return 12
    if normalized.startswith("D") or normalized.startswith("B"):
        return 7
    return 7


def _naive_forecast(train_values: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value across the forecast horizon."""
    last_value = float(train_values[-1])
    return np.repeat(last_value, int(horizon)).astype(float)


def _seasonal_naive_forecast(
    train_values: np.ndarray,
    horizon: int,
    seasonal_period: int,
) -> np.ndarray:
    """Seasonal naive forecast with fallback to naive when history is too short."""
    resolved_horizon = int(horizon)
    resolved_period = max(1, int(seasonal_period))
    if len(train_values) < resolved_period:
        return _naive_forecast(train_values=train_values, horizon=resolved_horizon)

    season_window = train_values[-resolved_period:].astype(float)
    return np.array(
        [float(season_window[index % resolved_period]) for index in range(resolved_horizon)],
        dtype=float,
    )


def normalize_panel_history(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    ticker_column: str = "ticker",
    value_column: str = "value",
) -> pd.DataFrame:
    """Normalize panel history to canonical timestamp/ticker/value columns."""
    required = {timestamp_column, ticker_column, value_column}
    if not required.issubset(df.columns):
        missing = sorted(required.difference(set(df.columns)))
        raise ValidationError(
            f"Panel data is missing required columns: {missing}."
        )

    normalized = df.copy()
    normalized[timestamp_column] = pd.to_datetime(normalized[timestamp_column], errors="coerce")
    normalized[value_column] = pd.to_numeric(normalized[value_column], errors="coerce")
    normalized[ticker_column] = normalized[ticker_column].astype(str).str.strip().str.upper()
    normalized = normalized.dropna(subset=[timestamp_column, ticker_column, value_column])
    normalized = normalized.rename(
        columns={
            timestamp_column: "timestamp",
            ticker_column: "ticker",
            value_column: "value",
        }
    )
    normalized = normalized.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    if normalized.empty:
        raise ValidationError("Panel data is empty after parsing timestamp/ticker/value.")
    return normalized


def build_panel_from_batch_items(
    history_items: list[BatchInputItem],
    timestamp_column: str = "timestamp",
    value_column: str = "value",
) -> pd.DataFrame:
    """Build canonical panel data from per-ticker CSV items."""
    rows: list[pd.DataFrame] = []
    for item in history_items:
        series_df = validate_timesfm_input(
            item.dataframe,
            timestamp_column=timestamp_column,
            value_column=value_column,
        )
        ticker = str(item.name).rsplit(".", 1)[0].strip().upper()
        if not ticker:
            raise ValidationError(f"Unable to infer ticker from file name: {item.name}")
        rows.append(series_df.assign(ticker=ticker)[["timestamp", "ticker", "value"]])
    if not rows:
        raise ValidationError("No valid per-ticker series were provided.")
    return pd.concat(rows, ignore_index=True).sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def run_timesfm_multi_asset_forecast(
    panel_df: pd.DataFrame,
    horizon: int,
    max_context: int,
    max_horizon: int,
    normalize_inputs: bool,
    include_quantiles: bool,
    model_id: str,
    backend: str,
    use_xreg: bool = False,
    covariate_columns: list[str] | None = None,
    xreg_mode: str = "xreg + timesfm",
    sentiment_scores: dict[str, float] | None = None,
    sentiment_strength: float = 0.0,
    sentiment_decay: float = 1.0,
    lora_adapter_path: str | None = None,
) -> MultiAssetForecastResult:
    """Forecast multiple assets from panel history with optional XReg and sentiment."""
    normalized_panel = normalize_panel_history(panel_df)
    covariates = covariate_columns or []

    rows: list[pd.DataFrame] = []
    for ticker, ticker_df in normalized_panel.groupby("ticker", sort=True):
        ticker_history = ticker_df.sort_values("timestamp").reset_index(drop=True)
        validated = validate_timesfm_input(ticker_history, "timestamp", "value")
        ticker_covariate_columns = [
            column
            for column in covariates
            if column in ticker_history.columns
        ]
        ticker_covariates = None
        if use_xreg:
            ticker_covariates = ticker_history[
                ["timestamp", *ticker_covariate_columns]
            ].copy()
        forecast_df = run_timesfm_forecast(
            history_df=validated,
            horizon=horizon,
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=normalize_inputs,
            include_quantiles=include_quantiles,
            model_id=model_id,
            backend=backend,
            use_xreg=use_xreg,
            ticker=str(ticker),
            covariates_df=ticker_covariates,
            covariate_columns=ticker_covariate_columns if use_xreg else None,
            xreg_mode=xreg_mode,
            lora_adapter_path=lora_adapter_path,
        ).forecast_df
        rows.append(forecast_df.assign(ticker=ticker)[["timestamp", "ticker", *forecast_df.columns.drop("timestamp")]])

    forecast_panel = pd.concat(rows, ignore_index=True).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    if sentiment_scores:
        forecast_panel = apply_sentiment_bias(
            forecast_df=forecast_panel,
            ticker_scores=sentiment_scores,
            strength=float(sentiment_strength),
            decay=float(sentiment_decay),
        )
    return MultiAssetForecastResult(forecast_df=forecast_panel)


def _build_fold_ranges(
    series_len: int,
    horizon: int,
    mode: str,
    folds: int,
    min_train_size: int,
    rolling_window: int,
) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    if mode == "walk_forward":
        for fold in range(folds):
            train_end = min_train_size + (fold * horizon)
            test_end = train_end + horizon
            if test_end > series_len:
                break
            ranges.append((0, train_end, test_end))
    elif mode == "rolling_window":
        for fold in range(folds):
            train_end = rolling_window + (fold * horizon)
            train_start = train_end - rolling_window
            test_end = train_end + horizon
            if train_start < 0 or test_end > series_len:
                break
            ranges.append((train_start, train_end, test_end))
    else:
        raise ValidationError("Backtesting mode must be 'walk_forward' or 'rolling_window'.")
    return ranges


def run_timesfm_backtesting_framework(
    panel_df: pd.DataFrame,
    mode: str,
    folds: int,
    horizon: int,
    max_context: int,
    max_horizon: int,
    normalize_inputs: bool,
    include_quantiles: bool,
    model_id: str,
    backend: str,
    min_train_size: int = 40,
    rolling_window: int = 120,
    lora_adapter_path: str | None = None,
) -> BacktestingFrameworkResult:
    """Run walk-forward or rolling-window framework backtesting for panel data."""
    normalized_panel = normalize_panel_history(panel_df)
    fold_rows: list[dict[str, object]] = []

    for ticker, ticker_df in normalized_panel.groupby("ticker", sort=True):
        validated = validate_timesfm_input(ticker_df, "timestamp", "value")
        ranges = _build_fold_ranges(
            series_len=len(validated),
            horizon=horizon,
            mode=mode,
            folds=folds,
            min_train_size=max(min_train_size, horizon + 5),
            rolling_window=max(rolling_window, horizon + 5),
        )
        if not ranges:
            continue

        for fold_index, (train_start, train_end, test_end) in enumerate(ranges, start=1):
            train_history = validated.iloc[train_start:train_end].copy()
            holdout = validated.iloc[train_end:test_end].copy().reset_index(drop=True)
            forecast = run_timesfm_forecast(
                history_df=train_history,
                horizon=len(holdout),
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=normalize_inputs,
                include_quantiles=include_quantiles,
                model_id=model_id,
                backend=backend,
                lora_adapter_path=lora_adapter_path,
            ).forecast_df
            train_values = train_history["value"].astype(float).to_numpy()
            actual_values = holdout["value"].astype(float).to_numpy()
            timesfm_predictions = forecast["prediction"].astype(float).to_numpy()
            timesfm_metrics = compute_regression_metrics(
                actual=actual_values.tolist(),
                predicted=timesfm_predictions.tolist(),
                direction_anchor=float(train_values[-1]),
                lower_quantile=forecast["p10"].tolist() if "p10" in forecast.columns else None,
                upper_quantile=forecast["p90"].tolist() if "p90" in forecast.columns else None,
            )

            naive_predictions = _naive_forecast(
                train_values=train_values,
                horizon=len(holdout),
            )
            naive_metrics = compute_regression_metrics(
                actual=actual_values.tolist(),
                predicted=naive_predictions.tolist(),
                direction_anchor=float(train_values[-1]),
            )

            inferred_seasonal_period = _infer_seasonal_period(train_history["timestamp"])
            seasonal_naive_predictions = _seasonal_naive_forecast(
                train_values=train_values,
                horizon=len(holdout),
                seasonal_period=inferred_seasonal_period,
            )
            seasonal_naive_metrics = compute_regression_metrics(
                actual=holdout["value"].astype(float).tolist(),
                predicted=seasonal_naive_predictions.astype(float).tolist(),
                direction_anchor=float(train_values[-1]),
            )
            common_window_payload = {
                "ticker": ticker,
                "mode": mode,
                "fold": fold_index,
                "historical_window_index": fold_index,
                "historical_window_label": f"{mode}_window_{fold_index}",
                "train_start": train_history["timestamp"].iloc[0],
                "train_end": train_history["timestamp"].iloc[-1],
                "test_start": holdout["timestamp"].iloc[0],
                "test_end": holdout["timestamp"].iloc[-1],
                "validation_start": holdout["timestamp"].iloc[0],
                "validation_end": holdout["timestamp"].iloc[-1],
            }
            fold_rows.extend(
                [
                    {
                        **common_window_payload,
                        "model": "timesfm",
                        "seasonal_period": None,
                        "mae": timesfm_metrics["mae"],
                        "rmse": timesfm_metrics["rmse"],
                        "mse": timesfm_metrics["mse"],
                        "mape": timesfm_metrics["mape"],
                        "smape": timesfm_metrics["smape"],
                        "wape": timesfm_metrics["wape"],
                        "directional_accuracy": timesfm_metrics["directional_accuracy"],
                        "quantile_coverage_error": timesfm_metrics["quantile_coverage_error"],
                    },
                    {
                        **common_window_payload,
                        "model": "naive",
                        "seasonal_period": None,
                        "mae": naive_metrics["mae"],
                        "rmse": naive_metrics["rmse"],
                        "mse": naive_metrics["mse"],
                        "mape": naive_metrics["mape"],
                        "smape": naive_metrics["smape"],
                        "wape": naive_metrics["wape"],
                        "directional_accuracy": naive_metrics["directional_accuracy"],
                        "quantile_coverage_error": naive_metrics["quantile_coverage_error"],
                    },
                    {
                        **common_window_payload,
                        "model": "seasonal_naive",
                        "seasonal_period": int(inferred_seasonal_period),
                        "mae": seasonal_naive_metrics["mae"],
                        "rmse": seasonal_naive_metrics["rmse"],
                        "mse": seasonal_naive_metrics["mse"],
                        "mape": seasonal_naive_metrics["mape"],
                        "smape": seasonal_naive_metrics["smape"],
                        "wape": seasonal_naive_metrics["wape"],
                        "directional_accuracy": seasonal_naive_metrics["directional_accuracy"],
                        "quantile_coverage_error": seasonal_naive_metrics["quantile_coverage_error"],
                    },
                ]
            )

    if not fold_rows:
        raise ValidationError(
            "Not enough observations to create backtesting folds with current settings."
        )

    fold_metrics_df = pd.DataFrame(fold_rows)
    fold_metrics_df = fold_metrics_df[
        [
            "ticker",
            "mode",
            "historical_window_index",
            "historical_window_label",
            "fold",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
            "model",
            "seasonal_period",
            "mae",
            "rmse",
            "mse",
            "mape",
            "smape",
            "wape",
            "directional_accuracy",
            "quantile_coverage_error",
        ]
    ]

    aggregate_metrics_df = (
        fold_metrics_df.groupby(["ticker", "mode", "model"], as_index=False)[
            [
                "mae",
                "rmse",
                "mse",
                "mape",
                "smape",
                "wape",
                "directional_accuracy",
                "quantile_coverage_error",
            ]
        ]
        .mean()
        .sort_values(["ticker", "mode", "model"])
        .reset_index(drop=True)
    )
    aggregate_metrics_df = aggregate_metrics_df[
        [
            "ticker",
            "mode",
            "model",
            "mae",
            "rmse",
            "mse",
            "mape",
            "smape",
            "wape",
            "directional_accuracy",
            "quantile_coverage_error",
        ]
    ]
    return BacktestingFrameworkResult(
        fold_metrics_df=fold_metrics_df,
        aggregate_metrics_df=aggregate_metrics_df,
    )


def derive_expected_returns(
    forecast_df: pd.DataFrame,
    panel_df: pd.DataFrame,
) -> dict[str, float]:
    """Estimate expected returns from forecast means and latest observed values."""
    latest_actual = (
        normalize_panel_history(panel_df)
        .sort_values("timestamp")
        .groupby("ticker")
        .tail(1)
        .set_index("ticker")["value"]
    )
    expected_by_ticker: dict[str, float] = {}
    for ticker, ticker_forecast in forecast_df.groupby("ticker"):
        last_value = float(latest_actual.get(ticker, np.nan))
        if np.isnan(last_value) or last_value == 0.0:
            continue
        forecast_col = "prediction"
        if "base_prediction" in ticker_forecast.columns:
            forecast_col = "prediction"
        expected_level = float(pd.to_numeric(ticker_forecast[forecast_col], errors="coerce").mean())
        expected_by_ticker[str(ticker)] = (expected_level / last_value) - 1.0
    if not expected_by_ticker:
        raise ValidationError("Unable to derive expected returns from forecast output.")
    return expected_by_ticker
