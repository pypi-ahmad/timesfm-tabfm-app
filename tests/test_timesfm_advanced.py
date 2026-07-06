"""Tests for advanced TimesFM orchestration."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from app_core.batch_io import BatchInputItem
from app_core.timesfm_advanced import (
    build_panel_from_batch_items,
    normalize_panel_history,
    run_timesfm_backtesting_framework,
    run_timesfm_multi_asset_forecast,
)


def test_normalize_panel_history_standardizes_columns() -> None:
    df = pd.DataFrame(
        {
            "dt": ["2024-01-01", "2024-01-02"],
            "symbol": ["aapl", "aapl"],
            "close": ["100", "102"],
        }
    )
    normalized = normalize_panel_history(
        df,
        timestamp_column="dt",
        ticker_column="symbol",
        value_column="close",
    )
    assert list(normalized.columns[:3]) == ["timestamp", "ticker", "value"]
    assert normalized["ticker"].tolist() == ["AAPL", "AAPL"]


def test_build_panel_from_batch_items_uses_file_stem_as_ticker() -> None:
    items = [
        BatchInputItem(
            name="aapl.csv",
            dataframe=pd.DataFrame(
                {"timestamp": pd.date_range("2024-01-01", periods=3), "value": [1.0, 2.0, 3.0]}
            ),
        )
    ]
    panel = build_panel_from_batch_items(items)
    assert panel["ticker"].tolist() == ["AAPL", "AAPL", "AAPL"]


def test_run_timesfm_multi_asset_forecast_with_sentiment_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5).tolist() * 2,
            "ticker": ["AAPL"] * 5 + ["MSFT"] * 5,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )

    def fake_run_timesfm_forecast(**kwargs: object) -> object:
        return type(
            "FakeForecast",
            (),
            {
                "forecast_df": pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2024-02-01", periods=2),
                        "prediction": [100.0, 101.0],
                    }
                )
            },
        )()

    monkeypatch.setattr("app_core.timesfm_advanced.run_timesfm_forecast", fake_run_timesfm_forecast)
    result = run_timesfm_multi_asset_forecast(
        panel_df=panel,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=False,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        sentiment_scores={"AAPL": 0.5, "MSFT": -0.25},
        sentiment_strength=0.1,
        sentiment_decay=1.0,
    )

    assert set(result.forecast_df["ticker"].unique()) == {"AAPL", "MSFT"}
    assert "base_prediction" in result.forecast_df.columns


def test_run_timesfm_multi_asset_forecast_continues_without_sentiment_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5).tolist() * 2,
            "ticker": ["AAPL"] * 5 + ["MSFT"] * 5,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )

    def fake_run_timesfm_forecast(**kwargs: object) -> object:
        return type(
            "FakeForecast",
            (),
            {
                "forecast_df": pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2024-02-01", periods=2),
                        "prediction": [100.0, 101.0],
                    }
                )
            },
        )()

    monkeypatch.setattr("app_core.timesfm_advanced.run_timesfm_forecast", fake_run_timesfm_forecast)
    result = run_timesfm_multi_asset_forecast(
        panel_df=panel,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=False,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        sentiment_scores={},
        sentiment_strength=0.1,
        sentiment_decay=1.0,
    )

    assert set(result.forecast_df["ticker"].unique()) == {"AAPL", "MSFT"}
    assert "base_prediction" not in result.forecast_df.columns


def test_run_timesfm_multi_asset_forecast_forwards_xreg_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "ticker": ["AAPL"] * 5,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "CPIAUCSL": [300.1, 300.2, 300.3, 300.4, 300.5],
        }
    )
    seen_kwargs: dict[str, object] = {}

    def fake_run_timesfm_forecast(**kwargs: object) -> object:
        nonlocal seen_kwargs
        seen_kwargs = kwargs
        return type(
            "FakeForecast",
            (),
            {
                "forecast_df": pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2024-02-01", periods=2, freq="D"),
                        "prediction": [100.0, 101.0],
                    }
                )
            },
        )()

    monkeypatch.setattr("app_core.timesfm_advanced.run_timesfm_forecast", fake_run_timesfm_forecast)
    result = run_timesfm_multi_asset_forecast(
        panel_df=panel,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=False,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        use_xreg=True,
        covariate_columns=["CPIAUCSL"],
        xreg_mode="timesfm + xreg",
    )

    assert result.forecast_df["ticker"].tolist() == ["AAPL", "AAPL"]
    assert seen_kwargs["use_xreg"] is True
    assert seen_kwargs["ticker"] == "AAPL"
    assert seen_kwargs["covariate_columns"] == ["CPIAUCSL"]
    assert seen_kwargs["xreg_mode"] == "timesfm + xreg"


def test_run_timesfm_backtesting_framework_returns_mse_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=15),
            "ticker": ["AAPL"] * 15,
            "value": [float(i) for i in range(15)],
        }
    )

    def fake_run_timesfm_forecast(**kwargs: object) -> object:
        history_df = kwargs["history_df"]
        horizon = int(kwargs["horizon"])
        step = timedelta(days=1)
        start = history_df["timestamp"].iloc[-1]
        return type(
            "FakeForecast",
            (),
            {
                "forecast_df": pd.DataFrame(
                    {
                        "timestamp": [start + step * (i + 1) for i in range(horizon)],
                        "prediction": [history_df["value"].iloc[-1]] * horizon,
                    }
                )
            },
        )()

    monkeypatch.setattr("app_core.timesfm_advanced.run_timesfm_forecast", fake_run_timesfm_forecast)
    result = run_timesfm_backtesting_framework(
        panel_df=panel,
        mode="walk_forward",
        folds=2,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=False,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        min_train_size=6,
        rolling_window=6,
    )

    assert not result.fold_metrics_df.empty
    assert set(result.fold_metrics_df["model"].unique()) == {
        "timesfm",
        "naive",
        "seasonal_naive",
    }
    assert "historical_window_index" in result.fold_metrics_df.columns
    assert "historical_window_label" in result.fold_metrics_df.columns
    assert "validation_start" in result.fold_metrics_df.columns
    assert "validation_end" in result.fold_metrics_df.columns
    assert "seasonal_period" in result.fold_metrics_df.columns
    assert "mse" in result.fold_metrics_df.columns
    assert "mape" in result.fold_metrics_df.columns
    assert "smape" in result.fold_metrics_df.columns
    assert "wape" in result.fold_metrics_df.columns
    assert "directional_accuracy" in result.fold_metrics_df.columns
    assert "quantile_coverage_error" in result.fold_metrics_df.columns
    assert not result.aggregate_metrics_df.empty
    assert set(result.aggregate_metrics_df["model"].unique()) == {
        "timesfm",
        "naive",
        "seasonal_naive",
    }
    assert "mse" in result.aggregate_metrics_df.columns
    assert "mape" in result.aggregate_metrics_df.columns
    assert "smape" in result.aggregate_metrics_df.columns
    assert "wape" in result.aggregate_metrics_df.columns
    assert "directional_accuracy" in result.aggregate_metrics_df.columns
    assert "quantile_coverage_error" in result.aggregate_metrics_df.columns


def test_run_timesfm_backtesting_framework_aggregate_matches_fold_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=16),
            "ticker": ["AAPL"] * 16,
            "value": [float(i) for i in range(16)],
        }
    )

    def fake_run_timesfm_forecast(**kwargs: object) -> object:
        history_df = kwargs["history_df"]
        horizon = int(kwargs["horizon"])
        step = timedelta(days=1)
        start = history_df["timestamp"].iloc[-1]
        return type(
            "FakeForecast",
            (),
            {
                "forecast_df": pd.DataFrame(
                    {
                        "timestamp": [start + step * (i + 1) for i in range(horizon)],
                        "prediction": [history_df["value"].iloc[-1]] * horizon,
                    }
                )
            },
        )()

    monkeypatch.setattr("app_core.timesfm_advanced.run_timesfm_forecast", fake_run_timesfm_forecast)
    result = run_timesfm_backtesting_framework(
        panel_df=panel,
        mode="walk_forward",
        folds=3,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=False,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        min_train_size=6,
        rolling_window=6,
    )

    grouped_fold_mean = (
        result.fold_metrics_df.groupby(["ticker", "mode", "model"], as_index=False)[
            [
                "mae",
                "rmse",
                "mse",
                "mape",
                "smape",
                "wape",
                "directional_accuracy",
            ]
        ]
        .mean()
        .sort_values(["ticker", "mode", "model"])
        .reset_index(drop=True)
    )
    aggregate_df = (
        result.aggregate_metrics_df.sort_values(["ticker", "mode", "model"])
        .reset_index(drop=True)
    )
    assert aggregate_df["mae"].tolist() == pytest.approx(grouped_fold_mean["mae"].tolist())
    assert aggregate_df["rmse"].tolist() == pytest.approx(grouped_fold_mean["rmse"].tolist())
    assert aggregate_df["mse"].tolist() == pytest.approx(grouped_fold_mean["mse"].tolist())
    assert aggregate_df["mape"].tolist() == pytest.approx(grouped_fold_mean["mape"].tolist())
    assert aggregate_df["smape"].tolist() == pytest.approx(grouped_fold_mean["smape"].tolist())
    assert aggregate_df["wape"].tolist() == pytest.approx(grouped_fold_mean["wape"].tolist())
    assert aggregate_df["directional_accuracy"].tolist() == pytest.approx(
        grouped_fold_mean["directional_accuracy"].tolist()
    )


def test_run_timesfm_backtesting_framework_seasonal_naive_falls_back_when_short_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-07", periods=12, freq="W"),
            "ticker": ["AAPL"] * 12,
            "value": [float(i) for i in range(12)],
        }
    )

    def fake_run_timesfm_forecast(**kwargs: object) -> object:
        history_df = kwargs["history_df"]
        horizon = int(kwargs["horizon"])
        step = timedelta(weeks=1)
        start = history_df["timestamp"].iloc[-1]
        return type(
            "FakeForecast",
            (),
            {
                "forecast_df": pd.DataFrame(
                    {
                        "timestamp": [start + step * (i + 1) for i in range(horizon)],
                        "prediction": [history_df["value"].iloc[-1]] * horizon,
                    }
                )
            },
        )()

    monkeypatch.setattr("app_core.timesfm_advanced.run_timesfm_forecast", fake_run_timesfm_forecast)
    result = run_timesfm_backtesting_framework(
        panel_df=panel,
        mode="walk_forward",
        folds=2,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=False,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        min_train_size=7,
        rolling_window=7,
    )

    seasonal_rows = result.fold_metrics_df[result.fold_metrics_df["model"] == "seasonal_naive"]
    naive_rows = result.fold_metrics_df[result.fold_metrics_df["model"] == "naive"]
    assert not seasonal_rows.empty
    assert not naive_rows.empty
    assert seasonal_rows["seasonal_period"].astype(int).tolist() == [52] * len(seasonal_rows)
    assert seasonal_rows["mae"].tolist() == pytest.approx(naive_rows["mae"].tolist())
    assert seasonal_rows["rmse"].tolist() == pytest.approx(naive_rows["rmse"].tolist())
    assert seasonal_rows["mse"].tolist() == pytest.approx(naive_rows["mse"].tolist())
    assert seasonal_rows["mape"].tolist() == pytest.approx(naive_rows["mape"].tolist())
    assert seasonal_rows["smape"].tolist() == pytest.approx(naive_rows["smape"].tolist())
    assert seasonal_rows["wape"].tolist() == pytest.approx(naive_rows["wape"].tolist())
    assert seasonal_rows["directional_accuracy"].tolist() == pytest.approx(
        naive_rows["directional_accuracy"].tolist()
    )
    assert seasonal_rows["quantile_coverage_error"].isna().all()
    assert naive_rows["quantile_coverage_error"].isna().all()


def test_run_timesfm_backtesting_framework_timesfm_qce_with_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=14, freq="D"),
            "ticker": ["AAPL"] * 14,
            "value": [float(i) for i in range(14)],
        }
    )

    def fake_run_timesfm_forecast(**kwargs: object) -> object:
        history_df = kwargs["history_df"]
        horizon = int(kwargs["horizon"])
        step = timedelta(days=1)
        start = history_df["timestamp"].iloc[-1]
        return type(
            "FakeForecast",
            (),
            {
                "forecast_df": pd.DataFrame(
                    {
                        "timestamp": [start + step * (i + 1) for i in range(horizon)],
                        "prediction": [history_df["value"].iloc[-1]] * horizon,
                        "p10": [-1_000_000.0] * horizon,
                        "p90": [1_000_000.0] * horizon,
                    }
                )
            },
        )()

    monkeypatch.setattr("app_core.timesfm_advanced.run_timesfm_forecast", fake_run_timesfm_forecast)
    result = run_timesfm_backtesting_framework(
        panel_df=panel,
        mode="walk_forward",
        folds=2,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=True,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        min_train_size=6,
        rolling_window=6,
    )

    timesfm_rows = result.fold_metrics_df[result.fold_metrics_df["model"] == "timesfm"]
    baseline_rows = result.fold_metrics_df[result.fold_metrics_df["model"] != "timesfm"]
    assert not timesfm_rows.empty
    assert timesfm_rows["quantile_coverage_error"].tolist() == pytest.approx([20.0] * len(timesfm_rows))
    assert baseline_rows["quantile_coverage_error"].isna().all()
