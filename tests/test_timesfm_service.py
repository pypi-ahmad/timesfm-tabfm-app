"""Tests for TimesFM service wrapper behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app_core.batch_io import BatchInputItem
from app_core.timesfm_service import (
    TimesFMRuntimeError,
    run_timesfm_backtest,
    run_timesfm_batch,
    run_timesfm_forecast,
)
from app_core.validators import ValidationError


@dataclass
class FakeForecastConfig:
    max_context: int
    max_horizon: int
    normalize_inputs: bool
    use_continuous_quantile_head: bool
    force_flip_invariance: bool
    infer_is_positive: bool
    fix_quantile_crossing: bool


class FakeTimesFMModel:
    loaded_adapter_path: str | None = None

    def __init__(self) -> None:
        self.compiled = False

    def compile(self, config: FakeForecastConfig) -> None:
        self.compiled = True

    def load_adapter(self, adapter_path: str) -> None:
        FakeTimesFMModel.loaded_adapter_path = adapter_path

    def forecast(self, horizon: int, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        point = np.array([[11.0, 12.0, 13.0]])
        # mean + p10..p90
        quantiles = np.array(
            [
                [
                    [11.0, 10.0, 10.5, 10.8, 11.0, 11.1, 11.2, 11.5, 11.8, 12.0],
                    [12.0, 11.0, 11.5, 11.8, 12.0, 12.1, 12.2, 12.5, 12.8, 13.0],
                    [13.0, 12.0, 12.5, 12.8, 13.0, 13.1, 13.2, 13.5, 13.8, 14.0],
                ]
            ]
        )
        return point[:, :horizon], quantiles[:, :horizon, :]


class FakeTimesFMClass:
    @staticmethod
    def from_pretrained(model_id: str) -> FakeTimesFMModel:
        return FakeTimesFMModel()


class FakeTimesFMNamespace:
    TimesFM_2p5_200M_torch = FakeTimesFMClass
    TimesFM_2p5_200M_flax = FakeTimesFMClass
    ForecastConfig = FakeForecastConfig


def _fake_load_timesfm() -> FakeTimesFMNamespace:
    return FakeTimesFMNamespace()


def test_run_timesfm_forecast_builds_future_timestamps_and_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app_core.timesfm_service._load_timesfm_module", _fake_load_timesfm)
    history_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    result = run_timesfm_forecast(
        history_df=history_df,
        horizon=3,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=True,
        model_id="google/timesfm-2.5-200m-pytorch",
    )

    assert result.forecast_df.shape[0] == 3
    assert list(result.forecast_df.columns) == ["timestamp", "prediction", "p10", "p90"]
    assert result.forecast_df["prediction"].tolist() == [11.0, 12.0, 13.0]


def test_run_timesfm_backtest_returns_metrics_and_holdout_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app_core.timesfm_service._load_timesfm_module", _fake_load_timesfm)
    history_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 11.0, 12.0, 13.0],
        }
    )

    result = run_timesfm_backtest(
        history_df=history_df,
        holdout_points=3,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=True,
        model_id="google/timesfm-2.5-200m-pytorch",
    )

    assert list(result.comparison_df.columns) == ["timestamp", "actual", "prediction", "p10", "p90"]
    assert result.metrics["mae"] == pytest.approx(0.0)
    assert result.metrics["rmse"] == pytest.approx(0.0)
    assert result.metrics["mse"] == pytest.approx(0.0)
    assert result.metrics["mape"] == pytest.approx(0.0)
    assert result.metrics["smape"] == pytest.approx(0.0)
    assert result.metrics["wape"] == pytest.approx(0.0)
    assert result.metrics["directional_accuracy"] == pytest.approx(100.0)
    assert result.metrics["quantile_coverage_error"] == pytest.approx(20.0)


def test_run_timesfm_batch_with_optional_backtest(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_backends: list[str] = []

    def fake_run_timesfm_forecast(**kwargs: object) -> object:
        seen_backends.append(str(kwargs["backend"]))
        return type(
            "FakeForecast",
            (),
            {"forecast_df": pd.DataFrame({"timestamp": pd.date_range("2024-01-10", periods=2), "prediction": [11.0, 12.0]})},
        )()

    def fake_run_timesfm_backtest(**kwargs: object) -> object:
        seen_backends.append(str(kwargs["backend"]))
        return type(
            "FakeBacktest",
            (),
            {
                "comparison_df": pd.DataFrame(
                    {"timestamp": pd.date_range("2024-01-08", periods=2), "actual": [10.0, 11.0], "prediction": [10.5, 11.5]}
                ),
                "metrics": {"mae": 0.5, "rmse": 0.5, "mape": 4.5},
            },
        )()

    monkeypatch.setattr("app_core.timesfm_service.run_timesfm_forecast", fake_run_timesfm_forecast)
    monkeypatch.setattr("app_core.timesfm_service.run_timesfm_backtest", fake_run_timesfm_backtest)

    items = [
        BatchInputItem(
            name="series_1.csv",
            dataframe=pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=8), "value": [1, 2, 3, 4, 5, 6, 7, 8]}),
        )
    ]

    result = run_timesfm_batch(
        history_items=items,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=False,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="jax",
        retry_count=1,
        run_backtest=True,
        holdout_points=2,
    )

    assert len(result.results) == 1
    assert result.results[0].status == "success"
    assert result.results[0].metrics == {"mae": 0.5, "rmse": 0.5, "mape": 4.5}
    assert seen_backends == ["jax", "jax"]


def test_run_timesfm_forecast_uses_jax_backend_when_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFlaxTimesFMClass:
        seen_model_id: str | None = None

        @staticmethod
        def from_pretrained(model_id: str) -> FakeTimesFMModel:
            FakeFlaxTimesFMClass.seen_model_id = model_id

            class FakeFlaxModel(FakeTimesFMModel):
                def forecast(
                    self,
                    horizon: int,
                    inputs: list[np.ndarray],
                ) -> tuple[np.ndarray, np.ndarray]:
                    point = np.array([[21.0, 22.0, 23.0]])
                    quantiles = np.array([[[21.0] * 10, [22.0] * 10, [23.0] * 10]])
                    return point[:, :horizon], quantiles[:, :horizon, :]

            return FakeFlaxModel()

    class FakeTimesFMNamespaceWithFlax:
        TimesFM_2p5_200M_torch = FakeTimesFMClass
        TimesFM_2p5_200M_flax = FakeFlaxTimesFMClass
        ForecastConfig = FakeForecastConfig

    monkeypatch.setattr(
        "app_core.timesfm_service._load_timesfm_module",
        lambda: FakeTimesFMNamespaceWithFlax(),
    )
    history_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    result = run_timesfm_forecast(
        history_df=history_df,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=False,
        model_id="google/timesfm-2.5-200m-flax",
        backend="jax",
    )

    assert result.forecast_df["prediction"].tolist() == [21.0, 22.0]
    assert FakeFlaxTimesFMClass.seen_model_id == "google/timesfm-2.5-200m-flax"


def test_run_timesfm_forecast_raises_actionable_error_when_jax_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTimesFMTorchOnlyNamespace:
        TimesFM_2p5_200M_torch = FakeTimesFMClass
        ForecastConfig = FakeForecastConfig

    monkeypatch.setattr(
        "app_core.timesfm_service._load_timesfm_module",
        lambda: FakeTimesFMTorchOnlyNamespace(),
    )
    history_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    with pytest.raises(TimesFMRuntimeError, match=r"Install `timesfm\[flax\]`"):
        run_timesfm_forecast(
            history_df=history_df,
            horizon=2,
            max_context=128,
            max_horizon=256,
            normalize_inputs=True,
            include_quantiles=False,
            model_id="google/timesfm-2.5-200m-flax",
            backend="jax",
        )


def test_run_timesfm_forecast_loads_lora_adapter_when_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("app_core.timesfm_service._load_timesfm_module", _fake_load_timesfm)
    history_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    adapter_dir = tmp_path / "adapter_dir"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    FakeTimesFMModel.loaded_adapter_path = None

    run_timesfm_forecast(
        history_df=history_df,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=False,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        lora_adapter_path=str(adapter_dir),
    )

    assert FakeTimesFMModel.loaded_adapter_path == str(adapter_dir.resolve())


def test_run_timesfm_forecast_raises_when_runtime_has_no_adapter_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeModelWithoutAdapter:
        def compile(self, config: FakeForecastConfig) -> None:
            _ = config

        def forecast(
            self,
            horizon: int,
            inputs: list[np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray]:
            _ = inputs
            return np.array([[1.0] * horizon]), np.array([[[1.0] * 10] * horizon])

    class FakeTimesFMClassWithoutAdapter:
        @staticmethod
        def from_pretrained(model_id: str) -> FakeModelWithoutAdapter:
            _ = model_id
            return FakeModelWithoutAdapter()

    class FakeNamespaceNoAdapter:
        TimesFM_2p5_200M_torch = FakeTimesFMClassWithoutAdapter
        ForecastConfig = FakeForecastConfig

    monkeypatch.setattr(
        "app_core.timesfm_service._load_timesfm_module",
        lambda: FakeNamespaceNoAdapter(),
    )
    history_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    adapter_dir = tmp_path / "adapter_dir"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(TimesFMRuntimeError, match="does not expose adapter loading APIs"):
        run_timesfm_forecast(
            history_df=history_df,
            horizon=2,
            max_context=128,
            max_horizon=256,
            normalize_inputs=True,
            include_quantiles=False,
            model_id="google/timesfm-2.5-200m-pytorch",
            backend="torch",
            lora_adapter_path=str(adapter_dir),
        )


def test_run_timesfm_forecast_with_xreg_uses_covariates_and_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeXRegTimesFMModel(FakeTimesFMModel):
        seen_kwargs: dict[str, object] = {}

        def forecast_with_covariates(self, horizon: int, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
            FakeXRegTimesFMModel.seen_kwargs = kwargs
            point = np.array([[31.0, 32.0, 33.0]])
            quantiles = np.array(
                [
                    [
                        [31.0, 30.0, 32.0],
                        [32.0, 31.0, 33.0],
                        [33.0, 32.0, 34.0],
                    ]
                ]
            )
            return point[:, :horizon], quantiles[:, :horizon, :]

    class FakeXRegTimesFMClass:
        @staticmethod
        def from_pretrained(model_id: str) -> FakeXRegTimesFMModel:
            return FakeXRegTimesFMModel()

    class FakeTimesFMNamespaceXReg:
        TimesFM_2p5_200M_torch = FakeXRegTimesFMClass
        TimesFM_2p5_200M_flax = FakeXRegTimesFMClass
        ForecastConfig = FakeForecastConfig

    monkeypatch.setattr(
        "app_core.timesfm_service._load_timesfm_module",
        lambda: FakeTimesFMNamespaceXReg(),
    )
    history_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            "value": [10.0, 11.0, 12.0, 13.0],
        }
    )
    covariates_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            "CPIAUCSL": [300.1, 300.2, 300.3, 300.4],
        }
    )

    result = run_timesfm_forecast(
        history_df=history_df,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=True,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        use_xreg=True,
        ticker="AAPL",
        covariates_df=covariates_df,
        covariate_columns=["CPIAUCSL"],
        xreg_mode="timesfm + xreg",
    )

    assert result.forecast_df["prediction"].tolist() == [31.0, 32.0]
    assert list(result.forecast_df.columns) == ["timestamp", "prediction", "p10", "p90"]
    assert (
        FakeXRegTimesFMModel.seen_kwargs["static_categorical_covariates"]  # type: ignore[index]
        == {"ticker": ["AAPL"]}
    )
    assert FakeXRegTimesFMModel.seen_kwargs["xreg_mode"] == "timesfm + xreg"
    dynamic_covariates = FakeXRegTimesFMModel.seen_kwargs["dynamic_numerical_covariates"]  # type: ignore[index]
    assert isinstance(dynamic_covariates, dict)
    assert "CPIAUCSL" in dynamic_covariates
    assert len(dynamic_covariates["CPIAUCSL"][0]) == len(history_df) + 2


def test_run_timesfm_forecast_with_xreg_requires_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app_core.timesfm_service._load_timesfm_module", _fake_load_timesfm)
    history_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            "value": [10.0, 11.0, 12.0, 13.0],
        }
    )
    covariates_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            "CPIAUCSL": [300.1, 300.2, 300.3, 300.4],
        }
    )

    with pytest.raises(ValidationError, match="Ticker is required"):
        run_timesfm_forecast(
            history_df=history_df,
            horizon=2,
            max_context=128,
            max_horizon=256,
            normalize_inputs=True,
            include_quantiles=False,
            model_id="google/timesfm-2.5-200m-pytorch",
            backend="torch",
            use_xreg=True,
            covariates_df=covariates_df,
        )


def test_run_timesfm_backtest_with_xreg_filters_covariates_to_train_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_covariates: pd.DataFrame | None = None

    def fake_run_timesfm_forecast(**kwargs: object) -> object:
        nonlocal seen_covariates
        seen_covariates = kwargs.get("covariates_df")  # type: ignore[assignment]
        history = kwargs["history_df"]  # type: ignore[index]
        horizon = int(kwargs["horizon"])  # type: ignore[index]
        last_value = float(history["value"].iloc[-1])
        return type(
            "FakeForecast",
            (),
            {
                "forecast_df": pd.DataFrame(
                    {
                        "timestamp": pd.date_range(
                            start=pd.to_datetime(history["timestamp"].iloc[-1]),
                            periods=horizon + 1,
                            freq="D",
                        )[1:],
                        "prediction": [last_value + idx + 1 for idx in range(horizon)],
                    }
                )
            },
        )()

    monkeypatch.setattr("app_core.timesfm_service.run_timesfm_forecast", fake_run_timesfm_forecast)
    history_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="D"),
            "value": [float(i) for i in range(1, 9)],
        }
    )
    covariates_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="D"),
            "macro_index": [100.0 + idx for idx in range(8)],
        }
    )

    result = run_timesfm_backtest(
        history_df=history_df,
        holdout_points=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=False,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        use_xreg=True,
        ticker="AAPL",
        covariates_df=covariates_df,
        covariate_columns=["macro_index"],
    )

    assert seen_covariates is not None
    assert seen_covariates["timestamp"].max() <= history_df["timestamp"].iloc[-3]
    assert result.metrics["mae"] == pytest.approx(0.0)
    assert result.metrics["quantile_coverage_error"] is None


def test_run_timesfm_batch_with_xreg_infers_ticker_and_passes_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_tickers: list[str] = []
    seen_modes: list[str] = []

    def fake_run_timesfm_forecast(**kwargs: object) -> object:
        seen_tickers.append(str(kwargs["ticker"]))
        seen_modes.append(str(kwargs["xreg_mode"]))
        assert kwargs["use_xreg"] is True
        assert kwargs["covariates_df"] is not None
        return type(
            "FakeForecast",
            (),
            {
                "forecast_df": pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2024-01-10", periods=2, freq="D"),
                        "prediction": [11.0, 12.0],
                    }
                )
            },
        )()

    def fake_run_timesfm_backtest(**kwargs: object) -> object:
        seen_tickers.append(str(kwargs["ticker"]))
        seen_modes.append(str(kwargs["xreg_mode"]))
        return type(
            "FakeBacktest",
            (),
            {
                "comparison_df": pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2024-01-08", periods=2, freq="D"),
                        "actual": [10.0, 11.0],
                        "prediction": [10.5, 11.5],
                    }
                ),
                "metrics": {"mae": 0.5, "rmse": 0.5, "mape": 4.5},
            },
        )()

    monkeypatch.setattr("app_core.timesfm_service.run_timesfm_forecast", fake_run_timesfm_forecast)
    monkeypatch.setattr("app_core.timesfm_service.run_timesfm_backtest", fake_run_timesfm_backtest)

    items = [
        BatchInputItem(
            name="aapl.csv",
            dataframe=pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=8, freq="D"),
                    "value": [1, 2, 3, 4, 5, 6, 7, 8],
                }
            ),
        )
    ]
    covariates_map = {
        "aapl.csv": pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=8, freq="D"),
                "macro_index": [200.0 + idx for idx in range(8)],
            }
        )
    }

    result = run_timesfm_batch(
        history_items=items,
        horizon=2,
        max_context=128,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=False,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        retry_count=1,
        run_backtest=True,
        holdout_points=2,
        use_xreg=True,
        covariates_df_by_file=covariates_map,
        covariate_columns=["macro_index"],
        xreg_mode="timesfm + xreg",
    )

    assert len(result.results) == 1
    assert result.results[0].status == "success"
    assert seen_tickers == ["AAPL", "AAPL"]
    assert seen_modes == ["timesfm + xreg", "timesfm + xreg"]
