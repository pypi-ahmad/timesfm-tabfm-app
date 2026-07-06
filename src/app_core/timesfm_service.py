"""Service wrapper for TimesFM forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from app_core.batch_io import BatchInputItem
from app_core.batch_models import BatchFileResult, BatchRunResult
from app_core.metrics import compute_regression_metrics
from app_core.validators import ValidationError, validate_timesfm_input


class TimesFMRuntimeError(RuntimeError):
    """Raised when TimesFM runtime dependencies are unavailable."""


SUPPORTED_BACKENDS = ("torch", "jax")
DEFAULT_TIMESFM_MODEL_IDS: dict[str, str] = {
    "torch": "google/timesfm-2.5-200m-pytorch",
    "jax": "google/timesfm-2.5-200m-flax",
}
BACKEND_CLASS_NAMES: dict[str, str] = {
    "torch": "TimesFM_2p5_200M_torch",
    "jax": "TimesFM_2p5_200M_flax",
}
SUPPORTED_XREG_MODES = ("xreg + timesfm", "timesfm + xreg")


@dataclass(frozen=True)
class TimesFMForecastResult:
    """Forecast output for UI rendering and downloads."""

    forecast_df: pd.DataFrame


@dataclass(frozen=True)
class TimesFMBacktestResult:
    """Backtest output with holdout comparison and metrics."""

    comparison_df: pd.DataFrame
    metrics: dict[str, float | None]


def _load_timesfm_module() -> Any:
    """Load TimesFM lazily to keep tests independent of heavy model deps."""
    try:
        return import_module("timesfm")
    except Exception as exc:  # pragma: no cover - exercised when env lacks model deps
        raise TimesFMRuntimeError(
            "TimesFM is not installed. Install model dependencies before forecasting."
        ) from exc


def _normalize_backend(backend: str) -> str:
    """Validate and normalize backend selection."""
    normalized = backend.strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        raise ValidationError(
            f"TimesFM backend must be one of {', '.join(SUPPORTED_BACKENDS)}."
        )
    return normalized


def _resolve_timesfm_model_class(timesfm: Any, backend: str) -> Any:
    """Resolve the backend-specific TimesFM class and return actionable errors."""
    class_name = BACKEND_CLASS_NAMES[backend]
    if hasattr(timesfm, class_name):
        return getattr(timesfm, class_name)

    dependency_hint = "timesfm[flax]" if backend == "jax" else "timesfm[torch]"
    raise TimesFMRuntimeError(
        f"TimesFM backend '{backend}' is unavailable. Install `{dependency_hint}` and retry."
    )


def _resolve_model_id(model_id: str, backend: str) -> str:
    """Resolve explicit model ID or fallback to backend defaults."""
    resolved_model_id = model_id.strip()
    if resolved_model_id:
        return resolved_model_id
    return DEFAULT_TIMESFM_MODEL_IDS[backend]


def _apply_lora_adapter(model: Any, adapter_path: str) -> None:
    """Attach LoRA adapter using whichever loader API the runtime exposes."""
    resolved_path = Path(adapter_path).expanduser().resolve()
    if not resolved_path.exists():
        raise TimesFMRuntimeError(f"LoRA adapter path does not exist: {resolved_path}")

    for method_name in ("load_lora_adapter", "load_adapter", "load_adapters"):
        if not hasattr(model, method_name):
            continue
        method = getattr(model, method_name)
        try:
            method(str(resolved_path))
        except Exception as exc:
            raise TimesFMRuntimeError(
                f"Failed to load LoRA adapter from '{resolved_path}' using {method_name}."
            ) from exc
        return

    raise TimesFMRuntimeError(
        "Selected TimesFM runtime does not expose adapter loading APIs "
        "(expected one of: load_lora_adapter, load_adapter, load_adapters)."
    )


def _normalize_xreg_mode(xreg_mode: str) -> str:
    """Validate and normalize XReg execution mode."""
    normalized = xreg_mode.strip().lower()
    if normalized not in SUPPORTED_XREG_MODES:
        raise ValidationError(
            "XReg mode must be one of: 'xreg + timesfm', 'timesfm + xreg'."
        )
    return normalized


def _infer_step(history: pd.Series) -> timedelta:
    """Infer timestamp delta from history and default to daily if ambiguous."""
    inferred = pd.infer_freq(history)
    if inferred:
        offset = pd.tseries.frequencies.to_offset(inferred)
        return timedelta(seconds=offset.nanos / 1_000_000_000)

    diffs = history.diff().dropna()
    if diffs.empty:
        return timedelta(days=1)
    return diffs.median().to_pytimedelta()


def _prepare_dynamic_numerical_covariates(
    history_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    covariate_columns: list[str] | None,
    horizon: int,
) -> dict[str, list[np.ndarray]]:
    """Build TimesFM dynamic numerical covariates aligned to context + horizon."""
    if "timestamp" not in covariates_df.columns:
        raise ValidationError("XReg covariates must include a 'timestamp' column.")

    normalized_covariates = covariates_df.copy()
    normalized_covariates["timestamp"] = pd.to_datetime(
        normalized_covariates["timestamp"],
        errors="coerce",
    )
    normalized_covariates = normalized_covariates.dropna(subset=["timestamp"]).sort_values(
        "timestamp"
    )
    if normalized_covariates.empty:
        raise ValidationError("XReg covariates are empty after parsing timestamps.")

    requested_columns = (
        [column for column in (covariate_columns or []) if column in normalized_covariates.columns]
        if covariate_columns is not None
        else [
            column
            for column in normalized_covariates.columns
            if column not in {"timestamp", "ticker", "value"}
        ]
    )
    if not requested_columns:
        raise ValidationError("No usable XReg covariate columns were provided.")

    covariates_by_timestamp = normalized_covariates[
        ["timestamp", *requested_columns]
    ].drop_duplicates(subset=["timestamp"], keep="last")
    aligned = history_df.merge(covariates_by_timestamp, on="timestamp", how="left")

    dynamic_numerical_covariates: dict[str, list[np.ndarray]] = {}
    for covariate in requested_columns:
        values = pd.to_numeric(aligned[covariate], errors="coerce").ffill().bfill()
        if values.isna().all():
            continue
        history_values = values.astype(float).to_numpy()
        future_values = np.repeat(history_values[-1], horizon)
        dynamic_numerical_covariates[covariate] = [
            np.concatenate([history_values, future_values])
        ]

    if not dynamic_numerical_covariates:
        raise ValidationError("No numeric XReg covariate values were available after alignment.")
    return dynamic_numerical_covariates


def _forecast_with_covariates(
    model: Any,
    history_values: np.ndarray,
    dynamic_numerical_covariates: dict[str, list[np.ndarray]],
    ticker: str,
    horizon: int,
    xreg_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    kwargs: dict[str, Any] = {
        "inputs": [history_values],
        "dynamic_numerical_covariates": dynamic_numerical_covariates,
        "dynamic_categorical_covariates": {},
        "static_categorical_covariates": {"ticker": [ticker]},
        "xreg_mode": xreg_mode,
    }
    try:
        return model.forecast_with_covariates(horizon=horizon, **kwargs)
    except TypeError:
        return model.forecast_with_covariates(**kwargs)


def run_timesfm_forecast(
    history_df: pd.DataFrame,
    horizon: int,
    max_context: int,
    max_horizon: int,
    normalize_inputs: bool,
    include_quantiles: bool,
    model_id: str,
    backend: str = "torch",
    use_xreg: bool = False,
    ticker: str | None = None,
    covariates_df: pd.DataFrame | None = None,
    covariate_columns: list[str] | None = None,
    xreg_mode: str = "xreg + timesfm",
    lora_adapter_path: str | None = None,
) -> TimesFMForecastResult:
    """Run TimesFM forecast on validated univariate history."""
    normalized_history = validate_timesfm_input(history_df)
    resolved_backend = _normalize_backend(backend)
    timesfm = _load_timesfm_module()
    model_class = _resolve_timesfm_model_class(timesfm, resolved_backend)
    resolved_model_id = _resolve_model_id(model_id, resolved_backend)
    normalized_xreg_mode = _normalize_xreg_mode(xreg_mode)
    try:
        model = model_class.from_pretrained(resolved_model_id)
    except Exception as exc:  # pragma: no cover - runtime import and hardware specific
        dependency_hint = "timesfm[flax]" if resolved_backend == "jax" else "timesfm[torch]"
        raise TimesFMRuntimeError(
            f"Failed to initialize TimesFM backend '{resolved_backend}' with model "
            f"'{resolved_model_id}'. Ensure `{dependency_hint}` is installed."
        ) from exc
    if lora_adapter_path:
        _apply_lora_adapter(model=model, adapter_path=lora_adapter_path)

    forecast_config_kwargs: dict[str, Any] = {
        "max_context": max_context,
        "max_horizon": max_horizon,
        "normalize_inputs": normalize_inputs,
        "use_continuous_quantile_head": include_quantiles,
        "force_flip_invariance": True,
        "infer_is_positive": False,
        "fix_quantile_crossing": include_quantiles,
    }
    if use_xreg:
        forecast_config_kwargs["return_backcast"] = True
    try:
        forecast_config = timesfm.ForecastConfig(**forecast_config_kwargs)
    except TypeError:
        # Older TimesFM variants may not accept return_backcast in ForecastConfig.
        forecast_config_kwargs.pop("return_backcast", None)
        forecast_config = timesfm.ForecastConfig(**forecast_config_kwargs)
    model.compile(forecast_config)

    if use_xreg:
        resolved_ticker = (ticker or "").strip().upper()
        if not resolved_ticker:
            raise ValidationError("Ticker is required when XReg covariates are enabled.")
        if covariates_df is None:
            raise ValidationError("XReg covariates are required when XReg is enabled.")
        dynamic_numerical_covariates = _prepare_dynamic_numerical_covariates(
            history_df=normalized_history,
            covariates_df=covariates_df,
            covariate_columns=covariate_columns,
            horizon=horizon,
        )
        try:
            point_forecast, quantile_forecast = _forecast_with_covariates(
                model=model,
                history_values=normalized_history["value"].astype(float).to_numpy(),
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                ticker=resolved_ticker,
                horizon=horizon,
                xreg_mode=normalized_xreg_mode,
            )
        except Exception as exc:
            raise TimesFMRuntimeError(
                "TimesFM XReg inference failed. Install `timesfm[xreg]` and verify "
                "covariates align with history timestamps."
            ) from exc
    else:
        point_forecast, quantile_forecast = model.forecast(
            horizon=horizon,
            inputs=[normalized_history["value"].astype(float).to_numpy()],
        )

    step = _infer_step(normalized_history["timestamp"])
    start_ts = normalized_history["timestamp"].iloc[-1]
    future_timestamps = [start_ts + step * (index + 1) for index in range(horizon)]

    forecast_df = pd.DataFrame(
        {
            "timestamp": future_timestamps,
            "prediction": point_forecast[0].astype(float).tolist(),
        }
    )

    if include_quantiles and quantile_forecast.shape[-1] >= 3:
        forecast_df["p10"] = quantile_forecast[0, :, 1].astype(float).tolist()
        forecast_df["p90"] = quantile_forecast[0, :, -1].astype(float).tolist()

    return TimesFMForecastResult(forecast_df=forecast_df)


def run_timesfm_backtest(
    history_df: pd.DataFrame,
    holdout_points: int,
    max_context: int,
    max_horizon: int,
    normalize_inputs: bool,
    include_quantiles: bool,
    model_id: str,
    backend: str = "torch",
    use_xreg: bool = False,
    ticker: str | None = None,
    covariates_df: pd.DataFrame | None = None,
    covariate_columns: list[str] | None = None,
    xreg_mode: str = "xreg + timesfm",
    lora_adapter_path: str | None = None,
) -> TimesFMBacktestResult:
    """Run rolling holdout backtest and return metrics."""
    normalized_history = validate_timesfm_input(history_df)
    if holdout_points < 1:
        raise ValidationError("Holdout points must be at least 1.")
    if holdout_points >= len(normalized_history):
        raise ValidationError("Holdout points must be less than total history length.")

    train_history = normalized_history.iloc[:-holdout_points].copy()
    holdout_history = normalized_history.iloc[-holdout_points:].copy().reset_index(drop=True)
    train_covariates: pd.DataFrame | None = None
    if use_xreg and covariates_df is not None:
        normalized_covariates = covariates_df.copy()
        if "timestamp" in normalized_covariates.columns:
            normalized_covariates["timestamp"] = pd.to_datetime(
                normalized_covariates["timestamp"],
                errors="coerce",
            )
            normalized_covariates = normalized_covariates.dropna(subset=["timestamp"])
            train_end = train_history["timestamp"].iloc[-1]
            train_covariates = normalized_covariates[
                normalized_covariates["timestamp"] <= train_end
            ].copy()

    forecast_result = run_timesfm_forecast(
        history_df=train_history,
        horizon=holdout_points,
        max_context=max_context,
        max_horizon=max_horizon,
        normalize_inputs=normalize_inputs,
        include_quantiles=include_quantiles,
        model_id=model_id,
        backend=backend,
        use_xreg=use_xreg,
        ticker=ticker,
        covariates_df=train_covariates if use_xreg else None,
        covariate_columns=covariate_columns,
        xreg_mode=xreg_mode,
        lora_adapter_path=lora_adapter_path,
    )
    forecast_df = forecast_result.forecast_df.reset_index(drop=True)

    comparison_df = pd.DataFrame(
        {
            "timestamp": holdout_history["timestamp"],
            "actual": holdout_history["value"].astype(float),
            "prediction": forecast_df["prediction"].astype(float),
        }
    )
    if include_quantiles and "p10" in forecast_df.columns and "p90" in forecast_df.columns:
        comparison_df["p10"] = forecast_df["p10"].astype(float)
        comparison_df["p90"] = forecast_df["p90"].astype(float)

    lower_quantile = comparison_df["p10"].tolist() if "p10" in comparison_df.columns else None
    upper_quantile = comparison_df["p90"].tolist() if "p90" in comparison_df.columns else None
    metrics = compute_regression_metrics(
        actual=comparison_df["actual"].tolist(),
        predicted=comparison_df["prediction"].tolist(),
        direction_anchor=float(train_history["value"].iloc[-1]),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    return TimesFMBacktestResult(comparison_df=comparison_df, metrics=metrics)


def run_timesfm_batch(
    history_items: list[BatchInputItem],
    horizon: int,
    max_context: int,
    max_horizon: int,
    normalize_inputs: bool,
    include_quantiles: bool,
    model_id: str,
    backend: str = "torch",
    retry_count: int = 1,
    run_backtest: bool = False,
    holdout_points: int = 3,
    use_xreg: bool = False,
    ticker_by_file: dict[str, str] | None = None,
    covariates_df_by_file: dict[str, pd.DataFrame] | None = None,
    covariate_columns: list[str] | None = None,
    xreg_mode: str = "xreg + timesfm",
    lora_adapter_path: str | None = None,
) -> BatchRunResult:
    """Run TimesFM batch forecasting with optional per-file backtesting."""
    results: list[BatchFileResult] = []
    for item in history_items:
        start = perf_counter()
        last_error = ""
        attempts = 0
        resolved_ticker = (
            (ticker_by_file or {}).get(item.name, Path(item.name).stem.strip().upper())
            if use_xreg
            else None
        )
        resolved_covariates_df = (
            (covariates_df_by_file or {}).get(item.name)
            if use_xreg
            else None
        )
        for attempt in range(1, retry_count + 2):
            attempts = attempt
            try:
                forecast = run_timesfm_forecast(
                    history_df=item.dataframe,
                    horizon=horizon,
                    max_context=max_context,
                    max_horizon=max_horizon,
                    normalize_inputs=normalize_inputs,
                    include_quantiles=include_quantiles,
                    model_id=model_id,
                    backend=backend,
                    use_xreg=use_xreg,
                    ticker=resolved_ticker,
                    covariates_df=resolved_covariates_df,
                    covariate_columns=covariate_columns,
                    xreg_mode=xreg_mode,
                    lora_adapter_path=lora_adapter_path,
                )
                metrics = None
                comparison_df = None
                if run_backtest:
                    backtest = run_timesfm_backtest(
                        history_df=item.dataframe,
                        holdout_points=holdout_points,
                        max_context=max_context,
                        max_horizon=max_horizon,
                        normalize_inputs=normalize_inputs,
                        include_quantiles=include_quantiles,
                        model_id=model_id,
                        backend=backend,
                        use_xreg=use_xreg,
                        ticker=resolved_ticker,
                        covariates_df=resolved_covariates_df,
                        covariate_columns=covariate_columns,
                        xreg_mode=xreg_mode,
                        lora_adapter_path=lora_adapter_path,
                    )
                    metrics = backtest.metrics
                    comparison_df = backtest.comparison_df

                results.append(
                    BatchFileResult(
                        file_name=item.name,
                        status="success",
                        attempts=attempts,
                        message="OK",
                        duration_seconds=perf_counter() - start,
                        output_df=forecast.forecast_df,
                        metrics=metrics,
                        comparison_df=comparison_df,
                    )
                )
                break
            except Exception as exc:  # pragma: no cover - covered by tests via monkeypatch
                last_error = str(exc)
                if attempt > retry_count:
                    results.append(
                        BatchFileResult(
                            file_name=item.name,
                            status="failed",
                            attempts=attempts,
                            message=last_error,
                            duration_seconds=perf_counter() - start,
                        )
                    )

    return BatchRunResult(results=results)
