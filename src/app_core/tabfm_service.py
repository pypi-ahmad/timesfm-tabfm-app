"""Service wrapper for TabFM classification and regression."""

from __future__ import annotations

from dataclasses import dataclass
import os
from importlib import import_module
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from app_core.batch_io import BatchInputItem
from app_core.batch_models import BatchFileResult, BatchRunResult
from app_core.validators import detect_tabfm_task, preprocess_tabular_features, validate_tabfm_inputs


class TabFMRuntimeError(RuntimeError):
    """Raised when TabFM runtime dependencies are unavailable."""


@dataclass(frozen=True)
class TabFMRunResult:
    """TabFM execution output for app rendering and download."""

    task: str
    output_df: pd.DataFrame


def _load_tabfm_symbols() -> tuple[Any, Any, Any]:
    """Load TabFM classes lazily so tests can run without the dependency."""
    try:
        tabfm_module = import_module("tabfm")
        tabfm_backend = getattr(tabfm_module, "tabfm_v1_0_0_pytorch")
    except Exception as exc:  # pragma: no cover - exercised when env lacks model deps
        raise TabFMRuntimeError(
            "TabFM is not installed. Install model dependencies before running predictions."
        ) from exc

    return tabfm_module.TabFMClassifier, tabfm_module.TabFMRegressor, tabfm_backend


def _maybe_resolve_local_tabfm_checkpoint(model_type: str) -> str | None:
    """Resolve a local Hugging Face snapshot path for TabFM if already cached.

    This avoids network calls in environments where model weights have already
    been warmed (or where outbound network is restricted).
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return None

    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    try:
        return snapshot_download(
            repo_id="google/tabfm-1.0.0-pytorch",
            allow_patterns=[f"{model_type}/**"],
            cache_dir=cache_dir,
            local_files_only=True,
            max_workers=1,
        )
    except Exception:
        return None


def _load_tabfm_backend_model(backend: Any, model_type: str) -> Any:
    """Load a TabFM model, preferring local cached checkpoints when available."""
    checkpoint_path = _maybe_resolve_local_tabfm_checkpoint(model_type)
    if checkpoint_path:
        try:
            return backend.load(model_type=model_type, checkpoint_path=checkpoint_path)
        except TypeError:
            # Test doubles or older backends may not accept checkpoint_path.
            return backend.load(model_type=model_type)

    try:
        return backend.load(model_type=model_type)
    except TypeError:
        # Some backends may default to classification if no model_type is supported.
        return backend.load()


def run_tabfm_prediction(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    target_column: str,
    task_mode: str = "auto",
) -> TabFMRunResult:
    """Run TabFM prediction for classification or regression."""
    x_train, y_train, x_predict = validate_tabfm_inputs(
        train_df=train_df,
        predict_df=predict_df,
        target_column=target_column,
    )
    x_train = preprocess_tabular_features(x_train)
    x_predict = preprocess_tabular_features(x_predict)

    resolved_task = detect_tabfm_task(y_train) if task_mode == "auto" else task_mode
    classifier_cls, regressor_cls, backend = _load_tabfm_symbols()

    output_df = x_predict.copy()
    if resolved_task == "classification":
        model = _load_tabfm_backend_model(backend=backend, model_type="classification")
        estimator = classifier_cls(model=model)
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_predict)
        probabilities = estimator.predict_proba(x_predict)
        output_df["prediction"] = predictions
        output_df["confidence"] = np.max(probabilities, axis=1)
    else:
        model = _load_tabfm_backend_model(backend=backend, model_type="regression")
        estimator = regressor_cls(model=model)
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_predict)
        output_df["prediction"] = predictions

    return TabFMRunResult(task=resolved_task, output_df=output_df)


def run_tabfm_batch(
    train_df: pd.DataFrame,
    predict_items: list[BatchInputItem],
    target_column: str,
    task_mode: str,
    retry_count: int = 1,
) -> BatchRunResult:
    """Run TabFM predictions for many input files with bounded retries."""
    results: list[BatchFileResult] = []
    for item in predict_items:
        start = perf_counter()
        last_error = ""
        attempts = 0
        for attempt in range(1, retry_count + 2):
            attempts = attempt
            try:
                predict_df = item.dataframe.copy()
                predict_df.attrs["batch_file_name"] = item.name
                result = run_tabfm_prediction(
                    train_df=train_df,
                    predict_df=predict_df,
                    target_column=target_column,
                    task_mode=task_mode,
                )
                results.append(
                    BatchFileResult(
                        file_name=item.name,
                        status="success",
                        attempts=attempts,
                        message="OK",
                        duration_seconds=perf_counter() - start,
                        output_df=result.output_df,
                        task=result.task,
                    )
                )
                break
            except Exception as exc:  # pragma: no cover - covered by tests with monkeypatch
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
