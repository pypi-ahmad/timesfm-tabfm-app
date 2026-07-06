"""Tests for TabFM service orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app_core.batch_io import BatchInputItem
from app_core.tabfm_service import run_tabfm_batch, run_tabfm_prediction


class FakeClassifier:
    def __init__(self, model: object) -> None:
        self.model = model

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "FakeClassifier":
        return self

    def predict(self, x_predict: pd.DataFrame) -> np.ndarray:
        return np.array(["low", "high"])

    def predict_proba(self, x_predict: pd.DataFrame) -> np.ndarray:
        return np.array([[0.9, 0.1], [0.2, 0.8]])


class FakeRegressor:
    def __init__(self, model: object) -> None:
        self.model = model

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "FakeRegressor":
        return self

    def predict(self, x_predict: pd.DataFrame) -> np.ndarray:
        return np.array([120.5, 140.2])


def _fake_load_tabfm_symbols() -> tuple[type[FakeClassifier], type[FakeRegressor], object]:
    class FakeBackend:
        @staticmethod
        def load(model_type: str | None = None) -> str:
            return "fake-model" if model_type is None else f"fake-model-{model_type}"

    return FakeClassifier, FakeRegressor, FakeBackend


def test_run_tabfm_prediction_classification_includes_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app_core.tabfm_service._load_tabfm_symbols", _fake_load_tabfm_symbols)
    train_df = pd.DataFrame({"age": [20, 30], "target": ["low", "high"]})
    predict_df = pd.DataFrame({"age": [22, 35]})

    result = run_tabfm_prediction(
        train_df=train_df,
        predict_df=predict_df,
        target_column="target",
        task_mode="classification",
    )

    assert result.task == "classification"
    assert result.output_df["prediction"].tolist() == ["low", "high"]
    assert result.output_df["confidence"].tolist() == [0.9, 0.8]


def test_run_tabfm_prediction_regression_omits_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app_core.tabfm_service._load_tabfm_symbols", _fake_load_tabfm_symbols)
    train_df = pd.DataFrame({"age": [20, 30], "target": [100.0, 150.0]})
    predict_df = pd.DataFrame({"age": [22, 35]})

    result = run_tabfm_prediction(
        train_df=train_df,
        predict_df=predict_df,
        target_column="target",
        task_mode="regression",
    )

    assert result.task == "regression"
    assert result.output_df["prediction"].tolist() == [120.5, 140.2]
    assert "confidence" not in result.output_df.columns


def test_run_tabfm_batch_retries_once_and_keeps_partial_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"good.csv": 0, "bad.csv": 0}

    def fake_run_tabfm_prediction(
        train_df: pd.DataFrame,
        predict_df: pd.DataFrame,
        target_column: str,
        task_mode: str,
    ) -> object:
        file_name = predict_df.attrs["batch_file_name"]
        state[file_name] += 1
        if file_name == "good.csv" and state[file_name] == 1:
            raise RuntimeError("transient")
        if file_name == "bad.csv":
            raise RuntimeError("permanent")
        return type(
            "FakeResult",
            (),
            {"task": "classification", "output_df": pd.DataFrame({"prediction": ["low"]})},
        )()

    monkeypatch.setattr("app_core.tabfm_service.run_tabfm_prediction", fake_run_tabfm_prediction)
    train_df = pd.DataFrame({"age": [20, 30], "target": ["low", "high"]})
    predict_items = [
        BatchInputItem(name="good.csv", dataframe=pd.DataFrame({"age": [25]})),
        BatchInputItem(name="bad.csv", dataframe=pd.DataFrame({"age": [26]})),
    ]

    result = run_tabfm_batch(
        train_df=train_df,
        predict_items=predict_items,
        target_column="target",
        task_mode="classification",
        retry_count=1,
    )

    assert len(result.results) == 2
    assert result.results[0].status == "success"
    assert result.results[0].attempts == 2
    assert result.results[1].status == "failed"
    assert result.results[1].attempts == 2
