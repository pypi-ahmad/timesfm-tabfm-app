"""Tests for tabular and time-series input validation."""

from __future__ import annotations

import pandas as pd
import pytest

from app_core.validators import (
    ValidationError,
    detect_tabfm_task,
    validate_tabfm_inputs,
    validate_timesfm_input,
)


def test_validate_tabfm_inputs_returns_aligned_train_and_predict_features() -> None:
    train_df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "job": ["eng", "mgr", "eng"],
            "target": ["low", "high", "low"],
        }
    )
    predict_df = pd.DataFrame({"job": ["eng"], "age": [35]})

    x_train, y_train, x_predict = validate_tabfm_inputs(
        train_df=train_df,
        predict_df=predict_df,
        target_column="target",
    )

    assert list(x_train.columns) == ["age", "job"]
    assert list(x_predict.columns) == ["age", "job"]
    assert y_train.tolist() == ["low", "high", "low"]


def test_validate_tabfm_inputs_raises_when_predict_feature_is_missing() -> None:
    train_df = pd.DataFrame(
        {
            "age": [20, 30],
            "job": ["eng", "mgr"],
            "target": ["low", "high"],
        }
    )
    predict_df = pd.DataFrame({"age": [28]})

    with pytest.raises(ValidationError, match="Missing required feature columns"):
        validate_tabfm_inputs(
            train_df=train_df,
            predict_df=predict_df,
            target_column="target",
        )


def test_detect_tabfm_task_infers_classification_for_non_numeric_target() -> None:
    y_train = pd.Series(["a", "b", "a", "c"])
    assert detect_tabfm_task(y_train) == "classification"


def test_detect_tabfm_task_infers_regression_for_continuous_numeric_target() -> None:
    y_train = pd.Series([10.1, 10.7, 11.9, 14.2, 17.5])
    assert detect_tabfm_task(y_train) == "regression"


def test_validate_timesfm_input_parses_sorts_and_coerces_values() -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "value": ["3.0", "1.0", "2.0"],
        }
    )

    validated = validate_timesfm_input(df)

    assert validated["timestamp"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert validated["value"].tolist() == [1.0, 2.0, 3.0]

