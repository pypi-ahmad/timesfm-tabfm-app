"""Validation helpers for tabular and time-series inputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ValidationError(ValueError):
    """Raised when user-provided input data is invalid."""


@dataclass(frozen=True)
class TabFMValidatedInput:
    """Canonical validated inputs for TabFM."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_predict: pd.DataFrame


def detect_tabfm_task(y_train: pd.Series) -> str:
    """Infer whether target variable indicates classification or regression."""
    if not pd.api.types.is_numeric_dtype(y_train):
        return "classification"

    unique_count = y_train.nunique(dropna=True)
    if unique_count <= 10 and unique_count <= max(2, int(len(y_train) * 0.05)):
        return "classification"

    return "regression"


def validate_tabfm_inputs(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Validate schema and return aligned train/predict features."""
    if target_column not in train_df.columns:
        raise ValidationError(f"Target column '{target_column}' was not found in training data.")

    feature_columns = [column for column in train_df.columns if column != target_column]
    if not feature_columns:
        raise ValidationError("Training data must include at least one feature column.")

    missing_columns = [column for column in feature_columns if column not in predict_df.columns]
    if missing_columns:
        raise ValidationError(
            f"Missing required feature columns in prediction data: {missing_columns}"
        )

    x_train = train_df[feature_columns].copy()
    y_train = train_df[target_column].copy()
    x_predict = predict_df[feature_columns].copy()
    return x_train, y_train, x_predict


def preprocess_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Make a best-effort, model-friendly version of tabular features.

    - Numeric: coerce to numeric and fill missing values with column median.
    - Non-numeric: cast to string and fill missing values with "missing".
    """
    processed = df.copy()
    for col in processed.columns:
        series = processed[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                fill = float(numeric.median())
            else:
                fill = 0.0
            processed[col] = numeric.fillna(fill)
        else:
            processed[col] = series.astype(str).replace({"nan": "missing"}).fillna("missing")
    return processed


def validate_timesfm_input(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    value_column: str = "value",
) -> pd.DataFrame:
    """Validate and normalize univariate history for TimesFM forecasting."""
    if timestamp_column not in df.columns or value_column not in df.columns:
        raise ValidationError(
            f"Input data must include '{timestamp_column}' and '{value_column}' columns."
        )

    validated = df[[timestamp_column, value_column]].copy()
    validated[timestamp_column] = pd.to_datetime(validated[timestamp_column], errors="coerce")
    validated[value_column] = pd.to_numeric(validated[value_column], errors="coerce")
    validated = validated.dropna(subset=[timestamp_column, value_column])
    # If duplicates exist, keep the most recent row deterministically.
    validated = validated.drop_duplicates(subset=[timestamp_column], keep="last")
    validated = validated.sort_values(by=timestamp_column)

    if validated.empty:
        raise ValidationError("No valid rows remained after parsing timestamp and value columns.")

    validated = validated.rename(
        columns={timestamp_column: "timestamp", value_column: "value"}
    ).reset_index(drop=True)
    return validated
