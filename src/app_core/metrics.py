"""Forecast metric utilities."""

from __future__ import annotations

import math
from typing import Iterable


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def compute_regression_metrics(
    actual: Iterable[float],
    predicted: Iterable[float],
    *,
    direction_anchor: float | None = None,
    lower_quantile: Iterable[float] | None = None,
    upper_quantile: Iterable[float] | None = None,
    target_coverage: float = 0.8,
) -> dict[str, float | None]:
    """Compute regression metrics, optional directional accuracy and quantile coverage error."""
    actual_values = [float(value) for value in actual]
    predicted_values = [float(value) for value in predicted]
    if len(actual_values) != len(predicted_values):
        raise ValueError("Actual and predicted series must have equal length.")
    if not actual_values:
        raise ValueError("Metrics require at least one point.")
    if not 0.0 <= float(target_coverage) <= 1.0:
        raise ValueError("target_coverage must be between 0.0 and 1.0.")

    absolute_errors = [abs(a - p) for a, p in zip(actual_values, predicted_values)]
    squared_errors = [(a - p) ** 2 for a, p in zip(actual_values, predicted_values)]
    percentage_errors = [
        abs((a - p) / a) * 100.0 for a, p in zip(actual_values, predicted_values) if a != 0
    ]
    smape_terms = []
    for a, p in zip(actual_values, predicted_values):
        denominator = abs(a) + abs(p)
        if denominator == 0.0:
            smape_terms.append(0.0)
            continue
        smape_terms.append((2.0 * abs(a - p) / denominator) * 100.0)

    mae = sum(absolute_errors) / len(absolute_errors)
    mse = sum(squared_errors) / len(squared_errors)
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    mape = sum(percentage_errors) / len(percentage_errors) if percentage_errors else 0.0
    smape = sum(smape_terms) / len(smape_terms)
    wape_denominator = sum(abs(a) for a in actual_values)
    wape = ((sum(absolute_errors) / wape_denominator) * 100.0) if wape_denominator != 0.0 else 0.0

    if direction_anchor is None:
        if len(actual_values) < 2:
            directional_accuracy = 0.0
        else:
            previous_actuals = actual_values[:-1]
            directional_actuals = actual_values[1:]
            directional_predictions = predicted_values[1:]
            direction_matches = [
                1.0
                if _sign(p - prev) == _sign(a - prev)
                else 0.0
                for a, p, prev in zip(
                    directional_actuals,
                    directional_predictions,
                    previous_actuals,
                )
            ]
            directional_accuracy = (
                (sum(direction_matches) / len(direction_matches)) * 100.0
                if direction_matches
                else 0.0
            )
    else:
        previous_actuals = [float(direction_anchor), *actual_values[:-1]]
        direction_matches = [
            1.0 if _sign(p - prev) == _sign(a - prev) else 0.0
            for a, p, prev in zip(actual_values, predicted_values, previous_actuals)
        ]
        directional_accuracy = (sum(direction_matches) / len(direction_matches)) * 100.0

    quantile_coverage_error: float | None = None
    if lower_quantile is not None or upper_quantile is not None:
        if lower_quantile is None or upper_quantile is None:
            raise ValueError("Both lower_quantile and upper_quantile must be provided together.")
        lower_values = [float(value) for value in lower_quantile]
        upper_values = [float(value) for value in upper_quantile]
        if len(lower_values) != len(actual_values) or len(upper_values) != len(actual_values):
            raise ValueError("Quantile bounds must match actual/predicted length.")
        inside_count = 0
        for actual_value, lower_value, upper_value in zip(actual_values, lower_values, upper_values):
            low = min(lower_value, upper_value)
            high = max(lower_value, upper_value)
            if low <= actual_value <= high:
                inside_count += 1
        observed_coverage = inside_count / len(actual_values)
        quantile_coverage_error = abs(observed_coverage - float(target_coverage)) * 100.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "mape": mape,
        "smape": smape,
        "wape": wape,
        "directional_accuracy": directional_accuracy,
        "quantile_coverage_error": quantile_coverage_error,
    }
