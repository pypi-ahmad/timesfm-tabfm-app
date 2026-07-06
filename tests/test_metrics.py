"""Tests for forecast evaluation metrics."""

from __future__ import annotations

import pytest

from app_core.metrics import compute_regression_metrics


def test_compute_regression_metrics_returns_expected_values() -> None:
    metrics = compute_regression_metrics(actual=[10.0, 12.0, 14.0], predicted=[9.0, 13.0, 15.0])

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mse"] == pytest.approx(1.0)
    assert metrics["mape"] == pytest.approx(8.492063, rel=1e-5)
    assert metrics["smape"] == pytest.approx(8.474289, rel=1e-5)
    assert metrics["wape"] == pytest.approx(8.333333, rel=1e-5)
    assert metrics["directional_accuracy"] == pytest.approx(100.0)
    assert metrics["quantile_coverage_error"] is None


def test_compute_regression_metrics_handles_zero_actuals_for_mape() -> None:
    metrics = compute_regression_metrics(actual=[0.0, 10.0], predicted=[1.0, 8.0])

    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(1.581138, rel=1e-5)
    assert metrics["mse"] == pytest.approx(2.5)
    assert metrics["mape"] == pytest.approx(20.0)
    assert metrics["smape"] == pytest.approx(111.111111, rel=1e-5)
    assert metrics["wape"] == pytest.approx(30.0)
    assert metrics["directional_accuracy"] == pytest.approx(100.0)
    assert metrics["quantile_coverage_error"] is None


def test_compute_regression_metrics_with_direction_anchor_and_quantiles() -> None:
    metrics = compute_regression_metrics(
        actual=[11.0, 12.0, 13.0],
        predicted=[11.0, 12.0, 13.0],
        direction_anchor=5.0,
        lower_quantile=[10.0, 11.0, 12.0],
        upper_quantile=[12.0, 13.0, 14.0],
    )

    assert metrics["directional_accuracy"] == pytest.approx(100.0)
    assert metrics["quantile_coverage_error"] == pytest.approx(20.0)
