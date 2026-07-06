"""Tests for batch summary model outputs."""

from __future__ import annotations

import pandas as pd

from app_core.batch_models import BatchFileResult, BatchRunResult


def test_batch_summary_includes_expanded_metric_columns() -> None:
    result = BatchRunResult(
        results=[
            BatchFileResult(
                file_name="series_1.csv",
                status="success",
                attempts=1,
                message="OK",
                duration_seconds=0.1234,
                output_df=pd.DataFrame({"prediction": [1.0, 2.0]}),
                metrics={
                    "mae": 0.1,
                    "rmse": 0.2,
                    "mse": 0.04,
                    "mape": 1.5,
                    "smape": 1.6,
                    "wape": 1.7,
                    "directional_accuracy": 100.0,
                    "quantile_coverage_error": None,
                },
            )
        ]
    )

    summary = result.to_summary_df()
    assert "smape" in summary.columns
    assert "wape" in summary.columns
    assert "directional_accuracy" in summary.columns
    assert "quantile_coverage_error" in summary.columns
    assert summary.loc[0, "smape"] == 1.6
    assert summary.loc[0, "wape"] == 1.7
    assert summary.loc[0, "directional_accuracy"] == 100.0
    assert pd.isna(summary.loc[0, "quantile_coverage_error"])
