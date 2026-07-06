"""Data models for batch run orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BatchFileResult:
    """Per-file outcome for a batch run."""

    file_name: str
    status: str
    attempts: int
    message: str
    duration_seconds: float
    output_df: pd.DataFrame | None = None
    metrics: dict[str, float | None] | None = None
    comparison_df: pd.DataFrame | None = None
    task: str | None = None


@dataclass(frozen=True)
class BatchRunResult:
    """Overall batch execution result."""

    results: list[BatchFileResult]

    def to_summary_df(self) -> pd.DataFrame:
        """Return a dataframe summary suitable for UI display and export."""
        rows = []
        for item in self.results:
            rows.append(
                {
                    "file_name": item.file_name,
                    "status": item.status,
                    "attempts": item.attempts,
                    "message": item.message,
                    "duration_seconds": round(item.duration_seconds, 4),
                    "mae": item.metrics.get("mae") if item.metrics else None,
                    "rmse": item.metrics.get("rmse") if item.metrics else None,
                    "mse": item.metrics.get("mse") if item.metrics else None,
                    "mape": item.metrics.get("mape") if item.metrics else None,
                    "smape": item.metrics.get("smape") if item.metrics else None,
                    "wape": item.metrics.get("wape") if item.metrics else None,
                    "directional_accuracy": item.metrics.get("directional_accuracy") if item.metrics else None,
                    "quantile_coverage_error": item.metrics.get("quantile_coverage_error") if item.metrics else None,
                }
            )
        return pd.DataFrame(rows)
