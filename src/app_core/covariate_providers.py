"""External covariate providers for TimesFM exogenous features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import requests


class CovariateProviderError(RuntimeError):
    """Raised when covariate provider APIs fail or return malformed payloads."""


def parse_csv_list(raw: str) -> list[str]:
    """Parse a comma-separated string into a deduplicated list."""
    values: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        values.append(normalized)
        seen.add(normalized)
    return values


def _to_epoch_seconds(value: datetime) -> int:
    return int(value.replace(tzinfo=timezone.utc).timestamp())


@dataclass(frozen=True)
class FredProvider:
    """FRED API provider for macroeconomic covariates."""

    api_key: str
    timeout_seconds: int = 20
    base_url: str = "https://api.stlouisfed.org/fred/series/observations"

    def fetch(self, series_ids: Iterable[str], start: datetime, end: datetime) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for series_id in series_ids:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start.strftime("%Y-%m-%d"),
                "observation_end": end.strftime("%Y-%m-%d"),
            }
            response = requests.get(self.base_url, params=params, timeout=self.timeout_seconds)
            if response.status_code != 200:
                raise CovariateProviderError(
                    f"FRED request failed for {series_id} with HTTP {response.status_code}."
                )
            payload = response.json()
            observations = payload.get("observations")
            if not isinstance(observations, list):
                raise CovariateProviderError(
                    f"FRED payload for {series_id} did not include observations."
                )

            rows: list[dict[str, object]] = []
            for item in observations:
                if not isinstance(item, dict):
                    continue
                ts = pd.to_datetime(item.get("date"), errors="coerce")
                value = pd.to_numeric(item.get("value"), errors="coerce")
                if pd.isna(ts) or pd.isna(value):
                    continue
                rows.append({"timestamp": ts, series_id: float(value)})
            frames.append(pd.DataFrame(rows))

        if not frames:
            return pd.DataFrame(columns=["timestamp"])

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="timestamp", how="outer")
        return merged.sort_values("timestamp").reset_index(drop=True)


@dataclass(frozen=True)
class YahooFinanceProvider:
    """Yahoo Finance chart API provider for sector-level covariates."""

    timeout_seconds: int = 20
    base_url: str = "https://query1.finance.yahoo.com/v8/finance/chart"

    def fetch(
        self,
        tickers: Iterable[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            params = {
                "interval": interval,
                "period1": _to_epoch_seconds(start),
                "period2": _to_epoch_seconds(end),
            }
            response = requests.get(
                f"{self.base_url}/{ticker}",
                params=params,
                timeout=self.timeout_seconds,
            )
            if response.status_code != 200:
                raise CovariateProviderError(
                    f"Yahoo Finance request failed for {ticker} with HTTP {response.status_code}."
                )
            payload = response.json()
            chart = payload.get("chart", {})
            result = chart.get("result") if isinstance(chart, dict) else None
            if not isinstance(result, list) or not result:
                raise CovariateProviderError(f"Yahoo Finance returned empty chart for {ticker}.")
            series = result[0]
            timestamps = series.get("timestamp", [])
            indicators = series.get("indicators", {})
            quote = indicators.get("quote", []) if isinstance(indicators, dict) else []
            closes = quote[0].get("close", []) if quote and isinstance(quote[0], dict) else []
            if len(timestamps) != len(closes):
                raise CovariateProviderError(f"Yahoo Finance close series mismatch for {ticker}.")

            rows: list[dict[str, object]] = []
            column_name = f"sector_{ticker}"
            for ts_raw, close in zip(timestamps, closes):
                ts = pd.to_datetime(ts_raw, unit="s", utc=True, errors="coerce")
                value = pd.to_numeric(close, errors="coerce")
                if pd.isna(ts) or pd.isna(value):
                    continue
                rows.append({"timestamp": ts.tz_convert(None), column_name: float(value)})
            frames.append(pd.DataFrame(rows))

        if not frames:
            return pd.DataFrame(columns=["timestamp"])

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="timestamp", how="outer")
        return merged.sort_values("timestamp").reset_index(drop=True)


def merge_covariates(
    panel_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
    override_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge macro/sector/override covariates into panel rows by timestamp."""
    merged = panel_df.copy()
    for frame in (macro_df, sector_df, override_df):
        if frame is None or frame.empty:
            continue
        normalized = frame.copy()
        if "timestamp" not in normalized.columns:
            raise CovariateProviderError("Covariate dataframe must include a timestamp column.")
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
        normalized = normalized.dropna(subset=["timestamp"]).sort_values("timestamp")
        merged = merged.merge(normalized, on="timestamp", how="left")

    merged = merged.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    covariate_columns = [
        column
        for column in merged.columns
        if column not in {"timestamp", "ticker", "value"}
    ]
    if covariate_columns:
        merged[covariate_columns] = merged[covariate_columns].ffill().bfill()
    return merged
