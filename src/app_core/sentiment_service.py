"""Sentiment ingestion and forecast-bias utilities for TimesFM."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import requests


class SentimentProviderError(RuntimeError):
    """Raised when sentiment providers fail or return malformed responses."""


@dataclass(frozen=True)
class SentimentFetchResult:
    """Live sentiment fetch output with diagnostics and fallback status."""

    scores_by_ticker: dict[str, float]
    scores_df: pd.DataFrame
    status: Literal["ok", "degraded"]
    error_message: str | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class AlphaVantageSentimentProvider:
    """News sentiment provider using Alpha Vantage NEWS_SENTIMENT endpoint."""

    api_key: str
    timeout_seconds: int = 20
    base_url: str = "https://www.alphavantage.co/query"

    def fetch_scores(self, tickers: Iterable[str], limit: int = 200) -> pd.DataFrame:
        ticker_list = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
        if not ticker_list:
            return pd.DataFrame(columns=["timestamp", "ticker", "sentiment_score"])

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(ticker_list),
            "sort": "LATEST",
            "limit": max(1, min(limit, 1000)),
            "apikey": self.api_key,
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout_seconds)
        except requests.Timeout as exc:
            raise SentimentProviderError(
                "Alpha Vantage sentiment request timed out."
            ) from exc
        except requests.RequestException as exc:
            raise SentimentProviderError(
                f"Alpha Vantage sentiment request failed: {exc}"
            ) from exc
        if response.status_code != 200:
            raise SentimentProviderError(
                f"Alpha Vantage sentiment request failed with HTTP {response.status_code}."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise SentimentProviderError(
                "Alpha Vantage sentiment payload was not valid JSON."
            ) from exc
        feed = payload.get("feed")
        if not isinstance(feed, list):
            message = payload.get("Information") or payload.get("Note") or "Missing feed."
            raise SentimentProviderError(f"Alpha Vantage sentiment payload invalid: {message}")

        rows: list[dict[str, object]] = []
        allowed = set(ticker_list)
        for item in feed:
            if not isinstance(item, dict):
                continue
            published = item.get("time_published")
            ts = pd.to_datetime(published, format="%Y%m%dT%H%M%S", errors="coerce", utc=True)
            if pd.isna(ts):
                continue
            ticker_sentiment = item.get("ticker_sentiment", [])
            if not isinstance(ticker_sentiment, list):
                continue
            for ticker_item in ticker_sentiment:
                if not isinstance(ticker_item, dict):
                    continue
                ticker = str(ticker_item.get("ticker", "")).upper().strip()
                score = pd.to_numeric(
                    ticker_item.get("ticker_sentiment_score"),
                    errors="coerce",
                )
                if ticker not in allowed or pd.isna(score):
                    continue
                score_value = float(score)
                if not np.isfinite(score_value):
                    continue
                rows.append(
                    {
                        "timestamp": ts.tz_convert(None),
                        "ticker": ticker,
                        "sentiment_score": score_value,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["timestamp", "ticker", "sentiment_score"])
        return pd.DataFrame(rows).sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def aggregate_recent_scores(
    scores_df: pd.DataFrame,
    tickers: Iterable[str],
    lookback_hours: int = 24,
) -> dict[str, float]:
    """Aggregate recent ticker sentiment into a single score per ticker."""
    if scores_df.empty:
        return {ticker.upper(): 0.0 for ticker in tickers}

    now = pd.Timestamp.utcnow().tz_localize(None)
    cutoff = now - timedelta(hours=max(1, lookback_hours))
    filtered = scores_df[scores_df["timestamp"] >= cutoff].copy()
    filtered["sentiment_score"] = pd.to_numeric(filtered["sentiment_score"], errors="coerce")
    filtered = filtered[np.isfinite(filtered["sentiment_score"])]
    if filtered.empty:
        return {ticker.upper(): 0.0 for ticker in tickers}

    grouped = filtered.groupby("ticker")["sentiment_score"].mean().to_dict()
    return {ticker.upper(): float(grouped.get(ticker.upper(), 0.0)) for ticker in tickers}


def fetch_sentiment_scores_with_diagnostics(
    provider: AlphaVantageSentimentProvider,
    tickers: Iterable[str],
    lookback_hours: int = 24,
    limit: int = 200,
    fail_open: bool = True,
) -> SentimentFetchResult:
    """Fetch and aggregate sentiment with diagnostics and optional fail-open fallback."""
    requested_tickers = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
    empty_scores_df = pd.DataFrame(columns=["timestamp", "ticker", "sentiment_score"])
    base_metadata: dict[str, Any] = {
        "requested_tickers": requested_tickers,
        "requested_ticker_count": len(requested_tickers),
        "lookback_hours": int(max(1, lookback_hours)),
        "limit": int(max(1, min(limit, 1000))),
    }
    if not requested_tickers:
        return SentimentFetchResult(
            scores_by_ticker={},
            scores_df=empty_scores_df,
            status="degraded",
            error_message="No tickers provided for sentiment fetch.",
            metadata={
                **base_metadata,
                "fetched_rows": 0,
                "scored_ticker_count": 0,
                "coverage_ratio": 0.0,
                "non_zero_score_count": 0,
            },
        )

    try:
        scores_df = provider.fetch_scores(
            tickers=requested_tickers,
            limit=int(max(1, min(limit, 1000))),
        )
        scores_by_ticker = aggregate_recent_scores(
            scores_df=scores_df,
            tickers=requested_tickers,
            lookback_hours=int(max(1, lookback_hours)),
        )
        scored_ticker_count = len([ticker for ticker, score in scores_by_ticker.items() if np.isfinite(score)])
        non_zero_score_count = len(
            [ticker for ticker, score in scores_by_ticker.items() if np.isfinite(score) and abs(score) > 1e-12]
        )
        coverage_ratio = (
            float(scored_ticker_count) / float(len(requested_tickers))
            if requested_tickers
            else 0.0
        )
        return SentimentFetchResult(
            scores_by_ticker=scores_by_ticker,
            scores_df=scores_df,
            status="ok",
            error_message=None,
            metadata={
                **base_metadata,
                "fetched_rows": int(len(scores_df)),
                "scored_ticker_count": int(scored_ticker_count),
                "coverage_ratio": float(coverage_ratio),
                "non_zero_score_count": int(non_zero_score_count),
            },
        )
    except SentimentProviderError as exc:
        if not fail_open:
            raise
        fallback_scores = {ticker: 0.0 for ticker in requested_tickers}
        return SentimentFetchResult(
            scores_by_ticker=fallback_scores,
            scores_df=empty_scores_df,
            status="degraded",
            error_message=str(exc),
            metadata={
                **base_metadata,
                "fetched_rows": 0,
                "scored_ticker_count": len(fallback_scores),
                "coverage_ratio": 1.0 if fallback_scores else 0.0,
                "non_zero_score_count": 0,
            },
        )


def apply_sentiment_bias(
    forecast_df: pd.DataFrame,
    ticker_scores: dict[str, float],
    strength: float,
    decay: float,
) -> pd.DataFrame:
    """Adjust forecast path using ticker sentiment and geometric decay."""
    adjusted = forecast_df.copy()
    if adjusted.empty:
        return adjusted

    adjusted["ticker"] = adjusted["ticker"].astype(str).str.upper()
    adjusted = adjusted.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    adjusted["base_prediction"] = adjusted["prediction"].astype(float)
    adjusted["prediction"] = adjusted["base_prediction"]

    if strength == 0.0:
        adjusted["sentiment_score"] = adjusted["ticker"].map(ticker_scores).fillna(0.0)
        adjusted["bias_multiplier"] = 1.0
        return adjusted

    adjusted["sentiment_score"] = adjusted["ticker"].map(ticker_scores).fillna(0.0).astype(float)
    adjusted["bias_multiplier"] = 1.0
    for ticker, group in adjusted.groupby("ticker", sort=False):
        score = float(ticker_scores.get(str(ticker).upper(), 0.0))
        multipliers = [
            1.0 + (strength * score * (decay**index))
            for index in range(len(group))
        ]
        adjusted.loc[group.index, "bias_multiplier"] = multipliers

    adjusted["prediction"] = adjusted["base_prediction"] * adjusted["bias_multiplier"]
    for quantile_column in ("p10", "p90"):
        if quantile_column in adjusted.columns:
            adjusted[quantile_column] = (
                adjusted[quantile_column].astype(float) * adjusted["bias_multiplier"]
            )
    return adjusted
