"""Tests for sentiment provider and forecast bias logic."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from app_core.sentiment_service import (
    AlphaVantageSentimentProvider,
    SentimentProviderError,
    aggregate_recent_scores,
    apply_sentiment_bias,
    fetch_sentiment_scores_with_diagnostics,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


def test_alpha_vantage_provider_parses_ticker_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            status_code=200,
            payload={
                "feed": [
                    {
                        "time_published": "20250101T101500",
                        "ticker_sentiment": [
                            {"ticker": "AAPL", "ticker_sentiment_score": "0.25"},
                            {"ticker": "MSFT", "ticker_sentiment_score": "-0.1"},
                        ],
                    }
                ]
            },
        )

    monkeypatch.setattr("requests.get", fake_get)
    provider = AlphaVantageSentimentProvider(api_key="test")
    result = provider.fetch_scores(["AAPL", "MSFT"])

    assert len(result) == 2
    assert set(result["ticker"].tolist()) == {"AAPL", "MSFT"}


def test_alpha_vantage_provider_raises_on_invalid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(status_code=200, payload={"Note": "rate limit"})

    monkeypatch.setattr("requests.get", fake_get)
    provider = AlphaVantageSentimentProvider(api_key="test")
    with pytest.raises(SentimentProviderError, match="payload invalid"):
        provider.fetch_scores(["AAPL"])


def test_apply_sentiment_bias_adjusts_predictions() -> None:
    forecast = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "prediction": [100.0, 101.0, 102.0],
        }
    )
    adjusted = apply_sentiment_bias(
        forecast_df=forecast,
        ticker_scores={"AAPL": 0.5},
        strength=0.1,
        decay=1.0,
    )
    assert adjusted["prediction"].iloc[0] == pytest.approx(105.0)
    assert adjusted["bias_multiplier"].iloc[0] == pytest.approx(1.05)


def test_aggregate_recent_scores_defaults_to_zero_when_empty() -> None:
    result = aggregate_recent_scores(
        scores_df=pd.DataFrame(columns=["timestamp", "ticker", "sentiment_score"]),
        tickers=["AAPL", "MSFT"],
        lookback_hours=24,
    )
    assert result == {"AAPL": 0.0, "MSFT": 0.0}


def test_fetch_sentiment_scores_with_diagnostics_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            status_code=200,
            payload={
                "feed": [
                    {
                        "time_published": "20250101T101500",
                        "ticker_sentiment": [
                            {"ticker": "AAPL", "ticker_sentiment_score": "0.25"},
                            {"ticker": "MSFT", "ticker_sentiment_score": "-0.1"},
                        ],
                    }
                ]
            },
        )

    monkeypatch.setattr("requests.get", fake_get)
    provider = AlphaVantageSentimentProvider(api_key="test")
    result = fetch_sentiment_scores_with_diagnostics(
        provider=provider,
        tickers=["AAPL", "MSFT"],
        lookback_hours=24,
        fail_open=True,
    )

    assert result.status == "ok"
    assert result.error_message is None
    assert set(result.scores_by_ticker.keys()) == {"AAPL", "MSFT"}
    assert result.metadata["requested_ticker_count"] == 2
    assert result.metadata["fetched_rows"] == 2


def test_fetch_sentiment_scores_with_diagnostics_fail_open_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_scores(self: AlphaVantageSentimentProvider, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise SentimentProviderError("rate limit")

    monkeypatch.setattr(AlphaVantageSentimentProvider, "fetch_scores", fake_fetch_scores)
    provider = AlphaVantageSentimentProvider(api_key="test")
    result = fetch_sentiment_scores_with_diagnostics(
        provider=provider,
        tickers=["AAPL", "MSFT"],
        lookback_hours=24,
        fail_open=True,
    )

    assert result.status == "degraded"
    assert result.error_message == "rate limit"
    assert result.scores_by_ticker == {"AAPL": 0.0, "MSFT": 0.0}
    assert result.metadata["fetched_rows"] == 0
    assert result.metadata["non_zero_score_count"] == 0


def test_fetch_sentiment_scores_with_diagnostics_fail_closed_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_scores(self: AlphaVantageSentimentProvider, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise SentimentProviderError("network down")

    monkeypatch.setattr(AlphaVantageSentimentProvider, "fetch_scores", fake_fetch_scores)
    provider = AlphaVantageSentimentProvider(api_key="test")
    with pytest.raises(SentimentProviderError, match="network down"):
        fetch_sentiment_scores_with_diagnostics(
            provider=provider,
            tickers=["AAPL"],
            fail_open=False,
        )
