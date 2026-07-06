"""Tests for TimesFM covariate provider adapters."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from app_core.covariate_providers import (
    FredProvider,
    YahooFinanceProvider,
    merge_covariates,
    parse_csv_list,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


def test_parse_csv_list_deduplicates_and_trims() -> None:
    assert parse_csv_list(" CPIAUCSL, UNRATE, CPIAUCSL , ,GDP") == [
        "CPIAUCSL",
        "UNRATE",
        "GDP",
    ]


def test_fred_provider_fetches_series(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        series_id = kwargs["params"]["series_id"]
        return FakeResponse(
            status_code=200,
            payload={
                "observations": [
                    {"date": "2024-01-01", "value": "100.1"},
                    {"date": "2024-01-02", "value": "100.3"},
                ],
                "series_id": series_id,
            },
        )

    monkeypatch.setattr("requests.get", fake_get)
    provider = FredProvider(api_key="test")
    result = provider.fetch(
        series_ids=["CPIAUCSL", "UNRATE"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 2),
    )

    assert {"timestamp", "CPIAUCSL", "UNRATE"}.issubset(set(result.columns))
    assert len(result) == 2


def test_yahoo_provider_fetches_close_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            status_code=200,
            payload={
                "chart": {
                    "result": [
                        {
                            "timestamp": [1704067200, 1704153600],
                            "indicators": {"quote": [{"close": [100.0, 101.0]}]},
                        }
                    ]
                }
            },
        )

    monkeypatch.setattr("requests.get", fake_get)
    provider = YahooFinanceProvider()
    result = provider.fetch(
        tickers=["XLK"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 2),
    )

    assert list(result.columns) == ["timestamp", "sector_XLK"]
    assert result["sector_XLK"].tolist() == [100.0, 101.0]


def test_merge_covariates_combines_macro_sector_and_override() -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["AAPL", "AAPL"],
            "value": [100.0, 102.0],
        }
    )
    macro = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "CPIAUCSL": [300.1, 300.2],
        }
    )
    sector = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "sector_XLK": [120.0, 121.0],
        }
    )
    override = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "promo_index": [0.1, 0.2],
        }
    )

    merged = merge_covariates(panel_df=panel, macro_df=macro, sector_df=sector, override_df=override)
    assert {"CPIAUCSL", "sector_XLK", "promo_index"}.issubset(merged.columns)
