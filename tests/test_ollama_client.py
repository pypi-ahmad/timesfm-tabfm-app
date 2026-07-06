"""Tests for the Ollama client wrapper."""

from __future__ import annotations

from typing import Any

import pytest

from app_core.ollama_client import OllamaClient, OllamaError


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error
        self.text = str(payload)

    def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def test_generate_insight_returns_response_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(status_code=200, payload={"response": "ok-summary"})

    monkeypatch.setattr("requests.post", fake_post)
    client = OllamaClient(base_url="http://localhost:11434/api/generate", model_name="qwen3:4b")

    result = client.generate_insight("hello")

    assert result == "ok-summary"


def test_generate_insight_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(status_code=503, payload={"error": "service unavailable"})

    monkeypatch.setattr("requests.post", fake_post)
    client = OllamaClient(base_url="http://localhost:11434/api/generate", model_name="qwen3:4b")

    with pytest.raises(OllamaError, match="HTTP 503"):
        client.generate_insight("hello")


def test_list_local_models_returns_unique_names(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        captured["url"] = args[0]
        return FakeResponse(
            status_code=200,
            payload={
                "models": [
                    {"name": "qwen3:4b"},
                    {"name": "llama3.2:3b"},
                    {"name": "qwen3:4b"},
                    {"name": " "},
                    {},
                ]
            },
        )

    monkeypatch.setattr("requests.get", fake_get)
    client = OllamaClient(base_url="http://localhost:11434/api/generate", model_name="qwen3:4b")

    result = client.list_local_models()

    assert captured["url"] == "http://localhost:11434/api/tags"
    assert result == ["qwen3:4b", "llama3.2:3b"]


def test_list_local_models_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(status_code=500, payload={"error": "broken"})

    monkeypatch.setattr("requests.get", fake_get)
    client = OllamaClient(base_url="http://localhost:11434/api/generate", model_name="qwen3:4b")

    with pytest.raises(OllamaError, match="HTTP 500"):
        client.list_local_models()


def test_list_local_models_raises_on_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(status_code=200, payload={"models": "not-a-list"})

    monkeypatch.setattr("requests.get", fake_get)
    client = OllamaClient(base_url="http://localhost:11434/api/generate", model_name="qwen3:4b")

    with pytest.raises(OllamaError, match="malformed"):
        client.list_local_models()


def test_list_local_models_returns_empty_if_no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(status_code=200, payload={"models": []})

    monkeypatch.setattr("requests.get", fake_get)
    client = OllamaClient(base_url="http://localhost:11434/api/generate", model_name="qwen3:4b")

    assert client.list_local_models() == []
