"""Thin client for local Ollama insight generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests


class OllamaError(RuntimeError):
    """Raised when insight generation fails."""


@dataclass(frozen=True)
class OllamaClient:
    """Client for calling local Ollama /api/generate endpoint."""

    base_url: str
    model_name: str
    timeout_seconds: int = 30

    def _tags_url(self) -> str:
        parsed = urlparse(self.base_url)
        return urlunparse((parsed.scheme, parsed.netloc, "/api/tags", "", "", ""))

    def generate_insight(self, prompt: str) -> str:
        """Generate a deterministic short summary with local Ollama."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        response = requests.post(self.base_url, json=payload, timeout=self.timeout_seconds)
        if response.status_code != 200:
            raise OllamaError(f"Ollama request failed with HTTP {response.status_code}.")
        result = response.json().get("response", "").strip()
        if not result:
            raise OllamaError("Ollama returned an empty response.")
        return result

    def list_local_models(self) -> list[str]:
        """List locally available Ollama models from /api/tags."""
        response = requests.get(self._tags_url(), timeout=self.timeout_seconds)
        if response.status_code != 200:
            raise OllamaError(f"Ollama model listing failed with HTTP {response.status_code}.")

        try:
            payload = response.json()
        except Exception as exc:
            raise OllamaError("Ollama model listing returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise OllamaError("Ollama model listing returned malformed payload.")
        models_payload = payload.get("models", [])
        if not isinstance(models_payload, list):
            raise OllamaError("Ollama model listing returned malformed payload.")

        model_names: list[str] = []
        seen: set[str] = set()
        for item in models_payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            model_names.append(name)

        return model_names
