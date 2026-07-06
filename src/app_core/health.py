"""Runtime health checks for dependencies and local services."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import requests


@dataclass(frozen=True)
class HealthStatus:
    """Simple health status for one runtime dependency/service."""

    name: str
    status: str
    detail: str


def check_python_dependency(module_name: str) -> HealthStatus:
    """Check if a Python dependency can be imported."""
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return HealthStatus(name=module_name, status="missing", detail=str(exc))
    return HealthStatus(name=module_name, status="ready", detail="Import successful.")


def check_ollama_health(generate_url: str, timeout_seconds: int = 2) -> HealthStatus:
    """Check Ollama availability by probing /api/tags."""
    parsed = urlparse(generate_url)
    tags_url = urlunparse((parsed.scheme, parsed.netloc, "/api/tags", "", "", ""))
    try:
        response = requests.get(tags_url, timeout=timeout_seconds)
        if response.status_code == 200:
            return HealthStatus(name="ollama", status="ready", detail="Reachable.")
        return HealthStatus(
            name="ollama",
            status="degraded",
            detail=f"HTTP {response.status_code} from {tags_url}",
        )
    except Exception as exc:
        return HealthStatus(name="ollama", status="missing", detail=str(exc))

