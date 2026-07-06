"""Tests for runtime health checks."""

from __future__ import annotations

from typing import Any

import pytest

from app_core.health import (
    check_ollama_health,
    check_python_dependency,
)


def test_check_python_dependency_reports_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str) -> object:
        return object()

    monkeypatch.setattr("importlib.import_module", fake_import_module)
    status = check_python_dependency("timesfm")
    assert status.status == "ready"


def test_check_python_dependency_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.import_module", fake_import_module)
    status = check_python_dependency("tabfm")
    assert status.status == "missing"


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_check_ollama_health_reports_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status_code=200)

    monkeypatch.setattr("requests.get", fake_get)
    status = check_ollama_health("http://localhost:11434/api/generate")
    assert status.status == "ready"

