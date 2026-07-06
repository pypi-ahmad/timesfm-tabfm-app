"""Tests for Ollama model selection helpers."""

from __future__ import annotations

from app_core.ollama_selector import build_fallback_models, resolve_model_selection


def test_build_fallback_models_dedupes_and_preserves_order() -> None:
    result = build_fallback_models(
        default_model="qwen3:4b",
        configured_models=["llama3.2:3b", "qwen3:4b", " "],
    )

    assert result == ["qwen3:4b", "llama3.2:3b"]


def test_resolve_model_selection_prefers_live_models() -> None:
    selection = resolve_model_selection(
        live_models=["qwen3:4b", "llama3.2:3b"],
        fallback_models=["mistral:7b"],
        preferred_model="llama3.2:3b",
    )

    assert selection.source == "local"
    assert selection.options == ["qwen3:4b", "llama3.2:3b"]
    assert selection.selected_model == "llama3.2:3b"


def test_resolve_model_selection_uses_fallback_when_live_unavailable() -> None:
    selection = resolve_model_selection(
        live_models=[],
        fallback_models=["qwen3:4b", "mistral:7b"],
        preferred_model="missing-model",
    )

    assert selection.source == "fallback"
    assert selection.options == ["qwen3:4b", "mistral:7b"]
    assert selection.selected_model == "qwen3:4b"


def test_resolve_model_selection_handles_empty_state() -> None:
    selection = resolve_model_selection(
        live_models=[],
        fallback_models=[],
        preferred_model=None,
    )

    assert selection.source == "none"
    assert selection.options == []
    assert selection.selected_model is None
