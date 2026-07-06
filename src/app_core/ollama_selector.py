"""Helpers for choosing an active Ollama model in the UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class OllamaModelSelection:
    """Resolved model choices and active model for the current session."""

    options: list[str]
    selected_model: str | None
    source: str


def _normalize_models(model_names: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for model_name in model_names:
        name = str(model_name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def build_fallback_models(default_model: str, configured_models: Iterable[str]) -> list[str]:
    """Build fallback model list from default plus optional configured models."""
    return _normalize_models([default_model, *configured_models])


def resolve_model_selection(
    live_models: Iterable[str],
    fallback_models: Iterable[str],
    preferred_model: str | None,
) -> OllamaModelSelection:
    """Resolve final model options and active model using precedence rules."""
    local_options = _normalize_models(live_models)
    fallback_options = _normalize_models(fallback_models)

    if local_options:
        options = local_options
        source = "local"
    elif fallback_options:
        options = fallback_options
        source = "fallback"
    else:
        return OllamaModelSelection(options=[], selected_model=None, source="none")

    selected_model = preferred_model if preferred_model in options else options[0]
    return OllamaModelSelection(
        options=options,
        selected_model=selected_model,
        source=source,
    )
