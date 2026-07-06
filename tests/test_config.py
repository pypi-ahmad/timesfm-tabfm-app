"""Tests for application settings parsing."""

from __future__ import annotations

import pytest

from app_core.config import AppSettings


def test_app_settings_parses_ollama_models_csv() -> None:
    settings = AppSettings(ollama_models="qwen3:4b, llama3.2:3b, qwen3:4b")

    assert settings.ollama_models == ["qwen3:4b", "llama3.2:3b"]


def test_app_settings_pdf_defaults() -> None:
    settings = AppSettings()

    assert settings.pdf_table_max_rows == 100
    assert settings.pdf_font_size == 10


def test_app_settings_default_backend_normalizes_case() -> None:
    settings = AppSettings(default_backend="JAX")

    assert settings.default_backend == "jax"


def test_app_settings_default_backend_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="default_backend"):
        AppSettings(default_backend="onnx")


def test_app_settings_validates_xreg_mode() -> None:
    settings = AppSettings(timesfm_xreg_mode="TimesFM + XReg")
    assert settings.timesfm_xreg_mode == "timesfm + xreg"

    with pytest.raises(ValueError, match="timesfm_xreg_mode"):
        AppSettings(timesfm_xreg_mode="invalid")


def test_app_settings_validates_backtest_default_mode() -> None:
    settings = AppSettings(backtest_default_mode="ROLLING_WINDOW")
    assert settings.backtest_default_mode == "rolling_window"

    with pytest.raises(ValueError, match="backtest_default_mode"):
        AppSettings(backtest_default_mode="holdout")


def test_app_settings_validates_timesfm_lora_mode() -> None:
    settings = AppSettings(timesfm_lora_default_mode="IN_APP")
    assert settings.timesfm_lora_default_mode == "in_app"

    with pytest.raises(ValueError, match="timesfm_lora_default_mode"):
        AppSettings(timesfm_lora_default_mode="invalid")


def test_app_settings_validates_timesfm_lora_retention_policy() -> None:
    settings = AppSettings(timesfm_lora_retention_policy="KEEP_RAW")
    assert settings.timesfm_lora_retention_policy == "keep_raw"

    with pytest.raises(ValueError, match="timesfm_lora_retention_policy"):
        AppSettings(timesfm_lora_retention_policy="archive")
