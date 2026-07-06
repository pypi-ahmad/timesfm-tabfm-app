"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime configuration for the unified TabFM/TimesFM app."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "qwen3:4b"
    ollama_models: list[str] = Field(default_factory=list)
    default_backend: str = "torch"
    timesfm_model_id: str = "google/timesfm-2.5-200m-pytorch"
    timesfm_jax_model_id: str = "google/timesfm-2.5-200m-flax"
    timesfm_xreg_mode: str = "xreg + timesfm"
    fred_api_key: str = ""
    alpha_vantage_api_key: str = ""
    covariate_default_macro_ids: str = "CPIAUCSL,UNRATE"
    covariate_default_sector_tickers: str = "XLK,XLF,XLV"
    sentiment_lookback_hours: int = 24
    sentiment_bias_strength: float = 0.1
    sentiment_bias_decay: float = 0.95
    portfolio_risk_aversion: float = 1.0
    portfolio_max_weight: float = 0.4
    backtest_default_mode: str = "walk_forward"
    backtest_default_folds: int = 3
    backtest_min_train_size: int = 40
    backtest_rolling_window: int = 120
    timesfm_lora_script_path: str = "timesfm-forecasting/examples/finetuning/finetune_lora.py"
    timesfm_lora_registry_path: str = ".timesfm/lora_jobs_registry.json"
    timesfm_lora_adapter_registry_path: str = ".timesfm/lora_adapters_registry.json"
    timesfm_lora_output_root: str = ".timesfm/lora_runs"
    timesfm_lora_default_mode: str = "external_script"
    timesfm_lora_retention_policy: str = "delete_raw"
    timesfm_lora_min_points_per_entity: int = 20
    timesfm_lora_validation_ratio: float = 0.2
    default_forecast_horizon: int = 24
    default_max_context: int = 1024
    default_max_horizon: int = 256
    normalize_inputs: bool = True
    batch_max_files: int = 25
    batch_retry_count: int = 1
    pdf_table_max_rows: int = 100
    pdf_font_size: int = 10
    output_markdown_path: str = "outputs.md"
    log_level: str = "INFO"

    @field_validator("ollama_models", mode="before")
    @classmethod
    def _parse_ollama_models(cls, value: Any) -> list[str]:
        """Parse comma-separated APP_OLLAMA_MODELS into a clean unique list."""
        if value is None:
            return []
        if isinstance(value, str):
            raw_models = value.split(",")
        elif isinstance(value, list):
            raw_models = value
        else:
            raise ValueError("ollama_models must be a comma-separated string or list.")

        parsed: list[str] = []
        seen: set[str] = set()
        for item in raw_models:
            model_name = str(item).strip()
            if not model_name or model_name in seen:
                continue
            seen.add(model_name)
            parsed.append(model_name)
        return parsed

    @field_validator("default_backend", mode="before")
    @classmethod
    def _parse_default_backend(cls, value: Any) -> str:
        """Normalize backend setting and restrict to supported values."""
        normalized = str(value).strip().lower()
        if normalized not in {"torch", "jax"}:
            raise ValueError("default_backend must be one of: torch, jax.")
        return normalized

    @field_validator("timesfm_xreg_mode", mode="before")
    @classmethod
    def _parse_timesfm_xreg_mode(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"xreg + timesfm", "timesfm + xreg"}:
            raise ValueError(
                "timesfm_xreg_mode must be one of: 'xreg + timesfm', 'timesfm + xreg'."
            )
        return normalized

    @field_validator("backtest_default_mode", mode="before")
    @classmethod
    def _parse_backtest_mode(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"walk_forward", "rolling_window"}:
            raise ValueError(
                "backtest_default_mode must be one of: walk_forward, rolling_window."
            )
        return normalized

    @field_validator("timesfm_lora_default_mode", mode="before")
    @classmethod
    def _parse_timesfm_lora_mode(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"external_script", "in_app"}:
            raise ValueError(
                "timesfm_lora_default_mode must be one of: external_script, in_app."
            )
        return normalized

    @field_validator("timesfm_lora_retention_policy", mode="before")
    @classmethod
    def _parse_timesfm_lora_retention_policy(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"delete_raw", "keep_raw"}:
            raise ValueError(
                "timesfm_lora_retention_policy must be one of: delete_raw, keep_raw."
            )
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache settings once per process."""
    return AppSettings()
