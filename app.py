from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, TypedDict

import pandas as pd
import streamlit as st

from app_core.batch_io import BatchInputItem, load_batch_items_from_bytes
from app_core.artifacts import build_artifact_bundle
from app_core.covariate_providers import (
    CovariateProviderError,
    FredProvider,
    YahooFinanceProvider,
    merge_covariates,
    parse_csv_list,
)
from app_core.charts import (
    build_timesfm_backtest_figure,
    build_timesfm_forecast_figure,
    build_timesfm_residual_figure,
)
from app_core.config import AppSettings, get_settings
from app_core.health import HealthStatus, check_ollama_health, check_python_dependency
from app_core.logging_config import configure_logging
from app_core.lora_jobs import (
    LoRAJobConfig,
    LoRAJobError,
    list_lora_jobs,
    refresh_lora_job,
    start_lora_job,
    stop_lora_job,
)
from app_core.timesfm_lora_adapters import (
    LoRAAdapterError,
    ensure_adapter_registered_from_job,
    list_lora_adapters,
    resolve_lora_adapter_path,
)
from app_core.timesfm_lora_data import (
    RETENTION_DELETE_RAW,
    RETENTION_KEEP_RAW,
    TransactionalDatasetSpec,
    materialize_lora_dataset_from_csv_bytes,
)
from app_core.ollama_client import OllamaClient, OllamaError
from app_core.ollama_selector import (
    OllamaModelSelection,
    build_fallback_models,
    resolve_model_selection,
)
from app_core.portfolio_optimization import (
    PortfolioOptimizationError,
    build_covariance_from_panel,
    optimize_mean_variance_long_only,
    validate_portfolio_forecast_inputs,
)
from app_core.pdf_export import (
    InsightPdfDocument,
    InsightPdfSection,
    build_batch_zip,
    build_pdf_bytes,
    build_timesfm_chart_png,
    can_export_insight,
)
from app_core.reporting import append_markdown_run
from app_core.tabfm_service import TabFMRuntimeError, run_tabfm_batch, run_tabfm_prediction
from app_core.sentiment_service import (
    AlphaVantageSentimentProvider,
    fetch_sentiment_scores_with_diagnostics,
)
from app_core.timesfm_advanced import (
    build_panel_from_batch_items,
    derive_expected_returns,
    normalize_panel_history,
    run_timesfm_backtesting_framework,
    run_timesfm_multi_asset_forecast,
)
from app_core.timesfm_service import (
    TimesFMRuntimeError,
    run_timesfm_backtest,
    run_timesfm_batch,
    run_timesfm_forecast,
)
from app_core.validators import ValidationError, validate_timesfm_input


def _load_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile | None) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


def _uploaded_to_named_bytes(
    uploaded_files: Iterable[st.runtime.uploaded_file_manager.UploadedFile] | None,
) -> list[tuple[str, bytes]]:
    if not uploaded_files:
        return []
    return [(uploaded.name, uploaded.getvalue()) for uploaded in uploaded_files]


def _demo_tabfm_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25.0, 45.0, 35.0, 50.0],
            "job": ["engineer", "manager", "engineer", "manager"],
            "income": [80000, 120000, 90000, 130000],
            "risk": ["low_risk", "high_risk", "low_risk", "high_risk"],
        }
    )


def _demo_tabfm_predict_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [30.0, 48.0],
            "job": ["engineer", "manager"],
            "income": [85000, 125000],
        }
    )


def _demo_timesfm_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=12, freq="D"),
            "value": [100, 102, 101, 103, 106, 108, 107, 110, 112, 111, 114, 116],
        }
    )


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _artifact_store() -> dict[str, bytes]:
    if "artifact_files" not in st.session_state:
        st.session_state["artifact_files"] = {}
    return st.session_state["artifact_files"]


def _register_artifact(filename: str, content: bytes) -> None:
    _artifact_store()[filename] = content


class PdfExportEntry(TypedDict):
    """Session-stored PDF export payload for local and global download controls."""

    label: str
    file_name: str
    mime: str
    data: bytes


def _pdf_export_store() -> dict[str, PdfExportEntry]:
    if "pdf_exports" not in st.session_state:
        st.session_state["pdf_exports"] = {}
    return st.session_state["pdf_exports"]


def _register_pdf_export(
    export_key: str,
    label: str,
    file_name: str,
    data: bytes,
    mime: str,
) -> None:
    _pdf_export_store()[export_key] = {
        "label": label,
        "file_name": file_name,
        "mime": mime,
        "data": data,
    }


def _set_pdf_context(context_key: str, context_payload: dict[str, object]) -> None:
    if "pdf_report_contexts" not in st.session_state:
        st.session_state["pdf_report_contexts"] = {}
    st.session_state["pdf_report_contexts"][context_key] = context_payload


def _utc_timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _render_export_disabled_message(reason: str) -> None:
    st.caption(f"Export disabled (no insight yet). {reason}")


def _toggle_batch_enrichment_controls(button_key: str, state_key: str) -> bool:
    if st.button("Configure Batch Enrichment", key=button_key):
        st.session_state[state_key] = not st.session_state.get(state_key, False)
    return bool(st.session_state.get(state_key, False))


def _status_emoji(status: str) -> str:
    if status == "ready":
        return "✅"
    if status == "degraded":
        return "⚠️"
    return "❌"


def _backend_label(backend: str) -> str:
    if backend == "jax":
        return "JAX"
    return "PyTorch"


def _resolve_timesfm_model_id(settings: AppSettings, backend: str) -> str:
    if backend == "jax":
        return settings.timesfm_jax_model_id
    return settings.timesfm_model_id


def _ollama_disabled_hint() -> None:
    st.caption(
        "Ollama insights are disabled until a model is available. "
        "Run `ollama pull qwen3:4b` or set `APP_OLLAMA_MODELS`."
    )


def _refresh_ollama_model_cache(settings: AppSettings, logger: logging.Logger) -> None:
    try:
        client = OllamaClient(base_url=settings.ollama_url, model_name=settings.ollama_model)
        st.session_state["ollama_live_models"] = client.list_local_models()
        st.session_state["ollama_model_error"] = ""
    except OllamaError as exc:
        logger.warning("Ollama model discovery failed: %s", exc)
        st.session_state["ollama_live_models"] = []
        st.session_state["ollama_model_error"] = str(exc)


def _render_ollama_model_selector(
    settings: AppSettings,
    logger: logging.Logger,
) -> OllamaModelSelection:
    st.sidebar.subheader("Ollama Model")
    fallback_models = build_fallback_models(settings.ollama_model, settings.ollama_models)

    if st.sidebar.button("Refresh Models", key="refresh_ollama_models"):
        _refresh_ollama_model_cache(settings, logger)

    if "ollama_live_models" not in st.session_state:
        _refresh_ollama_model_cache(settings, logger)

    preferred_model = st.session_state.get("ollama_selected_model", settings.ollama_model)
    selection = resolve_model_selection(
        live_models=st.session_state.get("ollama_live_models", []),
        fallback_models=fallback_models,
        preferred_model=preferred_model,
    )

    if selection.selected_model is not None:
        if st.session_state.get("ollama_selected_model") not in selection.options:
            st.session_state["ollama_selected_model"] = selection.selected_model
        selected_model = st.sidebar.selectbox(
            "Active model",
            options=selection.options,
            key="ollama_selected_model",
            help="Used for all TabFM and TimesFM Ollama summaries.",
        )
        selection = OllamaModelSelection(
            options=selection.options,
            selected_model=selected_model,
            source=selection.source,
        )
        source_label = "Local Ollama (/api/tags)" if selection.source == "local" else "Configured fallback"
        st.sidebar.caption(f"Model source: {source_label} ({len(selection.options)} available).")
    else:
        st.sidebar.warning(
            "No Ollama models available. Pull one locally or configure `APP_OLLAMA_MODELS`."
        )

    error_message = st.session_state.get("ollama_model_error")
    if error_message:
        st.sidebar.caption(f"Model discovery warning: {error_message}")

    return selection


def _render_health_panel(settings: AppSettings) -> None:
    st.sidebar.subheader("Runtime Health")
    if st.sidebar.button("Refresh Health", key="refresh_health"):
        st.session_state.pop("health_snapshot", None)

    if "health_snapshot" not in st.session_state:
        st.session_state["health_snapshot"] = [
            check_python_dependency("tabfm"),
            check_python_dependency("timesfm"),
            check_ollama_health(settings.ollama_url),
        ]

    statuses: list[HealthStatus] = st.session_state["health_snapshot"]
    for item in statuses:
        st.sidebar.write(f"{_status_emoji(item.status)} `{item.name}`: {item.detail}")


def _render_artifact_download(
    settings: AppSettings,
    ollama_selection: OllamaModelSelection,
) -> None:
    st.markdown("### Artifact Bundle")
    files = dict(_artifact_store())
    output_path = Path(settings.output_markdown_path)
    if output_path.exists():
        files[output_path.name] = output_path.read_bytes()

    if files:
        bundle = build_artifact_bundle(
            files=files,
            metadata={
                "ollama_model": ollama_selection.selected_model or "",
                "ollama_model_source": ollama_selection.source,
                "ollama_model_count": len(ollama_selection.options),
                "timesfm_default_backend": settings.default_backend,
                "timesfm_torch_model_id": settings.timesfm_model_id,
                "timesfm_jax_model_id": settings.timesfm_jax_model_id,
                "artifact_count": len(files),
            },
        )
        st.download_button(
            label="Download Artifact Bundle (ZIP)",
            data=bundle,
            file_name="tabfm_timesfm_artifacts.zip",
            mime="application/zip",
            key="download_artifact_bundle",
        )
    else:
        st.info("Run TabFM or TimesFM to generate downloadable artifacts.")

    pdf_exports = _pdf_export_store()
    if pdf_exports:
        st.markdown("### Latest Insight PDF Exports")
        for export_key in sorted(pdf_exports):
            item = pdf_exports[export_key]
            st.download_button(
                label=item["label"],
                data=item["data"],
                file_name=item["file_name"],
                mime=item["mime"],
                key=f"download_global_{export_key}",
            )


def _tabfm_insight_prompt(result_df: pd.DataFrame, task: str) -> str:
    return (
        f"TabFM task type: {task}\n"
        f"Prediction table:\n{result_df.to_string(index=False)}\n"
        "Provide a concise 2-3 sentence operational summary."
    )


def _timesfm_insight_prompt(forecast_df: pd.DataFrame, horizon: int) -> str:
    return (
        f"Forecast horizon: {horizon}\n"
        f"Forecast table:\n{forecast_df.to_string(index=False)}\n"
        "Provide a concise 2-3 sentence trend and uncertainty summary."
    )


def _build_insight_pdf(
    title: str,
    metadata: dict[str, object],
    sections: list[InsightPdfSection],
    settings: AppSettings,
) -> bytes:
    document = InsightPdfDocument(
        title=title,
        generated_at=_utc_timestamp_label(),
        metadata=metadata,
        sections=sections,
    )
    return build_pdf_bytes(
        document=document,
        table_max_rows=settings.pdf_table_max_rows,
        font_size=settings.pdf_font_size,
    )


def _build_timesfm_chart_or_none(
    logger: logging.Logger,
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> bytes | None:
    try:
        return build_timesfm_chart_png(history_df=history_df, forecast_df=forecast_df)
    except Exception as exc:
        logger.warning("TimesFM chart generation for PDF failed: %s", exc)
        return None


def _build_streamlit_forecast_plot_df(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    history_plot_df = history_df.rename(columns={"value": "history"})
    forecast_plot_df = forecast_df.rename(columns={"prediction": "forecast"})
    combined_plot_df = pd.concat(
        [
            history_plot_df[["timestamp", "history"]].assign(forecast=None),
            forecast_plot_df[["timestamp", "forecast"]].assign(history=None),
        ],
        ignore_index=True,
    ).sort_values("timestamp")
    return combined_plot_df


def _render_timesfm_forecast_chart(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    chart_mode: str,
    include_quantiles: bool,
) -> None:
    if chart_mode == "Plotly":
        figure = build_timesfm_forecast_figure(
            history_df=history_df,
            forecast_df=forecast_df,
            show_quantile_band=include_quantiles,
        )
        st.plotly_chart(figure, use_container_width=True)
        return

    streamlit_plot_df = _build_streamlit_forecast_plot_df(
        history_df=history_df,
        forecast_df=forecast_df,
    )
    st.line_chart(streamlit_plot_df.set_index("timestamp")[["history", "forecast"]])


def _render_timesfm_backtest_charts(
    comparison_df: pd.DataFrame,
    chart_mode: str,
) -> None:
    if chart_mode != "Plotly":
        return

    st.markdown("**Backtest Charts**")
    st.plotly_chart(
        build_timesfm_backtest_figure(comparison_df=comparison_df),
        use_container_width=True,
    )
    st.plotly_chart(
        build_timesfm_residual_figure(comparison_df=comparison_df),
        use_container_width=True,
    )


def _load_optional_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile | None) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


def _load_covariate_override_csv(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile | None,
) -> pd.DataFrame | None:
    override_df = _load_optional_csv(uploaded_file)
    if override_df is None:
        return None
    if "timestamp" not in override_df.columns:
        raise ValidationError("Override covariates CSV must include a 'timestamp' column.")
    override_df["timestamp"] = pd.to_datetime(override_df["timestamp"], errors="coerce")
    override_df = override_df.dropna(subset=["timestamp"])
    if override_df.empty:
        raise ValidationError("Override covariates CSV is empty after timestamp parsing.")
    return override_df


def _apply_timesfm_xreg_covariates(
    panel_df: pd.DataFrame,
    use_xreg: bool,
    macro_series_raw: str,
    sector_tickers_raw: str,
    fred_api_key: str,
    override_covariates_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    working_panel_df = panel_df.copy()
    if not use_xreg:
        return working_panel_df, []

    macro_series_ids = parse_csv_list(macro_series_raw)
    sector_tickers = parse_csv_list(sector_tickers_raw)
    if not macro_series_ids and not sector_tickers and override_covariates_df is None:
        raise ValidationError(
            "Enable at least one XReg source: macro series, sector tickers, or override CSV."
        )

    start_ts = pd.to_datetime(working_panel_df["timestamp"]).min().to_pydatetime()
    end_ts = pd.to_datetime(working_panel_df["timestamp"]).max().to_pydatetime()

    macro_df: pd.DataFrame | None = None
    if macro_series_ids:
        if not fred_api_key.strip():
            raise ValidationError("FRED API key is required to fetch macro covariates.")
        macro_df = FredProvider(api_key=fred_api_key.strip()).fetch(
            series_ids=macro_series_ids,
            start=start_ts,
            end=end_ts,
        )

    sector_df: pd.DataFrame | None = None
    if sector_tickers:
        sector_df = YahooFinanceProvider().fetch(
            tickers=sector_tickers,
            start=start_ts,
            end=end_ts,
        )

    working_panel_df = merge_covariates(
        panel_df=working_panel_df,
        macro_df=macro_df,
        sector_df=sector_df,
        override_df=override_covariates_df,
    )
    covariate_columns = [
        column
        for column in working_panel_df.columns
        if column not in {"timestamp", "ticker", "value"}
    ]
    if not covariate_columns:
        raise ValidationError("No usable XReg covariate columns were produced.")
    return working_panel_df, covariate_columns


def _infer_batch_ticker(file_name: str) -> str:
    ticker = PurePosixPath(file_name).stem.strip().upper()
    if not ticker:
        raise ValidationError(f"Unable to infer ticker from file name: {file_name}")
    return ticker


def _build_timesfm_panel_input(
    data_format: str,
    panel_file: st.runtime.uploaded_file_manager.UploadedFile | None,
    per_ticker_files: Iterable[st.runtime.uploaded_file_manager.UploadedFile] | None,
    per_ticker_zip: st.runtime.uploaded_file_manager.UploadedFile | None,
    max_files: int,
    timestamp_column: str,
    value_column: str,
    panel_ticker_column: str,
) -> pd.DataFrame:
    if data_format == "panel":
        if panel_file is None:
            raise ValidationError("Upload a panel CSV for advanced TimesFM runs.")
        panel_df = pd.read_csv(panel_file)
        return normalize_panel_history(
            panel_df,
            timestamp_column=timestamp_column,
            ticker_column=panel_ticker_column,
            value_column=value_column,
        )

    csv_files = _uploaded_to_named_bytes(per_ticker_files)
    zip_files = _uploaded_to_named_bytes([per_ticker_zip]) if per_ticker_zip is not None else []
    items = load_batch_items_from_bytes(
        csv_files=csv_files,
        zip_files=zip_files,
        max_files=max_files,
    )
    return build_panel_from_batch_items(
        history_items=items,
        timestamp_column=timestamp_column,
        value_column=value_column,
    )


def _sync_timesfm_lora_adapter_registry(
    job_registry_path: Path,
    adapter_registry_path: Path,
    logger: logging.Logger,
) -> None:
    """Register completed LoRA jobs as adapter entries when artifacts exist."""
    for status in list_lora_jobs(registry_path=job_registry_path):
        try:
            ensure_adapter_registered_from_job(
                job=status,
                registry_path=adapter_registry_path,
            )
        except LoRAAdapterError as exc:
            logger.warning("LoRA adapter registry sync skipped for job %s: %s", status.job_id, exc)


def _render_timesfm_lora_section(settings: AppSettings, logger: logging.Logger) -> None:
    st.markdown("#### LoRA Fine-Tuning Jobs")
    job_registry_path = Path(settings.timesfm_lora_registry_path)
    adapter_registry_path = Path(settings.timesfm_lora_adapter_registry_path)
    output_root = Path(settings.timesfm_lora_output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    lora_mode_options = ["external_script", "in_app"]
    lora_default_mode = (
        settings.timesfm_lora_default_mode
        if settings.timesfm_lora_default_mode in lora_mode_options
        else "external_script"
    )
    lora_mode_index = lora_mode_options.index(lora_default_mode)
    lora_mode = st.selectbox(
        "LoRA execution mode",
        options=lora_mode_options,
        index=lora_mode_index,
        key="timesfm_lora_mode",
        format_func=lambda value: "External script runner" if value == "external_script" else "In-app trainer",
        help="External script keeps compatibility with existing finetune_lora.py; in-app uses managed internal runner.",
    )
    lora_script_path = st.text_input(
        "LoRA script path",
        value=settings.timesfm_lora_script_path,
        key="timesfm_lora_script_path",
        help="Path to finetune_lora.py from official TimesFM examples or your wrapper script.",
        disabled=lora_mode != "external_script",
    )
    lora_base_model_id = st.text_input(
        "Base model ID",
        value=settings.timesfm_model_id,
        key="timesfm_lora_base_model_id",
        help="TimesFM checkpoint identifier used for adapter fine-tuning.",
    )
    lora_adapter_name = st.text_input(
        "Adapter name",
        value="timesfm_lora_adapter",
        key="timesfm_lora_adapter_name",
    )
    lora_dataset = st.file_uploader(
        "LoRA training dataset CSV (optional)",
        type=["csv"],
        key="timesfm_lora_dataset_csv",
    )
    if settings.timesfm_lora_retention_policy == RETENTION_DELETE_RAW:
        st.caption(
            "Raw proprietary upload will be auto-deleted after normalized train/validation artifacts are created."
        )
    else:
        st.caption("Raw proprietary upload retention is enabled by configuration.")

    dataset_preview = _load_csv(lora_dataset) if lora_dataset is not None else None
    dataset_columns = list(dataset_preview.columns) if dataset_preview is not None else []
    with st.expander("Transactional dataset mapping", expanded=dataset_preview is not None):
        if dataset_preview is None:
            st.caption("Upload a CSV to configure schema mapping.")
            lora_timestamp_col = "timestamp"
            lora_value_col = "value"
            lora_entity_col = "__none__"
            lora_feature_columns: list[str] = []
        else:
            st.dataframe(dataset_preview.head(20), use_container_width=True)
            timestamp_default = 0
            value_default = 1 if len(dataset_columns) > 1 else 0
            lora_timestamp_col = st.selectbox(
                "Timestamp column",
                options=dataset_columns,
                index=timestamp_default,
                key="timesfm_lora_timestamp_col",
            )
            lora_value_col = st.selectbox(
                "Value column",
                options=dataset_columns,
                index=value_default if dataset_columns[value_default] != lora_timestamp_col else 0,
                key="timesfm_lora_value_col",
            )
            lora_entity_col = st.selectbox(
                "Entity column (optional)",
                options=["__none__", *dataset_columns],
                index=0,
                key="timesfm_lora_entity_col",
                format_func=lambda value: "None (single series)" if value == "__none__" else value,
            )
            excluded_columns = {lora_timestamp_col, lora_value_col}
            if lora_entity_col != "__none__":
                excluded_columns.add(lora_entity_col)
            lora_feature_columns = st.multiselect(
                "Extra numeric feature columns (optional)",
                options=[column for column in dataset_columns if column not in excluded_columns],
                default=[],
                key="timesfm_lora_feature_columns",
            )

    lora_validation_ratio = st.number_input(
        "Validation ratio",
        min_value=0.05,
        max_value=0.5,
        value=float(settings.timesfm_lora_validation_ratio),
        step=0.05,
        key="timesfm_lora_validation_ratio",
    )
    lora_min_points = st.number_input(
        "Min points per entity",
        min_value=3,
        max_value=10000,
        value=int(settings.timesfm_lora_min_points_per_entity),
        step=1,
        key="timesfm_lora_min_points",
    )
    lora_epochs = st.number_input("LoRA epochs", min_value=1, max_value=200, value=10, key="timesfm_lora_epochs")
    lora_batch_size = st.number_input(
        "LoRA batch size",
        min_value=1,
        max_value=1024,
        value=64,
        key="timesfm_lora_batch_size",
    )
    lora_lr = st.number_input(
        "LoRA learning rate",
        min_value=1e-6,
        max_value=1.0,
        value=5e-5,
        step=1e-5,
        format="%0.6f",
        key="timesfm_lora_lr",
    )
    lora_r = st.number_input("LoRA rank (r)", min_value=1, max_value=256, value=8, key="timesfm_lora_r")
    lora_alpha = st.number_input("LoRA alpha", min_value=1, max_value=512, value=16, key="timesfm_lora_alpha")
    lora_context = st.number_input(
        "LoRA context length",
        min_value=8,
        max_value=4096,
        value=64,
        key="timesfm_lora_context_len",
    )
    lora_horizon = st.number_input(
        "LoRA horizon length",
        min_value=1,
        max_value=2048,
        value=13,
        key="timesfm_lora_horizon_len",
    )
    lora_eval_only = st.checkbox("Evaluation only", value=False, key="timesfm_lora_eval_only")
    lora_extra_args = st.text_input(
        "Additional LoRA args",
        value="",
        key="timesfm_lora_extra_args",
        help="Optional passthrough arguments appended to the command.",
    )

    if st.button("Start LoRA Job", key="timesfm_lora_start_job"):
        try:
            dataset_path: str | None = None
            train_dataset_path: str | None = None
            validation_dataset_path: str | None = None
            dataset_fingerprint = ""
            dataset_spec_payload: dict[str, object] | None = None
            if lora_dataset is not None:
                dataset_job_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                job_data_dir = output_root / "datasets" / f"dataset_{dataset_job_id}"
                spec = TransactionalDatasetSpec(
                    timestamp_column=lora_timestamp_col,
                    value_column=lora_value_col,
                    entity_column=None if lora_entity_col == "__none__" else lora_entity_col,
                    feature_columns=lora_feature_columns,
                    validation_ratio=float(lora_validation_ratio),
                    min_points_per_entity=int(lora_min_points),
                )
                materialized = materialize_lora_dataset_from_csv_bytes(
                    csv_bytes=lora_dataset.getvalue(),
                    output_dir=job_data_dir,
                    spec=spec,
                    retention_policy=settings.timesfm_lora_retention_policy,
                )
                dataset_path = materialized.train_path
                train_dataset_path = materialized.train_path
                validation_dataset_path = materialized.validation_path
                dataset_fingerprint = materialized.fingerprint
                dataset_spec_payload = {
                    "timestamp_column": spec.timestamp_column,
                    "value_column": spec.value_column,
                    "entity_column": spec.entity_column,
                    "feature_columns": spec.feature_columns,
                    "validation_ratio": spec.validation_ratio,
                    "min_points_per_entity": spec.min_points_per_entity,
                }

            job_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            output_dir = output_root / f"job_{job_id}"
            config = LoRAJobConfig(
                mode=lora_mode,
                script_path=lora_script_path,
                output_dir=str(output_dir),
                dataset_path=dataset_path,
                train_dataset_path=train_dataset_path,
                validation_dataset_path=validation_dataset_path,
                eval_only=lora_eval_only,
                epochs=int(lora_epochs),
                batch_size=int(lora_batch_size),
                learning_rate=float(lora_lr),
                lora_r=int(lora_r),
                lora_alpha=int(lora_alpha),
                context_len=int(lora_context),
                horizon_len=int(lora_horizon),
                backend="torch",
                base_model_id=lora_base_model_id,
                adapter_name=lora_adapter_name,
                dataset_fingerprint=dataset_fingerprint,
                dataset_spec=dataset_spec_payload,
                retention_policy=settings.timesfm_lora_retention_policy,
                extra_args=lora_extra_args,
            )
            status = start_lora_job(config=config, registry_path=job_registry_path)
            st.success(f"Started LoRA job `{status.job_id}`.")
            _register_artifact(
                f"timesfm_lora/{status.job_id}_config.json",
                json.dumps(status.config, indent=2).encode("utf-8"),
            )
        except (LoRAJobError, OSError) as exc:
            logger.error("TimesFM LoRA job start failed: %s", exc)
            st.error(str(exc))

    if st.button("Refresh LoRA Jobs", key="timesfm_lora_refresh_jobs"):
        try:
            for status in list_lora_jobs(registry_path=job_registry_path):
                refresh_lora_job(job_id=status.job_id, registry_path=job_registry_path)
            _sync_timesfm_lora_adapter_registry(
                job_registry_path=job_registry_path,
                adapter_registry_path=adapter_registry_path,
                logger=logger,
            )
        except LoRAJobError as exc:
            logger.error("TimesFM LoRA refresh failed: %s", exc)
            st.error(str(exc))

    jobs = list_lora_jobs(registry_path=job_registry_path)
    _sync_timesfm_lora_adapter_registry(
        job_registry_path=job_registry_path,
        adapter_registry_path=adapter_registry_path,
        logger=logger,
    )
    if jobs:
        jobs_df = pd.DataFrame(
            [
                {
                    "job_id": job.job_id,
                    "mode": job.mode,
                    "status": job.status,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                    "return_code": job.return_code,
                    "backend": job.backend,
                    "base_model_id": job.base_model_id,
                    "adapter_path": job.adapter_path,
                    "metrics_path": job.metrics_path,
                    "output_dir": job.output_dir,
                    "log_path": job.log_path,
                }
                for job in jobs
            ]
        )
        st.dataframe(jobs_df, use_container_width=True)
        selected_job_id = st.selectbox(
            "Select LoRA job",
            options=jobs_df["job_id"].tolist(),
            key="timesfm_lora_selected_job",
        )
        if st.button("Stop Selected LoRA Job", key="timesfm_lora_stop_job"):
            try:
                status = stop_lora_job(job_id=selected_job_id, registry_path=job_registry_path)
                st.warning(f"Stopped LoRA job `{status.job_id}`.")
            except LoRAJobError as exc:
                logger.error("TimesFM LoRA stop failed: %s", exc)
                st.error(str(exc))
    else:
        st.info("No LoRA jobs found yet.")

    adapters = list_lora_adapters(registry_path=adapter_registry_path)
    st.markdown("#### Registered LoRA Adapters")
    if adapters:
        adapters_df = pd.DataFrame(
            [
                {
                    "adapter_id": adapter.adapter_id,
                    "name": adapter.name,
                    "backend": adapter.backend,
                    "base_model_id": adapter.base_model_id,
                    "source_job_id": adapter.source_job_id,
                    "created_at": adapter.created_at,
                    "adapter_path": adapter.adapter_path,
                    "metrics_path": adapter.metrics_path,
                }
                for adapter in adapters
            ]
        )
        st.dataframe(adapters_df, use_container_width=True)
    else:
        st.info("No LoRA adapters registered yet.")


def _render_tabfm_tab(
    settings: AppSettings,
    logger: logging.Logger,
    ollama_selection: OllamaModelSelection,
) -> None:
    st.subheader("TabFM: Classification / Regression")
    st.selectbox(
        "Backend",
        options=["torch"],
        index=0,
        format_func=_backend_label,
        key="tabfm_backend",
        disabled=True,
        help="TabFM currently supports PyTorch only.",
    )
    st.caption("JAX backend is disabled for TabFM in this release.")
    ollama_available = ollama_selection.selected_model is not None
    batch_mode = st.checkbox(
        "Batch mode (bulk CSV processing)",
        value=False,
        key="tabfm_batch_mode",
    )
    st.caption(
        "Upload train and prediction CSVs, or use demo data."
        if not batch_mode
        else "Batch mode: one training CSV + many prediction CSV files (or ZIP)."
    )

    input_col, target_col = st.columns(2)
    with input_col:
        train_file = st.file_uploader(
            "Training CSV (features + target column)",
            type=["csv"],
            key="tabfm_train_csv",
        )
        if batch_mode:
            predict_files = st.file_uploader(
                "Prediction CSVs (multi-select)",
                type=["csv"],
                key="tabfm_predict_csv_batch",
                accept_multiple_files=True,
            )
            predict_zip = st.file_uploader(
                "Or upload ZIP with prediction CSV files",
                type=["zip"],
                key="tabfm_predict_zip_batch",
            )
        else:
            predict_file = st.file_uploader(
                "Prediction CSV (features only)",
                type=["csv"],
                key="tabfm_predict_csv",
            )

    with target_col:
        task_mode = st.selectbox(
            "Task mode",
            options=["auto", "classification", "regression"],
            index=0,
            key="tabfm_task_mode",
        )
        use_ollama = st.checkbox(
            "Generate Ollama insight",
            value=ollama_available,
            key="tabfm_ollama",
            disabled=not ollama_available,
        )
        if not ollama_available:
            use_ollama = False
            _ollama_disabled_hint()
        batch_use_ollama = False
        if batch_mode and _toggle_batch_enrichment_controls(
            button_key="tabfm_batch_enrichment_btn",
            state_key="tabfm_batch_enrichment_open",
        ):
            batch_use_ollama = st.checkbox(
                "Enable per-file Ollama summaries in batch",
                value=False,
                key="tabfm_batch_ollama",
                disabled=not ollama_available,
            )
            if not ollama_available:
                batch_use_ollama = False
                _ollama_disabled_hint()

    train_df = _load_csv(train_file) if train_file else _demo_tabfm_train_df()
    st.markdown("**Training Data Preview**")
    st.dataframe(train_df, use_container_width=True)

    default_target_index = (
        train_df.columns.get_loc("risk")
        if "risk" in train_df.columns
        else len(train_df.columns) - 1
    )
    target_column = st.selectbox(
        "Target column in training CSV",
        options=list(train_df.columns),
        index=default_target_index,
        key="tabfm_target_column",
    )

    if batch_mode:
        csv_files = _uploaded_to_named_bytes(predict_files)
        zip_files = (
            _uploaded_to_named_bytes([predict_zip]) if predict_zip is not None else []
        )
        raw_count = len(csv_files)
        if predict_zip is not None:
            raw_count += 1
        st.info(
            f"Batch sources selected: {len(csv_files)} direct CSV files"
            + (" + 1 ZIP file" if predict_zip is not None else "")
            + f" (max {settings.batch_max_files} parsed CSV files)."
        )

        if st.button("Run TabFM Batch", type="primary", key="run_tabfm_batch"):
            try:
                predict_items = load_batch_items_from_bytes(
                    csv_files=csv_files,
                    zip_files=zip_files,
                    max_files=settings.batch_max_files,
                )
                if not predict_items:
                    raise ValidationError("No prediction CSV files were provided for batch mode.")

                with st.spinner("Running TabFM batch..."):
                    batch_result = run_tabfm_batch(
                        train_df=train_df,
                        predict_items=predict_items,
                        target_column=target_column,
                        task_mode=task_mode,
                        retry_count=settings.batch_retry_count,
                    )

                summary_df = batch_result.to_summary_df()
                success_count = int((summary_df["status"] == "success").sum())
                failed_count = int((summary_df["status"] == "failed").sum())
                st.success(
                    f"TabFM batch completed: {success_count} succeeded, {failed_count} failed."
                )
                st.dataframe(summary_df, use_container_width=True)
                _set_pdf_context(
                    context_key="tabfm_batch",
                    context_payload={
                        "report_type": "batch",
                        "model_family": "tabfm",
                        "summary_rows": len(summary_df),
                        "insight_count": 0,
                    },
                )

                _register_artifact("tabfm_batch_train_input.csv", _to_csv_bytes(train_df))
                _register_artifact("tabfm_batch_summary.csv", _to_csv_bytes(summary_df))
                _register_artifact(
                    "tabfm_batch_summary.json",
                    summary_df.to_json(orient="records", indent=2).encode("utf-8"),
                )

                insight_rows: list[dict[str, str]] = []
                insight_sections: list[InsightPdfSection] = []
                per_file_pdf_exports: dict[str, bytes] = {}
                client = (
                    OllamaClient(
                        base_url=settings.ollama_url,
                        model_name=ollama_selection.selected_model or settings.ollama_model,
                    )
                    if batch_use_ollama and ollama_selection.selected_model is not None
                    else None
                )
                for file_result in batch_result.results:
                    if file_result.status != "success" or file_result.output_df is None:
                        continue
                    output_name = f"tabfm_batch_predictions/{Path(file_result.file_name).stem}_predictions.csv"
                    _register_artifact(output_name, _to_csv_bytes(file_result.output_df))

                    if client is not None:
                        try:
                            insight_text = client.generate_insight(
                                _tabfm_insight_prompt(
                                    result_df=file_result.output_df,
                                    task=file_result.task or task_mode,
                                )
                            )
                            insight_rows.append(
                                {"file_name": file_result.file_name, "insight": insight_text}
                            )
                            insight_name = (
                                f"tabfm_batch_insights/{Path(file_result.file_name).stem}_insight.txt"
                            )
                            _register_artifact(insight_name, insight_text.encode("utf-8"))
                            section = InsightPdfSection(
                                title=f"Insight: {file_result.file_name}",
                                insight_text=insight_text,
                                metadata={
                                    "Run Type": "TabFM Batch",
                                    "File": file_result.file_name,
                                    "Task": file_result.task or task_mode,
                                },
                                tables=[("Predictions", file_result.output_df)],
                            )
                            insight_sections.append(section)
                            per_file_pdf = _build_insight_pdf(
                                title=f"TabFM Insight Report: {file_result.file_name}",
                                metadata={
                                    "Ollama model": ollama_selection.selected_model or "Not configured",
                                    "Task": file_result.task or task_mode,
                                    "Scope": "Per-file batch insight",
                                },
                                sections=[section],
                                settings=settings,
                            )
                            pdf_file_name = (
                                f"tabfm_batch_insights/{Path(file_result.file_name).stem}_insight.pdf"
                            )
                            per_file_pdf_exports[pdf_file_name] = per_file_pdf
                            _register_artifact(pdf_file_name, per_file_pdf)
                        except OllamaError as exc:
                            logger.warning(
                                "TabFM batch Ollama insight failed for %s: %s",
                                file_result.file_name,
                                exc,
                            )

                if insight_rows:
                    insights_df = pd.DataFrame(insight_rows)
                    _register_artifact("tabfm_batch_insights.csv", _to_csv_bytes(insights_df))
                    st.markdown("**Batch Ollama Insights**")
                    st.dataframe(insights_df, use_container_width=True)

                    combined_sections = [
                        InsightPdfSection(
                            title="Batch Summary",
                            insight_text=(
                                "TabFM batch AI insight report. "
                                "See per-file sections for model observations."
                            ),
                            metadata={
                                "Run Type": "TabFM Batch",
                                "Files parsed": len(predict_items),
                                "Success count": success_count,
                                "Failed count": failed_count,
                                "Ollama model": ollama_selection.selected_model or "Not configured",
                            },
                            tables=[("Batch Status Summary", summary_df)],
                        ),
                        *insight_sections,
                    ]
                    combined_pdf = _build_insight_pdf(
                        title="TabFM Batch Insight Report",
                        metadata={
                            "Ollama model": ollama_selection.selected_model or "Not configured",
                            "Model source": ollama_selection.source,
                            "Generated sections": len(insight_sections),
                        },
                        sections=combined_sections,
                        settings=settings,
                    )
                    per_file_zip = build_batch_zip(per_file_pdf_exports)
                    _register_artifact("tabfm_batch_insights_report.pdf", combined_pdf)
                    _register_artifact("tabfm_batch_insights_per_file.zip", per_file_zip)
                    st.download_button(
                        label="Download Combined Insights PDF",
                        data=combined_pdf,
                        file_name="tabfm_batch_insights_report.pdf",
                        mime="application/pdf",
                        key="download_tabfm_batch_insights_pdf",
                    )
                    st.download_button(
                        label="Download Per-file Insight PDFs (ZIP)",
                        data=per_file_zip,
                        file_name="tabfm_batch_insights_per_file.zip",
                        mime="application/zip",
                        key="download_tabfm_batch_insights_zip",
                    )
                    _register_pdf_export(
                        export_key="tabfm_batch_combined_pdf",
                        label="Download Latest TabFM Batch Insights PDF",
                        file_name="tabfm_batch_insights_report.pdf",
                        data=combined_pdf,
                        mime="application/pdf",
                    )
                    _register_pdf_export(
                        export_key="tabfm_batch_per_file_zip",
                        label="Download Latest TabFM Per-file Insight PDFs (ZIP)",
                        file_name="tabfm_batch_insights_per_file.zip",
                        data=per_file_zip,
                        mime="application/zip",
                    )
                    _set_pdf_context(
                        context_key="tabfm_batch",
                        context_payload={
                            "report_type": "batch",
                            "model_family": "tabfm",
                            "summary_rows": len(summary_df),
                            "insight_count": len(insight_rows),
                        },
                    )
                else:
                    _render_export_disabled_message(
                        "Enable per-file Ollama summaries in batch and rerun."
                    )

                append_markdown_run(
                    output_path=settings.output_markdown_path,
                    section_title="TabFM Batch Run",
                    body_markdown=(
                        f"```text\n{summary_df.to_string(index=False)}\n```\n\n"
                        f"Batch files parsed: {len(predict_items)}\n"
                    ),
                )
            except (ValidationError, TabFMRuntimeError) as exc:
                logger.error("TabFM batch validation/runtime error: %s", exc)
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover - UI guardrail
                logger.exception("Unexpected TabFM batch failure")
                st.error(f"Unexpected TabFM batch error: {exc}")
    else:
        predict_df = _load_csv(predict_file) if predict_file else _demo_tabfm_predict_df()
        if train_file and predict_file is None:
            st.info("Prediction CSV missing: using demo prediction rows.")
        if predict_file and train_file is None:
            st.info("Training CSV missing: using demo training rows.")

        st.markdown("**Prediction Data Preview**")
        st.dataframe(predict_df, use_container_width=True)

        if st.button("Run TabFM Prediction", type="primary", key="run_tabfm"):
            try:
                with st.spinner("Running TabFM..."):
                    result = run_tabfm_prediction(
                        train_df=train_df,
                        predict_df=predict_df,
                        target_column=target_column,
                        task_mode=task_mode,
                    )

                st.success(f"TabFM completed ({result.task}).")
                st.dataframe(result.output_df, use_container_width=True)
                _set_pdf_context(
                    context_key="tabfm_single",
                    context_payload={
                        "report_type": "single",
                        "model_family": "tabfm",
                        "task": result.task,
                        "prediction_rows": len(result.output_df),
                        "insight_available": False,
                    },
                )
                _register_artifact("tabfm_train_input.csv", _to_csv_bytes(train_df))
                _register_artifact("tabfm_predict_input.csv", _to_csv_bytes(predict_df))
                _register_artifact("tabfm_predictions.csv", _to_csv_bytes(result.output_df))
                st.download_button(
                    label="Download TabFM Predictions CSV",
                    data=_to_csv_bytes(result.output_df),
                    file_name="tabfm_predictions.csv",
                    mime="text/csv",
                    key="download_tabfm_predictions",
                )

                insight_text = ""
                if use_ollama and ollama_selection.selected_model is not None:
                    try:
                        client = OllamaClient(
                            base_url=settings.ollama_url,
                            model_name=ollama_selection.selected_model or settings.ollama_model,
                        )
                        insight_text = client.generate_insight(
                            _tabfm_insight_prompt(result.output_df, result.task)
                        )
                        st.markdown("**Ollama Insight**")
                        st.success(insight_text)
                        _register_artifact("tabfm_insight.txt", insight_text.encode("utf-8"))
                        if can_export_insight(insight_text):
                            section = InsightPdfSection(
                                title="TabFM Prediction Insight",
                                insight_text=insight_text,
                                metadata={
                                    "Run Type": "TabFM Single",
                                    "Task": result.task,
                                },
                                tables=[("Predictions", result.output_df)],
                            )
                            insight_pdf = _build_insight_pdf(
                                title="TabFM Insight Report",
                                metadata={
                                    "Ollama model": ollama_selection.selected_model,
                                    "Model source": ollama_selection.source,
                                },
                                sections=[section],
                                settings=settings,
                            )
                            _register_artifact("tabfm_insight_report.pdf", insight_pdf)
                            st.download_button(
                                label="Download Insight PDF",
                                data=insight_pdf,
                                file_name="tabfm_insight_report.pdf",
                                mime="application/pdf",
                                key="download_tabfm_single_insight_pdf",
                            )
                            _register_pdf_export(
                                export_key="tabfm_single_pdf",
                                label="Download Latest TabFM Insight PDF",
                                file_name="tabfm_insight_report.pdf",
                                data=insight_pdf,
                                mime="application/pdf",
                            )
                            _set_pdf_context(
                                context_key="tabfm_single",
                                context_payload={
                                    "report_type": "single",
                                    "model_family": "tabfm",
                                    "task": result.task,
                                    "prediction_rows": len(result.output_df),
                                    "insight_available": True,
                                },
                            )
                    except OllamaError as exc:
                        logger.warning("TabFM insight generation failed: %s", exc)
                        st.warning(f"Ollama insight unavailable: {exc}")

                if not can_export_insight(insight_text):
                    _render_export_disabled_message("Generate an Ollama insight and rerun.")

                append_markdown_run(
                    output_path=settings.output_markdown_path,
                    section_title="TabFM Run",
                    body_markdown=(
                        f"### Task: {result.task}\n"
                        f"```text\n{result.output_df.to_string(index=False)}\n```\n\n"
                        f"### Ollama Insight\n{insight_text or 'Not generated.'}\n"
                    ),
                )
            except (ValidationError, TabFMRuntimeError) as exc:
                logger.error("TabFM validation/runtime error: %s", exc)
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover - UI guardrail
                logger.exception("Unexpected TabFM failure")
                st.error(f"Unexpected TabFM error: {exc}")


def _render_timesfm_tab(
    settings: AppSettings,
    logger: logging.Logger,
    ollama_selection: OllamaModelSelection,
) -> None:
    st.subheader("TimesFM: Forecasting")
    backend_options = ["torch", "jax"]
    default_backend_index = (
        backend_options.index(settings.default_backend)
        if settings.default_backend in backend_options
        else 0
    )
    selected_backend = st.selectbox(
        "Backend",
        options=backend_options,
        index=default_backend_index,
        format_func=_backend_label,
        key="timesfm_backend",
        help="Select PyTorch or JAX backend for TimesFM forecast and backtest.",
    )
    selected_model_id = _resolve_timesfm_model_id(settings, selected_backend)
    st.caption(f"Active TimesFM model: `{selected_model_id}`")
    if selected_backend == "jax":
        st.caption("JAX backend requires optional dependency `timesfm[flax]`.")

    adapter_registry_path = Path(settings.timesfm_lora_adapter_registry_path)
    selected_lora_adapter_path: str | None = None
    selected_lora_adapter_label = "None (base model only)"
    adapters_for_backend = list_lora_adapters(
        registry_path=adapter_registry_path,
        backend=selected_backend,
    )
    if adapters_for_backend:
        adapter_options = ["none", *[adapter.adapter_id for adapter in adapters_for_backend]]
        adapter_labels = {
            "none": "None (base model only)",
            **{
                adapter.adapter_id: f"{adapter.name} ({adapter.adapter_id})"
                for adapter in adapters_for_backend
            },
        }
        selected_adapter_id = st.selectbox(
            "LoRA adapter for inference",
            options=adapter_options,
            index=0,
            key="timesfm_inference_adapter",
            format_func=lambda value: adapter_labels.get(value, value),
            help="Optional LoRA adapter loaded on top of the selected TimesFM base model.",
        )
        selected_lora_adapter_label = adapter_labels.get(selected_adapter_id, selected_lora_adapter_label)
        if selected_adapter_id != "none":
            try:
                selected_lora_adapter_path = resolve_lora_adapter_path(
                    registry_path=adapter_registry_path,
                    adapter_id=selected_adapter_id,
                )
            except LoRAAdapterError as exc:
                logger.error("TimesFM LoRA adapter resolution failed: %s", exc)
                st.error(str(exc))
                selected_lora_adapter_path = None
                selected_lora_adapter_label = "None (adapter resolution failed)"
    else:
        st.caption("No LoRA adapters registered for the selected backend.")

    ollama_available = ollama_selection.selected_model is not None
    batch_mode = st.checkbox(
        "Batch mode (bulk CSV processing)",
        value=False,
        key="timesfm_batch_mode",
    )
    st.caption(
        "Upload univariate history CSV or use demo data."
        if not batch_mode
        else "Batch mode: many history CSV files (or ZIP) processed sequentially."
    )

    if batch_mode:
        history_files = st.file_uploader(
            "History CSV files (multi-select)",
            type=["csv"],
            key="timesfm_history_csv_batch",
            accept_multiple_files=True,
        )
        history_zip = st.file_uploader(
            "Or upload ZIP with history CSV files",
            type=["zip"],
            key="timesfm_history_zip_batch",
        )
        preview_df = _demo_timesfm_history_df()
    else:
        timesfm_file = st.file_uploader(
            "History CSV (timestamp + value)",
            type=["csv"],
            key="timesfm_history_csv",
        )
        preview_df = _load_csv(timesfm_file) if timesfm_file else _demo_timesfm_history_df()

    st.markdown("**History Preview**")
    st.dataframe(preview_df, use_container_width=True)

    timestamp_col, value_col = st.columns(2)
    with timestamp_col:
        timestamp_column = st.selectbox(
            "Timestamp column",
            options=list(preview_df.columns),
            index=0,
            key="timesfm_timestamp_column",
        )
    with value_col:
        default_value_index = 1 if len(preview_df.columns) > 1 else 0
        value_column = st.selectbox(
            "Value column",
            options=list(preview_df.columns),
            index=default_value_index,
            key="timesfm_value_column",
        )

    control_left, control_right = st.columns(2)
    with control_left:
        horizon = st.number_input(
            "Forecast horizon",
            min_value=1,
            max_value=1000,
            value=settings.default_forecast_horizon,
            step=1,
            key="timesfm_horizon",
        )
        max_context = st.number_input(
            "Max context",
            min_value=16,
            max_value=16384,
            value=settings.default_max_context,
            step=16,
            key="timesfm_max_context",
        )
    with control_right:
        max_horizon = st.number_input(
            "Max horizon",
            min_value=1,
            max_value=1000,
            value=settings.default_max_horizon,
            step=1,
            key="timesfm_max_horizon",
        )
        chart_mode = st.selectbox(
            "Chart mode",
            options=["Plotly", "Streamlit"],
            index=0,
            key="timesfm_chart_mode",
            help="Use Plotly for interactive charts, or fallback to Streamlit charts.",
        )
        include_quantiles = st.checkbox("Include quantile intervals (p10/p90)", value=True)
        use_ollama = st.checkbox(
            "Generate Ollama insight",
            value=ollama_available,
            key="timesfm_ollama",
            disabled=not ollama_available,
        )
        if not ollama_available:
            use_ollama = False
            _ollama_disabled_hint()
        run_backtest = st.checkbox("Run holdout backtest", value=False, key="timesfm_backtest")
        holdout_points = st.number_input(
            "Backtest holdout points",
            min_value=1,
            max_value=365,
            value=3,
            step=1,
            key="timesfm_holdout_points",
            disabled=not run_backtest,
        )

    with st.expander("TimesFM XReg Covariates (Single + Batch)", expanded=False):
        use_xreg_main = st.checkbox(
            "Enable XReg covariates for TimesFM single/batch runs",
            value=False,
            key="timesfm_main_use_xreg",
        )
        single_ticker = st.text_input(
            "Single run ticker",
            value="AAPL",
            key="timesfm_main_single_ticker",
            disabled=(not use_xreg_main) or batch_mode,
            help="Required when XReg is enabled in single mode.",
        )
        if batch_mode and use_xreg_main:
            st.caption("Batch mode infers ticker from each CSV filename stem (e.g., AAPL.csv -> AAPL).")
        macro_series_raw_main = st.text_input(
            "FRED macro series IDs (comma-separated)",
            value=settings.covariate_default_macro_ids,
            key="timesfm_main_macro_series",
            disabled=not use_xreg_main,
        )
        sector_tickers_raw_main = st.text_input(
            "Yahoo sector tickers (comma-separated)",
            value=settings.covariate_default_sector_tickers,
            key="timesfm_main_sector_tickers",
            disabled=not use_xreg_main,
        )
        fred_api_key_main = st.text_input(
            "FRED API key",
            value=settings.fred_api_key,
            type="password",
            key="timesfm_main_fred_api_key",
            disabled=not use_xreg_main,
        )
        xreg_mode_main = st.selectbox(
            "XReg mode",
            options=["xreg + timesfm", "timesfm + xreg"],
            index=0 if settings.timesfm_xreg_mode == "xreg + timesfm" else 1,
            key="timesfm_main_xreg_mode",
            disabled=not use_xreg_main,
        )
        override_covariate_file_main = st.file_uploader(
            "Override covariates CSV (optional)",
            type=["csv"],
            key="timesfm_main_cov_override_csv",
            disabled=not use_xreg_main,
        )

    with st.expander("Advanced TimesFM Features", expanded=False):
        st.markdown(
            "Run multi-asset forecasts with XReg covariates, sentiment bias, "
            "portfolio optimization, and framework backtesting."
        )
        advanced_data_format = st.selectbox(
            "Advanced dataset format",
            options=["panel", "per_ticker_files"],
            index=0,
            key="timesfm_advanced_data_format",
            format_func=lambda value: "Single panel CSV" if value == "panel" else "Per-ticker files/ZIP",
        )
        panel_ticker_column = st.text_input(
            "Panel ticker column",
            value="ticker",
            key="timesfm_advanced_panel_ticker_col",
            disabled=advanced_data_format != "panel",
        )
        advanced_panel_file = st.file_uploader(
            "Advanced panel CSV",
            type=["csv"],
            key="timesfm_advanced_panel_csv",
            disabled=advanced_data_format != "panel",
        )
        advanced_per_ticker_files = st.file_uploader(
            "Advanced per-ticker history CSV files",
            type=["csv"],
            key="timesfm_advanced_per_ticker_files",
            accept_multiple_files=True,
            disabled=advanced_data_format == "panel",
        )
        advanced_per_ticker_zip = st.file_uploader(
            "Advanced per-ticker ZIP",
            type=["zip"],
            key="timesfm_advanced_per_ticker_zip",
            disabled=advanced_data_format == "panel",
        )

        use_xreg = st.checkbox("Enable XReg covariates", value=False, key="timesfm_advanced_use_xreg")
        macro_series_raw = st.text_input(
            "FRED macro series IDs (comma-separated)",
            value=settings.covariate_default_macro_ids,
            key="timesfm_advanced_macro_series",
            disabled=not use_xreg,
        )
        sector_tickers_raw = st.text_input(
            "Yahoo sector tickers (comma-separated)",
            value=settings.covariate_default_sector_tickers,
            key="timesfm_advanced_sector_tickers",
            disabled=not use_xreg,
        )
        fred_api_key = st.text_input(
            "FRED API key",
            value=settings.fred_api_key,
            type="password",
            key="timesfm_advanced_fred_api_key",
            disabled=not use_xreg,
        )
        xreg_mode = st.selectbox(
            "XReg mode",
            options=["xreg + timesfm", "timesfm + xreg"],
            index=0 if settings.timesfm_xreg_mode == "xreg + timesfm" else 1,
            key="timesfm_advanced_xreg_mode",
            disabled=not use_xreg,
        )
        override_covariate_file = st.file_uploader(
            "Override covariates CSV (optional)",
            type=["csv"],
            key="timesfm_advanced_cov_override_csv",
            disabled=not use_xreg,
        )

        use_sentiment_bias = st.checkbox(
            "Enable real-time news sentiment bias",
            value=False,
            key="timesfm_advanced_use_sentiment",
        )
        alpha_vantage_api_key = st.text_input(
            "Alpha Vantage API key",
            value=settings.alpha_vantage_api_key,
            type="password",
            key="timesfm_advanced_alpha_vantage_key",
            disabled=not use_sentiment_bias,
        )
        sentiment_lookback_hours = st.number_input(
            "Sentiment lookback hours",
            min_value=1,
            max_value=168,
            value=settings.sentiment_lookback_hours,
            step=1,
            key="timesfm_advanced_sentiment_lookback",
            disabled=not use_sentiment_bias,
        )
        sentiment_bias_strength = st.number_input(
            "Sentiment bias strength",
            min_value=0.0,
            max_value=2.0,
            value=float(settings.sentiment_bias_strength),
            step=0.05,
            key="timesfm_advanced_sentiment_strength",
            disabled=not use_sentiment_bias,
        )
        sentiment_bias_decay = st.number_input(
            "Sentiment bias decay",
            min_value=0.5,
            max_value=1.0,
            value=float(settings.sentiment_bias_decay),
            step=0.01,
            key="timesfm_advanced_sentiment_decay",
            disabled=not use_sentiment_bias,
        )

        run_portfolio_optimization = st.checkbox(
            "Run multi-asset portfolio optimization",
            value=True,
            key="timesfm_advanced_run_portfolio_opt",
        )
        portfolio_risk_aversion = st.number_input(
            "Portfolio risk aversion",
            min_value=0.0001,
            max_value=100.0,
            value=float(settings.portfolio_risk_aversion),
            step=0.1,
            key="timesfm_advanced_portfolio_risk_aversion",
            disabled=not run_portfolio_optimization,
        )
        portfolio_max_weight = st.number_input(
            "Portfolio max single-asset weight",
            min_value=0.05,
            max_value=1.0,
            value=float(settings.portfolio_max_weight),
            step=0.05,
            key="timesfm_advanced_portfolio_max_weight",
            disabled=not run_portfolio_optimization,
        )

        run_framework_backtesting = st.checkbox(
            "Run framework backtesting",
            value=False,
            key="timesfm_advanced_run_framework_backtest",
        )
        framework_modes = st.multiselect(
            "Backtesting modes",
            options=["walk_forward", "rolling_window"],
            default=[settings.backtest_default_mode],
            key="timesfm_advanced_framework_modes",
            disabled=not run_framework_backtesting,
        )
        framework_folds = st.number_input(
            "Backtesting folds",
            min_value=1,
            max_value=20,
            value=settings.backtest_default_folds,
            step=1,
            key="timesfm_advanced_framework_folds",
            disabled=not run_framework_backtesting,
        )
        framework_min_train = st.number_input(
            "Backtesting min train size",
            min_value=10,
            max_value=5000,
            value=settings.backtest_min_train_size,
            step=1,
            key="timesfm_advanced_framework_min_train",
            disabled=not run_framework_backtesting,
        )
        framework_rolling_window = st.number_input(
            "Backtesting rolling window",
            min_value=10,
            max_value=5000,
            value=settings.backtest_rolling_window,
            step=1,
            key="timesfm_advanced_framework_rolling_window",
            disabled=not run_framework_backtesting,
        )

        if st.button("Run Advanced Multi-Asset Forecast", type="primary", key="run_timesfm_advanced"):
            try:
                panel_df = _build_timesfm_panel_input(
                    data_format=advanced_data_format,
                    panel_file=advanced_panel_file,
                    per_ticker_files=advanced_per_ticker_files,
                    per_ticker_zip=advanced_per_ticker_zip,
                    max_files=settings.batch_max_files,
                    timestamp_column=timestamp_column,
                    value_column=value_column,
                    panel_ticker_column=panel_ticker_column,
                )
                override_covariates_df = _load_covariate_override_csv(override_covariate_file)
                working_panel_df, covariate_columns = _apply_timesfm_xreg_covariates(
                    panel_df=panel_df,
                    use_xreg=use_xreg,
                    macro_series_raw=macro_series_raw,
                    sector_tickers_raw=sector_tickers_raw,
                    fred_api_key=fred_api_key,
                    override_covariates_df=override_covariates_df,
                )

                sentiment_scores: dict[str, float] = {}
                sentiment_scores_df: pd.DataFrame | None = None
                sentiment_diagnostics: dict[str, object] = {
                    "status": "disabled",
                    "error_message": None,
                    "requested_ticker_count": 0,
                    "requested_tickers": [],
                    "lookback_hours": int(sentiment_lookback_hours),
                    "fetched_rows": 0,
                    "scored_ticker_count": 0,
                    "coverage_ratio": 0.0,
                    "non_zero_score_count": 0,
                }
                if use_sentiment_bias:
                    if not alpha_vantage_api_key.strip():
                        raise ValidationError("Alpha Vantage API key is required for sentiment bias.")
                    sentiment_provider = AlphaVantageSentimentProvider(
                        api_key=alpha_vantage_api_key.strip()
                    )
                    sentiment_result = fetch_sentiment_scores_with_diagnostics(
                        provider=sentiment_provider,
                        tickers=working_panel_df["ticker"].unique().tolist()
                        if "ticker" in working_panel_df.columns
                        else [],
                        lookback_hours=int(sentiment_lookback_hours),
                        fail_open=True,
                    )
                    sentiment_scores_df = sentiment_result.scores_df
                    sentiment_scores = sentiment_result.scores_by_ticker
                    sentiment_diagnostics = {
                        "status": sentiment_result.status,
                        "error_message": sentiment_result.error_message,
                        **sentiment_result.metadata,
                    }
                    if sentiment_result.status == "degraded":
                        st.warning(
                            "Sentiment feed unavailable; continued without sentiment bias. "
                            f"Reason: {sentiment_result.error_message}"
                        )
                    st.markdown("**Sentiment Diagnostics**")
                    st.json(sentiment_diagnostics)

                advanced_result = run_timesfm_multi_asset_forecast(
                    panel_df=working_panel_df,
                    horizon=int(horizon),
                    max_context=int(max_context),
                    max_horizon=int(max_horizon),
                    normalize_inputs=settings.normalize_inputs,
                    include_quantiles=include_quantiles,
                    model_id=selected_model_id,
                    backend=selected_backend,
                    use_xreg=use_xreg,
                    covariate_columns=covariate_columns,
                    xreg_mode=xreg_mode,
                    sentiment_scores=sentiment_scores,
                    sentiment_strength=float(sentiment_bias_strength),
                    sentiment_decay=float(sentiment_bias_decay),
                    lora_adapter_path=selected_lora_adapter_path,
                )
                forecast_panel = advanced_result.forecast_df
                st.success(
                    f"Advanced TimesFM completed for {forecast_panel['ticker'].nunique()} assets."
                )
                st.dataframe(forecast_panel, use_container_width=True)

                _register_artifact("timesfm_advanced_panel_input.csv", _to_csv_bytes(working_panel_df))
                _register_artifact("timesfm_advanced_forecast.csv", _to_csv_bytes(forecast_panel))
                _register_artifact(
                    "timesfm_advanced_sentiment_scores.json",
                    json.dumps(sentiment_scores, indent=2).encode("utf-8"),
                )
                _register_artifact(
                    "timesfm_advanced_sentiment_diagnostics.json",
                    json.dumps(sentiment_diagnostics, indent=2).encode("utf-8"),
                )
                if sentiment_scores_df is not None and not sentiment_scores_df.empty:
                    _register_artifact(
                        "timesfm_advanced_sentiment_feed.csv",
                        _to_csv_bytes(sentiment_scores_df),
                    )

                selected_adv_ticker = st.selectbox(
                    "Advanced chart ticker",
                    options=sorted(forecast_panel["ticker"].astype(str).unique().tolist()),
                    key="timesfm_advanced_chart_ticker",
                )
                selected_history = working_panel_df[working_panel_df["ticker"] == selected_adv_ticker][
                    ["timestamp", "value"]
                ].copy()
                selected_forecast = forecast_panel[forecast_panel["ticker"] == selected_adv_ticker].copy()
                _render_timesfm_forecast_chart(
                    history_df=selected_history,
                    forecast_df=selected_forecast,
                    chart_mode=chart_mode,
                    include_quantiles=include_quantiles,
                )

                if run_portfolio_optimization:
                    validate_portfolio_forecast_inputs(
                        panel_df=working_panel_df,
                        forecast_df=forecast_panel,
                    )
                    expected_returns = derive_expected_returns(
                        forecast_df=forecast_panel,
                        panel_df=working_panel_df,
                    )
                    normalized_expected_returns = {
                        str(ticker).strip().upper(): float(value)
                        for ticker, value in expected_returns.items()
                    }
                    covariance = build_covariance_from_panel(working_panel_df)
                    covariance = covariance.copy()
                    covariance.index = covariance.index.map(lambda value: str(value).strip().upper())
                    covariance.columns = covariance.columns.map(lambda value: str(value).strip().upper())
                    aligned_tickers = sorted(
                        set(normalized_expected_returns).intersection(set(covariance.index)).intersection(
                            set(covariance.columns)
                        )
                    )
                    if len(aligned_tickers) < 2:
                        raise ValidationError(
                            "Portfolio optimization requires at least two overlapping tickers "
                            "between forecast expected returns and covariance history."
                        )
                    covariance_used = covariance.reindex(index=aligned_tickers, columns=aligned_tickers)
                    expected_returns_df = pd.DataFrame(
                        {
                            "ticker": aligned_tickers,
                            "expected_return": [
                                normalized_expected_returns[ticker] for ticker in aligned_tickers
                            ],
                        }
                    ).sort_values("expected_return", ascending=False, ignore_index=True)
                    portfolio_result = optimize_mean_variance_long_only(
                        expected_returns=normalized_expected_returns,
                        covariance_matrix=covariance_used,
                        risk_aversion=float(portfolio_risk_aversion),
                        max_weight=float(portfolio_max_weight),
                    )
                    st.markdown("**Portfolio Optimization (Mean-Variance Long-only)**")
                    met_col1, met_col2 = st.columns(2)
                    met_col1.metric("Expected Return", f"{portfolio_result.expected_return:.4f}")
                    met_col2.metric("Expected Volatility", f"{portfolio_result.expected_volatility:.4f}")
                    st.markdown("**Portfolio Diagnostics**")
                    diag_col1, diag_col2 = st.columns(2)
                    with diag_col1:
                        st.caption("Expected Returns Used")
                        st.dataframe(expected_returns_df, use_container_width=True)
                    with diag_col2:
                        st.caption("Covariance Matrix Used")
                        st.dataframe(covariance_used, use_container_width=True)
                    st.dataframe(portfolio_result.weights_df, use_container_width=True)
                    covariance_export = covariance_used.copy()
                    covariance_export.index.name = "ticker"
                    _register_artifact(
                        "timesfm_advanced_expected_returns.csv",
                        _to_csv_bytes(expected_returns_df),
                    )
                    _register_artifact(
                        "timesfm_advanced_covariance.csv",
                        _to_csv_bytes(covariance_export.reset_index()),
                    )
                    _register_artifact(
                        "timesfm_advanced_portfolio_weights.csv",
                        _to_csv_bytes(portfolio_result.weights_df),
                    )

                if run_framework_backtesting:
                    if not framework_modes:
                        raise ValidationError("Select at least one framework backtesting mode.")
                    for framework_mode in framework_modes:
                        framework_result = run_timesfm_backtesting_framework(
                            panel_df=working_panel_df,
                            mode=framework_mode,
                            folds=int(framework_folds),
                            horizon=int(horizon),
                            max_context=int(max_context),
                            max_horizon=int(max_horizon),
                            normalize_inputs=settings.normalize_inputs,
                            include_quantiles=include_quantiles,
                            model_id=selected_model_id,
                            backend=selected_backend,
                            min_train_size=int(framework_min_train),
                            rolling_window=int(framework_rolling_window),
                            lora_adapter_path=selected_lora_adapter_path,
                        )
                        st.markdown(f"**Framework Backtesting: {framework_mode}**")
                        st.caption(
                            "Historical Validation Windows (TimesFM vs naive vs seasonal_naive)"
                        )
                        st.dataframe(framework_result.fold_metrics_df, use_container_width=True)
                        st.caption("Aggregate Metrics (by model)")
                        st.dataframe(framework_result.aggregate_metrics_df, use_container_width=True)
                        _register_artifact(
                            f"timesfm_advanced_backtest_{framework_mode}_folds.csv",
                            _to_csv_bytes(framework_result.fold_metrics_df),
                        )
                        _register_artifact(
                            f"timesfm_advanced_backtest_{framework_mode}_historical_windows.csv",
                            _to_csv_bytes(framework_result.fold_metrics_df),
                        )
                        _register_artifact(
                            f"timesfm_advanced_backtest_{framework_mode}_aggregate.csv",
                            _to_csv_bytes(framework_result.aggregate_metrics_df),
                        )

                append_markdown_run(
                    output_path=settings.output_markdown_path,
                    section_title="TimesFM Advanced Run",
                    body_markdown=(
                        f"Advanced assets forecasted: {forecast_panel['ticker'].nunique()}\n\n"
                        f"Adapter: {selected_lora_adapter_label}\n\n"
                        f"```text\n{forecast_panel.head(20).to_string(index=False)}\n```\n"
                    ),
                )
            except (
                ValidationError,
                TimesFMRuntimeError,
                CovariateProviderError,
                PortfolioOptimizationError,
            ) as exc:
                logger.error("Advanced TimesFM validation/runtime error: %s", exc)
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover - UI guardrail
                logger.exception("Unexpected advanced TimesFM failure")
                st.error(f"Unexpected advanced TimesFM error: {exc}")

        _render_timesfm_lora_section(settings=settings, logger=logger)

    if batch_mode:
        csv_files = _uploaded_to_named_bytes(history_files)
        zip_files = _uploaded_to_named_bytes([history_zip]) if history_zip is not None else []
        st.info(
            f"Batch sources selected: {len(csv_files)} direct CSV files"
            + (" + 1 ZIP file" if history_zip is not None else "")
            + f" (max {settings.batch_max_files} parsed CSV files)."
        )

        batch_run_backtest = False
        batch_use_ollama = False
        if _toggle_batch_enrichment_controls(
            button_key="timesfm_batch_enrichment_btn",
            state_key="timesfm_batch_enrichment_open",
        ):
            batch_run_backtest = st.checkbox(
                "Enable per-file backtesting in batch",
                value=False,
                key="timesfm_batch_backtest",
            )
            batch_use_ollama = st.checkbox(
                "Enable per-file Ollama summaries in batch",
                value=False,
                key="timesfm_batch_ollama",
                disabled=not ollama_available,
            )
            if not ollama_available:
                batch_use_ollama = False
                _ollama_disabled_hint()
            if batch_run_backtest:
                holdout_points = st.number_input(
                    "Batch holdout points",
                    min_value=1,
                    max_value=365,
                    value=3,
                    step=1,
                    key="timesfm_batch_holdout_points",
                )

        if st.button("Run TimesFM Batch", type="primary", key="run_timesfm_batch"):
            try:
                history_items_raw = load_batch_items_from_bytes(
                    csv_files=csv_files,
                    zip_files=zip_files,
                    max_files=settings.batch_max_files,
                )
                if not history_items_raw:
                    raise ValidationError("No history CSV files were provided for batch mode.")
                history_items = [
                    BatchInputItem(
                        name=item.name,
                        dataframe=validate_timesfm_input(
                            item.dataframe,
                            timestamp_column=timestamp_column,
                            value_column=value_column,
                        ),
                    )
                    for item in history_items_raw
                ]

                batch_ticker_by_file: dict[str, str] = {}
                batch_covariates_by_file: dict[str, pd.DataFrame] = {}
                batch_covariate_columns: list[str] = []
                if use_xreg_main:
                    batch_rows: list[pd.DataFrame] = []
                    for item in history_items:
                        ticker = _infer_batch_ticker(item.name)
                        batch_ticker_by_file[item.name] = ticker
                        batch_rows.append(
                            item.dataframe.assign(ticker=ticker)[
                                ["timestamp", "ticker", "value"]
                            ]
                        )
                    batch_panel_df = pd.concat(batch_rows, ignore_index=True).sort_values(
                        ["ticker", "timestamp"]
                    )
                    override_covariates_df = _load_covariate_override_csv(
                        override_covariate_file_main
                    )
                    xreg_panel_df, batch_covariate_columns = _apply_timesfm_xreg_covariates(
                        panel_df=batch_panel_df,
                        use_xreg=use_xreg_main,
                        macro_series_raw=macro_series_raw_main,
                        sector_tickers_raw=sector_tickers_raw_main,
                        fred_api_key=fred_api_key_main,
                        override_covariates_df=override_covariates_df,
                    )
                    for item in history_items:
                        ticker = batch_ticker_by_file[item.name]
                        ticker_covariates = xreg_panel_df[
                            xreg_panel_df["ticker"] == ticker
                        ][["timestamp", *batch_covariate_columns]].copy()
                        batch_covariates_by_file[item.name] = ticker_covariates
                    _register_artifact(
                        "timesfm_batch_xreg_panel.csv",
                        _to_csv_bytes(xreg_panel_df),
                    )

                with st.spinner("Running TimesFM batch..."):
                    batch_result = run_timesfm_batch(
                        history_items=history_items,
                        horizon=int(horizon),
                        max_context=int(max_context),
                        max_horizon=int(max_horizon),
                        normalize_inputs=settings.normalize_inputs,
                        include_quantiles=include_quantiles,
                        model_id=selected_model_id,
                        backend=selected_backend,
                        retry_count=settings.batch_retry_count,
                        run_backtest=batch_run_backtest,
                        holdout_points=int(holdout_points),
                        use_xreg=use_xreg_main,
                        ticker_by_file=batch_ticker_by_file if use_xreg_main else None,
                        covariates_df_by_file=batch_covariates_by_file if use_xreg_main else None,
                        covariate_columns=batch_covariate_columns if use_xreg_main else None,
                        xreg_mode=xreg_mode_main,
                        lora_adapter_path=selected_lora_adapter_path,
                    )

                history_by_name = {item.name: item.dataframe for item in history_items}
                summary_df = batch_result.to_summary_df()
                success_count = int((summary_df["status"] == "success").sum())
                failed_count = int((summary_df["status"] == "failed").sum())
                st.success(
                    f"TimesFM batch completed: {success_count} succeeded, {failed_count} failed."
                )
                st.dataframe(summary_df, use_container_width=True)
                successful_results = [
                    result
                    for result in batch_result.results
                    if result.status == "success" and result.output_df is not None
                ]
                if successful_results:
                    st.markdown("**Batch Forecast Charts**")
                    selected_file_name = st.selectbox(
                        "Select successful file for chart preview",
                        options=[result.file_name for result in successful_results],
                        key="timesfm_batch_chart_file",
                    )
                    selected_result = next(
                        (result for result in successful_results if result.file_name == selected_file_name),
                        None,
                    )
                    if selected_result is not None:
                        selected_history_df = history_by_name.get(selected_result.file_name)
                        if selected_history_df is not None:
                            _render_timesfm_forecast_chart(
                                history_df=selected_history_df,
                                forecast_df=selected_result.output_df,
                                chart_mode=chart_mode,
                                include_quantiles=include_quantiles,
                            )
                        if batch_run_backtest and selected_result.comparison_df is not None:
                            _render_timesfm_backtest_charts(
                                comparison_df=selected_result.comparison_df,
                                chart_mode=chart_mode,
                            )
                _set_pdf_context(
                    context_key="timesfm_batch",
                    context_payload={
                        "report_type": "batch",
                        "model_family": "timesfm",
                        "summary_rows": len(summary_df),
                        "insight_count": 0,
                    },
                )
                _register_artifact("timesfm_batch_summary.csv", _to_csv_bytes(summary_df))
                _register_artifact(
                    "timesfm_batch_summary.json",
                    summary_df.to_json(orient="records", indent=2).encode("utf-8"),
                )

                insight_rows: list[dict[str, str]] = []
                insight_sections: list[InsightPdfSection] = []
                per_file_pdf_exports: dict[str, bytes] = {}
                client = (
                    OllamaClient(
                        base_url=settings.ollama_url,
                        model_name=ollama_selection.selected_model or settings.ollama_model,
                    )
                    if batch_use_ollama and ollama_selection.selected_model is not None
                    else None
                )
                for file_result in batch_result.results:
                    if file_result.status != "success" or file_result.output_df is None:
                        continue

                    forecast_name = (
                        f"timesfm_batch_forecasts/{Path(file_result.file_name).stem}_forecast.csv"
                    )
                    _register_artifact(forecast_name, _to_csv_bytes(file_result.output_df))

                    if file_result.comparison_df is not None:
                        backtest_name = (
                            f"timesfm_batch_backtests/{Path(file_result.file_name).stem}_backtest.csv"
                        )
                        _register_artifact(backtest_name, _to_csv_bytes(file_result.comparison_df))
                    if file_result.metrics is not None:
                        metrics_name = (
                            f"timesfm_batch_backtests/{Path(file_result.file_name).stem}_metrics.json"
                        )
                        _register_artifact(
                            metrics_name,
                            json.dumps(file_result.metrics, indent=2).encode("utf-8"),
                        )

                    if client is not None:
                        try:
                            insight_text = client.generate_insight(
                                _timesfm_insight_prompt(
                                    forecast_df=file_result.output_df,
                                    horizon=int(horizon),
                                )
                            )
                            insight_rows.append(
                                {"file_name": file_result.file_name, "insight": insight_text}
                            )
                            insight_name = (
                                f"timesfm_batch_insights/{Path(file_result.file_name).stem}_insight.txt"
                            )
                            _register_artifact(insight_name, insight_text.encode("utf-8"))
                            history_df_for_chart = history_by_name.get(file_result.file_name)
                            chart_png = (
                                _build_timesfm_chart_or_none(
                                    logger=logger,
                                    history_df=history_df_for_chart,
                                    forecast_df=file_result.output_df,
                                )
                                if history_df_for_chart is not None
                                else None
                            )
                            section_tables: list[tuple[str, pd.DataFrame]] = [
                                ("Forecast", file_result.output_df)
                            ]
                            if file_result.comparison_df is not None:
                                section_tables.append(("Backtest Comparison", file_result.comparison_df))
                            section = InsightPdfSection(
                                title=f"Insight: {file_result.file_name}",
                                insight_text=insight_text,
                                metadata={
                                    "Run Type": "TimesFM Batch",
                                    "File": file_result.file_name,
                                    "Horizon": int(horizon),
                                },
                                tables=section_tables,
                                metrics=file_result.metrics or {},
                                chart_png=chart_png,
                            )
                            insight_sections.append(section)
                            per_file_pdf = _build_insight_pdf(
                                title=f"TimesFM Insight Report: {file_result.file_name}",
                                metadata={
                                    "Ollama model": ollama_selection.selected_model or "Not configured",
                                    "Horizon": int(horizon),
                                    "Scope": "Per-file batch insight",
                                },
                                sections=[section],
                                settings=settings,
                            )
                            pdf_file_name = (
                                f"timesfm_batch_insights/{Path(file_result.file_name).stem}_insight.pdf"
                            )
                            per_file_pdf_exports[pdf_file_name] = per_file_pdf
                            _register_artifact(pdf_file_name, per_file_pdf)
                        except OllamaError as exc:
                            logger.warning(
                                "TimesFM batch Ollama insight failed for %s: %s",
                                file_result.file_name,
                                exc,
                            )

                if insight_rows:
                    insights_df = pd.DataFrame(insight_rows)
                    _register_artifact("timesfm_batch_insights.csv", _to_csv_bytes(insights_df))
                    st.markdown("**Batch Ollama Insights**")
                    st.dataframe(insights_df, use_container_width=True)

                    combined_sections = [
                        InsightPdfSection(
                            title="Batch Summary",
                            insight_text=(
                                "TimesFM batch AI insight report. "
                                "See per-file sections for detailed trend summaries."
                            ),
                            metadata={
                                "Run Type": "TimesFM Batch",
                                "Files parsed": len(history_items),
                                "Success count": success_count,
                                "Failed count": failed_count,
                                "Ollama model": ollama_selection.selected_model or "Not configured",
                            },
                            tables=[("Batch Status Summary", summary_df)],
                        ),
                        *insight_sections,
                    ]
                    combined_pdf = _build_insight_pdf(
                        title="TimesFM Batch Insight Report",
                        metadata={
                            "Ollama model": ollama_selection.selected_model or "Not configured",
                            "Model source": ollama_selection.source,
                            "Generated sections": len(insight_sections),
                        },
                        sections=combined_sections,
                        settings=settings,
                    )
                    per_file_zip = build_batch_zip(per_file_pdfs=per_file_pdf_exports)
                    _register_artifact("timesfm_batch_insights_report.pdf", combined_pdf)
                    _register_artifact("timesfm_batch_insights_per_file.zip", per_file_zip)
                    st.download_button(
                        label="Download Combined Insights PDF",
                        data=combined_pdf,
                        file_name="timesfm_batch_insights_report.pdf",
                        mime="application/pdf",
                        key="download_timesfm_batch_insights_pdf",
                    )
                    st.download_button(
                        label="Download Per-file Insight PDFs (ZIP)",
                        data=per_file_zip,
                        file_name="timesfm_batch_insights_per_file.zip",
                        mime="application/zip",
                        key="download_timesfm_batch_insights_zip",
                    )
                    _register_pdf_export(
                        export_key="timesfm_batch_combined_pdf",
                        label="Download Latest TimesFM Batch Insights PDF",
                        file_name="timesfm_batch_insights_report.pdf",
                        data=combined_pdf,
                        mime="application/pdf",
                    )
                    _register_pdf_export(
                        export_key="timesfm_batch_per_file_zip",
                        label="Download Latest TimesFM Per-file Insight PDFs (ZIP)",
                        file_name="timesfm_batch_insights_per_file.zip",
                        data=per_file_zip,
                        mime="application/zip",
                    )
                    _set_pdf_context(
                        context_key="timesfm_batch",
                        context_payload={
                            "report_type": "batch",
                            "model_family": "timesfm",
                            "summary_rows": len(summary_df),
                            "insight_count": len(insight_rows),
                        },
                    )
                else:
                    _render_export_disabled_message(
                        "Enable per-file Ollama summaries in batch and rerun."
                    )

                append_markdown_run(
                    output_path=settings.output_markdown_path,
                    section_title="TimesFM Batch Run",
                    body_markdown=(
                        f"```text\n{summary_df.to_string(index=False)}\n```\n\n"
                        f"Adapter: {selected_lora_adapter_label}\n"
                        f"Batch files parsed: {len(history_items)}\n"
                    ),
                )
            except (ValidationError, TimesFMRuntimeError, CovariateProviderError) as exc:
                logger.error("TimesFM batch validation/runtime error: %s", exc)
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover - UI guardrail
                logger.exception("Unexpected TimesFM batch failure")
                st.error(f"Unexpected TimesFM batch error: {exc}")
    else:
        raw_history_df = preview_df
        if st.button("Run TimesFM Forecast", type="primary", key="run_timesfm"):
            try:
                validated_history = validate_timesfm_input(
                    raw_history_df,
                    timestamp_column=timestamp_column,
                    value_column=value_column,
                )
                xreg_ticker: str | None = None
                single_covariates_df: pd.DataFrame | None = None
                single_covariate_columns: list[str] | None = None
                if use_xreg_main:
                    xreg_ticker = single_ticker.strip().upper()
                    if not xreg_ticker:
                        raise ValidationError("Single run ticker is required when XReg is enabled.")
                    single_panel_df = validated_history.assign(ticker=xreg_ticker)[
                        ["timestamp", "ticker", "value"]
                    ]
                    override_covariates_df = _load_covariate_override_csv(
                        override_covariate_file_main
                    )
                    xreg_panel_df, covariate_columns = _apply_timesfm_xreg_covariates(
                        panel_df=single_panel_df,
                        use_xreg=use_xreg_main,
                        macro_series_raw=macro_series_raw_main,
                        sector_tickers_raw=sector_tickers_raw_main,
                        fred_api_key=fred_api_key_main,
                        override_covariates_df=override_covariates_df,
                    )
                    single_covariate_columns = covariate_columns
                    single_covariates_df = xreg_panel_df[
                        ["timestamp", *covariate_columns]
                    ].copy()
                    _register_artifact(
                        "timesfm_single_xreg_covariates.csv",
                        _to_csv_bytes(single_covariates_df),
                    )
                with st.spinner("Running TimesFM..."):
                    result = run_timesfm_forecast(
                        history_df=validated_history,
                        horizon=int(horizon),
                        max_context=int(max_context),
                        max_horizon=int(max_horizon),
                        normalize_inputs=settings.normalize_inputs,
                        include_quantiles=include_quantiles,
                        model_id=selected_model_id,
                        backend=selected_backend,
                        use_xreg=use_xreg_main,
                        ticker=xreg_ticker,
                        covariates_df=single_covariates_df,
                        covariate_columns=single_covariate_columns,
                        xreg_mode=xreg_mode_main,
                        lora_adapter_path=selected_lora_adapter_path,
                    )

                st.success("TimesFM forecast completed.")
                st.dataframe(result.forecast_df, use_container_width=True)
                _set_pdf_context(
                    context_key="timesfm_single",
                    context_payload={
                        "report_type": "single",
                        "model_family": "timesfm",
                        "horizon": int(horizon),
                        "forecast_rows": len(result.forecast_df),
                        "backtest_enabled": run_backtest,
                        "insight_available": False,
                    },
                )
                _register_artifact("timesfm_history_input.csv", _to_csv_bytes(validated_history))
                _register_artifact("timesfm_forecast.csv", _to_csv_bytes(result.forecast_df))
                _render_timesfm_forecast_chart(
                    history_df=validated_history,
                    forecast_df=result.forecast_df,
                    chart_mode=chart_mode,
                    include_quantiles=include_quantiles,
                )

                st.download_button(
                    label="Download TimesFM Forecast CSV",
                    data=_to_csv_bytes(result.forecast_df),
                    file_name="timesfm_forecast.csv",
                    mime="text/csv",
                    key="download_timesfm_forecast",
                )

                if run_backtest:
                    backtest = run_timesfm_backtest(
                        history_df=validated_history,
                        holdout_points=int(holdout_points),
                        max_context=int(max_context),
                        max_horizon=int(max_horizon),
                        normalize_inputs=settings.normalize_inputs,
                        include_quantiles=include_quantiles,
                        model_id=selected_model_id,
                        backend=selected_backend,
                        use_xreg=use_xreg_main,
                        ticker=xreg_ticker,
                        covariates_df=single_covariates_df,
                        covariate_columns=single_covariate_columns,
                        xreg_mode=xreg_mode_main,
                        lora_adapter_path=selected_lora_adapter_path,
                    )
                    st.markdown("**Backtest (Holdout)**")
                    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                    met_col1.metric("MAE", f"{float(backtest.metrics['mae']):.4f}")
                    met_col2.metric("RMSE", f"{float(backtest.metrics['rmse']):.4f}")
                    met_col3.metric("MSE", f"{float(backtest.metrics['mse']):.4f}")
                    met_col4.metric("MAPE %", f"{float(backtest.metrics['mape']):.2f}")
                    met_col5, met_col6, met_col7, met_col8 = st.columns(4)
                    met_col5.metric("sMAPE %", f"{float(backtest.metrics['smape']):.2f}")
                    met_col6.metric("WAPE %", f"{float(backtest.metrics['wape']):.2f}")
                    met_col7.metric(
                        "Directional Accuracy %",
                        f"{float(backtest.metrics['directional_accuracy']):.2f}",
                    )
                    qce_value = backtest.metrics.get("quantile_coverage_error")
                    qce_display = (
                        "N/A"
                        if qce_value is None or pd.isna(qce_value)
                        else f"{float(qce_value):.2f} pp"
                    )
                    met_col8.metric("QCE (p10/p90)", qce_display)
                    st.dataframe(backtest.comparison_df, use_container_width=True)
                    _render_timesfm_backtest_charts(
                        comparison_df=backtest.comparison_df,
                        chart_mode=chart_mode,
                    )
                    _register_artifact("timesfm_backtest.csv", _to_csv_bytes(backtest.comparison_df))
                    _register_artifact(
                        "timesfm_backtest_metrics.json",
                        json.dumps(backtest.metrics, indent=2).encode("utf-8"),
                    )

                insight_text = ""
                if use_ollama and ollama_selection.selected_model is not None:
                    try:
                        client = OllamaClient(
                            base_url=settings.ollama_url,
                            model_name=ollama_selection.selected_model or settings.ollama_model,
                        )
                        insight_text = client.generate_insight(
                            _timesfm_insight_prompt(result.forecast_df, int(horizon))
                        )
                        st.markdown("**Ollama Insight**")
                        st.success(insight_text)
                        _register_artifact("timesfm_insight.txt", insight_text.encode("utf-8"))
                        if can_export_insight(insight_text):
                            section_tables: list[tuple[str, pd.DataFrame]] = [
                                ("Forecast", result.forecast_df)
                            ]
                            metrics_payload: dict[str, object] = {}
                            if run_backtest:
                                metrics_payload = {k.upper(): v for k, v in backtest.metrics.items()}
                                section_tables.append(("Backtest Comparison", backtest.comparison_df))
                            chart_png = _build_timesfm_chart_or_none(
                                logger=logger,
                                history_df=validated_history,
                                forecast_df=result.forecast_df,
                            )
                            section = InsightPdfSection(
                                title="TimesFM Forecast Insight",
                                insight_text=insight_text,
                                metadata={
                                    "Run Type": "TimesFM Single",
                                    "Horizon": int(horizon),
                                    "Backtest enabled": run_backtest,
                                },
                                tables=section_tables,
                                metrics=metrics_payload,
                                chart_png=chart_png,
                            )
                            insight_pdf = _build_insight_pdf(
                                title="TimesFM Insight Report",
                                metadata={
                                    "Ollama model": ollama_selection.selected_model,
                                    "Model source": ollama_selection.source,
                                },
                                sections=[section],
                                settings=settings,
                            )
                            _register_artifact("timesfm_insight_report.pdf", insight_pdf)
                            st.download_button(
                                label="Download Insight PDF",
                                data=insight_pdf,
                                file_name="timesfm_insight_report.pdf",
                                mime="application/pdf",
                                key="download_timesfm_single_insight_pdf",
                            )
                            _register_pdf_export(
                                export_key="timesfm_single_pdf",
                                label="Download Latest TimesFM Insight PDF",
                                file_name="timesfm_insight_report.pdf",
                                data=insight_pdf,
                                mime="application/pdf",
                            )
                            _set_pdf_context(
                                context_key="timesfm_single",
                                context_payload={
                                    "report_type": "single",
                                    "model_family": "timesfm",
                                    "horizon": int(horizon),
                                    "forecast_rows": len(result.forecast_df),
                                    "backtest_enabled": run_backtest,
                                    "insight_available": True,
                                },
                            )
                    except OllamaError as exc:
                        logger.warning("TimesFM insight generation failed: %s", exc)
                        st.warning(f"Ollama insight unavailable: {exc}")

                if not can_export_insight(insight_text):
                    _render_export_disabled_message("Generate an Ollama insight and rerun.")

                append_markdown_run(
                    output_path=settings.output_markdown_path,
                    section_title="TimesFM Run",
                    body_markdown=(
                        f"Adapter: {selected_lora_adapter_label}\n\n"
                        f"```text\n{result.forecast_df.to_string(index=False)}\n```\n\n"
                        f"### Ollama Insight\n{insight_text or 'Not generated.'}\n"
                    ),
                )
            except (ValidationError, TimesFMRuntimeError, CovariateProviderError) as exc:
                logger.error("TimesFM validation/runtime error: %s", exc)
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover - UI guardrail
                logger.exception("Unexpected TimesFM failure")
                st.error(f"Unexpected TimesFM error: {exc}")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger("unified_app")

    st.set_page_config(
        page_title="TabFM + TimesFM Unified Analyzer",
        layout="wide",
        page_icon="📊",
    )
    st.title("📊 TabFM + TimesFM Unified Analyzer")
    st.markdown(
        "Zero-shot **tabular prediction** with TabFM and **time-series forecasting** "
        "with TimesFM, plus optional local Ollama insights."
    )
    ollama_selection = _render_ollama_model_selector(settings=settings, logger=logger)
    _render_health_panel(settings)

    tab_tabfm, tab_timesfm = st.tabs(["TabFM Predictions", "TimesFM Forecasting"])
    with tab_tabfm:
        _render_tabfm_tab(settings=settings, logger=logger, ollama_selection=ollama_selection)
    with tab_timesfm:
        _render_timesfm_tab(settings=settings, logger=logger, ollama_selection=ollama_selection)

    _render_artifact_download(settings=settings, ollama_selection=ollama_selection)

    active_ollama_model = ollama_selection.selected_model or "Not configured"

    st.caption(
        f"Outputs are appended to: `{settings.output_markdown_path}` | "
        f"Ollama model: `{active_ollama_model}`"
    )


if __name__ == "__main__":
    main()
