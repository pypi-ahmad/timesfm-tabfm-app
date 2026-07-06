# TabFM + TimesFM Unified Streamlit App

Unified local app for:
- **TabFM**: zero-shot tabular classification/regression.
- **TimesFM**: univariate forecasting with configurable horizon/context.
- **Ollama**: optional local natural-language summaries of model outputs.

**Zero-to-mastery tutorial:** see `docs/tutorial/zero_to_mastery.md`.

## Features
- Two workflow tabs in one app: `TabFM Predictions` and `TimesFM Forecasting`.
- CSV-driven TabFM flow:
  - Training CSV: feature columns + target column.
  - Prediction CSV: feature columns only.
  - Batch mode: one training CSV + many prediction CSVs (multi-upload or ZIP).
- CSV-driven TimesFM flow:
  - History CSV with `timestamp` + `value` columns.
  - Configurable horizon, max context, max horizon, optional p10/p90 intervals.
  - Batch mode: many history CSVs (multi-upload or ZIP), sequential processing.
  - Optional XReg covariates (single + batch): macro indicators (FRED), sector indicators (Yahoo Finance), and optional override CSV.
  - Single-mode XReg requires a ticker input; batch-mode XReg infers ticker from CSV filename stem.
- Backend-aware model execution:
  - TimesFM backend toggle in UI (`PyTorch` / `JAX`) for single + batch runs.
  - TabFM backend surfaced in UI as PyTorch-only with explicit disabled JAX state.
- Advanced TimesFM-only features:
  - Multi-asset panel forecasting with optional XReg covariates.
  - Correlated ticker panel support for multi-asset runs.
  - Portfolio optimization (mean-variance, long-only, max-weight constrained).
  - Portfolio diagnostics (expected returns + covariance matrix) shown in UI and exported as artifacts.
  - Real-time sentiment bias (Alpha Vantage NEWS_SENTIMENT) with configurable strength/decay.
  - Sentiment diagnostics with fail-open fallback (forecast continues with neutral sentiment when feed fetch fails).
  - Framework backtesting modes: walk-forward and rolling-window, with explicit historical validation-window comparison reports (TimesFM vs naive vs seasonal_naive) and aggregate MAE/RMSE/**MSE**/MAPE/sMAPE/WAPE/directional-accuracy/QCE.
  - Managed local LoRA fine-tuning with two execution paths:
    - External script runner (`finetune_lora.py` compatible).
    - In-app trainer runner (`python -m app_core.timesfm_lora_runner`).
  - Transactional LoRA dataset mapping (timestamp/value + optional entity/features), deterministic train/validation split, and dataset fingerprinting.
  - Adapter registry with selectable adapter-aware inference in TimesFM forecast/backtest/batch flows.
- Batch execution policy:
  - Max 25 parsed CSV files per run.
  - One automatic retry per failed file.
  - Per-file status summary with success/failure and attempt counts.
- Optional batch enrichments (default OFF) via dedicated "Configure Batch Enrichment" controls:
  - Per-file Ollama summaries.
  - Per-file TimesFM backtesting.
- Sidebar Ollama model selector with refresh:
  - Auto-discovers local models from `/api/tags`.
  - Falls back to `APP_OLLAMA_MODELS` when local discovery is unavailable.
  - Persists selected model for the session and applies it across all tabs.
- Downloadable prediction/forecast CSV outputs.
- PDF export for AI insights:
  - Single runs: instant full-report PDF download near each generated insight.
  - Batch runs: combined report PDF plus per-file insight PDFs in ZIP.
  - TimesFM PDFs include forecast charts; TabFM PDFs include text/tables/metrics.
- Dynamic TimesFM charts with Plotly:
  - `Chart mode` switch (`Plotly` or `Streamlit`) in TimesFM controls.
  - Single runs: interactive forecast chart with optional p10/p90 shaded band.
  - Backtests: interactive actual-vs-prediction and residual charts.
  - Batch runs: per-file selector for interactive forecast/backtest chart preview.
- Holdout backtesting panel (MAE/RMSE/MSE/MAPE/sMAPE/WAPE/directional-accuracy/QCE) for TimesFM.
- Runtime health panel for `tabfm`, `timesfm`, and Ollama availability.
- Downloadable ZIP artifact bundle with run outputs and manifest metadata.
- Run artifacts appended to `outputs.md`.
- Environment-based config through `.env`.

## Project Layout
- `app.py`: Streamlit UI orchestration and tabs.
- `src/app_core/`: config, logging, validation, model services, output writer.
- `tests/`: validator/service/Ollama unit tests.
- `pyproject.toml`: canonical dependency + pytest config.
- `requirements.txt`: compatibility list for environments that still need it.

## Setup (uv)
```bash
./scripts/setup_env.sh
```

Install model backends:
```bash
./scripts/setup_env.sh --models
```

Optional JAX runtime for TimesFM:
```bash
./scripts/setup_env.sh --models-jax
```

Optional XReg runtime for TimesFM:
```bash
./scripts/setup_env.sh --models-xreg
```

Optional LoRA fine-tuning runtime:
```bash
./scripts/setup_env.sh --finetune
```

Pull local Ollama model (optional but recommended):
```bash
ollama pull qwen3:4b
```
If no local models are found and `APP_OLLAMA_MODELS` is empty, Ollama insight controls are disabled in the UI.

## Run
```bash
uv run streamlit run app.py
```

## Datasets (3 tabular + 3 time-series)

This repo includes tiny samples under `data/samples/` and provides a downloader/normalizer:

```bash
uv run python scripts/download_datasets.py --all
```

Data layout and input contracts are documented in `data/README.md`.

## Warm Model Weights (First Run)

```bash
uv run python scripts/warm_models.py
```

## E2E Smoke Check (Real Inference)

```bash
uv run python scripts/smoke_e2e.py
```

## Test
```bash
./.venv/bin/python -m pytest -q
```

## Environment Variables
Create `.env` (or copy from `.env.example`) and override defaults as needed:

```bash
APP_OLLAMA_URL=http://localhost:11434/api/generate
APP_OLLAMA_MODEL=qwen3:4b
APP_OLLAMA_MODELS=qwen3:4b,llama3.2:3b,mistral:7b
APP_DEFAULT_BACKEND=torch
APP_TIMESFM_MODEL_ID=google/timesfm-2.5-200m-pytorch
APP_TIMESFM_JAX_MODEL_ID=google/timesfm-2.5-200m-flax
APP_TIMESFM_XREG_MODE=xreg + timesfm
APP_FRED_API_KEY=
APP_ALPHA_VANTAGE_API_KEY=
APP_COVARIATE_DEFAULT_MACRO_IDS=CPIAUCSL,UNRATE
APP_COVARIATE_DEFAULT_SECTOR_TICKERS=XLK,XLF,XLV
APP_SENTIMENT_LOOKBACK_HOURS=24
APP_SENTIMENT_BIAS_STRENGTH=0.1
APP_SENTIMENT_BIAS_DECAY=0.95
APP_PORTFOLIO_RISK_AVERSION=1.0
APP_PORTFOLIO_MAX_WEIGHT=0.4
APP_BACKTEST_DEFAULT_MODE=walk_forward
APP_BACKTEST_DEFAULT_FOLDS=3
APP_BACKTEST_MIN_TRAIN_SIZE=40
APP_BACKTEST_ROLLING_WINDOW=120
APP_TIMESFM_LORA_SCRIPT_PATH=timesfm-forecasting/examples/finetuning/finetune_lora.py
APP_TIMESFM_LORA_REGISTRY_PATH=.timesfm/lora_jobs_registry.json
APP_TIMESFM_LORA_ADAPTER_REGISTRY_PATH=.timesfm/lora_adapters_registry.json
APP_TIMESFM_LORA_OUTPUT_ROOT=.timesfm/lora_runs
APP_TIMESFM_LORA_DEFAULT_MODE=external_script
APP_TIMESFM_LORA_RETENTION_POLICY=delete_raw
APP_TIMESFM_LORA_MIN_POINTS_PER_ENTITY=20
APP_TIMESFM_LORA_VALIDATION_RATIO=0.2
APP_DEFAULT_FORECAST_HORIZON=24
APP_DEFAULT_MAX_CONTEXT=1024
APP_DEFAULT_MAX_HORIZON=256
APP_NORMALIZE_INPUTS=true
APP_BATCH_MAX_FILES=25
APP_BATCH_RETRY_COUNT=1
APP_PDF_TABLE_MAX_ROWS=100
APP_PDF_FONT_SIZE=10
APP_OUTPUT_MARKDOWN_PATH=outputs.md
APP_LOG_LEVEL=INFO
```

## Notes
- TabFM dependency is installed from the official Google Research GitHub repo.
- TimesFM dependency uses the `timesfm` PyPI package (`>=2.0.2`).
- TabFM currently runs on PyTorch backend only.
- JAX backend is optional and applies to TimesFM paths (`timesfm[flax]`).
- XReg forecasting requires `timesfm[xreg]`.
- Sentiment bias requires a valid Alpha Vantage API key.
- Sentiment artifacts include feed CSV (when available), per-ticker scores, and diagnostics JSON.
- LoRA jobs support both external scripts and in-app execution mode.
- For external mode, verify `APP_TIMESFM_LORA_SCRIPT_PATH` exists in your environment.
- Default proprietary-data behavior is `APP_TIMESFM_LORA_RETENTION_POLICY=delete_raw` (raw upload removed after materialization).
- If model dependencies are missing, the app surfaces actionable runtime errors per tab.

## Cache Locations (Important For Sandboxed Environments)

Some environments mount `~/.cache` read-only. The setup script defaults these to `/tmp`:
Some environments also apply small `/tmp` quotas. The setup script defaults caches under the repo:
- `UV_CACHE_DIR=./.uv-cache`
- `XDG_CACHE_HOME=./.cache`
- `HF_HOME=./.cache/hf`
- `HUGGINGFACE_HUB_CACHE=./.cache/hf/hub`
