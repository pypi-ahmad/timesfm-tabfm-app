# Zero To Mastery Tutorial (TabFM + TimesFM)

This tutorial takes you from zero setup to reliable end-to-end runs, including:
- TabFM classification + regression (single + batch)
- TimesFM forecasting + backtesting (single + batch)
- Optional: XReg covariates, advanced multi-asset panel runs, LoRA, and Ollama insights

## 1) Setup

### 1.1 System prerequisites
- Linux
- Python 3.11+ (recommended: 3.12)
- `uv`

### 1.2 Create/Sync the environment (model deps included)

This repo defaults caches to `/tmp` because some environments mount `~/.cache` read-only.

```bash
cd TimesFM_TabFM_App
./scripts/setup_env.sh
```

Optional extras:
```bash
./scripts/setup_env.sh --models-xreg   # TimesFM XReg
./scripts/setup_env.sh --models-jax    # TimesFM JAX backend
./scripts/setup_env.sh --finetune      # LoRA fine-tuning dependencies
```

### 1.3 Download example datasets

```bash
uv run python scripts/download_datasets.py --all
```

Tiny samples live in `data/samples/` and are used throughout this tutorial.

### 1.4 Warm model weights (first run only)

This performs a tiny real inference to force model weights to download.

```bash
uv run python scripts/warm_models.py
```

## 2) Run The App

```bash
uv run streamlit run app.py
```

Open the printed URL (usually `http://localhost:8501`).

## 3) TabFM: Classification (Single)

In the app:
- Go to **TabFM Predictions**
- Upload:
  - Training CSV: `data/samples/tabular/iris/train.csv`
  - Prediction CSV: `data/samples/tabular/iris/predict.csv`
- Select target column: `target`
- Click **Run TabFM Prediction**

Expected outcome:
- A predictions table
- Download button for `tabfm_predictions.csv`

## 4) TabFM: Regression (Single)

Use:
- Training: `data/samples/tabular/wine_quality_red/train.csv`
- Predict: `data/samples/tabular/wine_quality_red/predict.csv`
- Target: `target`

## 5) TabFM: Batch Mode

Batch mode expects:
- One training CSV
- Many prediction CSVs (multi-upload) or a ZIP of CSVs

Tip: create a ZIP from multiple `predict.csv` files if you want to simulate batch quickly.

## 6) TimesFM: Forecast (Single)

Use:
- History: `data/samples/timeseries/air_passengers/history.csv`
- Timestamp column: `timestamp`
- Value column: `value`
- Horizon: 24 (or any positive integer)

Run:
- Click **Run TimesFM Forecast**

Expected:
- Forecast dataframe with `timestamp,prediction` (+ optional `p10,p90`)
- Forecast chart and CSV download

## 7) TimesFM: Backtest

Enable:
- **Run holdout backtest**
- Holdout points: e.g. 12

Expected:
- Comparison table with actual vs prediction
- Metrics panel (MAE/RMSE/MAPE etc.)

## 8) Advanced (Optional)

### 8.1 XReg covariates
Enable **TimesFM XReg Covariates** and provide:
- a ticker (single mode)
- FRED API key if you want macro series fetch
- optional override CSV for full control

### 8.2 Multi-asset panel forecasting
Use the **Advanced TimesFM Features** expander to run multi-ticker panel forecasts and portfolio optimization.

### 8.3 LoRA fine-tuning
LoRA is optional and requires `--finetune`. The app supports adapter registry and adapter-aware inference.

## 9) CLI Smoke Test (Recommended For CI/Debugging)

Runs real model inference against committed samples:

```bash
uv run python scripts/smoke_e2e.py
```

Outputs are written under `artifacts/smoke/` (gitignored).

## Troubleshooting

- If model imports fail: run `./scripts/setup_env.sh` again.
- If you see cache write errors: ensure `XDG_CACHE_HOME=/tmp/xdg-cache` and `HF_HOME=/tmp/hf`.
- If the app starts but models are missing: the **Runtime health** panel shows import status for `tabfm` and `timesfm`.
